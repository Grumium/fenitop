# SPDX-License-Identifier: GPL-3.0-or-later
"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7

Modified 2026 by Jan-David Förster for the TopoloGUI fork
(https://github.com/Grumium/fenitop):
solver tuning (GAMG defaults, near-nullspace, warm starts, adaptive
rebuilds), Dirichlet diagonal rescaling, MPC support, PETSc feature
fallback, VTKHDF export, Plotter removed, DOLFINx 0.11 port.
Full list of deviations from upstream: CHANGES_FROM_FENITOP.md
in https://github.com/Grumium/topologui
"""

import os
import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI as _MPI
from petsc4py import PETSc
import dolfinx.io
from dolfinx.fem import form, Function
from dolfinx import la
from dolfinx.fem.petsc import (create_vector, create_matrix,
                               assemble_vector, assemble_matrix, set_bc)

def create_mechanism_vectors(func_space, in_spring, out_spring, dof_coords=None, out_sign=1):
    """Create vectors for compliant mechanism design.

    Parameters
    ----------
    dof_coords : ndarray, optional
        Pre-computed ``func_space.tabulate_dof_coordinates()[:num_local]``.
        Avoids a redundant call when the caller already has them.
    out_sign : int, optional
        Sign for the output (effector) direction in ``l_vec``.
        +1 (default) maximises displacement in the positive axis direction,
        -1 maximises displacement in the negative direction.
    """
    index_map = func_space.dofmap.index_map
    block_size = func_space.dofmap.index_map_bs
    spring_vec_wrap = la.vector(index_map, block_size)
    l_vec_wrap = la.vector(index_map, block_size)
    
    # Get PETSc vectors for MPI-safe setValues operations
    spring_vec = spring_vec_wrap.petsc_vec
    l_vec = l_vec_wrap.petsc_vec

    local_range = index_map.local_range
    local_indices = np.arange(local_range[0], local_range[1]).astype(np.int32)
    num_local_nodes = index_map.size_local
    if dof_coords is None:
        local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]
    else:
        local_nodes = dof_coords

    for n, (locator, direction, value) in enumerate([in_spring, out_spring]):
        ctrl_nodes = local_indices[locator(local_nodes.T)]
        offset = ["x", "y", "z"].index(direction)
        ctrl_dofs = ctrl_nodes*block_size + offset
        # Use PETSc setValues for MPI-safe global index handling
        spring_vec.setValues(ctrl_dofs, [value,]*ctrl_dofs.size)
        if n == 1:
            l_vec.setValues(ctrl_dofs, [float(out_sign),]*ctrl_dofs.size)

    spring_vec.assemble()
    l_vec.assemble()
    return spring_vec_wrap, l_vec_wrap


def petsc_has_package(name):
    """True when this PETSc build provides *name* (mumps, hypre, superlu_dist).

    Returns True when the question cannot be answered, so an old petsc4py
    never blocks a solver that would in fact have worked.
    """
    try:
        return bool(PETSc.Sys.hasExternalPackage(name))
    except Exception:
        return True


#: Ordered fallbacks for solvers this PETSc build may not have been compiled
#: with.  A stand-alone run against a cluster's PETSc module is the case that
#: matters: without this, an unavailable package surfaces as an opaque error
#: deep inside KSPSetUp, which is a poor way to lose an overnight job.
_PC_FALLBACKS = {"hypre": ("gamg",)}
_FACTOR_FALLBACKS = {"superlu_dist": ("mumps", "petsc"),
                     "mumps": ("petsc",)}


def resolve_petsc_options(petsc_options, comm=None):
    """Downgrade solver choices this PETSc build cannot honour.

    Returns a copy; the input is left alone.  Emits one line on rank 0 naming
    the missing package and the fallback taken.  A no-op when everything the
    caller asked for is available, so a fully-featured build behaves exactly as
    before.
    """
    opts = dict(petsc_options)

    def _note(msg):
        if comm is None or comm.rank == 0:
            print(f"  ⚠️  {msg}", flush=True)

    pc = opts.get("pc_type")
    if pc in _PC_FALLBACKS and not petsc_has_package(pc):
        for cand in _PC_FALLBACKS[pc]:
            if cand == "gamg" or petsc_has_package(cand):
                _note(f"PETSc has no '{pc}'; falling back to pc_type={cand}.")
                opts["pc_type"] = cand
                break

    factor = opts.get("pc_factor_mat_solver_type")
    if factor and factor != "petsc" and not petsc_has_package(factor):
        for cand in _FACTOR_FALLBACKS.get(factor, ("petsc",)):
            if cand == "petsc" or petsc_has_package(cand):
                _note(f"PETSc has no '{factor}'; falling back to "
                      f"pc_factor_mat_solver_type={cand}.")
                if cand == "petsc":
                    opts.pop("pc_factor_mat_solver_type", None)
                else:
                    opts["pc_factor_mat_solver_type"] = cand
                break

    return opts


def _solver_defaults(petsc_options):
    """Preconditioner settings a topology optimization needs and PETSc's own
    defaults do not provide.  Returned rather than applied, so the caller can
    let explicit user options win and report only what it actually used.
    """
    if petsc_options.get("pc_type") == "gamg":
        # When beta doubles (Heaviside sharpening), material contrast jumps
        # and PETSc's GAMG defaults (V-cycle, 0 agg smooths, threshold 0)
        # produce poor coarse-grid operators -> KSP iterations explode.  This
        # is the standard recipe for SIMP-based topology optimization.
        return {
            "pc_gamg_agg_nsmooths": 1,     # smoother aggregation (default 0)
            "pc_gamg_threshold": 0.02,     # less aggressive coarsening
            "mg_levels_ksp_max_it": 2,     # extra smoother sweeps per level
            "pc_mg_cycle_type": "w",       # W-cycle for robustness
            "ksp_max_it": 500,             # cap runaway solves (default 10000)
        }
    if petsc_options.get("pc_type") == "hypre":
        # BoomerAMG's scalar defaults ignore that our unknowns come in blocks
        # of gdim displacement components per node.  Nodal coarsening plus
        # interpolation that preserves the rigid-body modes is the elasticity
        # recipe, and is hypre's counterpart to the near-nullspace handed to
        # GAMG below.
        #
        # Measured on beam_3d, serial, against cg-gamg at 2.91 s/iter: plain
        # BoomerAMG 19.2 (6.6x), these options 13.7 (4.5x), these options plus
        # vec_interp_variant and a near-nullspace 47.8 (16x).  So nodal
        # coarsening earns its place and RBM-preserving interpolation does
        # not.  hypre still loses to GAMG by 4.5x here and is deliberately NOT
        # offered in the GUI solver menu; these defaults exist for anyone who
        # selects it explicitly.
        return {
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_nodal_coarsen": 6,       # block coarsening
            "pc_hypre_boomeramg_strong_threshold": 0.7,  # 3D elasticity
            "pc_hypre_boomeramg_agg_nl": 1,
            "ksp_max_it": 500,
        }
    return {}


def _apply_solver_options(solver, mat, prefix, petsc_options, defaults):
    """Push *defaults*, then *petsc_options* over them, under *prefix*."""
    solver.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    opts.prefixPush(prefix)
    for key, value in defaults.items():
        if key not in petsc_options:
            opts[key] = value
    for key, value in petsc_options.items():
        opts[key] = value
    opts.prefixPop()
    solver.setFromOptions()
    mat.setOptionsPrefix(prefix)
    mat.setFromOptions()


def _dirichlet_rows(bcs):
    """Every local row the Dirichlet conditions in *bcs* hold at zero."""
    indices = []
    for bc in bcs:
        try:
            dofs, _ = bc.dof_indices()
            indices.append(np.asarray(dofs, dtype=np.int32))
        except Exception:
            pass
    return (np.unique(np.concatenate(indices)) if indices
            else np.empty(0, dtype=np.int32))


def _constrained_rows(bcs, func_space, mpc_slaves=None):
    """Owned local rows whose diagonal `_rescale_constrained_rows` may touch.

    Ghost entries are dropped: the diagonal vector only covers owned rows.
    """
    block_size = func_space.dofmap.index_map_bs
    n_owned = func_space.dofmap.index_map.size_local*block_size

    rows = _dirichlet_rows(bcs)
    if rows.size:
        rows = rows[rows < n_owned]
        # Keep only *partially* constrained nodes.  A full clamp binds every
        # component of its node, leaving a clean identity block that GAMG
        # handles without complaint -- rescaling those rows costs time and
        # buys nothing (beam_3d without symmetry: 9 Krylov iterations either
        # way, but 5.6% slower).  The damage comes from a node with some
        # components bound and others free, which is what a symmetry roller
        # produces, and what leaves a mixed block that defeats block
        # aggregation.
        if rows.size and block_size > 1:
            nodes, counts = np.unique(rows//block_size, return_counts=True)
            partial = set(nodes[counts < block_size].tolist())
            rows = rows[[int(d)//block_size in partial for d in rows]]

    # MPC slave rows bypass the test above.  A C4 constraint couples every
    # component of its node, so the node looks "fully constrained" while being
    # nothing like a clamp -- it is a coupling, not an identity block.
    # Measured no benefit on shell_3d, but the quadcopter with C4 is
    # unmeasured, so they stay in rather than be excluded on a guess.
    if mpc_slaves is not None:
        slaves = np.asarray(mpc_slaves, dtype=np.int32)
        slaves = slaves[slaves < n_owned]
        if slaves.size:
            rows = np.unique(np.concatenate([rows, slaves]))
    return rows


def _rescale_constrained_rows(mat, rows, comm):
    """Put the constrained rows on the same scale as the physical ones.

    dolfinx writes 1.0 into constrained rows, but SIMP scales the physical
    stiffness by rho^p, so the real diagonal sits near 1e-2 early on and
    drifts over orders of magnitude as the density evolves.  That mismatch
    wrecks GAMG's coarsening, and it gets worse the more of the boundary is
    constrained: on a symmetry-reduced beam it cost **65** Krylov iterations
    per solve instead of 8, which made exploiting symmetry *slower* than
    solving the whole domain despite halving the mesh.

    Every Dirichlet value here is zero, so ``set_bc`` writes 0 into the
    right-hand side and the constrained unknown comes out 0 for any non-zero
    diagonal.  The value is therefore free, and matching it to the physical
    scale costs one O(n) pass over the diagonal.
    """
    # NOTE: the reduction below is collective, so every rank must reach it.
    # Ranks owning no constrained rows still take part and contribute 0.0 --
    # returning early here deadlocks, because whether a rank holds any
    # Dirichlet dof depends on the partition.
    diag = mat.getDiagonal()
    arr = diag.array_w

    local = 0.0
    if arr.size:
        mask = np.ones(arr.size, dtype=bool)
        if rows.size:
            mask[rows] = False
        if mask.any():
            local = float(np.median(np.abs(arr[mask])))

    local_bc = float(np.median(np.abs(arr[rows]))) if rows.size else 0.0

    scale = comm.allreduce(local, op=_MPI.MAX)
    bc_scale = comm.allreduce(local_bc, op=_MPI.MAX)

    # Only ever scale the constrained rows *down*.  A constrained diagonal far
    # above the physical one breaks GAMG's aggregation -- that is the beam_3d
    # case, 63 Krylov iterations against 8.  Far *below* is harmless and
    # apparently even helpful: GAMG treats such a row as strongly constrained
    # and leaves it out of the coarse space.  Raising it measurably hurt the
    # quadcopter, whose E is 700x beam_3d's, at 10 iterations against 11.5.
    # So this triggers on the damaging direction only.
    if scale > 0.0 and rows.size and bc_scale > scale:
        arr[rows] = scale
        mat.setDiagonal(diag)
    diag.destroy()


def _set_rigid_body_modes(mat, func_space, constrained=None):
    """Set the near-nullspace (rigid body modes) on the matrix for GAMG.

    For 3D elasticity: 6 modes (3 translations + 3 rotations).
    For 2D elasticity: 3 modes (2 translations + 1 rotation).

    GAMG *requires* this information to build a good coarsening hierarchy --
    without it, convergence degrades catastrophically as material contrast
    grows.

    *constrained* lists the rows the Dirichlet conditions hold at zero, and
    they are zeroed in every mode.  This matters: a near-nullspace vector has
    to be one the operator *nearly annihilates*, and the constrained operator
    does no such thing to a mode that moves the rows it pins.  A clamp on one
    small face barely disturbs a rigid-body mode, but a symmetry roller pins
    one component along a whole mirror plane and removes a translation
    outright.  Handed the raw modes there, GAMG built its coarse space around
    the wrong subspace: measured on a high-contrast half beam, CG stopped with
    an indefinite preconditioner after six iterations and returned a field
    with a relative residual of **11.7**, which nothing downstream checked --
    the compliance came out 0.09% wrong at moderate contrast and 1.1% wrong at
    high contrast, silently.  With the modes projected it agrees with a direct
    solve to 5e-8.  A mode the constraints destroy entirely is dropped rather
    than handed over as round-off.

    GAMG only, deliberately.  Handing the same modes to hypre via BoomerAMG's
    vec_interp_variant measured *worse*: beam_3d went from 13.7 to 47.8
    s/iter.  Building interpolation that preserves six vectors is expensive
    setup, and topology optimization rebuilds the preconditioner every
    iteration (the density changes, so the matrix changes), so that setup is
    never amortized.
    """
    dim = func_space.mesh.geometry.dim
    bs = func_space.dofmap.index_map_bs
    num_local = func_space.dofmap.index_map.size_local
    coords = func_space.tabulate_dof_coordinates()[:num_local]

    def _make_vec(values_per_node):
        """Create a PETSc Vec and fill it from a (num_local, bs) array."""
        vec = mat.createVecLeft()
        arr = vec.getArray(readonly=False)
        arr[:num_local*bs] = values_per_node.ravel()
        return vec

    rows = np.empty(0, dtype=np.int32)
    if constrained is not None:
        rows = np.asarray(constrained, dtype=np.int32)
        rows = rows[rows < num_local*bs]

    zero, one = np.zeros(num_local), np.ones(num_local)
    modes = []
    if dim == 3:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        # Translations: Tx, Ty, Tz
        modes.append(_make_vec(np.column_stack([one, zero, zero])))
        modes.append(_make_vec(np.column_stack([zero, one, zero])))
        modes.append(_make_vec(np.column_stack([zero, zero, one])))
        # Rotations: Rx=(0,-z,y), Ry=(z,0,-x), Rz=(-y,x,0)
        modes.append(_make_vec(np.column_stack([zero, -z, y])))
        modes.append(_make_vec(np.column_stack([z, zero, -x])))
        modes.append(_make_vec(np.column_stack([-y, x, zero])))
    else:
        x, y = coords[:, 0], coords[:, 1]
        # Translations: Tx, Ty
        modes.append(_make_vec(np.column_stack([one, zero])))
        modes.append(_make_vec(np.column_stack([zero, one])))
        # Rotation: Rz=(-y, x)
        modes.append(_make_vec(np.column_stack([-y, x])))

    kept = []
    for vi in modes:
        if rows.size:
            vi.getArray(readonly=False)[rows] = 0.0
        full = vi.norm()
        for vj in kept:
            vi.axpy(-vi.dot(vj), vj)
        norm = vi.norm()
        # Dropped when the constraints leave nothing of the mode that the
        # earlier ones do not already span -- keeping it would hand GAMG a
        # direction made of round-off.
        if norm > 1e-6*max(full, 1.0):
            vi.scale(1.0/norm)
            kept.append(vi)
        else:
            vi.destroy()

    if kept:
        mat.setNearNullSpace(PETSc.NullSpace().create(
            vectors=kept, comm=func_space.mesh.comm))


class _DivergenceFallback:
    """Rescue a Krylov solve that stopped without converging.

    A diverged KSP still writes something into the solution vector, and PETSc
    reports the fact only through the converged reason -- which nothing
    downstream inspects.  On a high-contrast design that silence is expensive:
    measured on a half beam with rho clipped to 0.01, CG stopped with an
    indefinite preconditioner after seven iterations and returned a field
    whose relative residual was **199**, and the compliance built from it came
    out 1.8% wrong with no indication that anything had happened.

    So a failed solve is retried rather than believed.  First with the
    preconditioner rebuilt from scratch and the guess reset -- an AMG
    hierarchy carried over from an earlier density is the usual reason -- and
    if that fails too, by factorizing the operator outright.  The direct solve
    is slow, but it is once, and it is right.
    """

    def __init__(self, comm, label):
        self.comm = comm
        self.label = label
        self.direct = None
        self.reported = False

    def solve(self, solver, mat, rhs, out):
        solver.solve(rhs, out)
        if solver.getConvergedReason() >= 0:
            return
        first = solver.getConvergedReason()

        out.zeroEntries()
        solver.getPC().reset()
        solver.setOperators(mat, mat)
        solver.solve(rhs, out)
        if solver.getConvergedReason() >= 0:
            self._report(f"the {self.label} solve stopped early (KSP reason "
                         f"{first}); rebuilding the preconditioner recovered "
                         f"it")
            return
        second = solver.getConvergedReason()

        self._direct(mat).solve(rhs, out)
        reason = self._direct(mat).getConvergedReason()
        if reason < 0:
            raise RuntimeError(
                f"the {self.label} solve did not converge (KSP reason "
                f"{first}, {second} after a preconditioner rebuild) and the "
                f"direct fallback failed as well (reason {reason})")
        self._report(f"the {self.label} solve stopped early (KSP reason "
                     f"{first}, {second} after a preconditioner rebuild); "
                     f"falling back to a direct solve, which is slower")

    def _direct(self, mat):
        """A factorizing solver for *mat*, built once and reused."""
        if self.direct is None:
            ksp = PETSc.KSP().create(self.comm)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            for package in ("mumps", "superlu_dist"):
                if petsc_has_package(package):
                    pc.setFactorSolverType(package)
                    break
            else:
                if self.comm.size > 1:
                    raise RuntimeError(
                        f"the {self.label} solve did not converge and this "
                        "PETSc build has neither MUMPS nor SuperLU_DIST, so "
                        "there is no direct fallback in parallel.  Rerun on "
                        "one rank, or install one of them.")
            self.direct = ksp
        self.direct.setOperators(mat, mat)
        return self.direct

    def _report(self, message):
        if not self.reported and self.comm.rank == 0:
            print(f"  ⚠️  {message}.", flush=True)
            self.reported = True

    def destroy(self):
        if self.direct is not None:
            self.direct.destroy()
            self.direct = None


class _ParitySolver:
    """The same stiffness form under a different set of Dirichlet conditions.

    Used for the basis loads that are antisymmetric about a mirror plane: the
    operator differs from the symmetric one only in which rows the boundary
    conditions clear, so it shares the form, the density and the mesh, and
    needs its own matrix, its own factorization and nothing else.

    It needs everything LinearProblem does to make an iterative solve behave:
    the preconditioner defaults, the rigid-body modes GAMG coarsens with, and
    the rescaling of the constrained diagonal.  Without them a CG/GAMG run
    stops with an indefinite preconditioner within a handful of iterations --
    and this operator is the more exposed of the two, because an antisymmetric
    class pins every component on the mirror plane *but* the normal one, so
    the whole plane carries the partially constrained node pattern that
    defeats block aggregation.

    Deliberately without the GAMG rebuild heuristics of LinearProblem: those
    track a history per operator, and an antisymmetric solve is the same
    problem at the same contrast as the symmetric one it accompanies, so the
    main operator's judgement carries over.
    """

    def __init__(self, lhs_form, bcs, petsc_options, func_space):
        self.lhs_form = lhs_form
        self.bcs = bcs
        self.func_space = func_space
        self.mat = create_matrix(lhs_form)
        comm = func_space.mesh.comm
        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(self.mat)
        _apply_solver_options(self.solver, self.mat,
                              f"parity_solver_{id(self)}", petsc_options,
                              _solver_defaults(petsc_options))
        self._iterative = petsc_options.get("ksp_type", "preonly") != "preonly"
        if self._iterative:
            self.solver.setInitialGuessNonzero(True)
            if petsc_options.get("pc_type") == "gamg":
                _set_rigid_body_modes(self.mat, func_space,
                                      _dirichlet_rows(bcs))
        self.fallback = _DivergenceFallback(comm, "antisymmetric")
        self.rows = _constrained_rows(bcs, func_space)
        self.needs_rescale = bool(comm.allreduce(self.rows.size > 0,
                                                 op=_MPI.LOR))

    def assemble(self, spring_vec=None):
        self.mat.zeroEntries()
        assemble_matrix(self.mat, self.lhs_form, bcs=self.bcs)
        self.mat.assemble()
        if spring_vec is not None:
            self.mat.setDiagonal(self.mat.getDiagonal() + spring_vec)
        if self.needs_rescale:
            _rescale_constrained_rows(self.mat, self.rows,
                                      self.func_space.mesh.comm)

    def solve(self, rhs_vec, u_out):
        self.fallback.solve(self.solver, self.mat, rhs_vec, u_out)

    def destroy(self):
        self.fallback.destroy()
        self.solver.destroy()
        self.mat.destroy()


class LinearProblem:
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=[], petsc_options={},
                 mpc=None, direction_rhs=None, direction_bcs=None,
                 direction_parity=None):
        """Initialize a linear problem.

        Parameters
        ----------
        mpc : dolfinx_mpc.MultiPointConstraint, optional
            If provided, assembly and solve use dolfinx_mpc routines to
            enforce multi-point constraints (e.g. diagonal symmetry slip BCs).
        direction_rhs : list of ufl.Form, optional
            Right-hand sides spanning the loads a direction-uncertain problem
            admits: the deterministic part, then a basis per uncertain load.
            They are solved instead of the nominal one in
            solve_fem() and land in `self.u_dir`.  Each costs one more Krylov
            solve or back-substitution, not another assembly or factorization:
            the operator is shared, only the load differs.
        direction_bcs : list of list, optional
            One boundary condition set per parity class, for a symmetric half
            domain carrying an uncertain load.  A load antisymmetric about a
            mirror plane has an antisymmetric response, which the roller
            condition of the symmetric case cannot represent, so those basis
            loads are solved against their own operator.  Only the boundary
            conditions differ, so the extra cost is one assembly and one
            factorization per class in use -- both on the reduced domain.
        direction_parity : list of int, optional
            Which of those sets each entry of `direction_rhs` belongs to.
        """
        # Downgrade any solver this PETSc build was not compiled with, before
        # the choice reaches KSPSetUp where it would fail opaquely.
        petsc_options = resolve_petsc_options(
            petsc_options, u.function_space.mesh.comm)

        # Initialization
        self.u, self.lam = u, lam
        self.u_wrap = self.u.x.petsc_vec
        self.lam_wrap = self.lam.x.petsc_vec
        self.lhs_form, self.rhs_form = form(lhs), form(rhs)
        self.bcs = bcs
        self.l_vec_wrap = l_vec
        self.spring_vec_wrap = spring_vec
        self.l_vec = l_vec.petsc_vec if l_vec is not None else None
        self.spring_vec = spring_vec.petsc_vec if spring_vec is not None else None

        # MPC support
        self.mpc = mpc
        if mpc is not None:
            import dolfinx_mpc as _dmpc
            self._dmpc = _dmpc
            # Use dolfinx_mpc to create the matrix with the correct sparsity
            _comm = self.u.function_space.mesh.comm
            # Allocate with the MPC sparsity pattern, but do NOT assemble:
            # solve_fem() assembles on every call anyway, and the density has
            # not been initialised yet at this point, so anything assembled
            # here would be discarded unread.  dolfinx_mpc.assemble_matrix()
            # allocates and assembles together, so reach for the allocate-only
            # entry point it uses internally, falling back if it is not there.
            try:
                self.lhs_mat = _dmpc.cpp.mpc.create_matrix(
                    self.lhs_form._cpp_object, mpc._cpp_object,
                    mpc._cpp_object)
            except AttributeError:
                if _comm.rank == 0:
                    print("  [LinearProblem] allocate-only MPC matrix "
                          "unavailable; assembling instead", flush=True)
                self.lhs_mat = _dmpc.assemble_matrix(
                    self.lhs_form, mpc, bcs=self.bcs)
        else:
            self._dmpc = None
            self.lhs_mat = create_matrix(self.lhs_form)

        self.rhs_vec = create_vector(self.rhs_form.function_spaces[0])

        # Direction basis: one assembled load vector and one solution vector
        # per direction.  The loads do not depend on the density, so like
        # rhs_vec they are assembled once, below.
        self.direction_forms = [form(r) for r in (direction_rhs or [])]
        self.rhs_dir = [create_vector(f.function_spaces[0])
                        for f in self.direction_forms]
        self.u_dir = [self.u_wrap.copy() for _ in self.direction_forms]

        self.direction_bcs = direction_bcs
        # Parity classes.  Class 0 is the ordinary set held in self.bcs, so it
        # rides along on the main operator and only the others need one of
        # their own -- and only those actually carrying a basis load.
        self.direction_parity = list(direction_parity or
                                     [0]*len(self.direction_forms))
        self.parity_solvers = {}
        if direction_bcs and len(direction_bcs) > 1:
            for cls in sorted(set(self.direction_parity)):
                if cls == 0:
                    continue
                self.parity_solvers[cls] = _ParitySolver(
                    self.lhs_form, direction_bcs[cls], petsc_options,
                    u.function_space)

        # Construct a linear solver
        self.solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self.solver.setOperators(self.lhs_mat)
        prefix = f"linear_solver_{id(self)}"

        _pc_defaults = _solver_defaults(petsc_options)

        # Apply PETSc options (solver defaults first, then user options override)
        _apply_solver_options(self.solver, self.lhs_mat, prefix,
                              petsc_options, _pc_defaults)

        # Log the preconditioner defaults that were actually applied
        if _pc_defaults:
            applied = {k: v for k, v in _pc_defaults.items()
                       if k not in petsc_options}
            if applied and self.u.function_space.mesh.comm.rank == 0:
                items = ", ".join(f"{k}={v}" for k, v in applied.items())
                _pc = petsc_options.get("pc_type", "?").upper()
                print(f"  🔧 {_pc} high-contrast defaults: {items}", flush=True)

        # For iterative solvers (CG, GMRES, etc.), enable warm-starting from
        # the previous solution.  In topology optimization the design changes
        # incrementally, so the prior displacement field is an excellent initial
        # guess that typically halves the Krylov iteration count.
        self._iterative = petsc_options.get("ksp_type", "preonly") != "preonly"
        if self._iterative:
            self.solver.setInitialGuessNonzero(True)

            # ── Near-nullspace for GAMG ──
            # Elasticity problems have 6 near-kernel modes (3D) or 3 (2D):
            # translations + rotations.  GAMG *requires* this information to
            # build a good coarsening hierarchy — without it, convergence
            # degrades catastrophically as material contrast grows.
            #
            # GAMG only, deliberately.  Handing the same modes to hypre via
            # BoomerAMG's vec_interp_variant measured *worse*: beam_3d went
            # from 13.7 to 47.8 s/iter.  Building interpolation that preserves
            # six vectors is expensive setup, and topology optimization rebuilds
            # the preconditioner every iteration (the density changes, so the
            # matrix changes), so that setup is never amortized.
            if petsc_options.get("pc_type") == "gamg":
                _set_rigid_body_modes(self.lhs_mat, self.u.function_space,
                                      _dirichlet_rows(self.bcs))

        self._first_solve = True
        self._fallback = _DivergenceFallback(
            self.u.function_space.mesh.comm, "equilibrium")

        # Owned local row indices carrying a Dirichlet condition, cached for
        # _rescale_bc_diagonal.
        #
        # dolfinx_mpc writes diagval=1 into slave rows exactly as dolfinx does
        # for Dirichlet rows, so they need the same treatment.  Without this the
        # MPC paths -- diagonal and C4 symmetry, which cannot be expressed as a
        # plain Dirichlet condition -- keep the bad scaling: measured 63 Krylov
        # iterations against 8 on beam_3d.
        _mpc_slaves = None
        if mpc is not None:
            try:
                _mpc_slaves = np.asarray(mpc.slaves, dtype=np.int32)
            except Exception as _exc:
                if self.u.function_space.mesh.comm.rank == 0:
                    print(f"  ⚠️  MPC slave rows not available for diagonal "
                          f"rescaling: {_exc}", flush=True)
        self._bc_dof_indices = _constrained_rows(
            self.bcs, self.u.function_space, _mpc_slaves)

        # Decide once whether any rank has work to do, so an ordinary run pays
        # nothing per solve.  Collective, hence outside the branch above.
        self._needs_rescale = bool(self.u.function_space.mesh.comm.allreduce(
            self._bc_dof_indices.size > 0, op=_MPI.LOR))

        # GAMG hierarchy rebuild tracking: detect KSP iteration spikes and
        # monitor convergence failures to trigger proactive rebuilds.
        self._ksp_history = []          # all recent KSP iteration counts
        self._ksp_healthy = []          # only non-spike iterations (for clean baseline)
        self._ksp_baseline = None       # average KSP iters from healthy solves
        self._GAMG_REBUILD_FACTOR = 3.0 # rebuild when current > factor × baseline
        self._beta_changed = False      # set by notify_beta_change()
        self._consecutive_spikes = 0    # count consecutive spike/diverge solves
        self._REGIME_SHIFT_THRESHOLD = 5  # after this many consecutive spikes, accept new regime
        self._spike_logged = False      # rate-limit log spam (log once per spike series)


        assemble_vector(self.rhs_vec, self.rhs_form)
        if self.mpc is not None:
            self._dmpc.apply_lifting(
                self.rhs_vec, [self.lhs_form], [self.bcs], self.mpc)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)

        for vec, frm, cls in zip(self.rhs_dir, self.direction_forms,
                                 self.direction_parity):
            assemble_vector(vec, frm)
            if self.mpc is not None:
                self._dmpc.apply_lifting(vec, [self.lhs_form], [self.bcs], self.mpc)
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # Each basis load is cleared by *its own* conditions, not by the
            # symmetric set.  A load antisymmetric about a mirror plane pushes
            # along the plane normal, which is exactly the component the
            # symmetric roller pins -- clearing it here would silently drop
            # the load carried by the nodes on the plane, and with it a slice
            # of the compliance that grows with the mesh.
            set_bc(vec, self._class_bcs(cls))

    def _rescale_bc_diagonal(self):
        """Rescale this operator's constrained rows; see the module helper."""
        _rescale_constrained_rows(self.lhs_mat, self._bc_dof_indices,
                                  self.u.function_space.mesh.comm)

    def notify_beta_change(self, new_beta=None):
        """Proactively prepare solver for a Heaviside beta change.

        Must be called *before* the next solve_fem() after beta doubles.
        This forces a full GAMG hierarchy rebuild, zeros the initial guess
        (the old displacement reflects a smoother material distribution and
        misleads the Krylov solver), and resets the KSP baseline tracking
        so the new beta regime establishes its own clean baseline.

        Parameters
        ----------
        new_beta : float, optional
            The new beta value (for logging only).
        """
        if not self._iterative:
            return

        comm = self.u.function_space.mesh.comm

        # Force full GAMG hierarchy rebuild on next solve
        self._first_solve = True
        self._beta_changed = True

        # Zero the initial guess — the old displacement field reflects a
        # smoother density distribution and is a poor starting point for
        # the much sharper system.
        self.u_wrap.set(0.0)
        self.lam_wrap.set(0.0)

        # Reset KSP baseline tracking for the new beta regime.
        # Keep the last few healthy counts as a reference floor but clear
        # the spike history so the new regime builds its own baseline.
        self._ksp_history.clear()
        self._ksp_healthy.clear()
        self._ksp_baseline = None
        self._consecutive_spikes = 0
        self._spike_logged = False

        if comm.rank == 0:
            beta_str = f" (β={new_beta})" if new_beta is not None else ""
            print(f"  🔄 Beta change{beta_str}: forcing GAMG rebuild + "
                  f"initial-guess reset.", flush=True)

    def _class_bcs(self, cls):
        """The boundary conditions a basis load of parity *cls* is solved with."""
        if not self.direction_bcs or cls == 0:
            return self.bcs
        return self.direction_bcs[cls]

    def solve_fem(self):
        """Solve K*x=F for FEM."""
        comm = self.u.function_space.mesh.comm
        from fenitop.timing import stats

        stats.start('assembly')
        self.lhs_mat.zeroEntries()
        if self.mpc is not None:
            self._dmpc.assemble_matrix(self.lhs_form, self.mpc,
                                       bcs=self.bcs, A=self.lhs_mat)
        else:
            assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)

        self.lhs_mat.assemble()

        if self.spring_vec_wrap is not None:
            self.lhs_mat.setDiagonal(self.lhs_mat.getDiagonal() + self.spring_vec)

        if self._needs_rescale:
            self._rescale_bc_diagonal()

        # For iterative solvers: after the first solve, tell PETSc the
        # sparsity pattern hasn't changed so GAMG/AMG can reuse the
        # coarsening hierarchy and only recompute the smoother weights.
        if self._iterative and not self._first_solve:
            self.solver.setOperators(self.lhs_mat, self.lhs_mat)
        self._first_solve = False
        stats.stop('assembly')

        # The rhs_vec is already assembled in __init__
        set_bc(self.rhs_vec, self.bcs)

        stats.start('solve')
        if self.rhs_dir:
            # Direction-uncertain load: the basis solves below span every load
            # direction, the nominal one included, so solving rhs_vec as well
            # would be a wasted solve per iteration.  They share the operator,
            # hence the factorization or preconditioner, and cost only a
            # back-substitution or a Krylov run each.
            for solver in self.parity_solvers.values():
                solver.assemble(self.spring_vec)
            for vec, u_k, cls in zip(self.rhs_dir, self.u_dir,
                                     self.direction_parity):
                if cls in self.parity_solvers:
                    self.parity_solvers[cls].solve(vec, u_k)
                else:
                    set_bc(vec, self.bcs)
                    self._fallback.solve(self.solver, self.lhs_mat, vec, u_k)
            # Keep u_field a valid field: Sensitivity replaces it with the
            # nominal load's response, but callbacks may read it before that.
            self.u_dir[0].copy(self.u_wrap)
        else:
            self._fallback.solve(self.solver, self.lhs_mat,
                                 self.rhs_vec, self.u_wrap)
        stats.stop('solve')

        # ── GAMG hierarchy rebuild heuristic ──
        # Track KSP iteration counts and convergence status.  Two triggers
        # can force a full GAMG hierarchy rebuild on the next iteration:
        #   1. KSP iterations spike beyond 3× the healthy baseline
        #   2. KSP diverged (negative converged reason)
        # The baseline is computed ONLY from non-spike solves to prevent
        # poisoning — UNLESS we detect a regime shift (many consecutive
        # spikes), in which case we accept the new iteration level as the
        # new normal.
        if self._iterative:
            ksp_its = self.solver.getIterationNumber()
            ksp_reason = self.solver.getConvergedReason()
            self._ksp_history.append(ksp_its)
            # Keep bounded (need at least _REGIME_SHIFT_THRESHOLD entries)
            if len(self._ksp_history) > 20:
                self._ksp_history = self._ksp_history[-20:]

            # Check for divergence (negative reason = KSP_DIVERGED_*)
            diverged = ksp_reason < 0

            # Build or check baseline
            is_spike = False
            if self._ksp_baseline is None:
                # Still building baseline from first 5 solves.
                # Include diverged solves too — if the very first solves
                # after a beta change diverge, we still need a baseline.
                self._ksp_healthy.append(ksp_its)
                if len(self._ksp_healthy) >= 5:
                    self._ksp_baseline = (sum(self._ksp_healthy)
                                          / len(self._ksp_healthy))
            else:
                threshold = self._GAMG_REBUILD_FACTOR * self._ksp_baseline
                is_spike = ksp_its > threshold or diverged

                if is_spike:
                    self._consecutive_spikes += 1

                    # ── Regime shift detection ──
                    # After N consecutive spikes, the old baseline is stale.
                    # Accept the recent iteration counts as the new regime.
                    if (self._consecutive_spikes
                            >= self._REGIME_SHIFT_THRESHOLD):
                        recent = sorted(self._ksp_history[-self._REGIME_SHIFT_THRESHOLD:])
                        new_baseline = recent[len(recent) // 2]  # median
                        if (comm.rank == 0
                                and abs(new_baseline - self._ksp_baseline) > 1):
                            print(f"  📊 Regime shift: {self._consecutive_spikes} "
                                  f"consecutive high-iter solves.  Baseline "
                                  f"{self._ksp_baseline:.0f} → {new_baseline} "
                                  f"(accepting new level).", flush=True)
                        self._ksp_baseline = new_baseline
                        self._ksp_healthy = list(
                            self._ksp_history[-self._REGIME_SHIFT_THRESHOLD:])
                        self._consecutive_spikes = 0
                        self._spike_logged = False
                        # No rebuild — we already tried, the system is just
                        # harder now.  Let it converge at the higher iter count.
                        is_spike = False
                    else:
                        # Log once per spike series, not every iteration
                        if not self._spike_logged and comm.rank == 0:
                            if diverged:
                                print(f"  ⚠️  KSP diverged: reason={ksp_reason}, "
                                      f"iters={ksp_its}.  Rebuilding hierarchy.",
                                      flush=True)
                            else:
                                print(f"  ⚠️  KSP spike: {ksp_its} iters > "
                                      f"{self._GAMG_REBUILD_FACTOR:.0f}× baseline "
                                      f"({self._ksp_baseline:.0f}).  Rebuilding "
                                      f"preconditioner hierarchy.", flush=True)
                            self._spike_logged = True
                else:
                    # Healthy solve — update baseline and reset spike counter
                    self._consecutive_spikes = 0
                    self._spike_logged = False
                    self._ksp_healthy.append(ksp_its)
                    # Rolling window of last 10 healthy solves
                    if len(self._ksp_healthy) > 10:
                        self._ksp_healthy = self._ksp_healthy[-10:]
                    self._ksp_baseline = (sum(self._ksp_healthy)
                                          / len(self._ksp_healthy))

            # Trigger rebuild only on first 2 spikes of a series.
            # After that, rebuilding doesn't help — the matrix is just
            # harder at this contrast level.
            if is_spike and self._consecutive_spikes <= 2:
                self._first_solve = True

            # Clear beta_changed flag after first post-beta solve
            self._beta_changed = False

        # MPC: recover slave DOF values from the reduced solution.
        # scatter_forward() BEFORE backsubstitution ensures ghost master
        # DOF values are up-to-date on ranks that own slave DOFs — without
        # this, cross-rank master reads are stale and produce NaN/wrong values.
        if self.mpc is not None:
            self.u.x.scatter_forward()
            self.mpc.backsubstitution(self.u)
        self.u.x.scatter_forward()

    def solve_adjoint(self):
        """Solve K*lambda=-L for the adjoint equation."""
        from fenitop.timing import stats
        stats.start('solve')
        rhs = -self.l_vec
        self._fallback.solve(self.solver, self.lhs_mat, rhs, self.lam_wrap)
        rhs.destroy()
        stats.stop('solve')
        # MPC: recover slave DOF values (same pattern as solve_fem)
        if self.mpc is not None:
            self.lam.x.scatter_forward()
            self.mpc.backsubstitution(self.lam)
        self.lam.x.scatter_forward()

    def __del__(self):
        for solver in getattr(self, "parity_solvers", {}).values():
            solver.destroy()
        self._fallback.destroy()
        self.solver.destroy()
        self.lhs_mat.destroy()
        self.rhs_vec.destroy()
        for vec in getattr(self, "rhs_dir", []):
            vec.destroy()
        for vec in getattr(self, "u_dir", []):
            vec.destroy()
        self.u_wrap.destroy()
        self.lam_wrap.destroy()
        if self.spring_vec_wrap is not None:
            self.spring_vec.destroy()
            self.l_vec.destroy()


class Communicator():
    """Communicate information among different processes."""

    def __init__(self, func_space, mesh_serial, size=1):
        self.size = size
        self.comm = func_space.mesh.comm
        idx_map = func_space.dofmap.index_map

        num_local_nodes = idx_map.size_local
        num_global_nodes = idx_map.size_global
        num_nodal_dofs = func_space.dofmap.index_map_bs
        self.num_global_dofs = num_global_nodes * num_nodal_dofs

        local_nodal_range = np.asarray(idx_map.local_range, dtype=np.int32)  # [start, end]
        local_dof_range = local_nodal_range * num_nodal_dofs  # [start, end]
        local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]

        # Gather to Process 0
        local_nodal_range_gather = self.comm.gather(local_nodal_range, root=0)
        self.local_dof_range_gather = self.comm.gather(local_dof_range, root=0)
        local_nodes_gather = self.comm.gather(local_nodes, root=0)

        element = func_space.ufl_element()
        if self.comm.rank == 0:
            func_space_serial = dolfinx.fem.functionspace(mesh_serial, element)
            nodes_serial = func_space_serial.tabulate_dof_coordinates()

            nodes_collect = np.zeros((num_global_nodes, 3))
            for r, nodes in zip(local_nodal_range_gather, local_nodes_gather):
                nodes_collect[r[0]:r[1]] = nodes
            global_to_local_nodes = compare_matrices(nodes_serial, nodes_collect)
            local_to_global_nodes = compare_matrices(nodes_collect, nodes_serial)

            def node2dof(nodes, num_nodal_dofs):
                return (np.tile(nodes, (num_nodal_dofs, 1))*num_nodal_dofs
                        + np.arange(num_nodal_dofs).reshape(-1, 1)).ravel("F")

            global_to_local_dofs = node2dof(global_to_local_nodes, num_nodal_dofs)
            self.local_to_global_dofs = node2dof(local_to_global_nodes, num_nodal_dofs)
            self.local_to_global_dofs = (
                np.tile(self.local_to_global_dofs.reshape(-1, 1), (1, size))*size + np.arange(size)).ravel()
        else:
            global_to_local_dofs = None
        global_to_local_dofs = self.comm.bcast(global_to_local_dofs, root=0)
        self.idx = global_to_local_dofs[local_dof_range[0]:local_dof_range[1]]

    def bcast(self, func, global_values):
        """Broadcast data from Process 0 to all the other processes."""
        # global_values has size num_global_dofs; func.x.array may include ghost
        # DOFs and be larger — only fill the owned slice via self.idx.
        if global_values.size != self.num_global_dofs * self.size:
            raise ValueError(
                f"Mismatched sizes: global_values has {global_values.size} entries "
                f"but expected {self.num_global_dofs * self.size}.")
        func.x.array[:len(self.idx)] = global_values[self.idx]

    def gather(self, func):
        """Gather data to Process 0 from all the other processes."""
        if type(func) is Function:
            # Gather only owned DOFs (not ghosts) to match expected sizes
            owned_size = func.function_space.dofmap.index_map.size_local
            values_gather = self.comm.gather(func.x.array[:owned_size], root=0)
        elif type(func) is PETSc.Vec:
            # For PETSc Vec, also use only owned DOFs
            owned_size = func.getOwnershipRange()[1] - func.getOwnershipRange()[0]
            values_gather = self.comm.gather(func.array[:owned_size], root=0)
        elif type(func) is np.ndarray:
            values_gather = self.comm.gather(func, root=0)
        else:
            raise TypeError("Unsupported func.")

        if self.comm.rank == 0:
            values_collect = np.zeros(self.num_global_dofs*self.size)
            for r, local_values in zip(self.local_dof_range_gather, values_gather):
                values_collect[r[0]*self.size:r[1]*self.size] = local_values
            global_values = values_collect[self.local_to_global_dofs]
        else:
            global_values = None
        return global_values


def compare_matrices(array1, array2, precision=12, k=1):
    """Find the "args" such that array1[args] == array2."""
    kd_tree = cKDTree(array1.round(precision))
    return kd_tree.query(array2.round(precision), k=k)[1]



def save_xdmf(mesh, rho, path="", filename="optimized_design"):
    save_path = os.path.join(path, f"{filename}.xdmf")
    with dolfinx.io.XDMFFile(mesh.comm, save_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        rho.name = "density"
        xdmf.write_function(rho)


def save_vtkhdf(mesh, rho, path="", filename="optimized_design", time=0.0):
    """Write the design to VTKHDF, the format Kitware is standardising on.

    Complements :func:`save_xdmf` rather than replacing it -- XDMF stays the
    default because it is what the rest of the toolchain reads today.  VTKHDF
    writes a single self-contained file (no .xdmf/.h5 pair), handles mixed
    topology, and is the faster parallel writer.

    Handles the two spaces the optimizer uses: CG1 (``rho_phys``) is written as
    point data and DG0 (``rho``) as cell data.

    Note:
        ``write_point_data`` wants values ordered like the mesh *geometry
        nodes*, which is not the function's dofmap order.  Writing
        ``rho.x.array`` straight through produces a file that looks valid and
        has the densities scrambled, so the values are permuted through the two
        dofmaps below.
    """
    from dolfinx.io import vtkhdf

    V = rho.function_space
    family = V.ufl_element().family_name
    degree = V.ufl_element().degree
    save_path = os.path.join(path, f"{filename}.vtkhdf")

    vtkhdf.write_mesh(save_path, mesh)

    if degree == 0:
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        vtkhdf.write_cell_data(save_path, mesh,
                               rho.x.array[:num_cells].copy(), time)
        return save_path

    if not (family in ("Lagrange", "P", "CG") and degree == 1):
        raise ValueError(
            f"save_vtkhdf supports DG0 and CG1 densities, got "
            f"{family} degree {degree}.")

    # Scatter the CG1 dof values into geometry-node positions.  For P1
    # geometry both dofmaps have the same cell-local node ordering, so a
    # cell-wise gather/scatter is an exact permutation.  Refresh ghost dofs
    # first: a geometry node this rank owns can map to a dof it only ghosts,
    # since the two index maps are built independently.
    rho.x.scatter_forward()
    geom_dofmap = mesh.geometry.dofmap.reshape(-1)
    fn_dofmap = V.dofmap.list.reshape(-1)
    values = np.zeros(mesh.geometry.x.shape[0], dtype=rho.x.array.dtype)
    values[geom_dofmap] = rho.x.array[fn_dofmap]
    # The writer takes owned entries only; geometry.x also carries ghosts,
    # and owned nodes come first.
    vtkhdf.write_point_data(
        save_path, mesh, values[:mesh.geometry.index_map().size_local], time)
    return save_path

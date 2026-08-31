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
"""

import os
import numpy as np
from scipy.spatial import cKDTree
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


class LinearProblem:
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=[], petsc_options={}, mpc=None):
        """Initialize a linear problem.

        Parameters
        ----------
        mpc : dolfinx_mpc.MultiPointConstraint, optional
            If provided, assembly and solve use dolfinx_mpc routines to
            enforce multi-point constraints (e.g. diagonal symmetry slip BCs).
        """
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
            if _comm.rank == 0:
                print(f"  [LinearProblem] assembling MPC matrix ...", flush=True)
            self.lhs_mat = _dmpc.assemble_matrix(self.lhs_form, mpc, bcs=self.bcs)
            if _comm.rank == 0:
                print(f"  [LinearProblem] MPC matrix assembled", flush=True)
        else:
            self._dmpc = None
            self.lhs_mat = create_matrix(self.lhs_form)

        self.rhs_vec = create_vector(self.rhs_form.function_spaces[0])

        # Construct a linear solver
        self.solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self.solver.setOperators(self.lhs_mat)
        prefix = f"linear_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        # ── GAMG defaults for high-contrast topology optimization ──
        # When β doubles (Heaviside sharpening), material contrast jumps and
        # default GAMG settings (V-cycle, 0 agg smooths, threshold 0) produce
        # poor coarse-grid operators → KSP iterations explode.
        # These defaults are the standard recipe for SIMP-based topology
        # optimization and can be overridden by explicit user petsc_options.
        _gamg_topo_defaults = {}
        if petsc_options.get("pc_type") == "gamg":
            _gamg_topo_defaults = {
                "pc_gamg_agg_nsmooths": 1,    # smoother aggregation (default 0)
                "pc_gamg_threshold": 0.02,     # less aggressive coarsening
                "mg_levels_ksp_max_it": 2,     # extra smoother sweeps per level
                "pc_mg_cycle_type": "w",          # W-cycle for robustness
                "ksp_max_it": 500,             # cap runaway solves (PETSc default 10000)
            }

        # Apply PETSc options (GAMG defaults first, then user options override)
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in _gamg_topo_defaults.items():
            if key not in petsc_options:
                opts[key] = value
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        self.lhs_mat.setOptionsPrefix(prefix)
        self.lhs_mat.setFromOptions()

        # Log GAMG topology-optimization defaults
        if _gamg_topo_defaults:
            applied = {k: v for k, v in _gamg_topo_defaults.items()
                       if k not in petsc_options}
            if applied and self.u.function_space.mesh.comm.rank == 0:
                items = ", ".join(f"{k}={v}" for k, v in applied.items())
                print(f"  🔧 GAMG high-contrast defaults: {items}", flush=True)

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
            if petsc_options.get("pc_type") == "gamg":
                self._set_near_nullspace()

        self._first_solve = True

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

    def _set_near_nullspace(self):
        """Set the near-nullspace (rigid body modes) on the matrix for GAMG.

        For 3D elasticity: 6 modes (3 translations + 3 rotations).
        For 2D elasticity: 3 modes (2 translations + 1 rotation).

        This dramatically improves GAMG coarsening quality for elasticity
        problems, especially at high material contrast (late topology
        optimization iterations).
        """
        V = self.u.function_space
        dim = V.mesh.geometry.dim
        bs = V.dofmap.index_map_bs
        num_local = V.dofmap.index_map.size_local

        # Tabulate DOF coordinates (only owned DOFs)
        coords = V.tabulate_dof_coordinates()[:num_local]

        def _make_vec(values_per_node):
            """Create a PETSc Vec and fill it from (num_local, bs) array."""
            vec = self.lhs_mat.createVecLeft()
            arr = vec.getArray(readonly=False)
            arr[:num_local * bs] = values_per_node.ravel()
            return vec

        modes = []
        if dim == 3:
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            zero = np.zeros(num_local)
            one = np.ones(num_local)
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
            zero = np.zeros(num_local)
            one = np.ones(num_local)
            # Translations: Tx, Ty
            modes.append(_make_vec(np.column_stack([one, zero])))
            modes.append(_make_vec(np.column_stack([zero, one])))
            # Rotation: Rz=(-y, x)
            modes.append(_make_vec(np.column_stack([-y, x])))

        # Orthonormalize the modes
        for i, vi in enumerate(modes):
            for vj in modes[:i]:
                vi.axpy(-vi.dot(vj), vj)
            norm = vi.norm()
            if norm > 1e-10:
                vi.scale(1.0 / norm)

        nsp = PETSc.NullSpace().create(vectors=modes, comm=V.mesh.comm)
        self.lhs_mat.setNearNullSpace(nsp)

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
        self.solver.solve(self.rhs_vec, self.u_wrap)
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
        self.solver.solve(rhs, self.lam_wrap)
        rhs.destroy()
        stats.stop('solve')
        # MPC: recover slave DOF values (same pattern as solve_fem)
        if self.mpc is not None:
            self.lam.x.scatter_forward()
            self.mpc.backsubstitution(self.lam)
        self.lam.x.scatter_forward()

    def __del__(self):
        self.solver.destroy()
        self.lhs_mat.destroy()
        self.rhs_vec.destroy()
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

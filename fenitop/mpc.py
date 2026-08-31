# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jan-David Förster

"""
Multi-point constraints for symmetry-reduced domains.

Builds ``dolfinx_mpc`` constraints that enforce symmetry conditions the
plain Dirichlet path cannot express:

- **slip** (``u . n = 0``) for diagonal mirror planes
- **cyclic** (periodic) for C_n rotational symmetry

Axis-aligned mirror planes need no MPC — they reduce to a roller BC that
zeroes the normal displacement component, handled by the caller.
"""
from __future__ import annotations

import numpy as np


def build_mpc_slip_constraints(V, mesh, symmetry_bcs, bcs):
    """Build a ``dolfinx_mpc.MultiPointConstraint`` for diagonal symmetry.

    For each diagonal symmetry BC, a slip constraint ``u · n = 0`` is
    added using ``dolfinx_mpc``'s ``create_slip_constraint``.

    Parameters
    ----------
    V : dolfinx FunctionSpace
        The vector function space (CG1 vector).
    mesh : dolfinx Mesh
    symmetry_bcs : list of dict
        Output of :func:`build_symmetry_bc_locators` — only entries with
        ``"type": "diagonal"`` are processed here.
    bcs : list of dolfinx DirichletBC
        Existing Dirichlet BCs (passed to the slip constraint to avoid
        conflicts on shared DOFs).

    Returns
    -------
    dolfinx_mpc.MultiPointConstraint or None
        The MPC object (already finalized), or ``None`` if no diagonal
        symmetry planes are present.
    """
    diag_bcs = [s for s in symmetry_bcs if s.get("type") == "diagonal"]
    if not diag_bcs:
        return None

    try:
        import dolfinx_mpc
    except ImportError:
        import warnings
        warnings.warn(
            "[symmetry] dolfinx_mpc is required for diagonal symmetry planes "
            "but is not installed.  Diagonal symmetry BCs will be skipped.  "
            "Install with:  conda install -c conda-forge dolfinx_mpc",
            stacklevel=2,
        )
        return None

    from dolfinx.mesh import locate_entities_boundary, meshtags
    from dolfinx.fem import Function

    fdim = mesh.topology.dim - 1
    gdim = mesh.geometry.dim

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    for idx, sbc in enumerate(diag_bcs):
        locator_fn = sbc["locator"]
        normal_vec = np.asarray(sbc["normal"], dtype=float)

        # Locate boundary facets on the diagonal plane.
        sym_facets = locate_entities_boundary(mesh, fdim, locator_fn)

        if mesh.comm.rank == 0:
            print(f"  [slip_mpc] plane {idx}: n={normal_vec}, "
                  f"{len(sym_facets)} facets", flush=True)

        marker_val = 99
        mt = meshtags(mesh, fdim,
                      np.array(sym_facets, dtype=np.int32),
                      np.full(len(sym_facets), marker_val, dtype=np.int32))

        # Build a vector-valued Function on V holding the constant normal.
        n_func = Function(V)
        bs = V.dofmap.index_map_bs
        n_arr = n_func.x.array
        for i in range(gdim):
            n_arr[i::bs] = normal_vec[i] if i < len(normal_vec) else 0.0
        n_func.x.scatter_forward()

        mpc.create_slip_constraint(V, (mt, marker_val), n_func, bcs=bcs)

    mpc.finalize()
    return mpc


def build_mpc_cyclic_constraints(V, mesh, rot_sym, bcs, owned_dof_coords=None):
    """Build a ``dolfinx_mpc.MultiPointConstraint`` for C4 rotational symmetry.

    The mesh is assumed to be clipped to the 90° sector where x ≥ |y|
    (between the x = y and x = −y planes).  A periodic constraint maps
    DOFs on the x = −y face (slave) to the corresponding DOFs on the
    x = y face (master) via a 90° rotation.

    Parameters
    ----------
    V : dolfinx FunctionSpace
    mesh : dolfinx Mesh
    rot_sym : dict
        A rotational_c4 symmetry dict (from detect_rotational_symmetry).
    bcs : list of DirichletBC
    owned_dof_coords : ndarray, optional
        Pre-computed ``V.tabulate_dof_coordinates()[:num_local]``.

    Returns
    -------
    dolfinx_mpc.MultiPointConstraint or None
    """
    try:
        import dolfinx_mpc
        import dolfinx_mpc.cpp.mpc as _cpp_mpc
    except ImportError:
        import warnings
        warnings.warn(
            "[symmetry] dolfinx_mpc is required for cyclic symmetry but is "
            "not installed.  Install with:  conda install -c conda-forge dolfinx_mpc",
            stacklevel=2,
        )
        return None

    from mpi4py import MPI as _MPI
    gdim = mesh.geometry.dim
    comm = mesh.comm
    _rank = comm.rank

    bs = V.dofmap.index_map_bs            # block size = gdim

    tol = 1e-6
    corner_tol = tol * 1000

    # Phase 1: Use C++ bounding-box tree to resolve slave/master matching.
    ref_comp = min(2, gdim - 1)  # uz in 3-D, uy in 2-D

    def _slave_indicator(x):
        return ((np.abs(x[0] + x[1]) < tol)
                & (x[0] >= -tol)
                & (x[0] > corner_tol))

    def _slave_to_master(x):
        out = x.copy()
        out[1] = -x[1]   # (a, −a, z) → (a, +a, z)
        return out

    mpc_data_ref = _cpp_mpc.create_periodic_constraint_geometrical(
        V.sub(ref_comp)._cpp_object,
        _slave_indicator,
        _slave_to_master,
        [],                                        # no BC filter here
        1.0,                                       # scale (overridden below)
        True,                                      # sub_space = True
        float(500 * np.finfo(np.float64).eps),     # tol
    )

    ref_slaves  = np.asarray(mpc_data_ref.slaves).copy()
    ref_masters = np.asarray(mpc_data_ref.masters).copy()
    ref_owners  = np.asarray(mpc_data_ref.owners).copy()

    # Phase 2: Filter out slave DOFs whose node overlaps a Dirichlet BC.
    bc_dof_set = set()
    for _bc in bcs:
        try:
            bc_dofs_local, _ = _bc.dof_indices()
            bc_dof_set.update(bc_dofs_local.tolist())
        except Exception:
            pass
    bc_node_set = set(d // bs for d in bc_dof_set)

    n_before = len(ref_slaves)
    if n_before > 0 and bc_node_set:
        keep = np.array(
            [int(ref_slaves[i]) // bs not in bc_node_set
             for i in range(n_before)])
        if not np.all(keep):
            idx = np.where(keep)[0]
            ref_slaves  = ref_slaves[idx]
            ref_masters = ref_masters[idx]
            ref_owners  = ref_owners[idx]

    n_local = len(ref_slaves)
    total_slaves = comm.allreduce(n_local, op=_MPI.SUM)
    total_before = comm.allreduce(n_before, op=_MPI.SUM)
    if _rank == 0:
        excluded = total_before - total_slaves
        if excluded > 0:
            print(f"  [C4 MPC] Excluded {excluded} slave nodes "
                  f"that overlap with Dirichlet BCs")
        print(f"  [C4 MPC] {total_slaves} total slave nodes on x=−y face "
              f"({n_local} on rank {_rank})")

    # Phase 3: Build all 3 couplings by remapping the reference data.
    couplings = [(0, 1, +1.0), (1, 0, -1.0)]
    if gdim >= 3:
        couplings.append((2, 2, +1.0))

    mpc = dolfinx_mpc.MultiPointConstraint(V)

    for slave_comp, master_comp, coeff in couplings:
        s = (ref_slaves  + (slave_comp  - ref_comp)).astype(np.int32)
        m = (ref_masters + (master_comp - ref_comp)).astype(np.int64)
        c = np.full(len(s), coeff, dtype=np.float64)
        o = ref_owners.copy()
        off = np.arange(len(s) + 1, dtype=np.int32)
        mpc.add_constraint(V, s, m, c, o, off)

    mpc.finalize()
    return mpc

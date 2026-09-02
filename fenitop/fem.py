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
symmetry boundary conditions and MPC construction, objective scaling for
symmetry-reduced domains, precomputed-facet BC locators, DOLFINx 0.11
port.
Full list of deviations from upstream: CHANGES_FROM_FENITOP.md
in https://github.com/Grumium/topologui
"""

import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (functionspace, Function, Constant,
                         dirichletbc, locate_dofs_topological)

from fenitop.utility import create_mechanism_vectors
from fenitop.utility import LinearProblem


#: Which coordinate axes each admissible load plane is spanned by.
PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def direction_mode(value):
    """Normalize one load-direction setting to a supported mode.

    ``"fixed"``        the load acts exactly as entered (the default)
    ``"any"``          its direction is unknown, over every axis
    ``"xy"``/``"xz"``/``"yz"``
                       its direction stays in that coordinate plane, with a
                       bounded perpendicular share of the entered direction
    """
    mode = str(value if value is not None else "fixed").lower()
    if mode not in ("fixed", "any") and mode not in PLANE_AXES:
        raise ValueError(
            f"direction_mode must be 'fixed', 'any' or one of "
            f"{sorted(PLANE_AXES)}, got {mode!r}.")
    return mode


def direction_settings(fem):
    """One ``(mode, ratio)`` pair per traction.

    Read from ``fem["direction_modes"]``, a list parallel to
    ``traction_bcs``.  The older scalar spelling -- ``direction_mode`` and
    ``transverse_ratio`` on ``fem`` itself, from when only one load could be
    uncertain -- is accepted and applies to every traction, as does
    ``uncertain_direction: True`` for ``"any"``.
    """
    n = len(fem["traction_bcs"])
    entries = fem.get("direction_modes")
    if entries is None:
        scalar = fem.get("direction_mode")
        if scalar is None:
            scalar = "any" if fem.get("uncertain_direction") else "fixed"
        entries = [{"mode": scalar,
                    "transverse_ratio": fem.get("transverse_ratio", 0.0)}] * n
    if len(entries) != n:
        raise ValueError(f"direction_modes has {len(entries)} entries for "
                         f"{n} tractions.")
    settings = []
    for entry in entries:
        if isinstance(entry, dict):
            mode = direction_mode(entry.get("mode", entry.get("direction_mode")))
            ratio = float(entry.get("transverse_ratio", 0.0))
        else:                       # a bare mode string
            mode = direction_mode(entry)
            ratio = float(fem.get("transverse_ratio", 0.0))
        settings.append((mode, ratio))
    return settings


def _check_uncertain_direction(fem, opt, settings):
    """Reject the combinations the load-distribution objective cannot handle.

    Checked before anything is assembled, so the reason is the first thing
    the caller sees rather than a wrong result or a failure further in.

    The objective reads the moments of the compliance off a matrix over the
    load basis, which is linear in the design's response to each basis load
    and needs nothing about how many loads there are -- so unlike the
    worst-case formulation this replaced, other loads, a body force and
    several uncertain loads at once are all fine.  What remains are the
    genuine incompatibilities.
    """
    if all(mode == "fixed" for mode, _ in settings):
        return
    problems = []
    gdim = fem["mesh"].geometry.dim
    for index, (mode, ratio) in enumerate(settings):
        if mode == "fixed":
            continue
        value = np.asarray(fem["traction_bcs"][index][0], dtype=float)
        if np.linalg.norm(value) == 0.0:
            problems.append(f"load {index} is zero, so it has no direction")
        if mode not in PLANE_AXES:
            continue
        if not 0.0 <= ratio <= 1.0:
            problems.append(f"transverse_ratio of load {index} must be within "
                            f"[0, 1], got {ratio}")
        if max(PLANE_AXES[mode]) >= gdim:
            problems.append(f"the {mode} plane needs a {max(PLANE_AXES[mode])+1}D "
                            f"mesh, this one is {gdim}D")
            continue
        axes = list(PLANE_AXES[mode])
        out_of_plane = np.linalg.norm(np.delete(value, axes))
        in_plane = np.linalg.norm(value[axes])
        if in_plane <= 0.0:
            problems.append(f"load {index} has no component in the {mode} "
                            "plane, so it has no main direction there")
        elif out_of_plane > 1e-9 * max(np.linalg.norm(value), 1.0):
            problems.append(
                f"load {index} must lie in the {mode} plane, but its "
                f"out-of-plane component is {out_of_plane:.3g} "
                f"against {in_plane:.3g} in plane")
    kappa = float(opt.get("direction_kappa", 1.0))
    if not 0.0 <= kappa <= 1.0:
        problems.append(
            f"direction_kappa must be within [0, 1], got {kappa}: weighting "
            "the scatter above the mean forfeits the consistency proof")
    if not opt["opt_compliance"]:
        problems.append("the objective must be compliance")
    exotic = [sbc.get("type") for sbc in fem.get("symmetry_bcs", [])
              if sbc.get("type", "axis_aligned") != "axis_aligned"]
    if exotic:
        problems.append(
            f"symmetry of type {sorted(set(exotic))} cannot be exploited: the "
            "load basis is resolved into parities about coordinate axes, "
            "which a diagonal plane or a rotation does not provide")
    axes = [sbc.get("component") for sbc in fem.get("symmetry_bcs", [])
            if sbc.get("type", "axis_aligned") == "axis_aligned"]
    if len(set(axes)) != len(axes):
        problems.append("two symmetry planes share a normal axis, so their "
                        "parities are not independent")
    if problems:
        raise ValueError(
            "an uncertain load direction is incompatible with this setup: "
            + "; ".join(problems) + ".")


def _load_basis(mode, ratio, value, gdim):
    """Basis directions, nominal coefficients and moments for one load.

    Returns ``(directions, magnitude, nominal, mean, cov)``: the basis loads
    all carry the entered magnitude, so the coefficients are dimensionless.

    The basis is always made of **coordinate axes** -- every axis of the mesh
    for ``"any"``, the two axes of the plane for a plane mode.  Aligning it
    with the load instead would be the more natural choice for a plane, and
    was what this did at first, but a coordinate axis is the only direction
    with a definite parity about an axis-aligned symmetry plane: purely
    normal is antisymmetric, purely tangential is symmetric, and anything
    between is neither.  Keeping the basis on the axes is what lets a
    symmetric half domain carry an uncertain load at all.

    Nothing about the model changes with the basis: the entered direction
    becomes the mean's coefficients, and the perpendicular scatter becomes an
    off-diagonal covariance rather than a diagonal one.  ``ratio`` bounds that
    scatter and is read as two standard deviations, which is the interval an
    entered "+-10%" describes -- a hard bound is not something a normal
    distribution has room for, and reading it as one standard deviation
    instead would put a third of the draws outside what the user wrote down.
    """
    magnitude = float(np.linalg.norm(value))
    if mode == "any":
        directions = [row.copy() for row in np.eye(gdim)]
        return (directions, magnitude, value/magnitude,
                np.zeros(gdim), np.eye(gdim)/gdim)

    axes = list(PLANE_AXES[mode])
    directions = []
    for axis in axes:
        e = np.zeros(gdim)
        e[axis] = 1.0
        directions.append(e)

    # Coefficients of the entered direction, and of the perpendicular the
    # scatter acts along, in that two-axis basis.
    d = value[axes]/np.linalg.norm(value[axes])
    perpendicular = np.array([-d[1], d[0]])
    sigma = 0.5*ratio
    return (directions, magnitude, d.copy(), d.copy(),
            sigma*sigma*np.outer(perpendicular, perpendicular))


def form_fem(fem, opt):
    """Form an FEA problem."""
    direction_config = direction_settings(fem)
    _check_uncertain_direction(fem, opt, direction_config)


    # Function spaces and functions
    mesh = fem["mesh"]
    dim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (dim,)))  # Vector function space
    S0 = functionspace(mesh, ("DG", 0))         # Scalar DG space
    S = functionspace(mesh, ("Lagrange", 1))    # Scalar CG space

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_field = Function(V)  # Displacement field
    lambda_field = Function(V)  # Adjoint variable field
    rho_field = Function(S0)  # Density field
    rho_phys_field = Function(S)  # Physical density field


    # Material interpolation
    E0, nu = fem["young's modulus"], fem["poisson's ratio"]
    p, eps = opt["penalty"], opt["epsilon"]
    E = (eps + (1-eps)*rho_phys_field**p) * E0
    _lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

    # Kinematics
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):  # 3D or plane strain
        return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    disp_bc_fn = fem["disp_bc"]
    if hasattr(disp_bc_fn, '_facet_indices') and disp_bc_fn._facet_indices is not None:
        disp_facets = np.array(sorted(disp_bc_fn._facet_indices), dtype=np.int32)
    else:
        disp_facets = locate_entities_boundary(mesh, fdim, disp_bc_fn)
    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)),
                     locate_dofs_topological(V, fdim, disp_facets), V)

    bcs = [bc]

    # Symmetry roller BCs: constrain normal displacement to zero on each
    # symmetry plane.  Axis-aligned planes use a simple Dirichlet BC on a
    # single component.  Diagonal planes are handled via dolfinx_mpc slip
    # constraints (u · n = 0) further below.
    def _component_bc(facets, comp):
        """Pin one displacement component to zero on *facets*."""
        V_sub = V.sub(comp)
        V_collapsed, _ = V_sub.collapse()
        sym_dofs = locate_dofs_topological(
            (V_sub, V_collapsed), fdim, facets)
        sym_val = Function(V_collapsed)
        sym_val.x.array[:] = 0.0
        return dirichletbc(sym_val, sym_dofs, V_sub)

    # Axis-aligned planes, in the order their parities are numbered below.
    mirror_planes = []
    for sym_bc in fem.get("symmetry_bcs", []):
        if sym_bc.get("type") in ("diagonal", "rotational_c4"):
            continue  # handled via MPC below
        sym_facets = locate_entities_boundary(mesh, fdim, sym_bc["locator"])
        comp = sym_bc["component"]  # 0=x, 1=y, 2=z
        mirror_planes.append((comp, sym_facets))
        bcs.append(_component_bc(sym_facets, comp))

    # Build dolfinx_mpc MultiPointConstraint for non-axis-aligned symmetry
    mpc = None
    diag_sym_bcs = [s for s in fem.get("symmetry_bcs", [])
                    if s.get("type") == "diagonal"]
    rot_sym_bcs = [s for s in fem.get("symmetry_bcs", [])
                   if s.get("type") == "rotational_c4"]

    # Tabulate DOF coordinates once — reused by MPC builder and mechanism
    # vectors below to avoid redundant (identical) calls.
    num_local = V.dofmap.index_map.size_local
    owned_dof_coords = V.tabulate_dof_coordinates()[:num_local]

    if rot_sym_bcs:
        # Rotational C4 (cyclic periodic) takes priority
        try:
            from fenitop.mpc import build_mpc_cyclic_constraints
            rot_data = rot_sym_bcs[0].get("rot_sym", rot_sym_bcs[0])
            mpc = build_mpc_cyclic_constraints(V, mesh, rot_data, bcs,
                                               owned_dof_coords=owned_dof_coords)
            if mpc is not None and mesh.comm.rank == 0:
                print(f"  🔗 MPC cyclic constraint active (C4 rotational, "
                      f"{mpc.num_local_slaves} local slave DOFs)")
        except Exception as e:
            if mesh.comm.rank == 0:
                print(f"  ⚠️  Failed to build cyclic MPC: {e}")
                import traceback; traceback.print_exc()
            mpc = None
    elif diag_sym_bcs:
        try:
            from fenitop.mpc import build_mpc_slip_constraints
            mpc = build_mpc_slip_constraints(V, mesh, fem["symmetry_bcs"], bcs)
            if mpc is not None and mesh.comm.rank == 0:
                print(f"  🔗 MPC slip constraint active "
                      f"({len(diag_sym_bcs)} diagonal plane(s), "
                      f"{mpc.num_local_slaves} local slave DOFs)")
        except Exception as e:
            if mesh.comm.rank == 0:
                print(f"  ⚠️  Failed to build MPC for diagonal symmetry: {e}")
                import traceback; traceback.print_exc()
            mpc = None




    # Accumulate the traction acting on each facet.  A facet can carry more
    # than one traction -- two loads applied to the same face, or regions that
    # overlap in part -- and they superpose, so the facet's load is their sum.
    #
    # meshtags needs one marker per facet, so the markers are assigned per
    # distinct summed vector rather than per input traction.  Tagging by input
    # instead, and dropping duplicate facets to satisfy meshtags, would keep
    # whichever traction happened to be listed first and silently discard the
    # rest.
    settings = direction_config
    traction_facets = []
    for traction, traction_bc in fem["traction_bcs"]:
        if hasattr(traction_bc, '_facet_indices') and traction_bc._facet_indices is not None:
            current_facets = np.array(sorted(traction_bc._facet_indices), dtype=np.int32)
        else:
            current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
        traction_facets.append([int(f) for f in current_facets])

    # An uncertain load needs its own facet group: its basis loads act where
    # it acts, and a facet it shared with another load could not carry both
    # markers.  Summing them is not an option either -- the other load would
    # then be rotated along with this one.
    uncertain = [i for i, (mode, _) in enumerate(settings) if mode != "fixed"]
    for i in uncertain:
        own = set(traction_facets[i])
        for j, other in enumerate(traction_facets):
            if j != i and own.intersection(other):
                raise ValueError(
                    f"load {i} has an uncertain direction but shares facets "
                    f"with load {j}; give it a region of its own.")

    facet_load = {}
    for i, (traction, _) in enumerate(fem["traction_bcs"]):
        if i in uncertain:
            continue
        value = np.asarray(traction, dtype=float)
        for f in traction_facets[i]:
            if f in facet_load:
                facet_load[f] = facet_load[f] + value
            else:
                facet_load[f] = value.copy()

    # Group facets by the load they carry.  Rounding keeps facets that differ
    # only by floating-point noise in one group instead of one group each.
    groups = {}
    for f, value in facet_load.items():
        groups.setdefault(tuple(np.round(value, 12)), []).append(f)

    tractions, facets, markers = [], [], []
    for marker, (value, group_facets) in enumerate(groups.items()):
        tractions.append(Constant(mesh, np.array(value, dtype=float)))
        facets.extend(group_facets)
        markers.extend([marker] * len(group_facets))

    uncertain_markers = {}
    for i in uncertain:
        uncertain_markers[i] = len(tractions) + len(uncertain_markers)
        facets.extend(traction_facets[i])
        markers.extend([uncertain_markers[i]] * len(traction_facets[i]))

    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])



    metadata = {"quadrature_degree": fem["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    b = Constant(mesh, np.array(fem["body_force"], dtype=float))

    # Establish the equilibrium and adjoint equations
    lhs = ufl.inner(sigma(u), epsilon(v))*dx
    rhs = ufl.dot(b, v)*dx
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v)*ds(marker)

    # Loads with an uncertain direction.  The compliance is a quadratic form
    # in the applied load, so the response to any admissible load is a linear
    # combination of the responses to a basis: `dim` orthogonal loads on the
    # same facets for a free direction, two for a coordinate plane.  Sampling
    # angles is unnecessary, and so is a second FEM solve for the nominal
    # load -- it is a combination of the basis too, which is why the
    # deterministic part of the load (fixed tractions and the body force)
    # enters as leading basis vectors rather than as a separate solve.
    # Sensitivity._evaluate_direction() reads the objective off them.
    opt["direction_rhs"] = None
    opt["direction_modes"] = [mode for mode, _ in direction_config]
    opt["direction_kappa"] = float(opt.get("direction_kappa", 1.0))
    opt["direction_parity"] = None
    opt["direction_bcs"] = None
    if uncertain:
        gdim = mesh.geometry.dim
        forms, mean, cov_blocks, nominal, parity = [], [], [], [], []

        # Every basis load is one Cartesian component on one facet group, so
        # its parity about each mirror plane is decided by the axis alone:
        # along the normal it is antisymmetric, along anything else
        # symmetric.  A load can therefore be odd about at most one plane --
        # the planes have distinct normals -- so the classes are "even about
        # all of them" plus one per plane, not one per subset.
        mirror_axes = [comp for comp, _ in mirror_planes]

        def parity_of(axis):
            """Which class a load along *axis* falls into: 0 is even."""
            return mirror_axes.index(axis) + 1 if axis in mirror_axes else 0

        # The deterministic part splits the same way.  A traction t on a
        # mirror-symmetric region is neither even nor odd unless it happens to
        # be tangential or normal, but t = t_tangential + t_normal is exactly
        # that decomposition, and each piece extends across the plane with the
        # parity that reproduces the load the user entered.  Without symmetry
        # planes there is nothing to split and this is the whole right-hand
        # side, as before.
        body = np.asarray(fem["body_force"], dtype=float)
        deterministic = {}

        def _add(cls, form_term):
            deterministic[cls] = (deterministic.get(cls, 0) + form_term
                                  if cls in deterministic else form_term)

        def _split(vector, measure):
            """Add *vector*'s parity components over *measure*."""
            for cls in range(len(mirror_axes) + 1):
                part = np.zeros(gdim)
                if cls == 0:
                    keep = [a for a in range(gdim) if a not in mirror_axes]
                else:
                    keep = [mirror_axes[cls - 1]]
                part[keep] = vector[keep]
                if np.any(part):
                    _add(cls, ufl.dot(Constant(mesh, part), v)*measure)

        if np.any(body):
            _split(body, dx)
        for marker, t in enumerate(tractions):
            _split(np.asarray(t.value, dtype=float), ds(marker))

        for cls in sorted(deterministic):
            # Certain, hence mean 1 and no scatter, but still part of the
            # quadratic form: its cross terms with the uncertain loads are
            # what makes the objective see them acting together.
            forms.append(deterministic[cls])
            mean.append([1.0])
            nominal.append([1.0])
            cov_blocks.append(np.zeros((1, 1)))
            parity.append(cls)

        for i in uncertain:
            mode, ratio = direction_config[i]
            value = np.asarray(fem["traction_bcs"][i][0], dtype=float)
            directions, magnitude, nom, mean_i, cov_i = _load_basis(
                mode, ratio, value, gdim)
            marker = uncertain_markers[i]
            for d in directions:
                forms.append(ufl.dot(Constant(mesh, d*magnitude), v)*ds(marker))
                parity.append(parity_of(int(np.argmax(np.abs(d)))))
            mean.append(mean_i)
            nominal.append(nom)
            cov_blocks.append(cov_i)

        size = sum(len(block) for block in cov_blocks)
        covariance = np.zeros((size, size))
        offset = 0
        for block in cov_blocks:
            end = offset + len(block)
            covariance[offset:end, offset:end] = block
            offset = end

        opt["direction_rhs"] = forms
        opt["direction_mu"] = np.concatenate(mean)
        opt["direction_nominal"] = np.concatenate(nominal)
        opt["direction_cov"] = covariance
        opt["direction_parity"] = parity

        if mirror_planes:
            # One boundary condition set per class.  Even about a plane is
            # the usual roller -- normal component pinned, the rest free.
            # Odd is its complement: the response is antisymmetric, so it is
            # the tangential components that vanish on the plane while the
            # normal one is free to move.
            classes = []
            for cls in range(len(mirror_planes) + 1):
                class_bcs = [bc]
                for index, (comp, facets) in enumerate(mirror_planes):
                    if cls == index + 1:
                        for other in range(dim):
                            if other != comp:
                                class_bcs.append(_component_bc(facets, other))
                    else:
                        class_bcs.append(_component_bc(facets, comp))
                classes.append(class_bcs)
            opt["direction_bcs"] = classes

    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"],
            dof_coords=owned_dof_coords,
            out_sign=opt.get("out_spring_sign", 1))

        # When exploiting symmetry, nodes that sit exactly on a symmetry
        # plane appear in both the kept half and (conceptually) the
        # mirrored half.  The sym_factor will multiply spring/l_vec
        # contributions by 2^n, so we must halve the values at those
        # boundary nodes to avoid double-counting.
        sym_bcs = fem.get("symmetry_bcs", [])
        if sym_bcs and spring_vec is not None:
            block_size = V.dofmap.index_map_bs
            on_sym = np.zeros(num_local, dtype=bool)
            for sbc in sym_bcs:
                loc = sbc.get("locator")
                if loc is None:
                    continue  # rotational_c4 entries have no locator
                on_sym |= loc(owned_dof_coords.T)
            if np.any(on_sym):
                node_idx = np.where(on_sym)[0]
                sym_dofs = (node_idx[:, None] * block_size
                            + np.arange(block_size)).ravel().astype(np.int32)
                # Halve spring stiffness at symmetry-plane nodes
                sv = spring_vec.petsc_vec
                sv_arr = sv.array.copy()
                sv_arr[sym_dofs] *= 0.5
                sv.array[:] = sv_arr
                # Halve l_vec (output spring load vector) at symmetry-plane nodes
                lv = opt["l_vec"].petsc_vec
                lv_arr = lv.array.copy()
                lv_arr[sym_dofs] *= 0.5
                lv.array[:] = lv_arr

    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                   spring_vec, bcs, fem["petsc_options"],
                                   mpc=mpc, direction_rhs=opt["direction_rhs"],
                                   direction_bcs=opt["direction_bcs"],
                                   direction_parity=opt["direction_parity"])

    # When MPC is active, re-create u_field and lambda_field from the MPC's
    # function space.  After mpc.finalize(), the MPC replaces V's index map
    # with one that includes remote master DOFs as extra ghosts.  Without
    # this, scatter_forward() on u_field won't fetch the master values
    # needed by backsubstitution, causing NaN in MPI runs.
    if mpc is not None:
        V_mpc = mpc.function_space
        u_field_new = Function(V_mpc)
        lambda_field_new = Function(V_mpc)
        # Copy any existing data (should be zero at this point)
        n = min(len(u_field.x.array), len(u_field_new.x.array))
        u_field_new.x.array[:n] = u_field.x.array[:n]
        lambda_field_new.x.array[:n] = lambda_field.x.array[:n]
        u_field = u_field_new
        lambda_field = lambda_field_new
        # Update linear_problem to use the new functions
        linear_problem.u = u_field
        linear_problem.u_wrap = u_field.x.petsc_vec
        linear_problem.lam = lambda_field
        linear_problem.lam_wrap = lambda_field.x.petsc_vec


    # Define optimization-related variables
    # When symmetry planes reduce the computational domain, the integrals
    # (compliance, f_int) only cover a fraction of the full domain.  Scale
    # them so that objective values and sensitivities are consistent with
    # the full-domain problem.  Volume fraction (volume/total_volume) is a
    # ratio of integrals over the same domain and therefore needs no
    # correction.
    sym_factor = 1
    for sbc in fem.get("symmetry_bcs", []):
        if sbc.get("type") == "rotational_c4":
            sym_factor *= 4  # quarter domain (90° sector)
        else:
            sym_factor *= 2  # half domain (mirror plane)

    opt["f_int"] = sym_factor * ufl.inner(sigma(u_field), epsilon(v))*dx
    opt["compliance"] = sym_factor * ufl.inner(sigma(u_field), epsilon(u_field))*dx
    # The volume functional deliberately covers the whole mesh, passive zones
    # included: vol_frac is a budget for the finished part, so material fixed
    # into a solid zone spends from the same budget as the optimized structure
    # around it, which then has to come out correspondingly leaner.
    opt["volume"] = rho_phys_field*dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx
    opt["_sym_factor"] = sym_factor

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field


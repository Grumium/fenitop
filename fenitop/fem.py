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

import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (functionspace, Function, Constant,
                         dirichletbc, locate_dofs_topological)

from fenitop.utility import create_mechanism_vectors
from fenitop.utility import LinearProblem


def form_fem(fem, opt):
    """Form an FEA problem."""


    # Function spaces and functions
    mesh = fem["mesh"]
    dim = mesh.geometry.dim
    V = functionspace(mesh, ("CG", 1, (dim,)))  # Vector function space
    S0 = functionspace(mesh, ("DG", 0))         # Scalar DG space
    S = functionspace(mesh, ("CG", 1))          # Scalar CG space

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
    disp_facets = locate_entities_boundary(mesh, fdim, fem["disp_bc"])
    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)),
                     locate_dofs_topological(V, fdim, disp_facets), V)

    bcs = [bc]

    # Symmetry roller BCs: constrain normal displacement to zero on each
    # symmetry plane.  Each entry in "symmetry_bcs" is a dict with keys
    # "locator" (callable), "component" (int), "value" (float, usually 0).
    for sym_bc in fem.get("symmetry_bcs", []):
        sym_facets = locate_entities_boundary(mesh, fdim, sym_bc["locator"])
        comp = sym_bc["component"]  # 0=x, 1=y, 2=z
        # Constrain only the single DOF component (sub-space of V)
        V_sub = V.sub(comp)
        V_collapsed, _ = V_sub.collapse()
        sym_dofs = locate_dofs_topological(
            (V_sub, V_collapsed), fdim, sym_facets)
        sym_val = Function(V_collapsed)
        sym_val.x.array[:] = 0.0
        sym_dirichlet = dirichletbc(sym_val, sym_dofs, V_sub)
        bcs.append(sym_dirichlet)



    tractions, facets, markers = [], [], []
    for marker, (traction, traction_bc) in enumerate(fem["traction_bcs"]):

        tractions.append(Constant(mesh, np.array(traction, dtype=float)))
        current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
        facets.extend(current_facets)
        markers.extend([marker,]*len(current_facets))
    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
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
    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"])

        # When exploiting symmetry, nodes that sit exactly on a symmetry
        # plane appear in both the kept half and (conceptually) the
        # mirrored half.  The sym_factor will multiply spring/l_vec
        # contributions by 2^n, so we must halve the values at those
        # boundary nodes to avoid double-counting.
        sym_bcs = fem.get("symmetry_bcs", [])
        if sym_bcs and spring_vec is not None:
            mesh_dim = mesh.geometry.dim
            num_local = V.dofmap.index_map.size_local
            coords = V.tabulate_dof_coordinates()[:num_local]
            block_size = V.dofmap.index_map_bs
            on_sym = np.zeros(num_local, dtype=bool)
            for sbc in sym_bcs:
                on_sym |= sbc["locator"](coords.T)
            if np.any(on_sym):
                sym_dofs = []
                for node_idx in np.where(on_sym)[0]:
                    for d in range(block_size):
                        sym_dofs.append(node_idx * block_size + d)
                sym_dofs = np.array(sym_dofs, dtype=np.int32)
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
                                   gpu_accel=fem.get("gpu_accel", False))


    # Define optimization-related variables
    # When symmetry planes reduce the computational domain, the integrals
    # (compliance, f_int) only cover 1/2^n of the full domain.  Scale them
    # so that objective values and sensitivities are consistent with the
    # full-domain problem.  Volume fraction (volume/total_volume) is a ratio
    # of integrals over the same domain and therefore needs no correction.
    n_sym = len(fem.get("symmetry_bcs", []))
    sym_factor = 2 ** n_sym  # 1 if no symmetry

    opt["f_int"] = sym_factor * ufl.inner(sigma(u_field), epsilon(v))*dx
    opt["compliance"] = sym_factor * ufl.inner(sigma(u_field), epsilon(u_field))*dx
    opt["volume"] = rho_phys_field*dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx
    opt["_sym_factor"] = sym_factor

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field


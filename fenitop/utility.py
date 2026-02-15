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
import pyvista


def create_mechanism_vectors(func_space, in_spring, out_spring):
    """Create vectors for compliant mechanism design."""
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
    local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]

    for n, (locator, direction, value) in enumerate([in_spring, out_spring]):
        ctrl_nodes = local_indices[locator(local_nodes.T)]
        offset = ["x", "y", "z"].index(direction)
        ctrl_dofs = ctrl_nodes*block_size + offset
        # Use PETSc setValues for MPI-safe global index handling
        spring_vec.setValues(ctrl_dofs, [value,]*ctrl_dofs.size)
        if n == 1:
            l_vec.setValues(ctrl_dofs, [1.0,]*ctrl_dofs.size)
    
    spring_vec.assemble()
    l_vec.assemble()
    return spring_vec_wrap, l_vec_wrap


class LinearProblem:
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=[], petsc_options={}):
        """Initialize a linear problem."""
        # Initialization
        self.u, self.lam = u, lam
        self.u_wrap = self.u.x.petsc_vec
        self.lam_wrap = self.lam.x.petsc_vec
        self.lhs_form, self.rhs_form = form(lhs), form(rhs)
        self.lhs_mat = create_matrix(self.lhs_form)
        self.rhs_vec = create_vector(self.rhs_form.function_spaces[0])
        self.bcs = bcs
        self.l_vec_wrap = l_vec
        self.spring_vec_wrap = spring_vec
        self.l_vec = l_vec.petsc_vec if l_vec is not None else None
        self.spring_vec = spring_vec.petsc_vec if spring_vec is not None else None

        # Construct a linear solver
        self.solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self.solver.setOperators(self.lhs_mat)
        prefix = f"linear_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        # Apply PETSc options
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        for var in [self.lhs_mat, self.rhs_vec, self.l_vec]:
            if var is not None:
                var.setOptionsPrefix(prefix)
                var.setFromOptions()

        assemble_vector(self.rhs_vec, self.rhs_form)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)

    def solve_fem(self):
        """Solve K*x=F for FEM."""
        from fenitop.timing import stats
        
        stats.start('assembly')
        self.lhs_mat.zeroEntries()
        assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)
        self.lhs_mat.assemble()
        if self.spring_vec_wrap is not None:
            # Use PETSc vector directly - it handles owned DOFs correctly
            self.lhs_mat.setDiagonal(self.lhs_mat.getDiagonal() + self.spring_vec)
        stats.stop('assembly')
        
        # The rhs_vec is already assembled in __init__
        set_bc(self.rhs_vec, self.bcs)
        
        stats.start('solve')
        self.solver.solve(self.rhs_vec, self.u_wrap)
        stats.stop('solve')
        
        self.u.x.scatter_forward()

    def solve_adjoint(self):
        """Solve K*lambda=-L for the adjoint equation."""
        self.solver.solve(-self.l_vec, self.lam_wrap)
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
        if len(func.x.array) != global_values.size:
            raise ValueError("Mismatched sizes.")
        func.x.array[:] = global_values[self.idx]

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


class Plotter():
    def __init__(self, mesh):
        """Initialize a plotter."""
        import pyvista
        self.dim = mesh.topology.dim
        elements, cell_types, nodes = dolfinx.plot.vtk_mesh(mesh, self.dim)
        self.grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)

    def plot(self, density, threshold=0.49, smooth_iter=100, path="", filename="optimized_design"):
        import pyvista
        self.grid.point_data["density"] = np.hstack(density)
        if self.dim == 2:
            grid = self.grid
        else:
            # Adapt threshold if all densities are below the default threshold
            density_array = np.hstack(density)
            max_density = np.max(density_array)
            if max_density < threshold:
                # Use a threshold slightly below the maximum density
                adaptive_threshold = max(0.01, max_density * 0.8)
                print(f"   Adapting threshold: {threshold:.2f} → {adaptive_threshold:.2f} (max density: {max_density:.3f})")
                threshold = adaptive_threshold
            grid = self.grid.threshold(threshold).extract_surface(algorithm='dataset_surface')
        empty_mesh = (self.dim == 3 and grid.n_faces_strict == 0)

        if not empty_mesh:
            if self.dim == 3:
                grid = grid.smooth(n_iter=smooth_iter)
                grid.point_data["density"] = 0.4
            plotter = pyvista.Plotter(off_screen=True)
            plotter.background_color = "white"
            lighting = self.dim == 3
            plotter.add_mesh(grid, clim=[0, 1], cmap="Greys", lighting=lighting,
                             show_scalar_bar=False)
            if self.dim == 2:
                plotter.view_xy()
            
            # Construct file path safely
            save_path = os.path.join(path, f"{filename}.jpg")
            plotter.screenshot(save_path, window_size=(1000, 1000))
            plotter.close()
        else:
            print(f"   ⚠️  Warning: Mesh is empty after thresholding (threshold={threshold:.2f}). No JPG created.")


def save_xdmf(mesh, rho, path="", filename="optimized_design"):
    save_path = os.path.join(path, f"{filename}.xdmf")
    with dolfinx.io.XDMFFile(mesh.comm, save_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        rho.name = "density"
        xdmf.write_function(rho)

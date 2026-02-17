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
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=[], petsc_options={}, gpu_accel=False):
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
        # Only apply options to the matrix (not vectors — GPU types break ghost vectors)
        self.lhs_mat.setOptionsPrefix(prefix)
        self.lhs_mat.setFromOptions()

        # GPU acceleration (opt-in via --gpu flag)
        self.use_gpu = False
        self._gpu_active = False
        self._gpu_diag = None
        self._gpu_mat_obj = None
        if gpu_accel:
            try:
                # Test if HIP sparse matrices are supported
                test_mat = PETSc.Mat().create(self.u.function_space.mesh.comm)
                test_mat.setType("aijhipsparse")
                test_mat.setSizes([10, 10])
                test_mat.setUp()
                test_mat.assemble()
                test_mat.destroy()
                self.use_gpu = True
                # Override PC: GAMG/Hypre/SOR are all incompatible with aijhipsparse
                # Jacobi (diagonal scaling) is the only working preconditioner
                pc = self.solver.getPC()
                pc.setType("jacobi")
                self.solver.setTolerances(max_it=5000)
                if self.u.function_space.mesh.comm.rank == 0:
                    print("  🚀 HIP GPU acceleration enabled (PC: Jacobi)")
            except Exception as e:
                if self.u.function_space.mesh.comm.rank == 0:
                    print(f"  ⚠️  GPU requested but HIP not available, falling back to CPU")
                    print(f"      Reason: {type(e).__name__}: {e}")

        assemble_vector(self.rhs_vec, self.rhs_form)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)

    def solve_fem(self):
        """Solve K*x=F for FEM."""
        from fenitop.timing import stats
        
        stats.start('assembly')
        # Restore CPU type for assembly (if GPU mode changed it last iteration)
        if self.use_gpu and self._gpu_active:
            self.lhs_mat.setType("aij")
        self.lhs_mat.zeroEntries()
        assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)
        self.lhs_mat.assemble()
        if self.spring_vec_wrap is not None:
            self.lhs_mat.setDiagonal(self.lhs_mat.getDiagonal() + self.spring_vec)
        
        if self.use_gpu:
            comm = self.u.function_space.mesh.comm
            # Get diagonal for Jacobi BEFORE switching to GPU type
            self._gpu_diag = self.lhs_mat.getDiagonal()
            
            if self._gpu_mat_obj is None:
                self._gpu_mat_obj = self.lhs_mat.convert("aijhipsparse")
            else:
                self.lhs_mat.copy(self._gpu_mat_obj, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            
            self._gpu_mat_obj.assemble()
            self._gpu_active = True
        stats.stop('assembly')
        
        # The rhs_vec is already assembled in __init__
        set_bc(self.rhs_vec, self.bcs)
        
        stats.start('solve')
        if self.use_gpu:
            self._solve_cg_gpu(self.rhs_vec, self.u_wrap)
        else:
            self.solver.solve(self.rhs_vec, self.u_wrap)
        stats.stop('solve')
        
        self.u.x.scatter_forward()
    
    def _solve_cg_gpu(self, b, x, rtol=1e-8, max_it=5000):
        """Manual PCG: GPU SpMV (explicit p_gpu sync) + CPU Jacobi."""
        comm = self.u.function_space.mesh.comm
        # Jacobi preconditioner from pre-extracted diagonal
        diag_arr = self._gpu_diag.getArray().copy()
        diag_arr[np.abs(diag_arr) < 1e-30] = 1.0
        inv_diag = 1.0 / diag_arr
        
        b_norm = b.norm()
        if b_norm == 0.0:
            b_norm = 1.0
        
        # GPU workspace vectors (compatible with _gpu_mat_obj type)
        p_gpu = self._gpu_mat_obj.createVecRight()
        Ax_gpu = self._gpu_mat_obj.createVecLeft()
        # CPU scratch vector (standard type)
        Ax = b.duplicate()
        
        # Initial residual calculation: r = b - A*x
        if x.norm() > 1e-15:
            # Explicitly sync to GPU for mult (safer than relying on PETSc auto-sync)
            x.copy(p_gpu)
            self._gpu_mat_obj.mult(p_gpu, Ax_gpu)
            Ax_gpu.copy(Ax)
            r = b.copy()
            r.axpy(-1.0, Ax)
        else:
            r = b.copy()
        
        # z = M^{-1} * r (on CPU)
        z = r.duplicate()
        z.getArray()[:] = r.getArray() * inv_diag
        
        p = z.copy()
        rz = r.dot(z)
        
        for it in range(max_it):
            r_norm = r.norm()
            if comm.rank == 0 and it % 1000 == 0 and it > 0:
                print(f"  🔍 CG iter {it:4d}: r_norm/b_norm={r_norm/b_norm:.4e}")
            
            if r_norm / b_norm < rtol or it == max_it - 1:
                if comm.rank == 0 and it > 50: # Only print if not trivial solve
                    print(f"  🔍 CG {'Converged' if r_norm/b_norm < rtol else 'Reached Max It'} at iter {it}: r_norm/b_norm={r_norm/b_norm:.4e}")
                break
            
            # Ap = A * p
            p.copy(p_gpu) # CPU -> GPU
            self._gpu_mat_obj.mult(p_gpu, Ax_gpu)
            Ax_gpu.copy(Ax) # GPU -> CPU
            
            pAp = p.dot(Ax)
            if abs(pAp) < 1e-30:
                if comm.rank == 0:
                    print(f"  🔍 CG Breakdown at iter {it}: pAp={pAp:.4e}")
                break
            alpha = rz / pAp
            
            x.axpy(alpha, p)
            r.axpy(-alpha, Ax)
            
            # z = M^{-1} * r
            z.getArray()[:] = r.getArray() * inv_diag
            
            pAp_old = rz
            rz = r.dot(z)
            beta = rz / pAp_old
            p.aypx(beta, z)
        
        p_gpu.destroy()
        Ax_gpu.destroy()
        Ax.destroy()
        r.destroy()
        z.destroy()
        p.destroy()

    def solve_adjoint(self):
        """Solve K*lambda=-L for the adjoint equation."""
        from fenitop.timing import stats
        stats.start('solve')
        rhs = -self.l_vec
        if self.use_gpu:
            self._solve_cg_gpu(rhs, self.lam_wrap)
        else:
            self.solver.solve(rhs, self.lam_wrap)
        rhs.destroy()
        stats.stop('solve')
        self.lam.x.scatter_forward()

    def __del__(self):
        self.solver.destroy()
        self.lhs_mat.destroy()
        self.rhs_vec.destroy()
        self.u_wrap.destroy()
        self.lam_wrap.destroy()
        if self._gpu_diag is not None:
            self._gpu_diag.destroy()
        if self._gpu_mat_obj is not None:
            self._gpu_mat_obj.destroy()
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


def get_2d_refinement(base_grid, density, upsampling_factor=1, iso_smooth=0.0):
    """
    Helper to apply smoothing to 2D grid scalars (iterative averaging).
    Replaces previous Image-based resampling based on user request.
    """
    import pyvista
    import numpy as np
    
    # Create a copy to modify
    refined_grid = base_grid.copy(deep=False)
    refined_grid.point_data["density"] = np.hstack(density)

    if iso_smooth > 0:
        # Simulate Gaussian smoothing on unstructured grid via iterative averaging
        # Heuristic: n_iter ~ iso_smooth
        n_iter = int(max(1, iso_smooth * 10))
        
        for _ in range(n_iter):
             refined_grid = refined_grid.point_data_to_cell_data(pass_point_data=False)
             refined_grid = refined_grid.cell_data_to_point_data(pass_cell_data=False)
             
    return refined_grid




class Plotter():
    def __init__(self, mesh):
        """Initialize a plotter."""
        import pyvista
        self.dim = mesh.topology.dim
        elements, cell_types, nodes = dolfinx.plot.vtk_mesh(mesh, self.dim)
        
        # Map Lagrange types to standard linear types for better visualization
        # 70 (Lagrange Quad) -> 9 (Quad)
        # 72 (Lagrange Hex) -> 12 (Hex)
        cell_types[cell_types == 70] = 9
        cell_types[cell_types == 72] = 12
        
        self.grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)


    def plot(self, density, threshold=0.5, iso_smooth=0.0, smooth_iter=100, path="", filename="optimized_design"):
        import pyvista
        
        # Determine if data is Nodal (Point) or Elemental (Cell)
        n_points = self.grid.n_points
        n_cells = self.grid.n_cells
        data_len = len(density)
        
        is_cell_data = False
        if data_len == n_cells:
            is_cell_data = True
            self.grid.cell_data["density"] = np.hstack(density)
            # Remove point data if it exists to avoid confusion
            if "density" in self.grid.point_data:
                del self.grid.point_data["density"]
        elif data_len == n_points:
            self.grid.point_data["density"] = np.hstack(density)
            if "density" in self.grid.cell_data:
                del self.grid.cell_data["density"]
        else:
            print(f"⚠️  Plotter Error: Data size {data_len} does not match Points ({n_points}) or Cells ({n_cells})")
            return

        
        # Determine whether to use cell-based (discrete) or nodal-based (smooth iso-contour)
        # 2D: Use nodal thresholding for "sub-element resolution" (User request for ISO lines)
        # 3D: Use nodal thresholding for smooth iso-surface
        
        if self.dim == 2:
            # Re-implement optional refinement for smoothing
            if iso_smooth > 0:
                # Refinement requires Nodal Data
                grid_for_refinement = self.grid
                if is_cell_data:
                    grid_for_refinement = self.grid.cell_data_to_point_data()
                
                refined_grid = get_2d_refinement(grid_for_refinement, 
                                                 grid_for_refinement.point_data["density"], 
                                                 upsampling_factor=1, iso_smooth=iso_smooth)
                grid = refined_grid.threshold(threshold, scalars="density")
                grid_for_contour = refined_grid
            else:
                # If cell data, converting to point data usually gives better contours
                if is_cell_data:
                     grid_for_contour = self.grid.cell_data_to_point_data()
                else:
                     grid_for_contour = self.grid
                     
                grid = self.grid.threshold(threshold, scalars="density")
        else:
            # 3D Logic
            if is_cell_data:
                 # If we have cell data, we can use it directly for thresholding volume
                 density_array = self.grid.cell_data["density"]
            else:
                 # If point data, convert to cell for max check? Or just use point data
                 density_array = self.grid.point_data["density"]

            max_density = np.max(density_array)
            if max_density < threshold:
                adaptive_threshold = max(0.01, max_density * 0.8)
                print(f"   Adapting threshold: {threshold:.2f} → {adaptive_threshold:.2f} (max density: {max_density:.3f})")
                threshold = adaptive_threshold
            
            grid = self.grid.threshold(threshold, scalars="density")
            
        empty_mesh = (self.dim == 3 and grid.n_faces == 0)

        if not empty_mesh:
            plotter = pyvista.Plotter(off_screen=True)
            plotter.background_color = "white"
            lighting = self.dim == 3
            
            if self.dim == 2:
                # 1. Plot the actual density field (grayscale) to show material distribution
                # Use the original grid for the "mushy" background
                #plotter.add_mesh(self.grid, clim=[0, 1], cmap="Greys", show_edges=True, edge_color="lightgray", line_width=0.5)
                
                # 2. Explicitly plot the RED ISO-line (the boundary) for sub-element resolution
                try:
                    iso_line = grid_for_contour.contour(isosurfaces=[threshold], scalars="density")
                    plotter.add_mesh(iso_line, color="red", line_width=3, label=f"ISO {threshold:.2f}")
                except:
                    pass # Contour might fail if data is too uniform
                
                plotter.view_xy()
            else:
                # 3D Visualization
                if self.dim == 3:
                    grid = grid.smooth(n_iter=smooth_iter)
                    grid.point_data["density"] = 0.4
                
                plotter.add_mesh(grid, clim=[0, 1], cmap="Greys", lighting=lighting,
                                 show_scalar_bar=False, show_edges=False, edge_color="lightgray", line_width=0.5)
            
            # Construct file path safely
            save_path = os.path.join(path, f"{filename}.jpg")
            plotter.screenshot(save_path, window_size=(2000, 2000)) # Increased resolution
            plotter.close()
        else:
            print(f"   ⚠️  Warning: Mesh is empty after thresholding (threshold={threshold:.2f}). No JPG created.")


def save_xdmf(mesh, rho, path="", filename="optimized_design"):
    save_path = os.path.join(path, f"{filename}.xdmf")
    with dolfinx.io.XDMFFile(mesh.comm, save_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        rho.name = "density"
        xdmf.write_function(rho)

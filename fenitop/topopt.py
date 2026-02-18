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

import time
import os
import sys

import numpy as np
from mpi4py import MPI
import dolfinx.fem  # Added for rank-0 saving

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, Plotter, save_xdmf


def get_logical_rank(comm):
    """Determine rank even if mpi4py didn't initialize correctly."""
    if comm.size > 1:
        return comm.rank

    # Fallback checks for common MPI environment variables
    for key in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MPI_LOCALRANKID', 'MV2_COMM_WORLD_RANK', 'SLURM_PROCID']:
        if key in os.environ:
            try:
                return int(os.environ[key])
            except ValueError:
                pass
    return 0


def topopt(fem, opt):
    """Main function for topology optimization."""

    # Initialization
    comm = MPI.COMM_WORLD
    rank = get_logical_rank(comm)

    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])

    heaviside = Heaviside(rho_phys_field)

    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_field.function_space, fem["mesh_serial"])
    S_comm_phys = Communicator(rho_phys_field.function_space, fem["mesh_serial"])


    if rank == 0:  # Changed from comm.rank
        plotter = Plotter(fem["mesh_serial"])


    num_consts = 1 if opt["opt_compliance"] else 2
    # Get owned DOFs only (not ghosts) for array sizing
    num_elems = rho_field.function_space.dofmap.index_map.size_local
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # Apply passive zones
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T

    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    if "initial_density" in opt and opt["initial_density"] is not None:
        initial_rho = opt["initial_density"]
        if len(initial_rho) == num_elems:
            if rank == 0:
                 print("   ✅ Resuming optimization from previous state.")
            rho_ini = initial_rho
            rho_ini[solid], rho_ini[void] = 0.995, 0.005
            rho_field.x.petsc_vec.array[:num_elems] = rho_ini
        else:
            if rank == 0:
                 print(f"   ⚠️  Resume failed: Data size mismatch (Expected {num_elems}, Got {len(initial_rho)}). Starting fresh.")
            rho_ini = np.full(num_elems, opt["vol_frac"])
            rho_ini[solid], rho_ini[void] = 0.995, 0.005
            rho_field.x.petsc_vec.array[:num_elems] = rho_ini
    else:
        rho_ini = np.full(num_elems, opt["vol_frac"])
        rho_ini[solid], rho_ini[void] = 0.995, 0.005
        rho_field.x.petsc_vec.array[:num_elems] = rho_ini
    # Synchronize ghost DOFs across processes after initialization
    rho_field.x.scatter_forward()
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01


    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        # Density filter and Heaviside projection
        from fenitop.timing import stats
        stats.start('filter')
        density_filter.forward()

        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)
        stats.stop('filter')

        # Solve FEM
        # (Timing is handled inside solve_fem)
        linear_problem.solve_fem()

        # Compute function values and sensitivities
        stats.start('sensitivity')
        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])
        stats.stop('sensitivity')

        # Update the design variables (use only owned DOFs, not ghosts)
        stats.start('update')
        owned_size = rho_field.function_space.dofmap.index_map.size_local
        rho_values = rho_field.x.petsc_vec.array[:owned_size].copy()
        if opt["opt_compliance"] and opt["use_oc"]:
            rho_new, change = optimality_criteria(
                rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(
                num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy()
            rho_old1 = rho_values.copy()

        # Check for NaN values and stop optimization if detected
        if np.isnan(change) or np.any(np.isnan(rho_new)):
            if rank == 0:  # Changed from comm.rank
                print(f"\n⚠️  WARNING: NaN detected at iteration {opt_iter}. Stopping optimization.")
                print(f"   Using design from iteration {opt_iter-1}.")
            break

        # Update only owned DOFs (not ghosts)
        rho_field.x.array[:owned_size] = rho_new
        # Synchronize ghost DOFs across processes (critical for parallel correctness!)
        rho_field.x.scatter_forward()
        stats.stop('update')

        # Output the histories
        opt_time = time.perf_counter() - opt_start_time

        # Call progress callback or print status (Rank 0 only)
        if rank == 0:  # Changed from comm.rank
            if "progress_callback" in opt and opt["progress_callback"] is not None:
                try:
                    should_continue = opt["progress_callback"](opt_iter, C_value, V_value, change, field=rho_phys_field, comm=S_comm_phys)
                    if should_continue is False:
                        break
                except Exception:
                    pass  # Silently ignore callback errors
            else:
                print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                      f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                      f"U: {U_value:.3f}, change: {change:.3f}", flush=True)

    design_values = S_comm.gather(rho_field.x.petsc_vec)
    physical_values = S_comm_phys.gather(rho_phys_field.x.petsc_vec) 
    
    final_results = None
    if rank == 0:  # Changed from comm.rank
        final_results = {
            "design": design_values,
            "physical": physical_values
        }
        
        # 1. JPG Visualization
        try:
            print(f"\n📊 Creating JPG visualization...")
            print(f"   Density shape: {physical_values.shape if hasattr(physical_values, 'shape') else len(physical_values)}")
            print(f"   Density range: [{np.min(physical_values):.3f}, {np.max(physical_values):.3f}]")
            filename = opt.get("filename", "optimized_design")
            plotter.plot(physical_values, filename=filename)
            print(f"✅ JPG visualization created: {filename}.jpg")
        except Exception as e:
            print(f"⚠️  Warning: Failed to create JPG visualization: {e}")
            import traceback
            traceback.print_exc()

        # 2. XDMF Saving
        try:
            filename = opt.get("filename", "optimized_design")
            # Save results on rank 0 only to avoid file locking
            # Save Physical Density for Visualization compatibility (matches prior behavior)
            V_serial = dolfinx.fem.functionspace(fem["mesh_serial"], rho_phys_field.function_space.ufl_element())
            rho_serial = dolfinx.fem.Function(V_serial)
            rho_serial.x.array[:] = physical_values
            save_xdmf(fem["mesh_serial"], rho_serial, filename=filename)
        except Exception as e:
            #print(f"⚠️  Warning: Failed to save xdmf: {e}")
            import traceback
            traceback.print_exc()

    
    return final_results

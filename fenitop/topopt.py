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
on_setup/on_iteration/on_finish callbacks, resume support, phase timing,
NaN detection, DOLFINx 0.11 port.
Full list of deviations from upstream: CHANGES_FROM_FENITOP.md
in https://github.com/Grumium/topologui
"""

import time

import numpy as np
from mpi4py import MPI
import dolfinx.fem
import dolfinx.la

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.design_symmetry import DesignSymmetry
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, save_xdmf, save_vtkhdf


def topopt(fem, opt, on_iteration=None, on_setup=None, on_finish=None):
    """Main function for topology optimization.

    Parameters
    ----------
    fem : dict
        FEM problem definition.
    opt : dict
        Optimization settings.
    on_iteration : callable, optional
        Called on **all MPI ranks** after each iteration as
        ``on_iteration(iter, C, V, change) -> bool | None``.
        Return ``False`` to stop the optimization early.
        MPI collectives (e.g. field gathers) are safe inside this callback.
    on_setup : callable, optional
        Called once on **all MPI ranks** immediately after the FEM problem is
        set up (after form_fem, before the optimization loop).  Signature::

            on_setup(u_field, rho_field, rho_phys_field,
                     S_comm, S_comm_phys, V_comm, fem_dict)

        Use this to capture live field references for use inside on_iteration.
    on_finish : callable, optional
        Called once on **all MPI ranks** just before topopt returns, while all
        PETSc/MPI objects are still alive.  Signature::

            on_finish(u_field, rho_field, rho_phys_field,
                      S_comm, S_comm_phys, V_comm, fem_dict)

        Use this to perform any final field gathers safely.
    """

    # Initialization
    comm = MPI.COMM_WORLD
    rank = comm.rank

    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])

    heaviside = Heaviside(rho_phys_field)

    # ---- Passive zones -----------------------------------------------------
    num_elems = rho_field.function_space.dofmap.index_map.size_local
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)

    # Clamping the design variable (rho_min/rho_max, further down) is not
    # enough on its own: the PDE filter averages a solid zone's 0.99 against
    # the void beside it and the Heaviside projection then erodes that ~R-wide
    # boundary layer, so the zone thins out exactly where the structure
    # attaches to it.  Re-imposing the zones on the physical density inside the
    # loop fixes that.  Switchable because it changes converged designs near
    # the zones (identical objective, different local arrangement), so a run
    # can still be reproduced against a pre-fix reference.
    enforce_passive = bool(opt.get("enforce_passive_zones", True))
    solid_nodes = void_nodes = None
    num_nodes_owned = rho_phys_field.function_space.dofmap.index_map.size_local

    if enforce_passive:
        # The zones have to be expressed as node masks: the design variable rho
        # lives in DG0 and rho_phys in CG1, so the cell masks above cannot index
        # it.  Evaluating the zone predicate at the node coordinates would be
        # the obvious route, but it disagrees with the cell mask at the rim --
        # the nodal set reaches half a cell further into the design domain than
        # the cells whose centroids are in the zone, which dilates every zone by
        # that much.  So a node is taken to be in the zone only when *all* of
        # its adjacent cells are, which agrees with the cell-level clamp.
        S_node = rho_phys_field.function_space
        mesh_t = fem["mesh"].topology
        cell_nodes = np.asarray(S_node.dofmap.list)[:mesh_t.index_map(mesh_t.dim).size_local]

        def _interior_nodes(cell_mask):
            """Nodes every one of whose adjacent cells lies in the zone.

            Counted through owned cells and reduced across ranks the way an
            assembly is, so the mask does not depend on the partitioning: a
            node on a rank boundary would otherwise look interior on the rank
            that happens to hold only its zone-side cells.
            """
            total = dolfinx.fem.Function(S_node)
            in_zone = dolfinx.fem.Function(S_node)
            np.add.at(total.x.array, cell_nodes.ravel(), 1.0)
            np.add.at(in_zone.x.array, cell_nodes[np.asarray(cell_mask, dtype=bool)].ravel(), 1.0)
            for f in (total, in_zone):
                f.x.scatter_reverse(dolfinx.la.InsertMode.add)
                f.x.scatter_forward()
            return (total.x.array > 0) & np.isclose(in_zone.x.array, total.x.array)

        solid_nodes = _interior_nodes(solid)
        void_nodes = _interior_nodes(void)

    # Symmetric design without a reduced domain.  The planes here are the
    # ones the run does not exploit -- typically because an uncertain load
    # direction rules out halving -- so the symmetry the user asked for is
    # imposed on the design instead of on the solve.
    design_symmetry = DesignSymmetry(comm, rho_field,
                                     opt.get("design_symmetry_planes"))
    if design_symmetry and rank == 0:
        print(f"  ⚖️  Symmetric design enforced over {design_symmetry.group_size} "
              f"orbit position(s) per cell", flush=True)

    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_field.function_space, fem["mesh_serial"])
    S_comm_phys = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    V_comm = Communicator(u_field.function_space, fem["mesh_serial"])

    # Notify caller of live field references before the loop starts.
    if on_setup is not None:
        try:
            on_setup(u_field, rho_field, rho_phys_field, S_comm, S_comm_phys, V_comm, fem)
        except Exception:
            pass


    num_consts = 1 if opt["opt_compliance"] else 2
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # Apply passive zones (solid/void evaluated above, before Sensitivity)
    initial_rho = opt.get("initial_density")
    num_elems_global = rho_field.function_space.dofmap.index_map.size_global
    if initial_rho is not None:
        # initial_density is always the GLOBAL array (rank-0 gathered from a
        # previous run, size == num_elems_global).  Broadcast it to all ranks
        # then use S_comm.bcast to scatter each rank's local slice.
        if len(initial_rho) == num_elems_global:
            if rank == 0:
                print("   ✅ Resuming optimization from previous state.")
            # MPI-broadcast so all ranks have the global array
            initial_rho_global = comm.bcast(initial_rho, root=0)
            # S_comm.bcast is a local indexing op — distributes global→local DOFs
            S_comm.bcast(rho_field, initial_rho_global)
            rho_ini = rho_field.x.petsc_vec.array[:num_elems].copy()
            rho_ini[solid], rho_ini[void] = 0.995, 0.005
            rho_field.x.petsc_vec.array[:num_elems] = rho_ini
        else:
            if rank == 0:
                print(f"   ⚠️  Resume failed: size mismatch "
                      f"(expected {num_elems_global}, got {len(initial_rho)}). Starting fresh.")
            initial_rho = None  # fall through to fresh start
    if initial_rho is None:
        rho_ini = np.full(num_elems, opt["vol_frac"])
        rho_ini[solid], rho_ini[void] = 0.995, 0.005
        rho_field.x.petsc_vec.array[:num_elems] = rho_ini
    # Synchronize ghost DOFs across processes after initialization
    rho_field.x.scatter_forward()
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    # Restore optimizer state for resume.
    resume_opt_iter = int(opt.get("opt_iter", 0)) if initial_rho is not None else 0
    resume_beta     = int(opt.get("beta", 1))      if initial_rho is not None else 1

    if not opt["use_oc"]:
        ms = opt.get("mma_state") or {}
        _rho_old1 = ms.get("rho_old1")
        if ms and _rho_old1 is not None and len(_rho_old1) == num_elems_global:
            # Global arrays — scatter to each rank's local slice via S_comm.bcast
            import dolfinx.fem as _dfem
            _tmp = _dfem.Function(rho_field.function_space)
            for key, target in [("rho_old1", None), ("rho_old2", None),
                                 ("low", None), ("upp", None)]:
                val = ms.get(key)
                if val is not None:
                    g = comm.bcast(val, root=0)
                    S_comm.bcast(_tmp, g)
                    if key == "rho_old1":
                        rho_old1 = _tmp.x.petsc_vec.array[:num_elems].copy()
                    elif key == "rho_old2":
                        rho_old2 = _tmp.x.petsc_vec.array[:num_elems].copy()
                    elif key == "low":
                        low = _tmp.x.petsc_vec.array[:num_elems].copy()
                    elif key == "upp":
                        upp = _tmp.x.petsc_vec.array[:num_elems].copy()

    # Start topology optimization
    opt_iter, beta, change = resume_opt_iter, resume_beta, 2*opt["opt_tol"]
    stopped_early = False
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
            # Proactively prepare the linear solver for the new, much sharper
            # material contrast: force GAMG hierarchy rebuild + reset initial
            # guess before the first solve with the new beta.
            linear_problem.notify_beta_change(new_beta=beta)
        heaviside.forward(beta)

        if enforce_passive:
            # Hard-setting rho_phys makes drho vanish on those nodes, which is
            # also the consistent derivative of this modified density map --
            # zeroing it keeps the sensitivities exact rather than merely
            # suppressing them.
            rho_phys_field.x.array[solid_nodes] = 1.0
            rho_phys_field.x.array[void_nodes] = 0.0
            heaviside.drho[solid_nodes[:num_nodes_owned]] = 0.0
            heaviside.drho[void_nodes[:num_nodes_owned]] = 0.0
        stats.stop('filter')

        # Solve FEM
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
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt.get("compliance_bound", 1e9)])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])
        # Project the sensitivities onto the symmetric subspace before the
        # step, so the optimizer never proposes a step that leaves it.  The
        # projector is symmetric, which is exactly what the chain rule wants
        # for a design constrained to that subspace.
        if design_symmetry:
            dJdrho = design_symmetry.apply(dJdrho)
            dgdrho = np.vstack([design_symmetry.apply(row) for row in dgdrho])
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

        # The update is elementwise, so a symmetric iterate and symmetric
        # sensitivities keep it symmetric -- but the filter runs on an
        # unstructured mesh, and its round-off would accumulate over hundreds
        # of iterations.  Re-projecting costs one more MatMult.
        if design_symmetry:
            rho_new = design_symmetry.apply(rho_new)

        # Check for NaN values and stop optimization if detected
        if np.isnan(change) or np.any(np.isnan(rho_new)):
            if rank == 0:
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

        if on_iteration is not None:
            # Called on ALL ranks so that MPI collectives inside the callback
            # (e.g. Communicator.gather) work correctly across processes.
            try:
                should_continue = on_iteration(opt_iter, C_value, V_value, change)
            except Exception:
                should_continue = True
            # Rank 0 decides whether to stop; broadcast to all ranks.
            should_continue = comm.bcast(should_continue, root=0)
            if should_continue is False:
                stopped_early = True
                break
        else:
            if rank == 0:
                print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                      f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                      f"U: {U_value:.3f}, change: {change:.3f}", flush=True)

    # Save design_raw as the GLOBAL gathered array so it can be scattered back
    # to any number of MPI ranks on resume (local partition sizes vary by rank count).
    design_raw = S_comm.gather(rho_field.x.petsc_vec)

    # MMA history for resume — saved as GLOBAL arrays (rank 0 only) so they
    # can be scattered back to any number of MPI ranks on resume.
    mma_state = None
    if not opt["use_oc"]:
        mma_state = {
            "rho_old1": S_comm.gather(rho_old1),
            "rho_old2": S_comm.gather(rho_old2),
            "low":  S_comm.gather(low)  if low is not None else None,
            "upp":  S_comm.gather(upp)  if upp is not None else None,
        }

    # Skip MPI-collective gathers when stopped early — the process may be
    # killed shortly after on_iteration returns False, and calling a collective
    # while another rank is already dead causes a segfault.
    # The GUI adapter fills design/physical from _captured instead.
    if stopped_early:
        # Call on_finish before returning so the adapter can safely gather
        # fields while all PETSc objects are still alive.
        if on_finish is not None:
            try:
                on_finish(u_field, rho_field, rho_phys_field,
                          S_comm, S_comm_phys, V_comm, fem)
            except Exception:
                pass
        final_results = None
        if rank == 0:
            final_results = {
                "design": None,
                "design_raw": design_raw,
                "beta": beta,
                "opt_iter": opt_iter,
                "physical": None,
                "mma_state": mma_state,
            }
        return final_results

    design_values = S_comm.gather(rho_field.x.petsc_vec)
    physical_values = S_comm_phys.gather(rho_phys_field.x.petsc_vec)

    final_results = None
    if rank == 0:
        final_results = {
            "design": design_values,
            "design_raw": design_raw,
            "beta": beta,
            "opt_iter": opt_iter,
            "physical": physical_values,
            "mma_state": mma_state,
        }

        # Only produce file output when running in script/CLI mode.
        # When on_iteration is provided the caller manages its own output (e.g. GUI).
        if on_iteration is None:
            try:
                from topologui.visualization.plotter import Plotter
                plotter = Plotter(fem["mesh_serial"])
                filename = opt.get("filename", "optimized_design")
                print("\n📊 Creating JPG visualization...")
                plotter.plot(physical_values, filename=filename)
                print(f"✅ JPG visualization created: {filename}.jpg")
            except ImportError:
                pass  # topologui not installed (e.g. HPC / headless environment)
            except Exception as e:
                print(f"⚠️  Warning: Failed to create JPG visualization: {e}")

            filename = opt.get("filename", "optimized_design")
            rho_serial = None
            try:
                V_serial = dolfinx.fem.functionspace(fem["mesh_serial"], rho_phys_field.function_space.ufl_element())
                rho_serial = dolfinx.fem.Function(V_serial)
                rho_serial.x.array[:] = physical_values
            except Exception as e:
                print(f"⚠️  Warning: Failed to build serial density field: {e}")

            if rho_serial is not None:
                try:
                    save_xdmf(fem["mesh_serial"], rho_serial, filename=filename)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to save XDMF: {e}")

                # Alongside, not instead of: XDMF stays what the rest of the
                # toolchain reads, VTKHDF is the single-file format Kitware is
                # standardising on.  Separately guarded so that losing it never
                # looks like an XDMF failure.
                if opt.get("save_vtkhdf", True):
                    try:
                        save_vtkhdf(fem["mesh_serial"], rho_serial,
                                    filename=filename)
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to save VTKHDF: {e}")

    # Call on_finish before returning — PETSc objects are still alive here.
    if on_finish is not None:
        try:
            on_finish(u_field, rho_field, rho_phys_field,
                      S_comm, S_comm_phys, V_comm, fem)
        except Exception:
            pass

    return final_results


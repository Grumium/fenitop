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
symmetry scaling of the matrix-vector compliance and displacement
objectives, DOLFINx 0.11 port.
Full list of deviations from upstream: CHANGES_FROM_FENITOP.md
in https://github.com/Grumium/topologui
"""

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc
# from fenitop.deterministic_mpi import deterministic_sum  # DISABLED for speed comparison


class Sensitivity():
    def __init__(self, comm, opt, problem, u_field, lambda_field, rho_phys):
        # Symmetry scaling factor (2^n for n symmetry planes).
        # The UFL forms (compliance, f_int) already include this factor,
        # but the matrix-vector path (u^T K u) for opt_compliance=False
        # needs explicit scaling.
        self.sym_factor = opt.get("_sym_factor", 1)

        # Compliance
        if opt["opt_compliance"]:
            self.C_form = form(opt["compliance"])
        self.dCdrho_form = form(-ufl.derivative(opt["compliance"], rho_phys))
        self.dCdrho_vec = create_vector(self.dCdrho_form.function_spaces[0])

        # Direction-uncertain load.  The compliance of a load of fixed
        # magnitude acting in direction d is the quadratic form d^T A d, with
        # A[i,j] = f_i . u_j over the orthogonal basis loads assembled in
        # form_fem().  So the worst direction over the *entire* angular range
        # is the largest eigenvalue of a dim x dim matrix -- no angle sampling.
        #
        # Optimizing max(lambda) directly does not work: the eigenvalues
        # coalesce exactly when the design carries all directions equally,
        # which is the optimum this is aiming for, and max() is
        # non-differentiable there.  A p-norm over the eigenvalues is smooth,
        # and for dim <= 3 it overestimates the true worst case by at most
        # dim**(1/p) -- 9% for p = 8 in 2D, and only where the directions are
        # already balanced.
        self.direction_rhs = opt.get("direction_rhs") is not None
        self.u_field = u_field
        if self.direction_rhs:
            self.problem = problem
            self.direction_mode = opt.get("direction_mode", "any")
            self.transverse_ratio = float(opt.get("transverse_ratio", 0.0))
            self.p_norm = float(opt.get("direction_p_norm", 8.0))
            # Relative floor for the |A12| kink below, as a fraction of the
            # Cauchy-Schwarz bound sqrt(A11*A22) so it carries no units.
            self.transverse_smoothing = float(
                opt.get("transverse_smoothing", 1e-3))
            self.dCdrho_dir = create_vector(self.dCdrho_form.function_spaces[0])

        # Volume
        self.comm = comm
        # Use standard MPI sum for speed (non-deterministic but faster)
        self.total_volume = comm.allreduce(
            assemble_scalar(form(opt["total_volume"])), op=MPI.SUM)
        self.V_form = form(opt["volume"])
        dVdrho_form = form(ufl.derivative(opt["volume"], rho_phys))
        self.dVdrho_vec = create_vector(dVdrho_form.function_spaces[0])
        assemble_vector(self.dVdrho_vec, dVdrho_form)
        self.dVdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.dVdrho_vec /= self.total_volume

        # Displacement
        self.opt_compliance = opt["opt_compliance"]
        if not self.opt_compliance:
            self.dfdrho_form = form(ufl.adjoint(ufl.derivative(opt["f_int"], rho_phys)))
            self.dfdrho_mat = create_matrix(self.dfdrho_form)
            self.problem = problem
            self.l_vec_wrap = opt["l_vec"]
            self.l_vec = self.l_vec_wrap.petsc_vec if self.l_vec_wrap is not None else None
            self.u_field, self.lambda_field = u_field, lambda_field
            self.dUdrho_vec = rho_phys.x.petsc_vec.copy()
            self.prod_vec = u_field.x.petsc_vec.copy()

    def __del__(self):
        if not self.opt_compliance:
            self.prod_vec.destroy()

    def _combine(self, coefficients):
        """Put sum_i coefficients[i]*u_i into u_field and return -u^T K' u.

        That quantity is the ordinary compliance sensitivity of a design
        loaded by sum_i coefficients[i]*f_i, so every objective below is
        assembled from the one derivative form the fixed-load path uses.
        """
        u_vec = self.u_field.x.petsc_vec
        u_vec.zeroEntries()
        for coefficient, u_i in zip(coefficients, self.problem.u_dir):
            u_vec.axpy(float(coefficient), u_i)
        self.u_field.x.scatter_forward()
        with self.dCdrho_dir.localForm() as loc:
            loc.set(0)
        assemble_vector(self.dCdrho_dir, self.dCdrho_form)
        self.dCdrho_dir.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                    mode=PETSc.ScatterMode.REVERSE)
        return self.dCdrho_dir

    def _evaluate_transverse(self, A):
        """Main load plus an orthogonal perturbation of at most `eps` of it.

        With the basis aligned to the load, the admissible coefficient is
        c = (1, s) with |s| <= eps, so the compliance is the scalar parabola
        A11 + 2*A12*s + A22*s^2.  A22 >= 0 makes it convex, so the worst case
        sits on an endpoint and is closed form -- no eigenproblem, no root
        find:

            C = A11 + eps^2*A22 + 2*eps*|A12|

        |A12| kinks at zero, which is not a corner case but the norm: A12
        vanishes identically whenever the structure is symmetric about the
        load, and then both signs of the perturbation are equally bad.  The
        modulus is therefore floored by delta = c*sqrt(A11*A22), which at
        A12 = 0 yields the average of the two one-sided derivatives.

        Note that delta is itself a function of the design, so differentiating
        the floored modulus is not simply the sensitivity at the worst-case
        load -- treating delta as a constant leaves a relative error that
        grows with the smoothing (0.03% at c = 1e-2, which a finite-difference
        check picks up).  All three entries of A are therefore differentiated
        separately and recombined.
        """
        eps = self.transverse_ratio
        A11, A12, A22 = A[0, 0], A[0, 1], A[1, 1]
        c = self.transverse_smoothing

        W2 = max(A11*A22, 0.0)
        modulus = float(np.sqrt(A12*A12 + c*c*W2))
        C_value = A11 + eps*eps*A22 + 2.0*eps*modulus

        # dC/drho = w11*A11' + w22*A22' + w12*A12', from
        #   d|A12|_smoothed = [A12*A12' + c^2*(A11'*A22 + A11*A22')/2] / modulus
        if modulus > 0.0:
            w11 = 1.0 + eps*c*c*A22/modulus
            w22 = eps*eps + eps*c*c*A11/modulus
            w12 = 2.0*eps*A12/modulus
        else:                      # A12 == 0 and no smoothing: subgradient 0
            w11, w22, w12 = 1.0, eps*eps, 0.0

        # A11' and A22' come straight from the derivative form; the cross term
        # is recovered by polarization, since the form is quadratic in u:
        #   Q(u1 + u2) = Q(u1) + 2*A12' + Q(u2).
        # Three assemblies of an existing form, and no further FEM solves.
        d11 = self._combine([1.0, 0.0]).copy()
        d22 = self._combine([0.0, 1.0]).copy()
        d_sum = self._combine([1.0, 1.0])

        self._store(d_sum)
        self.dCdrho_vec.scale(0.5*w12)
        self.dCdrho_vec.axpy(w11 - 0.5*w12, d11)
        self.dCdrho_vec.axpy(w22 - 0.5*w12, d22)
        d11.destroy()
        d22.destroy()

        # Leave u_field on the worst-case load, which is what the viewer shows.
        sign = 1.0 if A12 >= 0.0 else -1.0
        self._combine([1.0, eps*sign])
        return C_value

    def _store(self, vector):
        with self.dCdrho_vec.localForm() as loc:
            loc.set(0)
        self.dCdrho_vec.axpy(1.0, vector)

    def _evaluate_direction(self):
        """Worst-case compliance over the admissible loads, and its gradient.

        The compliance of the load sum_i c_i f_i is the quadratic form c^T A c
        with A[i,j] = f_i . u_j, so once the basis is solved the worst case is
        a tiny optimization over c on a 2x2 or 3x3 matrix -- no further FEM
        solves, whatever shape the admissible set has.  Which shape it is
        depends on the mode: the whole unit sphere ("any"), or the main
        direction with a bounded orthogonal perturbation (a coordinate plane).

        Leaves u_field holding the worst case's displacement, which is the
        field worth looking at, and the one the viewer shows.
        """
        loads, disps = self.problem.rhs_dir, self.problem.u_dir
        n = len(loads)

        # PETSc's dot is already a global reduction, so A is the same on every
        # rank without a further allreduce.
        A = np.array([[loads[i].dot(disps[j]) for j in range(n)]
                      for i in range(n)])
        A = 0.5*(A + A.T)          # symmetric by construction, not by round-off

        if self.direction_mode != "any":
            return self._evaluate_transverse(A)

        lam, vecs = np.linalg.eigh(A)
        lam = np.clip(lam, 0.0, None)   # A is positive semidefinite

        peak = lam[-1]                  # eigh returns them ascending
        if peak <= 0.0:
            with self.dCdrho_vec.localForm() as loc:
                loc.set(0)
            return 0.0

        # Scaled p-norm: (sum lam^p)^(1/p) = peak * S^(1/p), which keeps
        # lam**p from overflowing for a stiff design.
        ratios = lam/peak
        S = float(np.sum(ratios**self.p_norm))
        C_value = peak * S**(1.0/self.p_norm)
        dCdlam = S**(1.0/self.p_norm - 1.0) * ratios**(self.p_norm - 1.0)

        # dlambda_k/drho is the ordinary compliance sensitivity evaluated at
        # the combined field sum_i vecs[i,k]*u_i, because
        # d(f^T K^-1 f)/drho = -u^T K' u carries over to the eigenpair.
        with self.dCdrho_vec.localForm() as loc:
            loc.set(0)
        for k in range(n):           # ascending, so u_field ends on the worst
            self.dCdrho_vec.axpy(float(dCdlam[k]), self._combine(vecs[:, k]))
        return C_value

    def evaluate(self):
        # Compliance
        if self.direction_rhs:
            C_value = self._evaluate_direction()
        elif self.opt_compliance:
            # Uses the (already symmetry-scaled) UFL form
            C_value = self.comm.allreduce(assemble_scalar(self.C_form), op=MPI.SUM)
        else:
            # Matrix-vector path: u^T K u integrates only over the reduced
            # domain — apply the symmetry factor explicitly.
            self.problem.lhs_mat.mult(self.u_field.x.petsc_vec, self.prod_vec)
            C_value = self.u_field.x.petsc_vec.dot(self.prod_vec) * self.sym_factor
        if not self.direction_rhs:
            with self.dCdrho_vec.localForm() as loc:
                loc.set(0)
            assemble_vector(self.dCdrho_vec, self.dCdrho_form)
            self.dCdrho_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                                        mode=PETSc.ScatterMode.REVERSE)
        # Use standard MPI sum for speed (non-deterministic but faster)
        actual_volume = self.comm.allreduce(assemble_scalar(self.V_form), op=MPI.SUM)
        V_value = actual_volume / self.total_volume
        self.dVdrho_vec_copy = self.dVdrho_vec.copy()

        # Displacement
        if not self.opt_compliance:
            U_value = self.u_field.x.petsc_vec.dot(self.l_vec) * self.sym_factor
            self.problem.solve_adjoint()
            self.dfdrho_mat.zeroEntries()
            assemble_matrix(self.dfdrho_mat, self.dfdrho_form)
            self.dfdrho_mat.assemble()
            self.dfdrho_mat.mult(self.lambda_field.x.petsc_vec, self.dUdrho_vec)
        else:
            U_value, self.dUdrho_vec = 0, None

        func_values = [C_value, V_value, U_value]
        sensitivities = [self.dCdrho_vec, self.dVdrho_vec_copy, self.dUdrho_vec]
        return func_values, sensitivities

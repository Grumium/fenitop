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

        # Load with an uncertain direction.  The compliance of the load
        # sum_i c_i f_i is the quadratic form c^T A c with A[i,j] = f_i . u_j
        # over the basis loads assembled in form_fem(), so once the basis is
        # solved every statistic of the compliance follows from a matrix of
        # side `len(direction_rhs)` -- no angle sampling, and no further FEM
        # solves whatever the load distribution is.
        #
        # Modelling c as a random vector of mean `mu` and covariance `cov`
        # rather than as a member of a set, both moments are closed form:
        #
        #     E[C]   = <A, mu mu^T + cov>
        #     Var[C] = 2 tr(A cov A cov) + 4 mu^T A cov A mu
        #
        # and the objective is their 2-norm, following Torii (CMAME 352,
        # 2019): J = sqrt(E^2 + (kappa*sd)^2).  The familiar weighted sum
        # E + kappa*sd is the same expression with the 1-norm, for which no
        # proof of physical consistency is available; for the 2-norm,
        # kappa <= 1 -- no more weight on the scatter than on the mean --
        # guarantees it.  Both moments are polynomial in A, so the objective
        # is smooth everywhere: no eigenvalues to coalesce, no |.| to floor,
        # and nothing whose sign can flip between iterations.
        self.direction_rhs = opt.get("direction_rhs") is not None
        self.u_field = u_field
        if self.direction_rhs:
            self.problem = problem
            self.dir_mu = np.asarray(opt["direction_mu"], dtype=float)
            self.dir_cov = np.asarray(opt["direction_cov"], dtype=float)
            self.dir_kappa = float(opt.get("direction_kappa", 1.0))
            # The load to show in the viewer.  The worst case is no longer
            # the field being optimized -- the objective covers a whole
            # distribution -- so the nominal load is both the honest choice
            # and a stable one from iteration to iteration.
            self.dir_nominal = np.asarray(
                opt.get("direction_nominal", self.dir_mu), dtype=float)
            # Which parity class each basis load belongs to on a symmetric
            # half domain; all zeros when the domain is whole.
            self.dir_parity = np.asarray(
                opt.get("direction_parity")
                or [0]*len(opt["direction_rhs"]), dtype=int)
            self.dCdrho_dir = create_vector(self.dCdrho_form.function_spaces[0])
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

    def _evaluate_direction(self):
        """Objective over the load distribution, and its gradient.

        Assembles A once from the basis solutions, reads both moments off it
        in closed form, and turns the derivative of the objective with
        respect to A into a design sensitivity.

        That last step is the only part worth explaining.  dJ/drho is
        sum_ij G_ij * dA_ij/drho with G = dJ/dA, and dA_ij/drho = -u_i^T K'
        u_j, which is not a form we have -- `_combine` gives the quadratic
        form Q(c) = sum_ij c_i c_j dA_ij/drho for one coefficient vector.
        Diagonalizing G = sum_k g_k w_k w_k^T therefore recovers exactly
        what is needed as sum_k g_k Q(w_k): `n` assemblies of an existing
        form, no further solves, and quadratic in each w_k, so the arbitrary
        sign of an eigenvector cannot reach the result.

        Leaves u_field on the nominal load, which is what the viewer shows.
        """
        loads, disps = self.problem.rhs_dir, self.problem.u_dir
        n = len(loads)

        # PETSc's dot is already a global reduction, so A is the same on every
        # rank without a further allreduce.
        A = np.array([[loads[i].dot(disps[j]) for j in range(n)]
                      for i in range(n)])
        A = 0.5*(A + A.T)          # symmetric by construction, not by round-off

        # On a reduced domain the dot products cover the kept fraction only,
        # while the derivative form already carries the factor -- so A has to
        # be scaled to match it, or objective and gradient describe different
        # structures.
        A *= self.sym_factor

        # Across parity classes the entry is exactly zero on the full domain:
        # a symmetric field paired with an antisymmetric load integrates to
        # nothing.  On the half domain the same product does not cancel, so
        # what the dot returns there is an artefact of the reduction and is
        # dropped rather than believed.  It follows that the gradient may only
        # combine fields within a class, which is what the block loop below
        # does -- combining across would assemble the same artefact into the
        # sensitivity.
        same_class = self.dir_parity[:, None] == self.dir_parity[None, :]
        A = np.where(same_class, A, 0.0)

        mu, cov, kappa = self.dir_mu, self.dir_cov, self.dir_kappa
        P = np.outer(mu, mu)
        A_cov = A @ cov

        mean = float(np.sum(A*(P + cov)))
        # Clipped because round-off can take a variance of zero -- a load with
        # no scatter left, or a design that carries every direction alike --
        # a little below it.
        variance = max(float(2.0*np.trace(A_cov @ A_cov)
                             + 4.0*(mu @ A_cov @ A @ mu)), 0.0)
        J = float(np.sqrt(mean*mean + kappa*kappa*variance))

        if J <= 0.0:               # no load reaches the structure
            with self.dCdrho_vec.localForm() as loc:
                loc.set(0)
            self._combine(self.dir_nominal)
            return 0.0

        # dJ/dA = [E*dE/dA + kappa^2*dVar/dA / 2] / J, with
        #   dE/dA   = mu mu^T + cov
        #   dVar/dA = 4 cov A cov + 4 (mu mu^T A cov + cov A mu mu^T)
        cross = P @ A_cov
        G = (mean*(P + cov)
             + 2.0*kappa*kappa*(cov @ A_cov + cross + cross.T)) / J

        G = np.where(same_class, 0.5*(G + G.T), 0.0)
        with self.dCdrho_vec.localForm() as loc:
            loc.set(0)
        for cls in np.unique(self.dir_parity):
            block = np.flatnonzero(self.dir_parity == cls)
            weights, vecs = np.linalg.eigh(G[np.ix_(block, block)])
            for k in range(len(block)):
                if weights[k] == 0.0:
                    continue
                coefficients = np.zeros(n)
                coefficients[block] = vecs[:, k]
                self.dCdrho_vec.axpy(float(weights[k]),
                                     self._combine(coefficients))

        self._combine(self.dir_nominal)
        return J

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

from petsc4py import PETSc
import argparse
import shutil
import time
from pathlib import Path
import numpy as np
from dolfinx import fem, io, default_scalar_type
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from basix.ufl import element, mixed_element
from ufl import (
    TestFunction, FacetNormal, Identity, div, inner, dx, ds,
    transpose, dot, as_vector, outer, lhs, rhs,
    TestFunctions, TrialFunctions, nabla_grad, derivative, CellDiameter, Measure
)
from .utils import compute_average_mesh_size, Q2D
import logging
from simulation import Simulator
import importlib.metadata
from .config import Config

# Configure logging
test_folder = Config().test_folder

class ActiveNematicSimulator(Simulator):
    """
    Simulates the evolution of flow and director field in an active nematic system using FEniCSx.
    """
    def pre_run(self):
        """Setup the simulation."""
        version = importlib.metadata.version("anfem")
        logging.info("============NEW SIMULATION============")
        logging.info(f"------------Version: {version}------------")
        logging.info(f"alpha = {self.params.alpha:.1f} | T = {self.params.T:.1f} | dt = {self.params.dt:.1f}")
        logging.info("++++++++++++++++++++++++++++++++++++++")
        self.save_params()
        self.test_folder = Path(test_folder).expanduser().resolve()

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.domain, self.cell_tags, self.facet_tags = self._load_mesh()
        self.ds = Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)

        self.constants = self._define_constants()
        self.function_spaces = self._define_function_spaces()
        self.functions = self._define_functions()

        self.solver_ns = self._setup_navier_stokes_solver()
        self.solver_q = self._setup_q_tensor_solver()

        self._initialize_states()

        self.avg_h = compute_average_mesh_size(self.domain)
        self.writer = io.VTXWriter(self.comm, self.data_dir, output=[self.functions["p_n"], self.functions["u_vis"],self.functions["Q_vis"], self.functions["S"]])

        
    def _load_mesh(self):
        """Loads the mesh from the specified directory."""
        
        mesh_dir_ori = self.test_folder / "mesh.msh" # always generate mesh in the test folder first, then copy to save_folder
        mesh_dir = self.save_folder / "mesh.msh"
        shutil.move(mesh_dir_ori, mesh_dir)

        logging.info(f"Loading mesh from {mesh_dir}")
        domain, cell_tags, facet_tags = io.gmshio.read_from_msh(mesh_dir, self.comm, 0, gdim=2)
        logging.info(f"All facet_tags: {np.unique(facet_tags.values)}")
        
        return domain, cell_tags, facet_tags

    def _define_constants(self):
        """Defines and returns simulation constants."""
        constants = {
            "alpha": fem.Constant(self.domain, PETSc.ScalarType(self.params.alpha)),
            "lambda_": fem.Constant(self.domain, PETSc.ScalarType(self.params.lambda_)),
            "rho": fem.Constant(self.domain, PETSc.ScalarType(self.params.rho_beta)),
            "beta1": fem.Constant(self.domain, PETSc.ScalarType(self.params.rho_beta - 1)),
            "beta2": fem.Constant(self.domain, PETSc.ScalarType((self.params.rho_beta + 1) / self.params.rho_beta**2)),
            "EA": fem.Constant(self.domain, PETSc.ScalarType(self.params.ea)),
            "k": fem.Constant(self.domain, PETSc.ScalarType(self.params.dt)),
            "gamma": fem.Constant(self.domain, PETSc.ScalarType(1.0e4)), # Penalty parameter for free-slip bc
        }
        return constants

    def _define_function_spaces(self):
        """Defines and returns the function spaces."""
        u_el = element("Lagrange", self.domain.topology.cell_name(), 2, shape=(2,))
        p_el = element("Lagrange", self.domain.topology.cell_name(), 1, shape=())
        Q1_el = element("Lagrange", self.domain.topology.cell_name(), 1, shape=(2, 2))
        Q_el = element("Lagrange", self.domain.topology.cell_name(), 1, shape=(2, 2))
        vis_el = element("Lagrange", self.domain.topology.cell_name(), 1, shape=(2,))
        w_el = mixed_element([u_el, p_el])

        W = fem.functionspace(self.domain, w_el)
        V_Q = fem.functionspace(self.domain, Q_el)
        V_Q1 = fem.functionspace(self.domain, Q1_el)
        V_vis = fem.functionspace(self.domain, vis_el)
        V_u, _ = W.sub(0).collapse()
        V_p, _ = W.sub(1).collapse()

        return {
            "W": W, "V_Q": V_Q, "V_Q1": V_Q1, "V_vis": V_vis,
            "V_u": V_u, "V_p": V_p
        }

    def _define_functions(self):
        """Defines and returns dolfinx functions."""
        V_u = self.function_spaces["V_u"]
        V_p = self.function_spaces["V_p"]
        V_Q = self.function_spaces["V_Q"]
        V_Q1 = self.function_spaces["V_Q1"]
        V_vis = self.function_spaces["V_vis"]
        W = self.function_spaces["W"]

        functions = {
            "u_n": fem.Function(V_u),
            "p_n": fem.Function(V_p, name="pressure"),
            "Q_": fem.Function(V_Q),
            "Q_n": fem.Function(V_Q),
            "Q1": fem.Function(V_Q1), # To convert higher order Q functions to order 1
            "u_vis": fem.Function(V_vis, name="velocity"),
            "Q_vis": fem.Function(V_vis, name="Q"), # To visualize Q (e.g. director field)
            "w": fem.Function(W),
            "S": fem.Function(V_p, name="scalar_order_parameter"),
        }
        return functions

    def _setup_navier_stokes_solver(self):
        """Sets up the Navier-Stokes linear problem."""
        u, p = TrialFunctions(self.function_spaces["W"])
        v, q = TestFunctions(self.function_spaces["W"])
        alpha = self.constants["alpha"]
        Q_n = self.functions["Q_n"]
        n = FacetNormal(self.domain)
        gamma = self.constants["gamma"]
        h = CellDiameter(self.domain)

        # Navier-Stokes equations
        F1 = (
            - inner(nabla_grad(u), nabla_grad(v)) * dx
            + p * div(v) * dx
            + q * div(u) * dx
            + alpha * inner(Q_n, nabla_grad(v)) * dx
        )

        bcs_u = []
        if self.params.noslip:
            logging.info("Applying NO-SLIP boundary condition at facets with tags: {}".format(self.params.wall_tags))
            noslip_value = fem.Function(self.function_spaces["V_u"])
            noslip_value.x.array[:] = 0
            for tag in self.params.wall_tags:
                # Check if the tag actually exists in the mesh
                if self.facet_tags.find(tag).size > 0:
                    wall_dofs = fem.locate_dofs_topological(
                        (self.function_spaces["W"].sub(0), self.function_spaces["V_u"]), 1, self.facet_tags.find(tag)
                    )
                    bcs_u.append(fem.dirichletbc(noslip_value, wall_dofs, self.function_spaces["W"].sub(0)))
                else:
                    logging.warning(f"Wall tag {tag} not found in mesh. Skipping no-slip BC for this tag.")
        else:
            logging.info("Applying FREE-SLIP boundary condition at facets with tags: {}".format(self.params.wall_tags))
            # Sum ds measures only for specified wall tags, handling potential empty list
            wall_measures = sum([self.ds(tag) for tag in self.params.wall_tags], start=self.ds(0))
            if wall_measures: # Only add if there are actual wall measures
                F1 += (gamma / h) * inner(dot(u, n), dot(v, n)) * wall_measures

        a = fem.form(lhs(F1))
        L = fem.form(rhs(F1))

        return fem.petsc.LinearProblem(a, L, bcs=bcs_u, u=self.functions["w"])

    def _setup_q_tensor_solver(self):
        """Sets up the Q-tensor evolution nonlinear problem."""
        Q_ = self.functions["Q_"]
        Q_n = self.functions["Q_n"]
        u_n = self.functions["u_n"]
        phi = TestFunction(self.function_spaces["V_Q"])

        k = self.constants["k"]
        lambda_ = self.constants["lambda_"]
        beta1 = self.constants["beta1"]
        beta2 = self.constants["beta2"]
        EA = self.constants["EA"]

        n = FacetNormal(self.domain)
        tangent = as_vector([-n[1], n[0]])
        Qb = outer(tangent, tangent) - 0.5 * Identity(self.domain.geometry.dim)

        u_grad = nabla_grad(u_n)
        Omega = 0.5 * (u_grad - transpose(u_grad)) # Vorticity tensor
        E = 0.5 * (u_grad + transpose(u_grad))     # Rate of strain tensor

        # Q-tensor evolution equation terms
        time_derivative_term = inner((Q_ - Q_n) / k, phi) * dx
        convection_term = inner(dot(u_n, nabla_grad(Q_)), phi) * dx
        rotation_term = - inner(dot(Q_, Omega) - dot(Omega, Q_), phi) * dx
        flow_alignment_term = - inner(lambda_ * E, phi) * dx
        elastic_term = inner(nabla_grad(Q_), nabla_grad(phi)) * dx
        bulk_free_energy_term = - inner(beta1 * Q_ - beta2 * inner(Q_, Q_) * Q_, phi) * dx

        # Surface anchoring term
        wall_measures = sum((self.ds(tag) for tag in self.params.wall_tags), start=self.ds(0))
        surface_anchoring_term = inner(EA * (Q_ - Qb), phi) * wall_measures if wall_measures else 0

        F2 = (
            time_derivative_term
            + convection_term
            + rotation_term
            + flow_alignment_term
            + elastic_term
            + bulk_free_energy_term
            + surface_anchoring_term
        )

        J = derivative(F2, Q_)
        problem = fem.petsc.NonlinearProblem(F2, Q_, J=J)
        solver = NewtonSolver(self.comm, problem)

        
        # solver.set_from_options()  # Must be called after setting PETSc options
        return solver

    def _initialize_states(self):
        """Initializes the Q-tensor."""
        Q_1_temp = fem.Function(self.function_spaces["V_Q1"])
        Q_1_temp.x.array[:] = self._initialize_Q(noise=self.params.init_noise)
        self.functions["Q_n"].interpolate(Q_1_temp)
        logging.info("Initial Q-tensor state set.")

    def _initialize_Q(self, noise=0.1):
        """Random initial Q-tensor."""
        # Get the mesh coordinates
        x = self.domain.geometry.x  # shape (n_points, 2)
        d = np.ones((x.shape[0], 2))
        d[:, 1] = noise * (0.5 - np.random.rand(x.shape[0]))
        d /= np.linalg.norm(d, axis=1)[:, None]  # Normalize d

        # Compute the Q-tensor: Q = S*(d⊗d - I/2)
        Q_vals = np.einsum("ni,nj->nij", d, d) - 0.5 * np.eye(2)

        # Reshape Q_vals to match the flattened dolfinx Function vector and set Q_n
        return Q_vals.flatten()

    def _run(self):
        """Executes the main simulation loop."""
        print(f"Simulation starts at {time.asctime()}")
        logging.info(f"Simulation starts at {time.asctime()}!")
        t0 = time.time()
        t = 0.0
        step_total = int(self.params.T / self.params.dt)

        try:
            for i in range(step_total):
                # Data visualization and output
                self.functions["u_vis"].interpolate(self.functions["u_n"])
                self.functions["Q1"].interpolate(self.functions["Q_n"]) # Interpolate Q_n to Q1
                Q_vals, Sv = Q2D(self.functions["Q1"])
                self.functions["Q_vis"].x.array[:] = Q_vals.flatten()
                self.functions["S"].x.array[:] = Sv.flatten()                
                self.writer.write(t)

                t = (i + 1) * self.params.dt # Update time for next step

                # Compute Courant number
                u_array = self.functions["u_vis"].x.array
                if u_array.size > 0:
                    vmax = np.linalg.norm(u_array.reshape(-1, 2), axis=1).max()
                    Cr = vmax * self.params.dt / self.avg_h
                else:
                    Cr = 0.0 # Handle case with no velocity data

                if self.rank == 0:
                    logging.info(f"{i+1}/{step_total}, Cr={Cr:.2f}, T_lapse={time.time()-t0:.0f} s")

                # Solve Navier-Stokes equations
                self.solver_ns.solve()
                u_sol, p_sol = self.functions["w"].split()
                self.functions["u_n"].interpolate(u_sol)
                self.functions["p_n"].interpolate(p_sol)

                # Solve Q-tensor evolution equation
                num_iterations, converged = self.solver_q.solve(self.functions["Q_"])
                if not converged:
                    logging.warning(f"Q-tensor solver did not converge at step {i+1}. Iterations: {num_iterations}")
                self.functions["Q_n"].interpolate(self.functions["Q_"])

            if self.rank == 0:
                logging.info(f"COMPLETED: {step_total}/{step_total} : Cr={Cr:.2f}, T_lapse={time.time()-t0:.0f} s")

        except Exception as e:
            logging.error(f"An unexpected error occurred during simulation at step {i+1}: {e}", exc_info=True)
            if self.rank == 0:
                logging.error(f"EXIT EARLY: {i+1}/{step_total}")
        finally:
            self.writer.close()
    
    def post_run(self):
        logging.info("Simulation completed.")
    
    def run(self):
        self.pre_run()
        self._run()
        self.post_run()

def main():
    parser = argparse.ArgumentParser(description="This script utilize the FEniCSx package to perform finite element method (FEM) to compute the evolution of the flow and the director field in an active nematic (AN) system, thus the name `anfem`.")
    parser.add_argument("mesh_dir", type=str, help="Input mesh.")
    parser.add_argument("-T", "--total_time", type=float, default=10, help="Total simulation time.")
    parser.add_argument("--dt", type=float, default=0.1, help="Step size in seconds.")
    parser.add_argument("--alpha", type=float, default=5, help="Dimensionless activity parameter.")
    parser.add_argument("--lambda_", type=float, default=0.7, help="Flow alignment parameter.")
    parser.add_argument("--rho_beta", type=float, default=1.6, help="parameters in LDG free energy functional, determines whether the system would favor nematic alignment (rho>1) or isotropic (rho<1).")
    parser.add_argument("--ea", type=float, default=0.1, help="anchor strength")
    parser.add_argument("--wall_tags", type=int, nargs="+", default=[1], help="channel wall tag number")
    parser.add_argument("-o", "--save_dir", type=str, default=".")
    parser.add_argument("--noslip", help="Enable noslip boundary condition on all boundaries with tag in wall_tags", action='store_true')
    parser.add_argument("-f", action="store_true", help="Force create the mesh. Overwrite if the file already exists.")
    parser.add_argument("--init_noise", type=float, default=0.1, help="Noise level of the initial director field. 0 for perfectly aligned in horizontal direction. 2 is close to totally random.")
    args = parser.parse_args()

    simulator = ActiveNematicSimulator(vars(args))
    simulator.run()

if __name__ == "__main__":
    main()
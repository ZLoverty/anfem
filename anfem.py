"""
anfem.py
========

This script utilize the FEniCSx package to perform finite element method (FEM) to compute the evolution of the flow and the director field in an active nematic (AN) system, thus the name `anfem`. The mesh is externally defined, usually with the software `gmsh`. 

Syntax
------

python anfem.py mesh_dir -t T --dt DT --alpha ALPHA --lambda LAMBDA --rho_beta RHO_BETA --ea EA

* -t: total simulation time (second)
* -dt: step time (second)
* --alpha: dimensionless activity 
* --lambda_: flow alignment parameter
* --rho_beta: parameters in LDG free energy functional, determines whether the system would favor nematic alignment (rho>1) or isotropic (rho<1).
* --ea: anchor strength
"""

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element, mixed_element
from ufl import TestFunction, FacetNormal, Identity, div, inner, dx, ds, transpose, dot, as_vector, outer, lhs, rhs, TestFunctions, TrialFunctions, nabla_grad, derivative
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="This script utilize the FEniCSx package to perform finite element method (FEM) to compute the evolution of the flow and the director field in an active nematic (AN) system, thus the name `anfem`.")

# Required argument
parser.add_argument("mesh_dir", type=str, help="Input mesh.")

# Optional flag
parser.add_argument("-t", "--total_time", type=float, default=10, help="Total simulation time in seconds.")
parser.add_argument("--dt", type=float, default=0.1, help="Step size in seconds.")
parser.add_argument("--alpha", type=float, default=5, help="Dimensionless activity parameter.")
parser.add_argument("--lambda_", type=float, default=0.7, help="Flow alignment parameter.")
parser.add_argument("--rho_beta", type=float, default=1.6, help="parameters in LDG free energy functional, determines whether the system would favor nematic alignment (rho>1) or isotropic (rho<1).")
parser.add_argument("--ea", type=float, default=0.1, help="anchor strength")
parser.add_argument("--wall_tag", type=int, default=5, help="channel wall tag number")
parser.add_argument("-o", "--save_dir", type=str, default="result.pvd")
args = parser.parse_args()

# Simulation parameters
T = args.total_time
dt = args.dt
mesh_dir = args.mesh_dir

# Define simulation domain: load from .msh file
domain, cell_tags, facet_tags = io.gmshio.read_from_msh(mesh_dir, MPI.COMM_WORLD, 0, gdim=2)

# Define constants
alpha = fem.Constant(domain, PETSc.ScalarType(args.alpha))
lambda_ = fem.Constant(domain, PETSc.ScalarType(args.lambda_))
rho = args.rho_beta
beta1 = fem.Constant(domain, PETSc.ScalarType(rho-1))
beta2 = fem.Constant(domain, PETSc.ScalarType((rho+1)/rho**2))
EA = fem.Constant(domain, PETSc.ScalarType(args.ea))
k = fem.Constant(domain, PETSc.ScalarType(dt))


# Define individual elements
u_el = element("Lagrange", domain.topology.cell_name(), 2, shape=(2,))
p_el = element("Lagrange", domain.topology.cell_name(), 1, shape=())
Q_el = element("Lagrange", domain.topology.cell_name(), 1, shape=(2, 2))
vis_el = element("Lagrange", domain.topology.cell_name(), 1, shape=(2,))
w_el = mixed_element([u_el, p_el])

# Define function spaces
W = fem.functionspace(domain, w_el)
V_Q = fem.functionspace(domain, Q_el)
V_vis = fem.functionspace(domain, vis_el)
V_u, _ = W.sub(0).collapse()
V_p, _ = W.sub(1).collapse()

# Define functions

# previous step
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
phi = TestFunction(V_Q)
u_n = fem.Function(V_u, )
p_n = fem.Function(V_p, name="pressure")
Q_ = fem.Function(V_Q)
Q_n = fem.Function(V_Q)
u_vis = fem.Function(V_vis, name="velocity")
Q_vis = fem.Function(V_vis, name="Q")
w = fem.Function(W)

# define boundary conditions - noslip velocity
# def walls(x):
#     return np.logical_or(np.isclose(np.linalg.norm(x, axis=0), 5.0), np.isclose(np.linalg.norm(x, axis=0), 10.0))

bcu = []
if facet_tags.find(args.wall_tag).size:
    print(f"Detect walls tagged as {args.wall_tag}")
    wall_dofs = fem.locate_dofs_topological(V_u, 1, facet_tags.find(args.wall_tag))
    noslip = fem.dirichletbc(PETSc.ScalarType((0, 0)), wall_dofs, V_u)
    bcu.append(noslip)

# Weak forms of governing equations

# Navier-Stokes equations
F1 = (
    - inner(nabla_grad(u), nabla_grad(v)) * dx
    + p * div(v) * dx
    - q * div(u) * dx
    + alpha * inner(Q_n, nabla_grad(v)) * dx
)

a = fem.form(lhs(F1))
L = fem.form(rhs(F1))

problem1 = fem.petsc.LinearProblem(a, L, bcs=bcu, u=w)

# Q-tensor evolution equation
u_grad = nabla_grad(u_n)
Omega = 0.5 * (u_grad - transpose(u_grad))
E = 0.5 * (u_grad + transpose(u_grad))
n = FacetNormal(domain)
tangent = as_vector([-n[1], n[0]])
Qb = outer(tangent, tangent) - 0.5 * Identity(domain.geometry.dim)

F2 = (
    inner((Q_ - Q_n) / k, phi) * dx
    + inner(dot(u_n, nabla_grad(Q_)) , phi) * dx
    - inner(dot(Q_, Omega) - dot(Omega, Q_), phi) * dx
    - inner(lambda_ * E, phi) * dx
    - inner(beta1 * Q_ - beta2 * inner(Q_, Q_) * Q_, phi) * dx
    + inner(nabla_grad(Q_), nabla_grad(phi)) * dx
    + inner(EA * (Q_ - Qb), phi) * ds 
)

# jacobian
J = derivative(F2, Q_)

problem = fem.petsc.NonlinearProblem(F2, Q_, J=J)
solver2 = NewtonSolver(domain.comm, problem)

# Set initial states

# 0-flow
u_n.x.array[:] = 0.0

# 0-pressure
p_n.x.array[:] = 0.0

# random initial Q-tensor
# Get the mesh coordinates
x = domain.geometry.x  # shape (n_points, 2)
d = np.random.rand(x.shape[0], 2)
d /= np.linalg.norm(d, axis=1)[:, None]  # Normalize d

# Compute the Q-tensor: Q = S*(d⊗d - I/2)
Q_vals = np.einsum("ni,nj->nij", d, d) - 0.5 * np.eye(2)

# Reshape Q_vals to match the flattened dolfinx Function vector and set Q_n
Q_n.x.array[:] = Q_vals.flatten()

# Q-tensor to director for visualization
def Q2D(Q_tensor):
    Q_e = Q_tensor.x.array.reshape(-1, 2, 2)
    num_pts = Q_e.shape[0]
    d_vals = np.zeros((num_pts, 2))

    for i in range(num_pts):
        # Get eigenvector with max eigenvalue
        vals, vecs = np.linalg.eigh(Q_e[i])
        d = vecs[:, np.argmax(vals)]

        d_vals[i, :] = d
    return d_vals

t = 0

writer = io.VTXWriter(domain.comm, f"{args.save_dir}", output=[p_n, u_vis, Q_vis]) 

rank = MPI.COMM_WORLD.Get_rank()

while t < T:
    u_vis.interpolate(u_n)
    Q_vis.x.array[:] = Q2D(Q_n).flatten()
    writer.write(t)

    t += dt
    if rank == 0:
        print(f"{time.asctime()}: t={t:.2f}", flush=True)

    problem1.solve()
    u_sol, p_sol = w.split()

    u_n.interpolate(u_sol)
    u_n.x.scatter_forward()
    p_n.interpolate(p_sol)
    p_n.x.scatter_forward()

    # solve for Q
    n, converged = solver2.solve(Q_)
    Q_n.interpolate(Q_)
    Q_n.x.scatter_forward()


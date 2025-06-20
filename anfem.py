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
from dolfinx import fem, io, default_scalar_type
from dolfinx.nls.petsc import NewtonSolver
from basix.ufl import element, mixed_element
from ufl import TestFunction, FacetNormal, Identity, div, inner, dx, ds, transpose, dot, as_vector, outer, lhs, rhs, TestFunctions, TrialFunctions, nabla_grad, derivative, CellDiameter
import numpy as np
import time
import argparse
from utils import *
import json
from pathlib import Path
import shutil

# Arguments
parser = argparse.ArgumentParser(description="This script utilize the FEniCSx package to perform finite element method (FEM) to compute the evolution of the flow and the director field in an active nematic (AN) system, thus the name `anfem`.")
parser.add_argument("mesh_dir", type=str, help="Input mesh.")
parser.add_argument("-t", "--total_time", type=float, default=10, help="Total simulation time in seconds.")
parser.add_argument("--dt", type=float, default=0.1, help="Step size in seconds.")
parser.add_argument("--alpha", type=float, default=5, help="Dimensionless activity parameter.")
parser.add_argument("--lambda_", type=float, default=0.7, help="Flow alignment parameter.")
parser.add_argument("--rho_beta", type=float, default=1.6, help="parameters in LDG free energy functional, determines whether the system would favor nematic alignment (rho>1) or isotropic (rho<1).")
parser.add_argument("--ea", type=float, default=0.1, help="anchor strength")
parser.add_argument("--wall_tags", type=int, nargs="+", default=[1], help="channel wall tag number")
parser.add_argument("-o", "--save_dir", type=str, default="result.pvd")
parser.add_argument("--noslip", help="Enable noslip boundary condition on all boundaries with tag in wall_tags", action='store_true')
args = parser.parse_args()

# Simulation parameters
T = args.total_time
dt = args.dt
mesh_dir = args.mesh_dir
wall_tags = args.wall_tags
save_dir = Path(args.save_dir).expanduser().resolve()

# file operations
save_dir.mkdir(exist_ok=True)
shutil.copy(mesh_dir, save_dir / "mesh.msh")

# Define simulation domain: load from .msh file
domain, cell_tags, facet_tags = io.gmshio.read_from_msh(mesh_dir, MPI.COMM_WORLD, 0, gdim=2)

all_tags = np.unique(facet_tags.values)
print("----------------------------------------------------------")
print(f"All unique facet tags found in mesh: {all_tags}")
print("Please ensure the --wall_tag you provide is in this list.")
print("----------------------------------------------------------")

# Define constants
alpha = fem.Constant(domain, PETSc.ScalarType(args.alpha))
lambda_ = fem.Constant(domain, PETSc.ScalarType(args.lambda_))
rho = args.rho_beta
beta1 = fem.Constant(domain, PETSc.ScalarType(rho-1))
beta2 = fem.Constant(domain, PETSc.ScalarType((rho+1)/rho**2))
EA = fem.Constant(domain, PETSc.ScalarType(args.ea))
k = fem.Constant(domain, PETSc.ScalarType(dt))
gamma = fem.Constant(domain, PETSc.ScalarType(1.0e4)) # Penalty parameter for free-slip bc

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
V_u, u_to_w_map = W.sub(0).collapse()
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

# Weak forms of governing equations

from ufl import Measure
ds = Measure("ds", domain=domain, subdomain_data=facet_tags)

n = FacetNormal(domain)
h = CellDiameter(domain)

# Navier-Stokes equations
F1 = (
    - inner(nabla_grad(u), nabla_grad(v)) * dx
    + p * div(v) * dx
    + q * div(u) * dx
    + alpha * inner(Q_n, nabla_grad(v)) * dx
)

# handle boundary conditions

bcu = []
if args.noslip:
    noslip_value = fem.Function(V_u)
    noslip_value.x.array[:] = 0
    for tag in wall_tags:
        if facet_tags.find(tag).size:
            print(f"Create noslip boundary at {tag}")
            wall_dofs = fem.locate_dofs_topological((W.sub(0), V_u), 1, facet_tags.find(tag))
            noslip_value = fem.Function(V_u)
            noslip = fem.dirichletbc(noslip_value, wall_dofs, W.sub(0))
            bcu.append(noslip)
else:
    print(f"Create freeslip boundary at {args.wall_tags}")
    F1 += (gamma / h) * inner(dot(u, n), dot(v, n)) * sum([ds(tag) for tag in args.wall_tags], start=ds(0)) 

a = fem.form(lhs(F1))
L = fem.form(rhs(F1))

solver1 = fem.petsc.LinearProblem(a, L, bcs=bcu, u=w)

# Q-tensor evolution equation
u_grad = nabla_grad(u_n)
Omega = 0.5 * (u_grad - transpose(u_grad))
E = 0.5 * (u_grad + transpose(u_grad))

########################################
# Although tangent vector theoretically should be defined as [-n[1], n[0]],
# my test suggests that such definition would make boundary vectors perpendicular to the boundary.
# Therefore, I make tangent = n here. The cause of this bug might be somewhere deeper in FEniCSx. 
# I set tangent this way just to give expected behavior.
tangent = n # as_vector([-n[1], n[0]]) # 
Qb = outer(tangent, tangent) - 0.5 * Identity(domain.geometry.dim)

# import pdb
# pdb.set_trace()

F2 = (
    inner((Q_ - Q_n) / k, phi) * dx
    + inner(dot(u_n, nabla_grad(Q_)) , phi) * dx
    - inner(dot(Q_, Omega) - dot(Omega, Q_), phi) * dx
    - inner(lambda_ * E, phi) * dx
    - inner(beta1 * Q_ - beta2 * inner(Q_, Q_) * Q_, phi) * dx
    + inner(nabla_grad(Q_), nabla_grad(phi)) * dx
    + inner(EA * (Q_ - Qb), phi) * sum([ds(tag) for tag in args.wall_tags], start=ds(0)) 
)

# jacobian
J = derivative(F2, Q_)

problem = fem.petsc.NonlinearProblem(F2, Q_, J=J)
solver2 = NewtonSolver(domain.comm, problem)

# Set initial states

Q_n.x.array[:] = initialize_Q(domain)

# write simulation parameters

params = {
    "alpha": float(alpha),
    "lambda": float(lambda_),
    "rho": rho,
    "EA": float(EA),
    "T": T,
    "dt": dt,
    "save_dir": str(save_dir)
}

print("Simulation parameters:")
for k, v in params.items():
    print(f"{k}: {v}")

with open(save_dir / "params.json", 'w') as json_file:
    json.dump(params, json_file, indent=4)

# ========= Begin the simulation ! ================
t = 0

writer = io.VTXWriter(domain.comm, save_dir / "results.pvd", output=[p_n, u_vis, Q_vis]) 

rank = MPI.COMM_WORLD.Get_rank()

while t < T:
    u_vis.interpolate(u_n)
    Q_vis.x.array[:] = Q2D(Q_n).flatten()
    writer.write(t)

    t += dt
    if rank == 0:
        print(f"{time.asctime()}: t={t:.2f}", flush=True)

    solver1.solve()

    u_sol, p_sol = w.split()

    u_n.interpolate(u_sol)
    # u_n.x.array[:] = apply_periodic_bc(u_n)
    u_n.x.scatter_forward()
    p_n.interpolate(p_sol)
    # p_n.x.array[:] = apply_periodic_bc(p_n)
    p_n.x.scatter_forward()

    # solve for Q
    n, converged = solver2.solve(Q_)
    Q_n.interpolate(Q_)
    # Q_n.x.array[:] = apply_periodic_bc(Q_n)
    Q_n.x.scatter_forward()
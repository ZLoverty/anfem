from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem, plot, mesh, io
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import pyvista
from ufl import TestFunction, TrialFunction, FacetNormal, Identity, grad, div, inner, dx, ds, transpose, dot, as_vector, outer, lhs, rhs, TestFunctions, TrialFunctions, split, nabla_grad, derivative
pyvista.set_jupyter_backend("static")
import numpy as np
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl

# Simulation parameters
T = 10
dt = 0.1

# Define simulation domain: load from .msh file
mesh_file = "annulus.msh"
domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)

# Define constants
alpha = fem.Constant(domain, PETSc.ScalarType(5.0))
mu = fem.Constant(domain, PETSc.ScalarType(1.0))
gamma = fem.Constant(domain, PETSc.ScalarType(1.0))
lambda_ = fem.Constant(domain, PETSc.ScalarType(0.7))
C = fem.Constant(domain, PETSc.ScalarType(1.0))
K = fem.Constant(domain, PETSc.ScalarType(1.0))
beta1 = fem.Constant(domain, PETSc.ScalarType(0.6))
beta2 = fem.Constant(domain, PETSc.ScalarType((1.6 + 1.0) / (1.6**2)))
EA = fem.Constant(domain, PETSc.ScalarType(0.01))
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
def walls(x):
    return np.logical_or(np.isclose(np.linalg.norm(x, axis=0), .5), np.isclose(np.linalg.norm(x, axis=0), 1.0))

wall_dofs = fem.locate_dofs_geometrical(V_u, walls)
noslip = fem.dirichletbc(PETSc.ScalarType((0, 0)), wall_dofs, V_u)
bcu = [noslip]

# Weak forms of governing equations

# Navier-Stokes equations
F1 = (
    inner(nabla_grad(u), nabla_grad(v)) * dx
    + inner(v, nabla_grad(p)) * dx
    + q * div(u) * dx
    - inner(alpha * div(Q_n), v) * dx
)

a = fem.form(lhs(F1))
L = fem.form(rhs(F1))

# assemble matrix A and b
A = assemble_matrix(a)
A.assemble()
b = create_vector(L)

# configure a solver for u
solver1 = PETSc.KSP().create(domain.comm)
solver1.setOperators(A)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Q-tensor evolution equation
u_grad = nabla_grad(u_n)
Omega = 0.5 * (u_grad + transpose(u_grad))
E = 0.5 * (u_grad - transpose(u_grad))
dFdQ_bulk = C * beta1 * Q_ + C * beta2 * inner(Q_, Q_) * Q_ + K * div(nabla_grad(Q_))
n = FacetNormal(domain)
Qb = outer(n, n) - 0.5 * Identity(domain.geometry.dim)
dFdQ_surf = EA * (Q_ - Qb)

F2 = (
    inner((Q_ - Q_n) / k, phi) * dx
    + inner(dot(u_n, nabla_grad(Q_)), phi) * dx
    - inner(Q_*Omega - Omega*Q_, phi) * dx
    - inner(lambda_ * E, phi) * dx
    - inner(dFdQ_bulk / gamma, phi) * dx
    - inner(dFdQ_surf / gamma, phi) * ds
)

# jacobian
dQ = TrialFunction(V_Q)
J = derivative(F2, Q_, dQ)

problem = NonlinearProblem(F2, Q_, J=J)
solver2 = NewtonSolver(domain.comm, problem)

# Set initial states

# 0-flow
u_n.x.array[:] = 0.0

# 0-pressure
p_n.x.array[:] = 0.0

# random initial Q-tensor
# Get the mesh coordinates
x = domain.geometry.x  # shape (n_points, 2)
r = np.sqrt(x[:, 0]**2 + x[:, 1]**2)

# Avoid division by zero (if any points are at the origin)
r[r < 1e-10] = 1.0

# Compute the tangent (director) d = (-y, x)/|r|
d = np.column_stack((-x[:, 1], x[:, 0])) / r[:, None]

# Compute the Q-tensor: Q = S*(d⊗d - I/2)
Q_vals = np.einsum("ni,nj->nij", d, d) - 0.5 * np.eye(2)

# Reshape Q_vals to match the flattened dolfinx Function vector and set Q_n
Q_n.x.array[:] = Q_vals.flatten()

# Time stepping
# helper function for visualizing Q

expr = ufl.as_vector([Q_n[0, 0], Q_n[0, 1]])
expr_interp = fem.Expression(expr, V_vis.element.interpolation_points())

t = 0

with io.XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)   
    while t < T:
        u_vis.interpolate(u_n)
        Q_vis.interpolate(expr_interp)
        xdmf.write_function(u_vis, t)
        xdmf.write_function(Q_vis, t)
        xdmf.write_function(p_n, t)
        
        t += dt
        print(f"Time: {t:.2f}")
        # solve for u
        b.zeroEntries()
        assemble_vector(b, L)
        apply_lifting(b, [a], [bcu])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcu)
        solver1.solve(b, w.x.petsc_vec)
        u_sol, p_sol = w.split()

        u_n.interpolate(u_sol)
        u_n.x.scatter_forward()
        p_n.interpolate(p_sol)
        p_n.x.scatter_forward()

        # solve for Q
        n, converged = solver2.solve(Q_)
        Q_n.interpolate(Q_)
        Q_n.x.scatter_forward()


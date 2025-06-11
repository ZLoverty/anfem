from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem, plot, mesh, io
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import pyvista
from ufl import TestFunction, TrialFunction, FacetNormal, Identity, grad, div, inner, dx, ds, transpose, dot, as_vector, outer, lhs, rhs, TestFunctions, TrialFunctions, split, nabla_grad
pyvista.set_jupyter_backend("static")
import numpy as np
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc

# Simulation parameters
T = 10
n_steps = 100
dt = T / n_steps

# Define simulation domain: load from .msh file
mesh_file = "annulus.msh"
domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)

# Define constants
alpha = fem.Constant(domain, PETSc.ScalarType(5.0)) # activity coef
mu = fem.Constant(domain, PETSc.ScalarType(1.0)) # viscosity
gamma = fem.Constant(domain, PETSc.ScalarType(1.0)) # dissipation rate ^-1
lambda_ = fem.Constant(domain, PETSc.ScalarType(0.7)) # flow alignment parameter
DE = fem.Constant(domain, PETSc.ScalarType(1.0)) # elastic coef / dissipation rate
Dr = fem.Constant(domain, PETSc.ScalarType(1.0)) # another elastic coef / dissipation rate
rho = 1.5 # determine isotropic / nematic 
b1 = fem.Constant(domain, PETSc.ScalarType(1 - rho)) 
b2 = fem.Constant(domain, PETSc.ScalarType((rho + 1.0) / (rho**2)))
EA = fem.Constant(domain, PETSc.ScalarType(0.01)) # anchoring strength / dissipation rate
k = fem.Constant(domain, PETSc.ScalarType(dt)) # dt as domain constant

# Define individual elements
p_el = element("Lagrange", domain.topology.cell_name(), 1)           
u_el = element("Lagrange", domain.topology.cell_name(), 2, shape=(2,))
Q_el = element("Lagrange", domain.topology.cell_name(), 1, shape=(2, 2))     
uvis_el = element("Lagrange", domain.topology.cell_name(), 1, shape=(2,))  
mix_el = mixed_element([p_el, u_el])

# Define function spaces
W_pu = fem.functionspace(domain, mix_el)
V_p, _ = W_pu.sub(0).collapse()
V_u, _ = W_pu.sub(1).collapse()
V_Q = fem.functionspace(domain, Q_el)
V_uvis = fem.functionspace(domain, uvis_el)

# Define functions

# previous step
p_n = fem.Function(V_p)
u_n = fem.Function(V_u)
Q_n = fem.Function(V_Q)

# current step
w_ = fem.Function(W_pu)
p_ = fem.Function(V_p)
u_ = fem.Function(V_u)
Q_ = fem.Function(V_Q)

# visualization, write to data
uvis = fem.Function(V_uvis)

# define boundary conditions - noslip velocity
def walls(x):
    return np.logical_or(np.isclose(np.linalg.norm(x, axis=0), .5), np.isclose(np.linalg.norm(x, axis=0), 1.0))

wall_dofs = fem.locate_dofs_geometrical(V_p, walls)
noslip = fem.dirichletbc(PETSc.ScalarType((0, 0)), wall_dofs, V_p)
bcu = [noslip]

# define  initial conditions -- 0 pressure, 0 velocity, randomized Q-tensor
# x = domain.geometry.x
# d = np.random.rand(x.shape[0], 2)
# d /= np.linalg.norm(d, axis=1, keepdims=True)
# S = 1.0
# Q_vals = S * (np.einsum("ni,nj->nij", d, d) - 0.5 * np.eye(2))
# Q_n.x.array[:] = Q_vals.flatten()
S = 0.5
Q_n.interpolate(lambda x: S * np.array([[1, 0], [0, -1]]))

# Weak forms of governing equations
# 
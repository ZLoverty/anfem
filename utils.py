import numpy as np
from dolfinx import geometry, fem

# Q-tensor to director for visualization
def Q2D(Q_tensor):
    """Convert Q-tensor to director.
    """
    Q_e = Q_tensor.x.array.reshape(-1, 2, 2)
    num_pts = Q_e.shape[0]
    d_vals = np.zeros((num_pts, 2))

    for i in range(num_pts):
        # Get eigenvector with max eigenvalue
        vals, vecs = np.linalg.eigh(Q_e[i])
        d = vecs[:, np.argmax(vals)]

        d_vals[i, :] = d
    return d_vals

def initialize_Q(domain):
    # random initial Q-tensor
    # Get the mesh coordinates
    x = domain.geometry.x  # shape (n_points, 2)
    d = np.random.rand(x.shape[0], 2)
    d /= np.linalg.norm(d, axis=1)[:, None]  # Normalize d

    # Compute the Q-tensor: Q = S*(d⊗d - I/2)
    Q_vals = np.einsum("ni,nj->nij", d, d) - 0.5 * np.eye(2)

    # Reshape Q_vals to match the flattened dolfinx Function vector and set Q_n
    return Q_vals.flatten()

def apply_periodic_bc(function: fem.Function):

    function_space = function.function_space
    domain = function_space.mesh

    # Compute bounding box
    coords = function_space.tabulate_dof_coordinates()
    xmin = coords.min(axis=0)
    xmax = coords.max(axis=0)

    def right_boundary(x): return np.isclose(x[0], xmax[0])
    def top_boundary(x): return np.isclose(x[1], xmax[1])

    dofs_right = fem.locate_dofs_geometrical(function_space, right_boundary)
    x_right = function_space.tabulate_dof_coordinates()[dofs_right]
    x_right_mapped = x_right.copy()
    x_right_mapped[:, 0] = xmin[0]

    dofs_top = fem.locate_dofs_geometrical(function_space, top_boundary)
    x_top = function_space.tabulate_dof_coordinates()[dofs_top]
    x_top_mapped = x_top.copy()
    x_top_mapped[:, 1] = xmin[1]

    tdim = domain.topology.dim
    tree = geometry.bb_tree(domain, tdim)

    def determine_cells(x_mapped):
        cell_candidates = geometry.compute_collisions_points(tree, x_mapped)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, x_mapped)
        cells = []
        for i, point in enumerate(x_mapped):
            if len(colliding_cells.links(i)) > 0:
                cells.append(colliding_cells.links(i)[0])
        return cells

    # update right boundary values
    cells = determine_cells(x_right_mapped)
    left_values = function.eval(x_right_mapped, cells)
    arr = function.x.array 
    arr = arr.reshape(-1, function_space.dofmap.bs)
    arr[dofs_right] = left_values

    # update top boundary values
    cells = determine_cells(x_top_mapped)
    bottom_values = function.eval(x_top_mapped, cells)
    arr[dofs_top] = bottom_values

    return arr.flatten()
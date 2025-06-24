import numpy as np
from dolfinx import geometry, fem
from adios2 import Stream
from scipy.interpolate import LinearNDInterpolator
from myimagelib.corrLib import corrS, distance_corr
from myimagelib import xy_bin
from scipy.optimize import curve_fit
import pandas as pd

def Q2D(Q_tensor: fem.Function) -> np.array:
    """Convert Q-tensor to director for visualization.
    Args:
    Q_tensor -- dolfinx Function object.

    Returns:
    d_vals -- director vector fields.
    """
    Q_e = Q_tensor.x.array.reshape(-1, 2, 2)
    num_pts = Q_e.shape[0]
    d_vals = np.zeros((num_pts, 2))
    S = np.zeros((num_pts, 1))
    for i in range(num_pts):
        # Get eigenvector with max eigenvalue
        vals, vecs = np.linalg.eigh(Q_e[i])
        d = vecs[:, np.argmax(vals)]

        d_vals[i, :] = d
        S[i, 0] = vals.max()
    return d_vals, S



def compute_average_mesh_size(domain):
    """Compute the average mesh size."""
    domain.topology.create_connectivity(1, 0)  # edges to vertices
    edges = domain.topology.connectivity(1, 0).array.reshape(-1, 2)
    edge_coords = domain.geometry.x[edges]
    edge_lengths = np.linalg.norm(edge_coords[:, 0, :] - edge_coords[:, 1, :], axis=1)
    return np.mean(edge_lengths)


def read_pvd(filename, variable):
    """Read data from .pvd data file.
    
    Args:
    filename -- .pvd data file name (a directory)
    variable -- variable name in the data set.
    
    Returns:
    velocity -- 3D velocity data, axis 0 is the time step.
    domain -- the mesh geometry, coordinates of nodes.
    """
    
    v_list = []
    filename = str(filename)
    with Stream(filename, "r") as s:
        # steps comes from the stream
        count = 0
        for _ in s.steps():

            # track current step
            print(f"Current step is {s.current_step()}", end="\r")

            if s.current_step() == 0:
                domain = s.read("geometry")
                
            v_list.append(s.read(variable))

            count += 1

    return np.stack(v_list), domain

def interpolate_to_grid(velocity_2D, domain, resolution=1.0):
    """Interpolate 2D velocity field to regular grid.
    
    Args:
    velocity_2D -- a matrix of shape (npt, 2).
    domain -- the mesh geometry, coordinates of nodes.
    resolution -- grid size to interpolate. The number of points in each direction is inferred from the domain bounds and resolution. 
    
    Returns:
    X, Y -- the regular grid coordintates.
    u_grid -- interpolated velocity u.
    v_grid -- interpolated velocity v.
    """

    points_on_mesh = domain[:, :2] # Assuming 2D, take first two columns

     # Create a uniform grid for interpolation
    x_min, y_min = np.min(points_on_mesh, axis=0)
    x_max, y_max = np.max(points_on_mesh, axis=0)

    nx = int((x_max - x_min) // resolution)
    ny = int((y_max - y_min) // resolution)

    grid_x = np.linspace(x_min, x_max, nx)
    grid_y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(grid_x, grid_y)

    interp_u = LinearNDInterpolator(points_on_mesh, velocity_2D[:, 0])
    interp_v = LinearNDInterpolator(points_on_mesh, velocity_2D[:, 1])

    u_grid = interp_u(X, Y)
    v_grid = interp_v(X, Y)

    return X, Y, u_grid, v_grid

def corr_length(X, Y, U, V, p0=20):
    """Compute the correlation length of a 2D velocity field.
    
    Args:
    X, Y, U, V -- coordinates and velocity field.
    p0 -- initial guess of the correlation length, to facilitate fitting.
    Return:
    CL -- correlation length.
    VACF -- velocity autocorrelation function, DataFrame [R, C]. 
    """

    x, y, ca, cv = corrS(X, Y, U, V)
    dc = distance_corr(x, y, cv)
    
    def exp(x, tau):
        return np.exp(-x/tau)
    
    try:
        xbin, ybin = xy_bin(dc.R, dc.C, mode="lin")
        popt, _ = curve_fit(exp, xbin, ybin, p0=[p0])
        return popt[0], pd.DataFrame({"R": xbin, "C": ybin})
    except:
        return np.nan, dc

def apply_periodic_bc(function: fem.Function):
    """This function helps to implement periodic boundary condition by setting values at top and right boundaries with the values at bottom and left boundaries.
    
    To apply, use
    
    ```
    u_n.x.array[:] = apply_periodic_bc(u_n)
    ```
    """

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
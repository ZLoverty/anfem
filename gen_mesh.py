import gmsh
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="This script draws the mesh of ratchet channels with different dimensions and number of ratchets.")
parser.add_argument("-H", type=float, default=5, help="Ratchet height.")
parser.add_argument("-N", type=int, default=3, help="Number of rathets.")
parser.add_argument("-l", type=float, default=None, help="Ratchet length.")
parser.add_argument("-m", type=float, default=None, help="Ratchet channel padding.")
parser.add_argument("-w", type=float, default=None, help="Channel width -- the distance between ratchet tips.")
parser.add_argument("--cl_outer", type=float, default=3, help="Characeteristic length of outer boundary.")
parser.add_argument("--cl_inner", type=float, default=1, help="Characteristic length of inner boundary.")
parser.add_argument("--w_total", type=int, default=200, help="Total width of the pool.")
parser.add_argument("--h_total", type=int, default=100, help="Total height of the pool.")
parser.add_argument("-o", type=str, default="mesh.msh", help="Output dir of the .msh file.")
args = parser.parse_args()

# Initialize Gmsh
gmsh.initialize()

# Set the geometry kernel to OpenCASCADE
# This is equivalent to SetFactory("OpenCASCADE");
gmsh.model.add("ratchet_channel")
occ = gmsh.model.occ

# --- 1. Parameters ---
W_TOTAL = args.w_total
H_TOTAL = args.h_total
h = args.H
m = h / 2 if args.m is None else args.m
l = 2 * h if args.l is None else args.l
w = 2 * h if args.w is None else args.w
N = args.N
cl_outer = args.cl_outer  # Outer boundary characteristic length
cl_inner = args.cl_inner   # Channel characteristic length
save_dir = Path(args.o).expanduser().resolve()
if save_dir.exists():
    print(f"Mesh {save_dir} already exists, abort ...")
    exit()
if save_dir.parent.exists() == False:
    save_dir.parent.mkdir(parents=True)

# check length compatibility
assert(H_TOTAL > 4*h + w)
assert(W_TOTAL > N*l + 2*m)

# --- 2. Create the outer rectangle ---
outer_rectangle = occ.addRectangle(0, 0, 0, W_TOTAL, H_TOTAL)
occ.synchronize()

# --- 3. Create the ratchet "tool" shapes using a helper function ---
def create_ratchet(x_start, y_start, is_top):
    """
    Creates a ratchet surface tool.
    Returns the tag of the Plane Surface.
    """
    p_current = gmsh.model.occ.getMaxTag(0) + 1 # Get a fresh point tag
    
    # Starting points
    p1 = occ.addPoint(x_start, y_start, 0, cl_inner, p_current)
    y_offset = 2 * h if not is_top else -2 * h
    p2 = occ.addPoint(x_start, y_start + y_offset, 0, cl_inner, p_current + 1)

    points = [p1, p2]
    
    # Loop over teeth
    for i in range(N):
        y_direction = 1 if not is_top else -1
        # Use p_current for unique tags
        p_current = gmsh.model.occ.getMaxTag(0) + 1
        
        # Ratchet tooth corner points
        pt1 = occ.addPoint(x_start + m + i * l, y_start + y_offset, 0, cl_inner, p_current)
        pt2 = occ.addPoint(x_start + m + i * l, y_start + y_offset/2, 0, cl_inner, p_current + 1)
        points.extend([pt1, pt2])

    # End points of the ratchet
    p_current = gmsh.model.occ.getMaxTag(0) + 1
    end_x = x_start + m + N * l # Correctly use N instead of the loop variable i
    
    pt_end1 = occ.addPoint(end_x, y_start + y_offset, 0, cl_inner, p_current)
    pt_end2 = occ.addPoint(end_x + m, y_start + y_offset, 0, cl_inner, p_current+1)
    pt_end3 = occ.addPoint(end_x + m, y_start, 0, cl_inner, p_current+2)
    points.extend([pt_end1, pt_end2, pt_end3])

    # Create lines connecting all the points
    lines = []
    for i in range(len(points) - 1):
        lines.append(occ.addLine(points[i], points[i+1]))
    # Add the final line to close the loop
    lines.append(occ.addLine(points[-1], points[0]))
    
    # Create the surface
    cloop = occ.addCurveLoop(lines)
    surface = occ.addPlaneSurface([cloop])
    return surface

# Create the bottom and top ratchets
x01 = 0.5 * (W_TOTAL - N * l - 2 * m)
y01 = 0.5 * (H_TOTAL - 4 * h - w)
bottom_ratchet = create_ratchet(x01, y01, is_top=False)

x02 = x01
y02 = H_TOTAL - y01
top_ratchet = create_ratchet(x02, y02, is_top=True)

# --- 4. Perform the Boolean Operation ---
# Subtract the two ratchet tools from the outer rectangle
final_shape, _ = occ.cut([(2, outer_rectangle)], [(2, bottom_ratchet), (2, top_ratchet)], removeObject=True, removeTool=True)

# IMPORTANT: Synchronize the CAD kernel with the Gmsh model
occ.synchronize()

# --- 5. Identify Boundaries and Assign Physical Groups ---
final_surface_tag = final_shape[0][1]

boundary_curves = gmsh.model.getBoundary([(2, final_surface_tag)], combined=False)
boundary_curves_tags = [c[1] for c in boundary_curves]

# Identify boundary points and assign cl_outer
boundary_curves = gmsh.model.getBoundary([(2, outer_rectangle)])
points = []
for curve in boundary_curves:
    boundary_points = gmsh.model.getBoundary([curve])
    for point in boundary_points:
        points.append(point)

gmsh.model.mesh.setSize(points, cl_outer)

# Assign Physical Groups
gmsh.model.addPhysicalGroup(2, [final_surface_tag], 1, "domain")
gmsh.model.addPhysicalGroup(1, boundary_curves_tags, 1, "boundary")

occ.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write(os.path.join(save_dir))

gmsh.finalize()
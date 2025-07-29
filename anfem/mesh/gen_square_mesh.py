import gmsh
import sys
import argparse
from pathlib import Path

# parser = argparse.ArgumentParser(description="This script draws a square mesh with specified size and mesh resolution.")
# parser.add_argument("-L", type=int, default=100, help="Mesh total dimension.")
# parser.add_argument("--cl", type=float, default=3, help="Mesh resolution")
# parser.add_argument("-o", type=str, default="mesh.msh", help="Output dir of the .msh file.")
# args = parser.parse_args()

# L = args.L
# cl = args.cl
# save_dir = Path(args.o).expanduser().resolve()
# if save_dir.parent.exists() == False:
#     save_dir.parents.mkdir()
class SquareMeshGenerator:

    def __init__(self, save_folder, params, exist_ok=False):
        self.save_folder = Path(save_folder).expanduser().resolve()
        self.params = params
        self.exist_ok = exist_ok
        self.setup_directories()

    def setup_directories(self):
        """Check existence of the .msh file."""
        self.mesh_dir = self.save_folder / "mesh.msh"
        if self.mesh_dir.exists() and not self.exist_ok:
            print(f"Mesh already exists, abort ...")
            exit()
        else:
            if self.save_folder.exists() == False:
                self.save_folder.mkdir(parents=True)

    def run(self):
        cl = self.params.cl
        L = self.params.L
        gmsh.initialize()
        gmsh.model.add("square")

        # Define corner points (tags are assigned automatically)
        p1 = gmsh.model.geo.addPoint(0, 0, 0, cl)
        p2 = gmsh.model.geo.addPoint(L, 0, 0, cl)
        p3 = gmsh.model.geo.addPoint(L, L, 0, cl)
        p4 = gmsh.model.geo.addPoint(0, L, 0, cl)

        # Define lines
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        # Line loop and surface
        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surface = gmsh.model.geo.addPlaneSurface([loop])

        # Physical groups
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], tag=1)  # boundary
        gmsh.model.setPhysicalName(1, 1, "Boundary")

        gmsh.model.addPhysicalGroup(2, [surface], tag=2)  # domain
        gmsh.model.setPhysicalName(2, 2, "Domain")

        # Mesh generation
        gmsh.model.mesh.generate(2)
        gmsh.write(str(self.save_folder / "mesh.msh"))     # Save as msh2 by default
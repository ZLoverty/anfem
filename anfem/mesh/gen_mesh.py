import gmsh
import argparse
from pathlib import Path

class RatchetMeshGenerator:

    def __init__(self, save_folder, params, exist_ok=False):
        self.save_folder = Path(save_folder).expanduser().resolve()
        self.params = params
        self.exist_ok = exist_ok
        self.setup_directories()
        self.check_dimensions()
    
    def setup_directories(self):
        """Check existence of the .msh file."""
        self.mesh_dir = self.save_folder / "mesh.msh"
        if self.mesh_dir.exists() and not self.exist_ok:
            print(f"Mesh already exists, abort ...")
            exit()
        else:
            if self.save_folder.exists() == False:
                self.save_folder.mkdir(parents=True)
        
    def check_dimensions(self):
        """Check if the channel exceeds the bounds of the pool."""
        assert(self.params.h_total > 4*self.params.h + self.params.w)
        assert(self.params.w_total > self.params.N*self.params.l + 2*self.params.m)

    def _find_starting_point(self):
        """Determine the starting point (x0, y0) of the ratchet."""
        self.x0_bottom = 0.5 * (self.params.w_total - self.params.N * self.params.l - 2 * self.params.m)
        self.y0_bottom = 0.5 * (self.params.h_total - 4 * self.params.h - self.params.w)
        self.x0_top = self.x0_bottom
        self.y0_top = self.params.h_total - self.y0_bottom

    def create_ratchet(self, x_start, y_start, is_top):
        """
        Creates a ratchet surface tool.
        Returns the tag of the Plane Surface.
        """
        h = self.params.h
        m = self.params.m
        l = self.params.l
        cl_inner = self.params.cl_inner
        cl_outer = self.params.cl_outer
        N = self.params.N

        p_current = gmsh.model.occ.getMaxTag(0) + 1 # Get a fresh point tag
        
        # Starting points
        p1 = self.occ.addPoint(x_start, y_start, 0, cl_inner, p_current)
        y_offset = 2 * h if not is_top else -2 * h
        p2 = self.occ.addPoint(x_start, y_start + y_offset, 0, cl_inner, p_current + 1)

        points = [p1, p2]
        
        # Loop over teeth
        for i in range(N):
            y_direction = 1 if not is_top else -1
            # Use p_current for unique tags
            p_current = gmsh.model.occ.getMaxTag(0) + 1
            
            # Ratchet tooth corner points
            pt1 = self.occ.addPoint(x_start + m + i * l, y_start + y_offset, 0, cl_inner, p_current)
            pt2 = self.occ.addPoint(x_start + m + i * l, y_start + y_offset/2, 0, cl_inner, p_current + 1)
            points.extend([pt1, pt2])

        # End points of the ratchet
        p_current = gmsh.model.occ.getMaxTag(0) + 1
        end_x = x_start + m + N * l # Correctly use N instead of the loop variable i
        
        pt_end1 = self.occ.addPoint(end_x, y_start + y_offset, 0, cl_inner, p_current)
        pt_end2 = self.occ.addPoint(end_x + m, y_start + y_offset, 0, cl_inner, p_current+1)
        pt_end3 = self.occ.addPoint(end_x + m, y_start, 0, cl_inner, p_current+2)
        points.extend([pt_end1, pt_end2, pt_end3])

        # Create lines connecting all the points
        lines = []
        for i in range(len(points) - 1):
            lines.append(self.occ.addLine(points[i], points[i+1]))
        # Add the final line to close the loop
        lines.append(self.occ.addLine(points[-1], points[0]))
        
        # Create the surface
        cloop = self.occ.addCurveLoop(lines)
        surface = self.occ.addPlaneSurface([cloop])
        return surface
    
    def write_roi(self):
        with open(self.mesh_dir, "a") as f:
            f.write("$ROI\n")
            f.write(f"{self.x0_bottom:f} {self.y0_bottom:f} {self.y0_top:f} {self.x0_bottom+self.params.N*self.params.l:f}\n")
            f.write("$ROI")

    def run(self):
        """Create the mesh."""
        # Initialize Gmsh
        gmsh.initialize()

        # Set the geometry kernel to OpenCASCADE
        # This is equivalent to SetFactory("OpenCASCADE");
        gmsh.model.add("ratchet_channel")
        self.occ = gmsh.model.occ

        # --- 2. Create the outer rectangle ---
        outer_rectangle = self.occ.addRectangle(0, 0, 0, self.params.w_total, self.params.h_total)
        self.occ.synchronize()

        # --- 3. Create the ratchet "tool" shapes using a helper function ---
        self._find_starting_point()
        bottom_ratchet = self.create_ratchet(self.x0_bottom, self.y0_bottom, is_top=False)
        top_ratchet = self.create_ratchet(self.x0_top, self.y0_top, is_top=True)

        # --- 4. Perform the Boolean Operation ---
        # Subtract the two ratchet tools from the outer rectangle
        final_shape, _ = self.occ.cut([(2, outer_rectangle)], [(2, bottom_ratchet), (2, top_ratchet)], removeObject=True, removeTool=True)

        # IMPORTANT: Synchronize the CAD kernel with the Gmsh model
        self.occ.synchronize()

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

        gmsh.model.mesh.setSize(points, self.params.cl_outer)

        # Assign Physical Groups
        gmsh.model.addPhysicalGroup(2, [final_surface_tag], 1, "domain")
        gmsh.model.addPhysicalGroup(1, boundary_curves_tags, 1, "boundary")

        self.occ.synchronize()

        gmsh.model.mesh.generate(2)
        gmsh.write(str(self.save_folder / "mesh.msh"))

        gmsh.finalize()

        self.write_roi()

def main():
    parser = argparse.ArgumentParser(description="This script draws the mesh of ratchet channels with different dimensions and number of ratchets.")
    parser.add_argument("-H", type=float, default=5, help="Ratchet height.")
    parser.add_argument("-N", type=int, default=3, help="Number of rathets.")
    parser.add_argument("-l", type=float, default=None, help="Ratchet length.")
    parser.add_argument("-m", type=float, default=None, help="Ratchet channel padding.")
    parser.add_argument("-w", type=float, default=None, help="Channel width -- the distance between ratchet tips.")
    parser.add_argument("--cl_outer", type=float, default=2, help="Characeteristic length of outer boundary.")
    parser.add_argument("--cl_inner", type=float, default=1, help="Characteristic length of inner boundary.")
    parser.add_argument("--w_total", type=int, default=80, help="Total width of the pool.")
    parser.add_argument("--h_total", type=int, default=80, help="Total height of the pool.")
    parser.add_argument("-o", type=str, default="mesh.msh", help="Output dir of the .msh file.")
    parser.add_argument("-f", action="store_true", help="Force create the mesh. Overwrite if the file already exists.")
    args = parser.parse_args()

    generator = RatchetMeshGenerator(args)
    generator.run()

if __name__ == "__main__":
    main()
"""
test_mesh_resolution.py
===================

The results from Q tensor correlation length suggest that the correlation is close to the length of the mesh size. That side, the correlation measured cannot be trusted because the mesh is too coarse. The correlation length is within one mesh size. 

In this section, we test a smaller mesh, with finer resolution. The plan is to test a square of 20x20, with mesh size ranging from 0.2 to 0.5.
"""

from subprocess import Popen, run
from pathlib import Path

save_folder = "~/Documents/RATSIM/mesh_resolution"
save_folder = Path(save_folder).expanduser().resolve()
if save_folder.exists() == False:
    save_folder.mkdir(parents=True)

mesh_folder = "~/Documents/RATSIM/MESHES"
mesh_folder = Path(mesh_folder).expanduser().resolve()

ress = [.8, 1]
for res in ress:
    mesh_dir = str(mesh_folder / f"square_20_{res:.1f}.msh")
    cmd_gen_mesh = [
        "python", "gen_square_mesh.py",
        "-L", "20",
        "--cl", f"{res:.1f}",
         "-o", mesh_dir
    ]
    run(cmd_gen_mesh)
    save_dir = save_folder / f"RES={res:.1f}"
    cmd = [
        "python", "anfem.py", mesh_dir,
        "--noslip", "-T", "20",
        "--dt", "0.1",
        "-o", save_dir,
    ]
    Popen(cmd)
"""
test_channel_width.py
===================

Test the flow behavior in ratchet channels of different width.
"""

from subprocess import Popen, run
from pathlib import Path

save_folder = "~/Documents/RATSIM/channel_width"
save_folder = Path(save_folder).expanduser().resolve()
if save_folder.exists() == False:
    save_folder.mkdir(parents=True)

mesh_folder = "~/Documents/RATSIM/MESHES"
mesh_folder = Path(mesh_folder).expanduser().resolve()

Ws = [8, 10, 12, 14]
for W in Ws:
    mesh_dir = str(mesh_folder / f"W_{W:d}.msh")
    cmd_gen_mesh = [
        "python", "gen_mesh.py",
        "-H", "5",
        "-w", f"{W:d}",
        "--cl_outer", "2",
        "--cl_inner", "01",
        "--w_total", "80",
        "--h_total", "80",
        "-N", "3",
         "-o", mesh_dir
    ]
    run(cmd_gen_mesh)
    save_dir = save_folder / f"W={W:d}"
    cmd = [
        "python", "anfem.py", mesh_dir,
        "--noslip", "-T", "20",
        "--dt", "0.1",
        "-o", save_dir,
    ]
    Popen(cmd)
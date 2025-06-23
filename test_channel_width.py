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

Ws = [2, 3, 4, 5, 6]
for W in Ws:
    mesh_dir = str(mesh_folder / f"W_{W:d}_H_2_N_4.msh")
    cmd_gen_mesh = [
        "python", "gen_mesh.py",
        "-H", "2",
        "-w", f"{W:d}",
        "--cl_outer", "1",
        "--cl_inner", "0.5",
        "--w_total", "50",
        "--h_total", "50",
        "-N", "4",
         "-o", mesh_dir
    ]
    run(cmd_gen_mesh)
    save_dir = save_folder / f"W={W:d}"
    cmd = [
        "python", "anfem.py", mesh_dir,
        "--noslip", "-T", "100",
        "--dt", "0.02",
        "-o", save_dir,
    ]
    Popen(cmd)
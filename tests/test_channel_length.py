"""
We have identified a good ratchet geometry: $h=5$ and $w=10$. Now, we are ready to test the ultimate parameter: the length of the channel! Here, we are going to vary the length ranging from 3 to 10. The width of the pool is chosen to be 200 to accommodate the long channel.
"""

from subprocess import Popen, run
from pathlib import Path

save_folder = "~/Documents/RATSIM/channel_length"
save_folder = Path(save_folder).expanduser().resolve()
if save_folder.exists() == False:
    save_folder.mkdir(parents=True)

mesh_folder = "~/Documents/RATSIM/MESHES"
mesh_folder = Path(mesh_folder).expanduser().resolve()

Ns = range(8, 12)
for N in Ns:
    mesh_dir = str(mesh_folder / f"N_{N:d}.msh")
    cmd_gen_mesh = [
        "python", "gen_mesh.py",
        "-H", "5",
        "-w", "10",
        "--cl_outer", "2",
        "--cl_inner", "1",
        "--w_total", "200",
        "--h_total", "80",
        "-N", f"{N:d}",
         "-o", mesh_dir
    ]
    run(cmd_gen_mesh)
    save_dir = save_folder / f"N={N:d}"
    cmd = [
        "python", "anfem.py", mesh_dir,
        "--noslip", "-T", "50",
        "--dt", "0.1",
        "-o", save_dir,
    ]
    Popen(cmd)
"""
test_system_size.py
===================

In this section, we try larger systems, ranging from 100 to 300. The goal is to see whether larger systems give rise to more random handness of the vorticity.
"""

from subprocess import Popen, run
from pathlib import Path

save_folder = "~/Documents/RATSIM/system_size"

save_folder = Path(save_folder).expanduser().resolve()
save_folder.mkdir(parents=True)

Ls = [100, 150, 200, 250, 300]
for i in range(5):
    L = Ls[i]
    cmd_gen_mesh = [
        "python", "gen_square_mesh.py", f"{L:d}"
    ]
    run(cmd_gen_mesh)
    save_dir = save_folder / f"L={L:d}"
    cmd = [
        "python", "anfem.py", f"square_{L:d}.msh",
        "--noslip",
        "-o", save_dir,
    ]
    Popen(cmd)
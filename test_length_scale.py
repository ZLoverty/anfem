"""
test_mpi_speed.py
=================

This script tests the length scale of the velocity field of the result of `anfem.py` in a simple square domain, using different activity parameters alpha (1, 2, 4, 8, 16, 32).
"""

from subprocess import Popen
from pathlib import Path

save_folder = "~/Documents/RATSIM/alpha_length_scale"

save_folder = Path(save_folder).expanduser().resolve()
save_folder.mkdir(parents=True, exist_ok=True)

for i in range(6):
    alpha = 2 ** i
    pvd_file = save_folder / f"alpha={alpha:d}"
    cmd = [
        "python", "anfem.py", "mesh.msh",
        "--alpha", str(alpha),
        "--noslip",
        "-o", pvd_file
    ]
    Popen(cmd)
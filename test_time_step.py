"""
test_time_step.py
=================

This script tests the time step of the velocity field of the result of `anfem.py` in a simple square domain, using different activity parameters dt (0.01, 0.02, 0.04, 0.08).
"""

from subprocess import Popen
from pathlib import Path

save_folder = "~/Documents/RATSIM/time_step"

save_folder = Path(save_folder).expanduser().resolve()
save_folder.mkdir(parents=True)

dts = [0.01, 0.02, 0.04, 0.08]
for i in range(4):
    dt = dts[i]
    save_dir = save_folder / f"dt={dt:.2f}"
    cmd = [
        "python", "anfem.py", "square_50.msh",
        "--dt", str(dt),
        "--noslip",
        "-o", save_dir,
    ]
    Popen(cmd)
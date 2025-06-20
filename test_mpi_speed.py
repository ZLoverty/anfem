"""
test_mpi_speed.py
=================

This script tests the time it takes to compute 10 s of the same `anfem.py` using different numbers of processes. Naively, higher process number gives higher speed, but the communication overhead could diminish the advantage of parallel computing, especially for small meshes. 
"""

from subprocess import run
import time
import os

save_folder = os.path.expanduser("~/Documents/RATSIM/MPI_speed_test")
print(save_folder)
os.makedirs(save_folder, exist_ok=True)
result_file = os.path.join(save_folder, "result.txt")
with open(result_file, "w") as f:
    pass

for i in range(1, 10):
    t0 = time.monotonic()
    run(["mpirun", "-n", f"{i:d}", "python", "anfem.py", "mesh.msh", "-o", os.path.join(save_folder, f"n={i:d}.pvd")])
    t = time.monotonic() - t0
    with open(result_file, "a+") as f:
        f.write(f"n={i:d}: {t:.1f} s\n")



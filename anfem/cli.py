"""
cli.py
======
This is the main entry of the active nematics simulation. It is a ready to run script that can take command line arguments to run simulations with different parameters. Install the package either directly from the GitHub repo:

```
pip install git+https://github.com/ZLoverty/anfem.git
```

or, clone the git repo and install as editable package for development

```
git clone https://github.com/ZLoverty/anfem.git
conda env create -f environment.yaml
conda activate ansim
pip install -e .
```

Once the `anfem` package is installed, use the following script to run the bubble bouncing simulation.

```
ansim --save_folder FOLDER [-f] [--arg ARG]
```

For a full list of available arguments, see

```
ansim -h
```
"""

from .config import Config
from pathlib import Path
import argparse
import logging
from simulation import parse_params, available_keys
from .actnem import SimulationParams, MeshParams
from .anfem import ActiveNematicSimulator
from .mesh import RatchetMeshGenerator

test_folder = Config().test_folder

def main():
    parser = argparse.ArgumentParser(f"This script takes in the save_folder as a positional argument. Optional arguments can be passed to set simulation parameters. Available arguments are {available_keys(SimulationParams())}.")
    parser.add_argument("--save_folder", type=str, default=test_folder, help="folder to save simulation data.")
    parser.add_argument("-f", action="store_true")
    parser.add_argument('--log-level', '-l',
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    # Parse known args; leave unknown ones (like --R, --T, etc.)
    args, unknown = parser.parse_known_args()
    save_folder = Path(args.save_folder).expanduser().resolve()
    args_dict = parse_params(unknown, SimulationParams())
    params = SimulationParams(**args_dict)
    log_level_str = args.log_level

    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level_str}")

    sim = ActiveNematicSimulator(save_folder, params, exist_ok=args.f, short_name="an", results_suffix=".pvd")

    logging.basicConfig(
        filename = sim.log_file,
        level = numeric_level,        # Set the logging level
        format = '%(asctime)s - %(levelname)s - %(message)s',  # Log format
    )

    sim.run()

def genmesh():
    """Generate ratchet channel mesh."""
    
    parser = argparse.ArgumentParser(f"This script generates ratchet channel mesh in a rectangular pool. Available options are {available_keys(MeshParams())}.")
    parser.add_argument("--save_folder", type=str, default=test_folder, help="folder to save the mesh (temporarily).")
    parser.add_argument("-f", action="store_true")
    parser.add_argument('--log-level', '-l',
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    # Parse known args; leave unknown ones (like --H, --N, etc.)
    args, unknown = parser.parse_known_args()
    save_folder = Path(args.save_folder).expanduser().resolve()
    args_dict = parse_params(unknown, MeshParams())
    params = MeshParams(**args_dict)
    generator = RatchetMeshGenerator(save_folder, params, exist_ok=args.f)
    generator.run()


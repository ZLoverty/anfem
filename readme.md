# Active Nematics (AN) Finite Element Method (FEM)

Use FEM to solve the dynamics of active nematics bounded in channels. 

## Installation

Requires `conda`.

#### Option 1: clone the repo and run setup script

```
git clone https://github.com/ZLoverty/anfem.git
cd anfem
./setup.sh
```

## Run simulation 

The package provides two CLI commands: 

- `genmesh`: generate ratchet channel mesh.
- `ansim`: run active nematics simulation.
- `genmesh_square`: generate a simple square mesh. 

### With default parameters

```
conda activate ansim
genmesh
ansim --save_folder ~/Documents/test
```

The results can be found in `~/Documents/test`. The folder consists 
- `ansim.log`: simulation log, containing run time and Courant number;
- `mesh.msh`: the mesh file, can be viewed by `gmsh`;
- `params.yaml`: simulation parameters;
- `results.pvd`: results data folder, can be loaded by ParaView.

### With custom parameters

To see the available options of these commands, use the `-h` flag to show the help string

```
genmesh -h
ansim -h
genmesh_square -h
```

Example output:

```
usage: This script generates ratchet channel mesh in a rectangular pool. Available options are 
--h: Ratchet height, default 5
--N: number of ratchets, default 3
--l: ratchet length, default 10
--m: Ratchet channel padding, default 2.5
--w: Channel width, the distance between ratchet tips, default 10
--cl_outer: outer boundary mesh characteristic length, default 2
--cl_inner: inner boundary (channel) mesh characteristic length, default 1
--w_total: total width of the pool, default 80
--h_total: total height of the pool, default 80.
```

These are all the available parameters that can be tuned for the command. As an example, if we want to generate a mesh of 4 pairs of ratchets, and channel width of 8, use

```
genmesh --N 4 --w 8
```

Note that the software checks the existence of target files. If they exist, the simulation code aborts. To force overwrite existing files, use flag `-f`. The mesh is always generated in the `test_folder` in `config.py` (default `~/.ansim`). Can be previewed by `gmsh`

```
gmsh mesh.msh
```

## To do

- Graphical user guide 
- pip install from GitHub
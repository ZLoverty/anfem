# Active Nematics (AN) Finite Element Method (FEM)

Use FEM to solve the dynamics of active nematics bounded in channels. 

## Installation

Install FEniCSx

```
conda env create -f environment.yaml
conda activate ansim
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

Packages only available on pip

```
pip install myimagelib
pip install git+https://github.com/ZLoverty/simulation.git
pip install gmsh
```


## To do

- adapt to the code style based on the Simulator class
- find a way to use list for wall_tags parameter
- if default mesh_dir "" is passed, automatically generate a sample mesh and save it to file
- implement CLI commands for easy use 
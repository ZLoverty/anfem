from dataclasses import dataclass, field

@dataclass
class SimulationParams:
    """Dataclass to hold the default active nematics simulation parameters."""
    # Physical constants:
    alpha: float = field(default=5.0, metadata={"help": "activity coefficient, default 5"})
    lambda_: float = field(default=0.7, metadata={"help": "flow alignment parameter, default 0.7"})
    rho_beta: float = field(default=1.6, metadata={"help": "parameter related to the isotropic to nematic transition, default 1.6"})
    ea: float = field(default=0.1, metadata={"help": "boundary anchoring strength, default 0.1"})
    noslip: bool = field(default=True, metadata={"help": "whether no-slip boundary condition is applied, default True"})
    wall_tags: list[int] = field(default_factory=lambda: [1], metadata={"help": "tags of facets that are identified as boundaries, default [1]"})

    # Initial conditions:
    init_noise: float = field(default=0.1, metadata={"help": "initial noise level for the nematic alignment, default 0.1"})

    # Simulation control:
    T: float = field(default=10.0, metadata={"help": "total simulation time (s), default 10"})
    dt: float = field(default=0.1, metadata={"help": "step size (s), data is saved at each step, default 0.1"})

@dataclass
class MeshParams:
    """Dataclass to hold the default mesh parameters."""
    # Ratchet geometries:
    h: float = field(default=5.0, metadata={"help": "Ratchet height, default 5"})
    N: int = field(default=3, metadata={"help": "number of ratchets, default 3"})
    l: float = field(default=10.0, metadata={"help": "ratchet length, default 10"})
    m: float = field(default=2.5, metadata={"help": "Ratchet channel padding, default 2.5"}) 
    w: float = field(default=10.0, metadata={"help": "Channel width, the distance between ratchet tips, default 10"}) # 
    cl_outer: float = field(default=2.0, metadata={"help": "outer boundary mesh characteristic length, default 2"}) 
    cl_inner: float = field(default=1.0, metadata={"help": "inner boundary (channel) mesh characteristic length, default 1"}) # 
    w_total: float = field(default=80., metadata={"help": "total width of the pool, default 80"})  
    h_total: float = field(default=80., metadata={"help": "total height of the pool, default 80"}) 


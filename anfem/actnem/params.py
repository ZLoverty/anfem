from dataclasses import dataclass, field

@dataclass
class SimulationParams:
    """Dataclass to hold the default active nematics simulation parameters."""
    # Physical constants:
    alpha: float = field(default=5.0, metadata={"help": "activity coefficient"})
    lambda_: float = field(default=0.7, metadata={"help": "flow alignment parameter"})
    rho_beta: float = field(default=1.6, metadata={"help": "parameter related to the isotropic to nematic transition"})
    ea: float = field(default=0.1, metadata={"help": "boundary anchoring strength"})
    noslip: bool = field(default=True, metadata={"help": "whether no-slip boundary condition is applied"})
    wall_tags: list[int] = field(default_factory=lambda: [1], metadata={"help": "tags of facets that are identified as boundaries"})

    # Initial conditions:
    init_noise: float = field(default=0.1, metadata={"help": "initial noise level for the nematic alignment"})

    # Simulation control:
    T: float = field(default=10.0, metadata={"help": "total simulation time (s)"})
    dt: float = field(default=0.1, metadata={"help": "step size (s), data is saved at each step."})

@dataclass
class MeshParams:
    """Dataclass to hold the default mesh parameters."""
    # Ratchet geometries:
    h: float = field(default=5.0, metadata={"help": "Ratchet height"})
    N: int = field(default=3, metadata={"help": "number of ratchets"})
    l: float = field(default=10.0, metadata={"help": "ratchet length"})
    m: float = field(default=2.5, metadata={"help": "Ratchet channel padding"}) 
    w: float = field(default=10.0, metadata={"help": "Channel width -- the distance between ratchet tips"}) # 
    cl_outer: float = field(default=2.0, metadata={"help": "outer boundary mesh characteristic length"}) 
    cl_inner: float = field(default=1.0, metadata={"help": "inner boundary (channel) mesh characteristic length"}) # 
    w_total: float = field(default=80., metadata={"help": "total width of the pool"})  
    h_total: float = field(default=80., metadata={"help": "total height of the pool"}) 


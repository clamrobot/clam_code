from clam.models.clam.clam import IDM, ContinuousLAM
from clam.models.clam.diffusion_clam import DiffusionCLAM
from clam.models.clam.space_time_clam import SpaceTimeCLAM
from clam.models.clam.transformer_clam import TransformerCLAM, TransformerIDM
from clam.models.hierarchical_clam.v1 import HierarchicalCLAMv1
from clam.models.hierarchical_clam.v2 import HierarchicalCLAMv2


def get_la_dim(cfg):
    if "hierarchical_clam" in cfg.name:
        if cfg.model.idm.add_continuous_offset:
            # if we add them, the discrete and continuous part have the same dimension
            la_dim = cfg.model.la_dim
        else:
            la_dim = cfg.model.la_dim + cfg.model.idm.discrete_latent_dim
    else:
        la_dim = cfg.model.la_dim

    return la_dim


def get_clam_cls(name):
    if name == "clam":
        return ContinuousLAM
    elif name == "transformer_clam":
        return TransformerCLAM
    elif name == "st_vivit_clam":
        return SpaceTimeCLAM
    elif name == "hierarchical_clam_v1":
        return HierarchicalCLAMv1
    elif name == "hierarchical_clam_v2":
        return HierarchicalCLAMv2
    elif name == "diffusion_clam":
        return DiffusionCLAM
    else:
        raise ValueError(f"Unknown CLAM model {name}")


def get_idm_cls(name):
    if name == "mlp":
        return IDM
    elif name == "transformer":
        return TransformerIDM
    else:
        raise ValueError(f"Unknown IDM model {name}")

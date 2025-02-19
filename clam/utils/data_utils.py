from dataclasses import dataclass
from itertools import repeat
from typing import Dict

import numpy as np
from jaxtyping import Array, Float, Key, Union

BT = Float[Array, "B T"]
BTD = Float[Array, "B T D"]
BTHWC = Float[Array, "B T H W C"]
BTX = Union[BTD, BTHWC]
PRNGKeyDict = Dict[str, Key]


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    # assert not torch.is_floating_point(obs)
    return obs.astype(np.float64) / 255 - 0.5


def unnormalize_obs(obs: np.ndarray) -> np.ndarray:
    return ((obs + 0.5) * 255).astype(np.uint8)


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    image: np.ndarray = None
    info: Dict[str, np.ndarray] = None


@dataclass
class Batch:
    states: BTX
    observations: BTX
    actions: BT
    rewards: BT = None
    next_observations: np.ndarray = None
    dones: np.ndarray = None
    tasks: np.ndarray = None
    mask: BT = None
    timestep: BT = None
    scene_obs: BT = None
    traj_index: np.ndarray = None
    latent_actions: np.ndarray = None
    is_first: BT = None
    is_last: BT = None
    is_terminal: BT = None
    discount: BT = None
    images: BTHWC = None
    embeddings: BT = None
    env_state: np.ndarray = None


def get_latent_action_dim(idm_config):
    if idm_config.apply_quantization:
        latent_action_dim = idm_config.vq.code_dim
    else:
        latent_action_dim = idm_config.latent_action_dim

    if idm_config.separate_categorical_la:
        # add the categorical part
        latent_action_dim += idm_config.vq.code_dim

    return latent_action_dim


def repeat_loader(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

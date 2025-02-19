import collections
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig


def format_dict_keys(dictionary, format_fn):
    """Returns new dict with `format_fn` applied to keys in `dictionary`."""
    return collections.OrderedDict(
        [(format_fn(key), value) for key, value in dictionary.items()]
    )


def prefix_dict_keys(dictionary, prefix):
    """Add `prefix` to keys in `dictionary`."""
    return format_dict_keys(dictionary, lambda key: "%s%s" % (prefix, key))


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)


def to_device(x, device):
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: torch.tensor(v).to(device) for k, v in x.items()}
    else:
        import ipdb

        ipdb.set_trace()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, dict):
        return {k: v.cpu().detach().numpy() for k, v in x.items()}
    else:
        return x

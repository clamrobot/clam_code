from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from clam.utils.logger import log


class BaseModel(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int = None):
        super(BaseModel, self).__init__()
        self.name = ""
        self.cfg = cfg
        self.input_dim = input_dim

    def load_from_ckpt(
        self, ckpt_path, ckpt_step: int = None, key: str = "model", strict: bool = True
    ):
        cfg_file = Path(ckpt_path) / "config.yaml"
        cfg = OmegaConf.load(cfg_file)

        if ckpt_step is None:
            ckpt_file = "latest.pkl"
        else:
            ckpt_file = f"ckpt_{ckpt_step}.pkl"

        ckpt_path = Path(ckpt_path) / "model_ckpts" / ckpt_file
        log(f"loading {self.name} model from {ckpt_path}", "green")
        ckpt = torch.load(open(ckpt_path, "rb"))

        params = ckpt[key]
        # turns out this happens when you use a torch.compile model
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(params.items()):
            if k.startswith(unwanted_prefix):
                params[k[len(unwanted_prefix) :]] = params.pop(k)
        self.load_state_dict(params, strict=strict)
        return cfg, ckpt

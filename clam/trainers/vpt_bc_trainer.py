import collections
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from clam.models.utils.clam_utils import get_idm_cls
from clam.trainers.bc_trainer import BCTrainer
from clam.utils.logger import log


class VPTBCTrainer(BCTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_model(self):
        cfg_file = Path(self.cfg.vpt_ckpt) / "config.yaml"
        log(f"loading config from {cfg_file}", "blue")
        vpt_cfg = OmegaConf.load(cfg_file)
        self.vpt_cfg = vpt_cfg

        self.use_transformer = "transformer" in self.vpt_cfg.model.idm.net.name
        log(f"Using transformer: {self.use_transformer}")

        if not self.cfg.data.load_relabelled_actions:
            log("Using trained VPT IDM to compute latent actions", "red")

            idm_cls = get_idm_cls(self.vpt_cfg.model_cls)

            self.vpt = idm_cls(self.vpt_cfg.model.idm, input_dim=self.obs_shape)
            self.vpt.load_from_ckpt(
                self.cfg.vpt_ckpt, self.cfg.vpt_ckpt_step, key="model"
            )
            self.vpt = self.vpt.to(self.device)
            log("VPT model loaded successfully", "green")

        else:
            log("Loaded annotated actions from data", "red")

        return super().setup_model()

    def compute_loss(self, batch, train: bool = True):
        idm_output = self.vpt(batch.observations)
        annotated_action = idm_output.la
        if self.use_transformer:
            action_pred = self.model(batch.observations[:, 1:])
        else:
            action_pred = self.model(batch.observations[:, 1])
        loss = self.loss_fn(action_pred, annotated_action)
        loss = loss.mean()
        return {"action_loss": loss.item()}, loss

import collections
import copy

import numpy as np
import torch
import torch.nn as nn

from clam.models.utils.clam_utils import get_idm_cls
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.logger import log


class VPTTrainer(OfflineTrainer):
    def __init__(self, cfg):
        self.use_transformer = "transformer" in cfg.model.idm.net.name
        log(f"Using transformer: {self.use_transformer}")

        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

    def setup_model(self):
        idm_cls = get_idm_cls(self.cfg.model_cls)
        model = idm_cls(self.cfg.model.idm, input_dim=self.obs_shape)

        if self.cfg.load_from_ckpt:
            cfg, ckpt = model.load_from_ckpt(
                self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step, key="model"
            )

        return model

    def compute_loss(self, batch, train: bool = True):
        idm_output = self.model(batch.observations)
        action_pred = idm_output.la
        if self.use_transformer:
            gt_action = batch.actions[:, 1:]
        else:
            gt_action = batch.actions[:, 1]

        loss = self.loss_fn(action_pred, gt_action)
        loss = loss.mean()
        return {"action_loss": loss.item()}, loss

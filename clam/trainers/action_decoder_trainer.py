from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from clam.models.clam.clam import ContinuousLAM
from clam.models.clam.transformer_clam import TransformerCLAM
from clam.models.mlp_policy import MLPPolicy
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.logger import log


class ActionDecoderTrainer(OfflineTrainer):
    def __init__(self, cfg):
        # load LAM model
        lam_cfg_file = Path(cfg.lam_ckpt) / "config.yaml"
        log(f"loading LAM config from {lam_cfg_file}", "blue")
        self.lam_cfg = OmegaConf.load(lam_cfg_file)
        self.la_dim = self.lam_cfg.model.idm.la_dim
        self.use_transformer = "transformer" in self.lam_cfg.model.idm.net.name

        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

        if not self.cfg.data.load_latent_actions:
            if cfg.lam_ckpt_step is None:
                ckpt_file = "latest.pkl"
            else:
                ckpt_file = f"ckpt_{cfg.lam_ckpt_step}.pkl"

            ckpt_path = Path(cfg.lam_ckpt) / "model_ckpts" / ckpt_file
            log(f"loading LAM model from {ckpt_path}", "blue")
            ckpt = torch.load(open(ckpt_path, "rb"))

            if self.use_transformer:
                self.clam = TransformerCLAM(
                    self.lam_cfg.model, input_dim=self.obs_shape
                )
            else:
                self.clam = ContinuousLAM(self.lam_cfg.model, input_dim=self.obs_shape)

            self.clam.load_state_dict(ckpt["model"])
            self.clam = self.clam.to(self.device)

    def setup_model(self):
        model = MLPPolicy(
            self.cfg.model,
            input_dim=self.la_dim,
            output_dim=self.cfg.env.action_dim,
        )
        return model

    def compute_loss(self, batch, train: bool = True):
        if hasattr(batch, "latent_actions"):
            latent_action = batch.latent_actions
            gt_actions = batch.actions
        else:
            with torch.no_grad():
                latent_action, *_ = self.clam(batch.observations)

            if self.use_transformer:
                action_pred = action_pred[:, :-1]
                gt_actions = batch.actions[:, :-1]
            else:
                gt_actions = batch.actions[:, -2]

        action_pred = self.model(latent_action)

        loss = self.loss_fn(action_pred, gt_actions)
        loss = loss.mean()

        return {"action_loss": loss.item()}, loss

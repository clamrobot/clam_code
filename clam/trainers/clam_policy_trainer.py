import copy
from pathlib import Path

import einops
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf

from clam.models.mlp_policy import MLPPolicy
from clam.models.utils.clam_utils import get_la_dim
from clam.trainers.clam_trainer import get_clam_cls
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.logger import log


class CLAMPolicy(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        action_decoder,
        input_dim: int,
        output_dim: int,
        la_dim: int,
        distributional_la: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_decoder = action_decoder
        self.la_dim = la_dim
        self.distributional_la = distributional_la
        self.latent_policy = MLPPolicy(cfg, input_dim=input_dim, output_dim=output_dim)

    def forward(self, observations):
        latent_action = self.latent_policy(observations)
        if self.distributional_la:
            latent_action_mean = latent_action[:, : self.la_dim]
            latent_action_logvar = latent_action[:, self.la_dim :]

            if self.cfg.use_latent_mean:
                latent_action = latent_action_mean
            else:
                latent_action = latent_action_mean + torch.exp(
                    latent_action_logvar * 0.5
                ) * torch.randn_like(latent_action_mean)

        action = self.action_decoder(latent_action)
        return action

    def get_latent_action(self, observations):
        return self.latent_policy(observations)


class CLAMPolicyTrainer(OfflineTrainer):
    """
    Policy that maps from latent actions to environment actions
    """

    def __init__(self, cfg):
        # call the superclass constructor
        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

    def setup_model(self):
        cfg_file = Path(self.cfg.lam_ckpt) / "config.yaml"
        log(f"loading config from {cfg_file}", "blue")
        lam_cfg = OmegaConf.load(cfg_file)
        self.lam_cfg = lam_cfg

        la_dim = get_la_dim(lam_cfg)
        self.use_transformer = (
            "transformer" in lam_cfg.name or lam_cfg.name == "st_vivit_clam"
        )

        if not self.cfg.data.load_latent_actions:
            log("Using trained CLAM to compute latent actions", "red")

            clam_cls = get_clam_cls(lam_cfg.name)
            self.clam = clam_cls(lam_cfg.model, input_dim=self.obs_shape, la_dim=la_dim)
            self.clam.load_from_ckpt(
                self.cfg.lam_ckpt, self.cfg.lam_ckpt_step, key="model"
            )
            self.clam = self.clam.to(self.device)
            log("LAM model loaded successfully", "green")

        else:
            log("Loaded latent actions from data", "red")

        if (
            "action_decoder_ckpt" in self.cfg
            and self.cfg.action_decoder_ckpt is not None
        ):
            action_decoder_ckpt = self.cfg.action_decoder_ckpt

            cfg_file = Path(action_decoder_ckpt) / "config.yaml"
            action_decoder_cfg = OmegaConf.load(cfg_file)

            self.action_decoder = MLPPolicy(
                action_decoder_cfg.model,
                input_dim=la_dim,
                output_dim=self.cfg.env.action_dim,
            )
            self.action_decoder.load_from_ckpt(
                action_decoder_ckpt, self.cfg.action_decoder_ckpt_step, key="model"
            )
            log("Action decoder separately trained!", "red")
        else:
            self.action_decoder = MLPPolicy(
                lam_cfg.model.action_decoder,
                input_dim=la_dim,
                output_dim=self.cfg.env.action_dim,
            )
            self.action_decoder.load_from_ckpt(
                self.cfg.lam_ckpt, self.cfg.lam_ckpt_step, key="action_decoder"
            )
            log("Action decoder loaded from checkpoint!", "red")

        self.action_decoder = self.action_decoder.to(self.device)

        model = CLAMPolicy(
            self.cfg.model,
            action_decoder=self.action_decoder,
            input_dim=self.obs_shape,
            output_dim=la_dim,
            la_dim=la_dim,
            distributional_la=lam_cfg.model.distributional_la,
        )
        return model

    def compute_loss(self, batch, train: bool = True):
        if self.use_transformer:
            if not self.cfg.data.load_latent_actions:
                # need to include the timestep for positional encodings
                with torch.no_grad():
                    clam_output = self.clam(
                        batch.observations, timesteps=batch.timestep
                    )

                if self.cfg.env.image_obs:
                    B, T = batch.observations.shape[:2]
                    # flatten [B, T]
                    observations = einops.rearrange(
                        batch.observations[:, :-1], "B T C H W -> (B T) C H W"
                    )
                    latent_action_pred = self.model.get_latent_action(observations)
                    # reshape back to [B, T, D]
                    latent_action_pred = einops.rearrange(
                        latent_action_pred, "(B T) D -> B T D", B=B, T=T - 1
                    )

                else:
                    # TODO: check this, but we predict the latent action between every step
                    latent_action_pred = self.model.get_latent_action(
                        batch.observations[:, :-1]
                    )
                latent_action = clam_output.la[:, 1:]
            else:
                latent_action_pred = self.model.get_latent_action(batch.observations)
                latent_action = batch.latent_actions
        else:
            if not self.cfg.data.load_latent_actions:
                with torch.no_grad():
                    clam_output = self.clam(batch.observations)

                latent_action = clam_output.la
                latent_action_pred = self.model.get_latent_action(
                    batch.observations[:, 1]
                )
            else:
                latent_action = batch.latent_actions
                latent_action_pred = self.model.get_latent_action(batch.observations)

        action_loss = self.loss_fn(latent_action_pred, latent_action)
        action_loss = action_loss.mean()

        loss = action_loss
        metrics = {"latent_action_loss": loss.item()}
        return metrics, loss

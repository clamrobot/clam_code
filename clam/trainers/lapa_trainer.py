import copy
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig, OmegaConf

from clam.models.base import BaseModel
from clam.models.mlp_policy import MLPPolicy
from clam.models.utils.utils import make_mlp
from clam.trainers.clam_trainer import get_clam_cls, get_labelled_dataloader
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.data_utils import Batch
from clam.utils.logger import log


class LAPA(BaseModel):
    """
    LAPA is a latent action policy that maps from o_t to z_t
    """

    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super().__init__(cfg=cfg, input_dim=input_dim)
        self.cfg = cfg
        self.activation = nn.LeakyReLU(0.2)

        # this spits out the representation which is then inputted to
        # either the latent policy head or the action head
        self.backbone, _ = make_mlp(
            cfg.backbone, input_dim=input_dim, output_dim=cfg.embedding_dim
        )

        self.latent_head, _ = make_mlp(
            net_kwargs=cfg.latent_head,
            input_dim=cfg.embedding_dim,
            output_dim=la_dim,
        )

        self.action_head, _ = make_mlp(
            net_kwargs=cfg.action_head,
            input_dim=cfg.embedding_dim,
            output_dim=cfg.action_dim,
        )

        if cfg.action_activation:
            self.action_activation = getattr(nn, cfg.action_activation)()
        else:
            self.action_activation = None

    def forward(self, observations):
        embed = self.backbone(observations)
        embed = self.activation(embed)
        action = self.action_head(embed)  # finetune an action head
        if self.action_activation:
            action = self.action_activation(action)

            # scale action
            if self.cfg.action_scale:
                action = action * self.cfg.action_scale
        return action

    def get_latent_action(self, observations):
        embed = self.backbone(observations)
        embed = self.activation(embed)
        return self.latent_head(embed)


class LAPATrainer(OfflineTrainer):
    """
    LAPA takes the trained LAM model and first learns a policy that maps from
    observations to latent actions.
    """

    def __init__(self, cfg):
        # stage 1: train LAP
        # stage 2: train LAPA
        super().__init__(cfg)

        self.loss_fn = nn.MSELoss(reduction="none")

        log(f"LAPA trainer, stage: {self.cfg.name}", "blue")
        if self.cfg.name == "lap":
            log("Training LAP, just training the latent action policy", "blue")
        else:
            log(
                "Training LAPA, finetuning the whole policy with the action labelled data",
                "blue",
            )
            # also grab labelled data
            log("loading labelled data for finetuning model")
            cfg_cpy = copy.deepcopy(cfg)
            self.train_dataloader = get_labelled_dataloader(cfg_cpy)

    def setup_model(self):
        if self.cfg.name == "lap":
            # load trained LAM model
            lam_cfg_file = Path(self.cfg.lam_ckpt) / "config.yaml"
            log(f"loading LAM config from {lam_cfg_file}", "blue")
            self.lam_cfg = OmegaConf.load(lam_cfg_file)

            clam_cls = get_clam_cls(self.lam_cfg.name)
            self.clam = clam_cls(self.lam_cfg.model, input_dim=self.obs_shape)
            self.clam.load_from_ckpt(
                self.cfg.lam_ckpt, self.cfg.lam_ckpt_step, key="model"
            )
            self.clam = self.clam.to(self.device)
            self.use_transformer_clam = "transformer" in self.lam_cfg.model.idm.net.name

            log("LAM model loaded successfully", "green")
            # update the action dim in the model config
            model = LAPA(
                self.cfg.model,
                input_dim=self.obs_shape,
                la_dim=self.lam_cfg.model.idm.la_dim,
            )

        elif self.cfg.name == "lapa":
            if self.cfg.load_lap_ckpt:
                lap_cfg_file = Path(self.cfg.lap_ckpt) / "config.yaml"
                log(f"loading LAP config from {lap_cfg_file}", "blue")
                self.lap_cfg = OmegaConf.load(lap_cfg_file)

                model = LAPA(
                    self.lap_cfg.model,
                    input_dim=self.obs_shape,
                    la_dim=self.lap_cfg.model.la_dim,
                )

                # load the latent policy part, but we finetune the whole model
                model.load_from_ckpt(
                    self.cfg.lap_ckpt,
                    self.cfg.lap_ckpt_step,
                    key="model",
                    strict=False,
                )
                log("LAP model loaded successfully", "green")
            else:
                model = LAPA(self.cfg.model, input_dim=self.obs_shape)
                log("Running ablation, not loading LAP model", "red")

        return model

    def compute_loss(self, batch, train: bool = True):
        if self.cfg.name == "lap":
            # train LAP
            clam_output = self.clam(batch.observations, timesteps=batch.timestep)
            if self.use_transformer_clam:
                latent_action_pred = self.model.get_latent_action(
                    batch.observations[:, 1:]
                )
                latent_action = clam_output.la[:, 1:]
            else:
                latent_action_pred = self.model.get_latent_action(
                    batch.observations[:, 1]
                )
                latent_action = clam_output.la[:, 1]

            la_pred_loss = self.loss_fn(latent_action_pred, latent_action)
            la_pred_loss = la_pred_loss.mean()

            loss = la_pred_loss
            metrics = {"latent_action_pred_loss": loss.item()}
        elif self.cfg.name == "lapa":
            action_pred = self.model(batch.observations)
            loss = self.loss_fn(action_pred, batch.actions)
            loss = loss.mean()

            metrics = {"action_loss": loss.item()}
        else:
            raise ValueError(f"Unknown stage {self.cfg.name}")

        return metrics, loss

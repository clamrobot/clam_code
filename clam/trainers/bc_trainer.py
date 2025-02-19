from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from clam.models.dynamo import DynaMo
from clam.models.mlp_policy import MLPPolicy
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.logger import log


# TODO: move this function out
def gaussian_nll_loss(actions, means, logvars):
    """
    Computes the negative log-likelihood loss for a Gaussian policy when the model predicts log-variance.

    Args:
        actions: Tensor of shape (batch_size, action_dim), the sampled actions.
        means: Tensor of shape (batch_size, action_dim), the predicted means.
        logvars: Tensor of shape (batch_size, action_dim), the predicted log-variances.

    Returns:
        Tensor of shape (batch_size,), the NLL loss for each sample.
    """
    # Compute variance from log-variance
    var = torch.exp(logvars)

    # Compute the log probability
    log_prob = -0.5 * (
        ((actions - means) ** 2 / var) + logvars + torch.log(torch.tensor(2 * torch.pi))
    )

    # Negative log-likelihood
    nll = -torch.sum(log_prob, dim=-1)  # Sum over action dimensions
    return nll


class BCTrainer(OfflineTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

    def setup_model(self):
        # setup vision encoder if needed to extract features from images
        # when we have image observations, i.e. LAPA, Dynamo

        if "feature_extractor" in self.cfg.model and self.cfg.model.feature_extractor:
            # load pretrained vision model encoder
            if self.cfg.model.feature_extractor == "dynamo":
                # load ckpt file
                cfg_file = Path(self.cfg.model.feature_extractor_ckpt) / "config.yaml"
                vision_cfg = OmegaConf.load(cfg_file)

                dynamo = DynaMo(vision_cfg.model, input_dim=self.obs_shape)
                dynamo.load_from_ckpt(
                    self.cfg.model.feature_extractor_ckpt, key="model"
                )
                feature_extractor = dynamo.obs_embedding
            else:
                raise ValueError(
                    f"Unknown vision model {self.cfg.model.feature_extractor}"
                )

            if not self.cfg.model.finetune_feature_extractor:
                log("Freezing feature extractor, not training these weights", "blue")
                for param in feature_extractor.parameters():
                    param.requires_grad = False

        else:
            feature_extractor = None

        model = MLPPolicy(
            self.cfg.model,
            input_dim=self.obs_shape,
            output_dim=self.cfg.env.action_dim,
            feature_extractor=feature_extractor,
        )
        return model

    def compute_loss(self, batch, train: bool = True):
        if self.cfg.model.gaussian_policy:
            action_mean, action_logvar = self.model(batch.observations)
            loss = gaussian_nll_loss(batch.actions, action_mean, action_logvar)
        else:
            action_pred = self.model(batch.observations)
            loss = self.loss_fn(action_pred, batch.actions)

        loss = loss.mean()
        return {"action_loss": loss.item()}, loss

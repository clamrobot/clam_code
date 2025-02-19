import torch
import torch.nn as nn

from clam.models.dynamo import DynaMo
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.logger import log


class DynaMoTrainer(OfflineTrainer):
    """
    Rough implementation of https://arxiv.org/pdf/2409.12192

    DynaMo is a transformer encoder-decoder trained with the LAM
    pretraining objective and then uses the learned encoder
    to learn a policy mapping from state->actions
    """

    @staticmethod
    def covar_reg_loss_fn(obs, state_dim):
        cov_mat = torch.cov(obs)

        # sum of all off-diagonal elements
        off_diag = torch.sum(cov_mat - torch.diag(torch.diagonal(cov_mat)))

        # divide by the state dimension
        return off_diag.sum() ** 2 / state_dim

    @staticmethod
    def dynamics_loss_fn(s_t, s_t_pred):
        # see Formula 1 in the paper
        # 1 - x * y / (||x|| * ||y||)
        return 1 - nn.CosineSimilarity(dim=1)(s_t, s_t_pred)

    def setup_model(self):
        model = DynaMo(self.cfg.model, input_dim=self.obs_shape)
        return model

    def compute_loss(self, batch, train: bool = True):
        # [B, T-1, D]
        forward_predictions = self.model(batch.observations, timesteps=batch.timestep)

        # compute the embedding of the next frame to use as target recon
        # [B, T-1, D]
        target_next_frame_embed = (
            self.model.obs_embedding(batch.observations[:, 1:])
        ).detach()

        assert forward_predictions.shape == target_next_frame_embed.shape

        # the dynamics objective predicts the target embedding at the next step
        dynamics_loss = self.dynamics_loss_fn(
            forward_predictions, target_next_frame_embed
        )
        dynamics_loss = dynamics_loss.mean()

        # computes covariance of a minibatch of observation embeddings
        # this is just some regularization objective for the encoding outputs
        obs_embed = self.model.obs_embedding(batch.observations)
        cov_reg_loss = self.covar_reg_loss_fn(
            obs_embed.flatten(0, 1), state_dim=self.obs_shape
        )

        cov_reg_loss = cov_reg_loss * self.cfg.model.cov_reg_weight
        loss = dynamics_loss + cov_reg_loss

        metrics = {
            "dynamics_loss": dynamics_loss.item(),
            "cov_reg_loss": cov_reg_loss.item(),
            "total_loss": loss.item(),
        }

        return metrics, loss

import torch
import torch.nn as nn
from omegaconf import DictConfig

from clam.models.base import BaseModel
from clam.models.clam.transformer_clam import TransformerCLAM
from clam.models.utils.utils import make_mlp


class DynaMo(BaseModel):
    """
    DynaMo jointly learns the obs embedding, IDM, and FDM.
    Mainly it uses the IDM and FDM as pretraining objectives for learning the visual encoder.
    """

    def __init__(self, cfg: DictConfig, input_dim: int):
        super(DynaMo, self).__init__(cfg, input_dim)

        # TODO: upgrade this to convnet for image input
        if cfg.image_obs:
            raise NotImplementedError(
                "DynaMo only supports vector observations for now"
            )
        else:
            self.obs_embedding, _ = make_mlp(
                net_kwargs=cfg.encoder,
                input_dim=input_dim,
                output_dim=cfg.embedding_dim,
            )

        self.lam = TransformerCLAM(cfg, input_dim=cfg.embedding_dim)

    def forward(self, observations, timesteps=None):
        """
        Args:
            observations: [B, T, D]

        Returns:
            forward_predictions: [B, T-1, D], the predicted next frame
        """
        obs_embedding = self.obs_embedding(observations)
        forward_predictions = self.lam(obs_embedding, timesteps=timesteps)
        return forward_predictions.reconstructed_obs

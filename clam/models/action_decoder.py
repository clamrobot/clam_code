import einops
import torch
import torch.nn as nn

from omegaconf import DictConfig
from clam.utils.logger import log
from clam.models.utils import CLAMOutput
from x_transformers import ContinuousTransformerWrapper, Decoder


class ActionChunkingDecoder(nn.Module):
    """
    Action Chunking Transformer decoder
    or this could just be some diffusion head.

    For ACT, we take the first observation and predict
    a_t:t+k
    """

    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.cfg = cfg
        self.observation_embed = nn.Linear(input_dim, cfg.embedding_dim)

        self.transformer_decoder = ContinuousTransformerWrapper(
            dim_in=cfg.embedding_dim,
            dim_out=output_dim,  # environment action dim
            max_seq_len=1024,
            # TODO: make these hyperparameters
            attn_layers=Decoder(dim=512, depth=4, heads=4, cross_attend=True),
        )

    def forward(self, o_t: torch.Tensor, clam_output: CLAMOutput) -> torch.Tensor:
        """
        First embed the observation.

        Then apply transformer decoder to predict action sequence.
        The query and keys are the embedded observations and the values are the
        latent actions concatenated together.

        Args:
            clam_output
        """
        import ipdb

        ipdb.set_trace()

        discrete_latent_action = clam_output.idm_output.discrete_latent_action
        continuous_latent_action = clam_output.idm_output.continuous_latent_action

        obs_embed = self.observation_embed(o_t)

        # apply transformer decoder
        # the input is just positional tokens
        self.transformer_decoder(context=obs_embed)

        return

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from clam.models.act.models import ACTDecoder, ACTEncoder
from clam.models.base import BaseModel
from clam.models.clam.clam import get_vq_cls
from clam.models.utils.transformer_utils import get_pos_encoding
from clam.models.utils.utils import CLAMOutput, IDMOutput, compute_perplexity
from clam.utils.logger import log


class TransformerIDM(nn.Module):
    """
    TransformerIDM takes sequence of observations and returns z_t
    between each pair of observations.

    (o_t-1, o_t, o_t+1) -> (z_t-1, z_t)
    """

    def __init__(self, cfg: DictConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg

        self.input_dim = input_dim

        # first embed the input
        self.input_embed = nn.Linear(input_dim, cfg.input_embed_dim)
        self.activation = nn.LeakyReLU(0.2)

        self.encoder = ACTEncoder(cfg.net)
        self.latent_action = nn.Linear(cfg.input_embed_dim, cfg.la_dim)
        self.pos_embed = get_pos_encoding(
            cfg.net.pos_enc, embedding_dim=cfg.input_embed_dim, max_len=200
        )

        if self.cfg.quantize_la:
            vq_cls = get_vq_cls(self.cfg.vq.name)
            log(f"Using vq {self.cfg.vq.name}", "green")
            self.cfg.vq.kwargs.dim = self.cfg.la_dim
            self.vq = vq_cls(**self.cfg.vq.kwargs)
        else:
            log("Not using vq, continuous latent action space", "red")

    def forward(self, observations, timesteps=None, causal: bool = False, **kwargs):
        """
        Args:
            observations: [B, T, D]

        Returns:
            latent_action: [B, T, D]
            obs_embed: [B, T, D]
            timesteps: [B, T]
        """
        # first embed the input
        obs_embed = self.input_embed(observations)
        obs_embed = self.activation(obs_embed)

        self.pos_embed = self.pos_embed.to(obs_embed.device)

        if self.cfg.net.pos_enc == "learned":
            pos_embed = self.pos_embed(timesteps.int())
        else:
            pos_embed = self.pos_embed[timesteps]

        # embed(o_t-1), embed(o_t), embed(o_t+1)
        obs_embed = self.encoder(obs_embed, pos_embed=pos_embed)
        la = self.latent_action(obs_embed)

        # TODO: clean this up, duplicated with regular CLAM
        if self.cfg.quantize_la:
            quantized_la, indices, vq_loss = self.vq(la)

            if self.cfg.use_quantized_las:  # replace la with quantized_la
                la = quantized_la

            vq_loss = vq_loss.mean()

            vq_outputs = {"indices": indices}
            vq_metrics = {
                "vq_loss": vq_loss.item(),
                "perplexity": compute_perplexity(
                    indices, self.cfg.vq.kwargs.codebook_size
                ),
            }

            return IDMOutput(
                la=la,
                quantized_la=quantized_la,
                vq_loss=vq_loss,
                vq_metrics=vq_metrics,
                vq_outputs=vq_outputs,
            )

        return IDMOutput(la=la, encoder_out=obs_embed)


class TransformerFDM(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, target_dim: int = None):
        super().__init__()
        self.cfg = cfg

        self.input_dim = input_dim
        self.activation = nn.LeakyReLU(0.2)

        # first embed the input
        self.input_embed = nn.Linear(input_dim, cfg.input_embed_dim)
        self.la_embed = nn.Linear(cfg.la_dim, cfg.input_embed_dim)

        self.decoder = ACTDecoder(cfg.net)
        self.decoder_pos_embed = get_pos_encoding(
            cfg.net.pos_enc, embedding_dim=cfg.input_embed_dim, max_len=200
        )
        self.encoder_pos_embed = get_pos_encoding(
            cfg.net.pos_enc, embedding_dim=cfg.input_embed_dim, max_len=200
        )

        # map the decoder output to the observation
        target_dim = input_dim if target_dim is None else target_dim
        self.to_observation = nn.Linear(cfg.input_embed_dim, target_dim)

    def forward(self, observations, idm_output: IDMOutput, timesteps=None, **kwargs):
        """
        Args:
            observations: [B, T, D]
            idm_output: IDMOutput
            timesteps: [B, T]

        Returns:
            reconstructed_obs: [B, T, D]
        """
        context = observations[:, :-1]
        obs_embed = self.input_embed(context)
        obs_embed = self.activation(obs_embed)

        self.decoder_pos_embed = self.decoder_pos_embed.to(obs_embed.device)
        self.encoder_pos_embed = self.encoder_pos_embed.to(obs_embed.device)

        if self.cfg.net.pos_enc == "learned":
            decoder_pos_embed = self.decoder_pos_embed(timesteps[:, :-1].int())
            encoder_pos_embed = self.encoder_pos_embed(timesteps[:, 1:].int())
        else:
            decoder_pos_embed = self.decoder_pos_embed[timesteps[:, :-1]]
            encoder_pos_embed = self.encoder_pos_embed[timesteps[:, 1:]]

        # embed latent actions, ignore the first
        encoder_out = self.la_embed(idm_output.la[:, 1:])

        # TODO: currently no causal mask applied
        obs_embed = self.decoder(
            x=obs_embed,
            encoder_out=encoder_out,
            decoder_pos_embed=decoder_pos_embed,
            encoder_pos_embed=encoder_pos_embed,
        )

        # reconstruct the observation
        return self.to_observation(obs_embed)


class TransformerCLAM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int = None):
        super().__init__(cfg, input_dim)
        self.name = "TransformerCLAM"

        self.idm = TransformerIDM(cfg.idm, input_dim=input_dim)
        self.fdm = TransformerFDM(cfg.fdm, input_dim=input_dim)

    def forward(self, observations, timesteps=None, **kwargs):
        idm_output = self.idm(observations, timesteps=timesteps, **kwargs)
        reconstructed_obs = self.fdm(
            observations, idm_output=idm_output, timesteps=timesteps, **kwargs
        )

        return CLAMOutput(
            la=idm_output.la, reconstructed_obs=reconstructed_obs, idm_output=idm_output
        )

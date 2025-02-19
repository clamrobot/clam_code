import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from clam.models.utils.utils import get_activation_fn
from clam.utils.logger import log


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [ACTEncoderLayer(config) for _ in range(config.n_encoder_layers)]
        )
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.dim_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        # Feed forward layers.
        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.activation = get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x)[
            0
        ]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList(
            [ACTDecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)]
        )
        self.norm = nn.LayerNorm(cfg.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.dim_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            cfg.dim_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        # Feed forward layers.
        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.norm1 = nn.LayerNorm(cfg.dim_model)
        self.norm2 = nn.LayerNorm(cfg.dim_model)
        self.norm3 = nn.LayerNorm(cfg.dim_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

        self.activation = get_activation_fn(cfg.feedforward_activation)
        self.pre_norm = cfg.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Batch, Decoder Sequence, Channel) tensor of input tokens.
            encoder_out: (B, Encoder Sequence, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        causal_mask = torch.full(
            (x.size(1), x.size(1)), -np.inf, device=x.device, dtype=x.dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        x = self.self_attn(q, k, value=x, attn_mask=causal_mask, is_causal=True)[
            0
        ]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # this is the cross-attention part
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
            attn_mask=causal_mask,
            is_causal=True,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x

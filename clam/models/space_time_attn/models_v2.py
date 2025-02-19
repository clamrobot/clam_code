import math
from typing import Any, Dict

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from clam.models.utils.utils import get_activation_fn


class STBlock(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # input here is (B*T, N, D)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.dim_model,
            num_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )

        # input here is (B*N, T, D)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.dim_model,
            num_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )

        # apply cross attention to the latent actions
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.dim_model,
            num_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(cfg.dim_model, cfg.dim_feedforward)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.dim_model)

        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)
        self.dropout4 = nn.Dropout(cfg.dropout)

        self.norm = lambda x: F.layer_norm(x, x.size()[1:])
        self.activation = get_activation_fn(cfg.feedforward_activation)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        causal: bool,
        cond: torch.Tensor = None,
        cond_pos_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, N, D) where T is the sequence length, N is the number of patches, and D is the feature dimension.
            pos_embed: Positional embedding tensor of shape (B, T, N, D)
            cond: (B, T, 1, D) tensor representing the latent action
            cond_pos_embed: (B, T, N, D)
        """
        # --- Spatial attention ---
        B, T, N, D = x.size()
        x = x.flatten(start_dim=0, end_dim=1)  # (B, T, N, D) -> (B*T, N, D)
        pos_embed_flat = (
            pos_embed.flatten(start_dim=0, end_dim=1) if pos_embed is not None else None
        )

        skip = x
        if self.cfg.pre_norm:
            x = self.norm(x)

        if x.shape != pos_embed_flat.shape:
            import ipdb

            ipdb.set_trace()

        q = k = x if pos_embed_flat is None else x + pos_embed_flat

        x, _ = self.spatial_attn(q, k, value=x)
        x = skip + self.dropout1(x)
        if self.cfg.pre_norm:
            skip = x
            x = self.norm(x)
        else:
            x = self.norm(x)
            skip = x

        # --- Temporal attention ---

        # reshape back
        x = einops.rearrange(x, "(B T) N D -> B T N D", B=B, T=T)
        x = x.transpose(1, 2)  # Swapaxes to (B, N, T, D)
        x = x.flatten(start_dim=0, end_dim=1)  # (B, N, T, D) -> (B*N, T, D)

        pos_embed_flat = pos_embed.transpose(1, 2) if pos_embed is not None else None
        pos_embed_flat = (
            pos_embed_flat.flatten(start_dim=0, end_dim=1)
            if pos_embed is not None
            else None
        )

        skip = x
        if self.cfg.pre_norm:
            x = self.norm(x)

        q = k = x if pos_embed_flat is None else x + pos_embed_flat

        if causal:
            causal_mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)
        else:
            causal_mask = None

        x, _ = self.temporal_attn(
            q, k, value=x, attn_mask=causal_mask, is_causal=causal
        )
        x = skip + self.dropout2(x)
        if self.cfg.pre_norm:
            skip = x
            x = self.norm(x)
        else:
            x = self.norm(x)
            skip = x

        if cond is not None:  # cross attention with the cond (latent action)
            cond = einops.repeat(cond, "B T 1 D -> (B N) T D", N=N)
            # apply cross attention
            skip = x
            if self.cfg.pre_norm:
                x = self.norm(x)

            q = x if pos_embed_flat is None else x + pos_embed_flat
            # reshape cond embed
            cond_pos_embed = einops.rearrange(cond_pos_embed, "B T N D -> (B N) T D")
            k = cond + cond_pos_embed
            v = cond
            x, _ = self.cross_attn(
                q, k, value=v, attn_mask=causal_mask, is_causal=causal
            )
            x = skip + self.dropout3(x)

            if self.cfg.pre_norm:
                skip = x
                x = self.norm(x)
            else:
                x = self.norm(x)
                skip = x

        x = einops.rearrange(x, "(B N) T D -> B T N D", B=B, T=T)
        skip = x

        # --- Feedforward ---
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout4(x)
        if not self.cfg.pre_norm:
            x = self.norm(x)

        return x


class STTransformer(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.layers = nn.ModuleList([STBlock(cfg=cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.dim_model) if cfg.pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        causal: bool = False,
        cond: torch.Tensor | None = None,
        cond_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                pos_embed=pos_embed,
                causal=causal,
                cond=cond,
                cond_pos_embed=cond_pos_embed,
            )
        x = self.norm(x)
        return x

import math
from typing import Any, Dict
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class STBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, num_heads: int, dropout: float):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=seq_len, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Linear(dim, dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D).
        """
        # --- Spatial attention ---
        z = F.layer_norm(x, x.size()[1:])
        z, _ = self.spatial_attn(z, z, z)
        x = x + self.dropout1(z)

        # --- Temporal attention ---
        x = x.transpose(1, 2)  # Swapaxes to (B, D, T)
        z = F.layer_norm(x, x.size()[1:])
        if causal:
            causal_mask = torch.triu(torch.ones(z.size(1), z.size(1)), diagonal=1).to(
                x.device
            )
        else:
            causal_mask = None

        z, _ = self.temporal_attn(z, z, z, attn_mask=causal_mask)
        x = x + self.dropout2(z)
        x = x.transpose(1, 2)  # Swapaxes back to (B, T, D)

        # --- Feedforward ---
        z = F.layer_norm(x, x.size()[1:])
        z = self.ff(z)
        z = F.gelu(z)
        x = x + self.dropout3(z)

        return x


class STTransformer(nn.Module):
    def __init__(self, cfg: DictConfig, causal: bool = False):
        super().__init__()
        self.cfg = cfg

        self.input_embed = nn.Linear(cfg.input_dim, cfg.model_dim)

        self.blocks = nn.ModuleList(
            [
                STBlock(
                    dim=cfg.model_dim,
                    seq_len=cfg.seq_len,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_blocks)
            ]
        )

        self.causal = causal
        self.output_embed = nn.Linear(cfg.model_dim, cfg.output_dim)

    def forward(self, x, pos_embed: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D_in).
        Returns:
            Tensor of shape (B, T, D_out).
        """
        # Input normalization and projection
        x = F.layer_norm(x, x.size()[1:])
        x = self.input_embed(x)
        x = F.layer_norm(x, x.size()[1:])

        # Add positional encoding
        x = x + pos_embed

        # Pass through STBlocks
        for block in self.blocks:
            x = block(x, causal=self.causal)

        # Output projection
        x = self.output_embed(x)
        return x

"""1D Spectral Transformer for FTIR features."""
from __future__ import annotations
import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpectralTransformer(nn.Module):
    def __init__(self, n_ftir_features: int = 40, embed_dim: int = 64,
                 out_dim: int = 128, n_heads: int = 4, n_layers: int = 3,
                 ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_ftir = n_ftir_features
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Per-wavenumber embedding
        self.input_proj = nn.Linear(1, embed_dim)
        # Learnable positional encoding
        self.pos_enc = nn.Parameter(
            torch.randn(1, n_ftir_features, embed_dim) / math.sqrt(embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_ftir) -> (B, out_dim)"""
        B, N = x.shape
        # Each wavenumber becomes a token: (B, N, 1) -> (B, N, embed_dim)
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_enc[:, :N, :]
        tokens = self.transformer(tokens)  # (B, N, embed_dim)
        pooled = tokens.mean(dim=1)  # (B, embed_dim)
        out = self.output_proj(pooled)  # (B, out_dim)
        assert out.shape[1] == self.out_dim, f"Expected (B,{self.out_dim}), got {out.shape}"
        return out

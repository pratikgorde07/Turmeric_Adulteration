from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class SpectralTransformer(nn.Module):
    """1D Spectral Transformer for FTIR features. Output: (B, 128)."""

    def __init__(self, n_ftir_features: int, config):
        super().__init__()
        self.n_ftir = n_ftir_features
        self.config = config

        d_token = 64  # per-wavenumber embedding dimension

        # Per-wavenumber projection: (B, n_ftir) -> (B, n_ftir, d_token)
        self.input_proj = nn.Linear(1, d_token)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_ftir_features, d_token) / (d_token ** 0.5)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Output projection: d_token -> 128
        self.output_proj = nn.Sequential(
            nn.Linear(d_token, 128),
            nn.LayerNorm(128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_ftir_features) FTIR feature tensor
        Returns:
            (B, 128) spectral embedding
        """
        B, n = x.shape
        assert n == self.n_ftir, (
            f"Expected n_ftir={self.n_ftir} features, got {n}"
        )

        # (B, n_ftir) -> (B, n_ftir, 1) -> (B, n_ftir, d_token)
        x_tok = x.unsqueeze(-1)  # (B, n_ftir, 1)
        x_tok = self.input_proj(x_tok)  # (B, n_ftir, 64)

        # Add positional encoding
        x_tok = x_tok + self.pos_encoding  # (B, n_ftir, 64)

        # Transformer
        x_tok = self.transformer(x_tok)  # (B, n_ftir, 64)

        # Global mean pool across wavenumber dimension
        x_pooled = x_tok.mean(dim=1)  # (B, 64)

        # Output projection
        out = self.output_proj(x_pooled)  # (B, 128)

        assert out.shape == (B, 128), f"Expected (B,128), got {out.shape}"
        return out

"""Colorimetric MLP encoder."""
from __future__ import annotations
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ColorimetricEncoder(nn.Module):
    def __init__(self, n_color_features: int = 20, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_color_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_color_features) -> (B, out_dim)"""
        out = self.encoder(x)
        assert out.shape[1] == self.out_dim, f"Expected (B,{self.out_dim}), got {out.shape}"
        return out

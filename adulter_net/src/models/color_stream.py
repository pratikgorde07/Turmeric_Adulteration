from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class ColorimetricEncoder(nn.Module):
    """MLP encoder for colorimetric (HunterLab) features. Output: (B, 64)."""

    def __init__(self, n_color_features: int, config):
        super().__init__()
        self.n_color = n_color_features

        # BatchNorm1d used (not LayerNorm): population-level norm appropriate
        # for physically correlated colorimetric measurements
        self.encoder = nn.Sequential(
            nn.Linear(n_color_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_color_features) colorimetric tensor
        Returns:
            (B, 64) color embedding
        """
        B = x.shape[0]
        out = self.encoder(x)
        assert out.shape == (B, 64), f"Expected (B,64), got {out.shape}"
        return out

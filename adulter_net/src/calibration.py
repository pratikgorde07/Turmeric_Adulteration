from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_logger

logger = get_logger(__name__)


class TemperatureScaler(nn.Module):
    """Post-hoc calibration via temperature scaling on validation logits."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Optimise temperature to minimise NLL on validation logits.

        Args:
            logits: (N, n_classes) raw (uncalibrated) logits on validation set
            labels: (N,) integer class indices
        """
        assert logits.shape[0] == labels.shape[0], "Batch size mismatch"
        device = logits.device
        self.to(device)

        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_closure():
            optimizer.zero_grad()
            scaled = self.scale(logits)
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)

        # Clamp temperature to a reasonable range
        with torch.no_grad():
            self.temperature.clamp_(min=0.1, max=10.0)

        nll_before = nll(logits, labels).item()
        nll_after = nll(self.scale(logits), labels).item()
        logger.info(
            f"Temperature calibration: T={self.temperature.item():.4f} | "
            f"NLL before={nll_before:.4f}, after={nll_after:.4f}"
        )

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by learned temperature."""
        return logits / self.temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale(logits)

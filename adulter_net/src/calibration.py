"""Temperature scaling calibration."""
from __future__ import annotations
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll = nn.CrossEntropyLoss()

        def eval_closure():
            optimizer.zero_grad()
            loss = nll(self.scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_closure)
        logger.info(f"Temperature calibrated to {self.temperature.item():.4f}")

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

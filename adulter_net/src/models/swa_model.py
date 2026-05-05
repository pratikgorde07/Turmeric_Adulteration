from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class SWAWrapper:
    """Stochastic Weight Averaging wrapper."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, config):
        self.config = config
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            optimizer,
            swa_lr=config.swa_lr,
            anneal_epochs=config.swa_anneal_epochs,
        )
        logger.info(
            f"SWAWrapper initialized: swa_lr={config.swa_lr}, "
            f"anneal_epochs={config.swa_anneal_epochs}"
        )

    def update(self, model: nn.Module) -> None:
        """Update SWA model with current model weights."""
        self.swa_model.update_parameters(model)

    def step_scheduler(self) -> None:
        """Step SWA learning rate scheduler."""
        self.swa_scheduler.step()

    def update_bn(self, train_loader, device: str) -> None:
        """Update batch normalization statistics on training data."""
        logger.info("Updating SWA BatchNorm statistics...")
        update_bn(train_loader, self.swa_model, device=device)
        logger.info("SWA BatchNorm update complete.")

    def get_model(self) -> AveragedModel:
        return self.swa_model

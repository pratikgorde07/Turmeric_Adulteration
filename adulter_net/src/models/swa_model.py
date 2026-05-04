"""SWA model wrapper."""
from __future__ import annotations
import logging

import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

logger = logging.getLogger(__name__)


class SWAWrapper:
    def __init__(self, model, optimizer, config):
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            optimizer,
            swa_lr=config.swa_lr,
            anneal_epochs=config.swa_anneal_epochs,
        )
        logger.info("SWAWrapper initialised")

    def update(self, model) -> None:
        self.swa_model.update_parameters(model)

    def update_bn(self, train_loader, device: str) -> None:
        logger.info("Updating BatchNorm statistics for SWA model...")
        update_bn(train_loader, self.swa_model, device=device)
        logger.info("SWA BatchNorm update complete")

    def step_scheduler(self) -> None:
        self.swa_scheduler.step()

    def get_model(self) -> AveragedModel:
        return self.swa_model

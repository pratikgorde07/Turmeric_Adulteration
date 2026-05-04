"""Fold ensemble logic."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FoldEnsemble:
    def __init__(self, model_paths: List[str], config):
        self.config = config
        self.models = []
        for path in model_paths:
            from src.models.adulter_net import AdulterNet
            # Dimensions will be loaded from checkpoint
            checkpoint = torch.load(path, map_location=config.device)
            model_kwargs = checkpoint.get("model_kwargs", {})
            model = AdulterNet(
                config,
                n_img_features=model_kwargs.get("n_img_features", config.img_feat_dim_raw),
                n_ftir_features=model_kwargs.get("n_ftir_features", config.spa_n_components),
                n_color_features=model_kwargs.get("n_color_features", config.mrmr_n_features),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.models.append(model.to(config.device))
            logger.info(f"Loaded model from {path}")

    def predict_proba(self, batch: dict, use_tta: bool = True) -> torch.Tensor:
        from src.tta import tta_predict, TTA_TRANSFORMS
        all_probs = []
        for model in self.models:
            if use_tta:
                probs = tta_predict(model, batch, TTA_TRANSFORMS, self.config)
            else:
                with torch.no_grad():
                    probs = F.softmax(model(batch)["logits"], dim=-1)
            all_probs.append(probs)
        return torch.stack(all_probs).mean(0)

    def predict(self, batch: dict, use_tta: bool = True) -> torch.Tensor:
        return self.predict_proba(batch, use_tta).argmax(-1)


def calibrated_ensemble_predict(
    ensemble: FoldEnsemble,
    calibrators: List,
    batch: dict,
    use_tta: bool = True,
) -> torch.Tensor:
    """Ensemble with per-fold temperature calibration."""
    from src.tta import tta_predict, TTA_TRANSFORMS
    all_probs = []
    for model, calibrator in zip(ensemble.models, calibrators):
        if use_tta:
            probs = tta_predict(model, batch, TTA_TRANSFORMS, ensemble.config)
        else:
            with torch.no_grad():
                logits = model(batch)["logits"]
            logits = calibrator.scale(logits)
            probs = F.softmax(logits, dim=-1)
        all_probs.append(probs)
    return torch.stack(all_probs).mean(0)

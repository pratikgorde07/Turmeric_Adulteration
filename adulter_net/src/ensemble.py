from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_logger
from src.tta import TTA_TRANSFORMS, tta_predict

logger = get_logger(__name__)


class FoldEnsemble:
    """Ensemble of models from all CV folds for final inference."""

    def __init__(self, model_paths: List[str], config):
        from src.models.adulter_net import AdulterNet

        self.config = config
        self.models = []
        for path in model_paths:
            model = AdulterNet(config)
            checkpoint = torch.load(path, map_location=config.device)
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            model.to(config.device)
            self.models.append(model)
            logger.info(f"Loaded ensemble model: {path}")

        logger.info(f"FoldEnsemble ready with {len(self.models)} models")

    def predict_proba(
        self,
        batch: Dict,
        use_tta: bool = True,
        calibrators: Optional[List] = None,
    ) -> torch.Tensor:
        """Soft voting ensemble over all fold models.

        Args:
            batch: dict of tensors (already on device)
            use_tta: whether to apply TTA
            calibrators: optional list of TemperatureScaler per fold

        Returns:
            (B, n_classes) mean probability tensor
        """
        all_probs = []
        for i, model in enumerate(self.models):
            if use_tta:
                probs = tta_predict(model, batch, TTA_TRANSFORMS, self.config)
            else:
                with torch.no_grad():
                    out = model(batch)
                    logits = out["logits"]
                    if calibrators is not None and i < len(calibrators):
                        logits = calibrators[i].scale(logits)
                    probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        return torch.stack(all_probs).mean(0)

    def predict(self, batch: Dict, use_tta: bool = True) -> torch.Tensor:
        """Return predicted class indices (B,)."""
        return self.predict_proba(batch, use_tta).argmax(-1)

    def predict_with_uncertainty(
        self, batch: Dict, use_tta: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Return predictions with uncertainty estimate from fold disagreement."""
        all_probs = []
        for model in self.models:
            if use_tta:
                probs = tta_predict(model, batch, TTA_TRANSFORMS, self.config)
            else:
                with torch.no_grad():
                    out = model(batch)
                    probs = F.softmax(out["logits"], dim=-1)
            all_probs.append(probs)

        probs_stack = torch.stack(all_probs)  # (n_folds, B, n_classes)
        mean_probs = probs_stack.mean(0)
        std_probs = probs_stack.std(0)

        return {
            "proba": mean_probs,
            "pred": mean_probs.argmax(-1),
            "confidence": mean_probs.max(-1).values,
            "uncertainty_std": std_probs.max(-1).values,
        }


def calibrated_ensemble_predict(
    ensemble: FoldEnsemble,
    calibrators: List,
    batch: Dict,
    use_tta: bool = True,
) -> torch.Tensor:
    """Ensemble with per-fold temperature calibration."""
    all_probs = []
    for i, model in enumerate(ensemble.models):
        with torch.no_grad():
            out = model(batch)
            logits = out["logits"]
            if i < len(calibrators):
                logits = calibrators[i].scale(logits)
            if use_tta:
                probs = tta_predict(model, batch, TTA_TRANSFORMS, ensemble.config)
            else:
                probs = F.softmax(logits, dim=-1)
        all_probs.append(probs)

    return torch.stack(all_probs).mean(0)

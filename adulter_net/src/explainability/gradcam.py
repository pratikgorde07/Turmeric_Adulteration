from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class VisualFeatureGradCAM:
    """Grad-CAM on EfficientNet-B0 visual stream (or gradient-based feature importance)."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def generate_gradcam(self, batch: dict, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for raw images, or gradient importance for features."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        image_raw = batch.get("image_raw")
        has_raw = (image_raw is not None and
                   image_raw.shape[1:] == (3, 224, 224) and
                   not (image_raw == 0).all())

        if has_raw:
            return self._gradcam_image(batch, target_class)
        else:
            return self._gradient_feature_importance(batch, target_class)

    def _gradcam_image(self, batch: dict, target_class: Optional[int]) -> np.ndarray:
        """Spatial Grad-CAM on EfficientNet-B0 last conv block."""
        import torch
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            # Target layer: last conv block of EfficientNet
            if hasattr(self.model.visual_stream.base_model, "blocks"):
                target_layers = [self.model.visual_stream.base_model.blocks[-1]]
            else:
                target_layers = [list(self.model.visual_stream.base_model.children())[-3]]

            cam = GradCAM(model=self.model.visual_stream.base_model, target_layers=target_layers)
            images = batch["image_raw"]
            targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
            grayscale_cam = cam(input_tensor=images, targets=targets)
            return grayscale_cam  # (B, H, W)
        except Exception as e:
            logger.warning(f"pytorch_grad_cam failed: {e}, falling back to gradient importance")
            return self._gradient_feature_importance(batch, target_class)

    def _gradient_feature_importance(self, batch: dict, target_class: Optional[int]) -> np.ndarray:
        """Gradient of predicted logit w.r.t. image feature vector."""
        import torch
        x_feat = batch["image_feat"].requires_grad_(True)
        batch_grad = {**batch, "image_feat": x_feat,
                      "image_raw": torch.zeros(x_feat.shape[0], 3, 224, 224, device=self.config.device)}
        out = self.model(batch_grad)
        logits = out["logits"]
        if target_class is None:
            target_class = logits.argmax(-1)[0].item()
        score = logits[:, target_class].sum()
        score.backward()
        importance = x_feat.grad.abs().cpu().numpy()  # (B, n_features)
        return importance

    def save_gradcam_figure(self, importance: np.ndarray, sample_id: str, config) -> None:
        """Save gradient importance bar chart."""
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            top_n = min(30, importance.shape[-1])
            top_idx = np.argsort(-importance[0])[:top_n]
            ax.bar(range(top_n), importance[0, top_idx])
            ax.set_xlabel("Feature index (sorted by gradient magnitude)")
            ax.set_ylabel("|Gradient|")
            ax.set_title(f"Gradient Feature Importance — {sample_id}")
            plt.tight_layout()
            path = config.output_dir / "figures" / f"gradcam_feature_importance_{sample_id}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Grad-CAM figure saved: {path}")
        except Exception as e:
            logger.warning(f"Could not save Grad-CAM figure: {e}")

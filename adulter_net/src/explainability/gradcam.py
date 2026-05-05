"""Grad-CAM explainability for visual stream."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class VisualFeatureGradCAM:
    """Gradient-based saliency for image stream features or raw images."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def explain_feature_vector(self, batch: dict, target_class: int, sample_id: str) -> None:
        """Gradient of target logit w.r.t. input feature vector."""
        import torch
        import matplotlib.pyplot as plt

        self.model.eval()
        img_feat = batch["image_feat"].clone().requires_grad_(True)
        batch_copy = {**batch, "image_feat": img_feat}
        out = self.model(batch_copy)
        logit = out["logits"][0, target_class]
        logit.backward()
        grad = img_feat.grad[0].abs().cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(grad)), grad, color="steelblue", width=1.0)
        ax.set_xlabel("Feature Dimension")
        ax.set_ylabel("|Gradient|")
        ax.set_title(f"Feature Gradient Importance — {sample_id} (class {target_class})")
        out_path = (
            Path(self.config.output_dir) / "figures" / f"gradcam_{sample_id}.png"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Grad-CAM feature plot saved: {out_path}")

    def explain_batch(self, val_loader, n_samples_per_class: int = 3) -> None:
        import torch
        seen_classes = set()
        for batch in val_loader:
            for i in range(len(batch["label"])):
                cls = int(batch["label"][i])
                if cls in seen_classes:
                    continue
                sid = batch["sample_id"][i]
                single = {
                    k: v[i:i+1].to(self.config.device) if isinstance(v, torch.Tensor) else [v[i]]
                    for k, v in batch.items()
                }
                try:
                    self.explain_feature_vector(single, cls, sid)
                    seen_classes.add(cls)
                except Exception as e:
                    logger.warning(f"Grad-CAM failed for {sid}: {e}")
            if len(seen_classes) >= self.config.n_classes:
                break

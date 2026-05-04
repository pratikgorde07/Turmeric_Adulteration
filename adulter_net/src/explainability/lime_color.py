"""LIME explanations for colorimetric stream."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class ColorimetricLIME:
    def __init__(self, config, feature_names: List[str] = None):
        self.config = config
        self.feature_names = feature_names or []

    def explain_class(self, model, X_color: np.ndarray, y: np.ndarray,
                      class_idx: int, class_name: str) -> None:
        try:
            import lime.lime_tabular
            import torch
            import torch.nn.functional as F
            import matplotlib.pyplot as plt

            def predict_fn(x):
                model.eval()
                with torch.no_grad():
                    t = torch.tensor(x, dtype=torch.float32).to(self.config.device)
                    out = model.color_stream(t)
                    logits = model.classification_head(
                        model.cmt_fusion(
                            torch.zeros(len(x), model.visual_stream.embed_dim).to(self.config.device),
                            torch.zeros(len(x), model.spectral_stream.out_dim).to(self.config.device),
                            out,
                        )
                    )
                    return F.softmax(logits, -1).cpu().numpy()

            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_color,
                feature_names=self.feature_names,
                class_names=self.config.class_names,
                mode="classification",
            )
            # Find hardest correct prediction for this class
            class_mask = y == class_idx
            if not class_mask.any():
                return
            X_cls = X_color[class_mask]
            probs = predict_fn(X_cls)
            confidences = probs[:, class_idx]
            hardest_idx = np.argmin(confidences)

            exp = explainer.explain_instance(
                X_cls[hardest_idx], predict_fn, num_features=10, labels=(class_idx,)
            )
            fig = exp.as_pyplot_figure(label=class_idx)
            fig.set_size_inches(10, 6)
            out_path = Path(self.config.output_dir) / "figures" / f"lime_color_class{class_idx}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"LIME colorimetric plot saved: {out_path}")
        except Exception as e:
            logger.warning(f"LIME explanation for class {class_idx} failed: {e}")

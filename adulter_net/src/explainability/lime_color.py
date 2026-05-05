from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class ColorimetricLIME:
    """LIME explainer for colorimetric stream."""

    def __init__(self, config):
        self.config = config

    def explain_class(
        self,
        X_color: np.ndarray,
        y: np.ndarray,
        predict_fn,
        feature_names: List[str],
        class_idx: int,
        save_dir: Optional[Path] = None,
    ) -> None:
        """Generate LIME explanation for the hardest correct prediction in class_idx."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
            import matplotlib.pyplot as plt
        except ImportError as e:
            logger.warning(f"LIME unavailable: {e}")
            return

        save_dir = save_dir or self.config.output_dir / "figures"

        explainer = LimeTabularExplainer(
            X_color,
            feature_names=feature_names,
            class_names=self.config.class_names,
            mode="classification",
        )

        # Find hardest correct prediction in this class (lowest confidence)
        class_mask = y == class_idx
        if not class_mask.any():
            logger.warning(f"No samples of class {class_idx} found")
            return

        X_cls = X_color[class_mask]
        probs = predict_fn(X_cls)
        correct_mask = probs.argmax(-1) == class_idx
        if not correct_mask.any():
            idx = 0
        else:
            probs_correct = probs[correct_mask, class_idx]
            idx = np.where(correct_mask)[0][probs_correct.argmin()]

        sample = X_cls[idx]

        try:
            explanation = explainer.explain_instance(
                sample, predict_fn, num_features=len(feature_names), top_labels=1
            )
            fig = explanation.as_pyplot_figure(label=class_idx)
            path = save_dir / f"lime_color_class{class_idx}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"LIME colorimetric plot saved: {path}")
        except Exception as e:
            logger.warning(f"LIME explanation for class {class_idx} failed: {e}")

    def explain_all_classes(
        self, X_color: np.ndarray, y: np.ndarray, predict_fn, feature_names: List[str],
        save_dir: Optional[Path] = None,
    ) -> None:
        for cls_idx in range(self.config.n_classes):
            self.explain_class(X_color, y, predict_fn, feature_names, cls_idx, save_dir)

"""SHAP explanations for FTIR stream."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class FTIRShapExplainer:
    def __init__(self, config, wavenumber_labels: List[str] = None):
        self.config = config
        self.wavenumber_labels = wavenumber_labels or []
        self.surrogate = None
        self.explainer = None

    def fit(self, X_ftir: np.ndarray, y: np.ndarray) -> None:
        try:
            import xgboost as xgb
            import shap
            self.surrogate = xgb.XGBClassifier(
                n_estimators=200, max_depth=4,
                eval_metric="mlogloss", verbosity=0
            )
            self.surrogate.fit(X_ftir, y)
            self.explainer = shap.TreeExplainer(self.surrogate)
            logger.info("FTIR SHAP surrogate fitted")
        except Exception as e:
            logger.warning(f"FTIR SHAP fit failed: {e}")

    def plot_beeswarm(self, X_ftir: np.ndarray, save_path: str) -> None:
        try:
            import shap
            import matplotlib.pyplot as plt
            if self.explainer is None:
                return
            sv = self.explainer.shap_values(X_ftir)
            if isinstance(sv, list):
                sv = sv[0]
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                sv, X_ftir,
                feature_names=self.wavenumber_labels or [f"wn_{i}" for i in range(X_ftir.shape[1])],
                show=False,
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"FTIR SHAP beeswarm saved: {save_path}")
        except Exception as e:
            logger.warning(f"FTIR SHAP beeswarm failed: {e}")

    def generate_all_plots(self, X_ftir: np.ndarray) -> None:
        figures_dir = Path(self.config.output_dir) / "figures"
        self.plot_beeswarm(X_ftir, str(figures_dir / "shap_ftir_beeswarm.png"))

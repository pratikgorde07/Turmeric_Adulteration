from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)

# Known chemically significant wavenumber ranges (cm⁻¹)
CURCUMIN_CO_RANGE = (1600, 1650)  # ~1628 cm⁻¹ curcumin C=O stretch
STARCH_CO_RANGE = (980, 1020)     # ~1000 cm⁻¹ starch C-O stretch


class FTIRShapExplainer:
    """SHAP explainer for FTIR stream using XGBoost surrogate."""

    def __init__(self, config):
        self.config = config

    def explain(
        self,
        X_ftir_selected: np.ndarray,
        y: np.ndarray,
        wavenumber_labels: List[str],
        save_dir: Optional[Path] = None,
    ) -> None:
        """Train XGBoost surrogate, compute SHAP, generate plots."""
        try:
            import xgboost as xgb
            import shap as shap_lib
            import matplotlib.pyplot as plt
        except ImportError as e:
            logger.warning(f"SHAP FTIR explainer requires xgboost and shap: {e}")
            return

        save_dir = save_dir or self.config.output_dir / "figures"

        # Train surrogate
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42,
                                  eval_metric="mlogloss", verbosity=0)
        clf.fit(X_ftir_selected, y)

        explainer = shap_lib.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_ftir_selected)
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv).mean(0) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals).mean(0)

        # Log chemically significant wavenumbers if present
        self._log_key_wavenumbers(wavenumber_labels, mean_abs)

        # Summary beeswarm
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            top_n = min(20, len(wavenumber_labels))
            top_idx = np.argsort(-mean_abs)[:top_n]
            ax.barh(range(top_n), mean_abs[top_idx], color="steelblue")
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([wavenumber_labels[i] for i in top_idx], fontsize=8)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("FTIR Feature SHAP Importance (Surrogate XGBoost)")
            plt.tight_layout()
            path = save_dir / "shap_ftir_beeswarm.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"FTIR SHAP beeswarm saved: {path}")
        except Exception as e:
            logger.warning(f"Could not save SHAP beeswarm: {e}")

    def _log_key_wavenumbers(self, labels: List[str], mean_abs: np.ndarray) -> None:
        """Log chemically relevant wavenumbers if selected."""
        for i, label in enumerate(labels):
            try:
                wn = float(str(label).replace("wn_", "").replace("cm", "").strip())
                if CURCUMIN_CO_RANGE[0] <= wn <= CURCUMIN_CO_RANGE[1]:
                    logger.info(f"Key wavenumber at ~{wn:.0f} cm⁻¹ (curcumin C=O stretch): SHAP={mean_abs[i]:.4f}")
                if STARCH_CO_RANGE[0] <= wn <= STARCH_CO_RANGE[1]:
                    logger.info(f"Key wavenumber at ~{wn:.0f} cm⁻¹ (starch C-O stretch): SHAP={mean_abs[i]:.4f}")
            except (ValueError, TypeError):
                pass

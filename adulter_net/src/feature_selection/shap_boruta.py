from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class SHAPBorutaSelector:
    """Feature selector combining SHAP values with Boruta algorithm for image features."""

    def __init__(self, top_k: int = 200, max_iter: int = 150, random_state: int = 42):
        self.top_k = top_k
        self.max_iter = max_iter
        self.random_state = random_state
        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.shap_values_mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SHAPBorutaSelector":
        """Fit selector using SHAP + Boruta on training data.

        Falls back to top-K by SHAP importance if Boruta confirms < 50 features.
        """
        import lightgbm as lgb
        try:
            import shap as shap_lib
        except ImportError:
            logger.warning("shap not available, falling back to LightGBM importance")
            shap_lib = None

        n_features = X.shape[1]
        logger.info(f"SHAPBoruta: fitting on X={X.shape}, {n_features} features")

        # Step 1: Fit LightGBM to get SHAP values
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X, y)

        if shap_lib is not None:
            explainer = shap_lib.TreeExplainer(lgb_model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_vals], axis=0)
            else:
                mean_abs_shap = np.abs(shap_vals).mean(0)
        else:
            # Fallback to LightGBM feature importance
            mean_abs_shap = lgb_model.feature_importances_.astype(float)

        self.shap_values_mean_ = mean_abs_shap

        # Step 2: Boruta iterations
        rng = np.random.RandomState(self.random_state)
        confirmed = np.zeros(n_features, dtype=bool)
        rejected = np.zeros(n_features, dtype=bool)
        tentative = np.ones(n_features, dtype=bool)

        for iteration in range(self.max_iter):
            active = tentative & ~confirmed & ~rejected
            if not active.any():
                break

            # Create shadow features
            X_shadow = X[:, active].copy()
            for col in range(X_shadow.shape[1]):
                rng.shuffle(X_shadow[:, col])

            X_aug = np.hstack([X[:, active], X_shadow])
            y_iter = y.copy()

            lgb_iter = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state + iteration,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_iter.fit(X_aug, y_iter)
            importances = lgb_iter.feature_importances_

            n_active = active.sum()
            real_imp = importances[:n_active]
            shadow_imp = importances[n_active:]

            shadow_max = shadow_imp.max() if len(shadow_imp) > 0 else 0.0
            shadow_min = shadow_imp.min() if len(shadow_imp) > 0 else 0.0

            active_indices = np.where(active)[0]
            for li, gi in enumerate(active_indices):
                if real_imp[li] > shadow_max:
                    confirmed[gi] = True
                    tentative[gi] = False
                elif real_imp[li] < shadow_min:
                    rejected[gi] = True
                    tentative[gi] = False

            if iteration % 20 == 0:
                logger.info(
                    f"  Boruta iter {iteration}: confirmed={confirmed.sum()}, "
                    f"rejected={rejected.sum()}, tentative={tentative.sum()}"
                )

        # For remaining tentative features, use SHAP ranking
        all_confirmed = confirmed.copy()
        # Add tentative features sorted by SHAP as fallback
        tentative_sorted = np.argsort(-mean_abs_shap)
        tentative_sorted = [i for i in tentative_sorted if tentative[i]]
        all_confirmed[tentative_sorted[:max(0, 50 - int(confirmed.sum()))]] = True

        confirmed_indices = np.where(all_confirmed)[0]
        logger.info(
            f"Boruta: {confirmed.sum()} confirmed, adding {len(tentative_sorted[:50-int(confirmed.sum())])} tentative"
        )

        # Fallback: ensure at least 50 features
        if len(confirmed_indices) < 50:
            top_by_shap = np.argsort(-mean_abs_shap)[:50]
            confirmed_indices = top_by_shap
            logger.warning("Boruta confirmed < 50 features, using top-50 by SHAP")

        # Trim to top-K
        if len(confirmed_indices) > self.top_k:
            confirmed_indices = confirmed_indices[
                np.argsort(-mean_abs_shap[confirmed_indices])[:self.top_k]
            ]

        self.selected_indices_ = np.sort(confirmed_indices)
        self.feature_importances_ = mean_abs_shap

        logger.info(
            f"SHAPBoruta selected {len(self.selected_indices_)} features "
            f"(from {n_features})"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return self.selected_indices_

    def get_feature_importance_df(self):
        import pandas as pd
        assert self.shap_values_mean_ is not None, "Call fit() first"
        return pd.DataFrame({
            "feature_idx": np.arange(len(self.shap_values_mean_)),
            "mean_abs_shap": self.shap_values_mean_,
            "selected": np.isin(np.arange(len(self.shap_values_mean_)), self.selected_indices_),
        }).sort_values("mean_abs_shap", ascending=False)

    def plot_importance(self, save_path: str, top_n: int = 30) -> None:
        """Plot top-N feature importances."""
        try:
            import matplotlib.pyplot as plt
            df = self.get_feature_importance_df().head(top_n)
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ["green" if s else "red" for s in df["selected"]]
            ax.barh(range(len(df)), df["mean_abs_shap"].values, color=colors)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels([f"feat_{i}" for i in df["feature_idx"]])
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP-Boruta Feature Importance (top {top_n})")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Importance plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"Could not create importance plot: {e}")

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"SHAPBorutaSelector saved: {path}")

    @classmethod
    def load(cls, path: str) -> "SHAPBorutaSelector":
        obj = joblib.load(path)
        logger.info(f"SHAPBorutaSelector loaded: {path}")
        return obj

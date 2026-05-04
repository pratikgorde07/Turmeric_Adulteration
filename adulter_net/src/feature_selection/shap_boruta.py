"""SHAP-Boruta feature selector for image stream."""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SHAPBorutaSelector:
    """SHAP-augmented Boruta selector for high-dimensional image features."""

    def __init__(self, top_k: int = 200, n_estimators: int = 200, max_iter: int = 150,
                 random_state: int = 42):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.selected_indices_: Optional[List[int]] = None
        self.feature_importance_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SHAPBorutaSelector":
        try:
            import lightgbm as lgb
            import shap
        except ImportError as e:
            logger.warning(f"LightGBM/SHAP not available: {e}. Using RF fallback.")
            return self._fit_rf_fallback(X, y)

        n_features = X.shape[1]
        logger.info(f"SHAP-Boruta: fitting on {X.shape}, top_k={self.top_k}")

        # Fit initial LightGBM for SHAP values
        model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(0)

        self.feature_importance_ = mean_abs_shap

        # Boruta iterations
        confirmed = set()
        tentative = set(range(n_features))
        rejected = set()

        for iteration in range(min(self.max_iter, 50)):
            if not tentative:
                break
            active = list(tentative | confirmed)
            X_active = X[:, active]

            # Shadow features
            X_shadow = X_active.copy()
            for col in range(X_shadow.shape[1]):
                np.random.shuffle(X_shadow[:, col])

            X_combined = np.hstack([X_active, X_shadow])
            shadow_model = lgb.LGBMClassifier(
                n_estimators=50, learning_rate=0.1,
                random_state=self.random_state + iteration, n_jobs=-1, verbose=-1
            )
            shadow_model.fit(X_combined, y)
            importances = shadow_model.feature_importances_
            real_imp = importances[:len(active)]
            shadow_imp = importances[len(active):]

            threshold = shadow_imp.max()
            shadow_min = shadow_imp.min()

            newly_confirmed = set()
            newly_rejected = set()
            for local_i, global_i in enumerate(active):
                if global_i in tentative:
                    if real_imp[local_i] > threshold:
                        newly_confirmed.add(global_i)
                    elif real_imp[local_i] < shadow_min:
                        newly_rejected.add(global_i)

            confirmed |= newly_confirmed
            rejected |= newly_rejected
            tentative -= newly_confirmed | newly_rejected

            if iteration % 10 == 0:
                logger.info(
                    f"Boruta iter {iteration}: confirmed={len(confirmed)}, "
                    f"tentative={len(tentative)}, rejected={len(rejected)}"
                )

        # Merge tentative into confirmed (conservative)
        all_confirmed = list(confirmed | tentative)

        if len(all_confirmed) < 50:
            logger.warning(
                f"Boruta confirmed only {len(all_confirmed)} features. "
                f"Falling back to top-50 by SHAP."
            )
            all_confirmed = list(np.argsort(mean_abs_shap)[::-1][:50])

        # Trim to top_k
        if len(all_confirmed) > self.top_k:
            sorted_confirmed = sorted(all_confirmed, key=lambda i: mean_abs_shap[i], reverse=True)
            all_confirmed = sorted_confirmed[:self.top_k]

        self.selected_indices_ = sorted(all_confirmed)
        logger.info(f"SHAP-Boruta selected {len(self.selected_indices_)} features from {n_features}")
        return self

    def _fit_rf_fallback(self, X: np.ndarray, y: np.ndarray) -> "SHAPBorutaSelector":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        self.feature_importance_ = importances
        top_idx = np.argsort(importances)[::-1][:self.top_k]
        self.selected_indices_ = sorted(top_idx.tolist())
        logger.info(f"RF fallback selected {len(self.selected_indices_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return X[:, self.selected_indices_]

    def get_selected_indices(self) -> List[int]:
        assert self.selected_indices_ is not None, "Call fit() first"
        return self.selected_indices_

    def get_feature_importance_df(self):
        import pandas as pd
        assert self.feature_importance_ is not None, "Call fit() first"
        df = pd.DataFrame({
            "feature_index": range(len(self.feature_importance_)),
            "mean_abs_shap": self.feature_importance_,
            "selected": [i in set(self.selected_indices_) for i in range(len(self.feature_importance_))],
        }).sort_values("mean_abs_shap", ascending=False)
        return df

    def plot_importance(self, save_path: str, top_n: int = 30) -> None:
        try:
            import matplotlib.pyplot as plt
            df = self.get_feature_importance_df().head(top_n)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["green" if s else "gray" for s in df["selected"]]
            ax.barh(range(len(df)), df["mean_abs_shap"].values, color=colors)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels([f"feat_{i}" for i in df["feature_index"]])
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title(f"Top-{top_n} Image Features (green=selected)")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP importance plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"Could not plot importance: {e}")

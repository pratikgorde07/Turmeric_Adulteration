from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class mRMRSHAPSelector:
    """mRMR + SHAP elbow thresholding for colorimetric feature selection."""

    def __init__(
        self,
        n_mrmr: int = 20,
        elbow_threshold: float = 0.01,
        random_state: int = 42,
    ):
        self.n_mrmr = n_mrmr
        self.elbow_threshold = elbow_threshold
        self.random_state = random_state
        self.selected_feature_names_: Optional[List[str]] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.shap_values_mean_: Optional[np.ndarray] = None
        self.mrmr_features_: Optional[List[str]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        col_names: List[str],
    ) -> "mRMRSHAPSelector":
        """Fit mRMR + SHAP selector.

        X: (N, n_features)
        y: (N,) integer class indices
        col_names: list of column names length n_features
        """
        assert len(col_names) == X.shape[1], (
            f"col_names length {len(col_names)} != X.shape[1] {X.shape[1]}"
        )
        logger.info(f"mRMR+SHAP: fitting on X={X.shape}, {len(col_names)} features")

        # Step 1: mRMR
        try:
            from mrmr import mrmr_classif
            df_X = pd.DataFrame(X, columns=col_names)
            df_y = pd.Series(y)
            n_select = min(self.n_mrmr, len(col_names))
            mrmr_features = mrmr_classif(X=df_X, y=df_y, K=n_select)
            logger.info(f"mRMR selected {len(mrmr_features)} features: {mrmr_features[:5]}...")
        except Exception as e:
            logger.warning(f"mRMR failed ({e}), using correlation-based fallback")
            # Fallback: select by mutual information
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X, y, random_state=self.random_state)
            top_k = min(self.n_mrmr, len(col_names))
            mrmr_features = [col_names[i] for i in np.argsort(-mi)[:top_k]]

        self.mrmr_features_ = mrmr_features

        # Get mRMR subset
        mrmr_idx = [col_names.index(f) for f in mrmr_features if f in col_names]
        X_mrmr = X[:, mrmr_idx]
        mrmr_col_names = [col_names[i] for i in mrmr_idx]

        # Step 2: SHAP on mRMR subset
        try:
            import xgboost as xgb
            import shap as shap_lib
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                random_state=self.random_state,
                eval_metric="mlogloss",
                verbosity=0,
                use_label_encoder=False,
            )
            xgb_model.fit(X_mrmr, y)
            explainer = shap_lib.TreeExplainer(xgb_model)
            shap_vals = explainer.shap_values(X_mrmr)
            if isinstance(shap_vals, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_vals], axis=0)
            else:
                mean_abs_shap = np.abs(shap_vals).mean(0)
        except Exception as e:
            logger.warning(f"SHAP on mRMR subset failed ({e}), using RF importance fallback")
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state, n_jobs=-1
            )
            rf.fit(X_mrmr, y)
            mean_abs_shap = rf.feature_importances_

        self.shap_values_mean_ = mean_abs_shap

        # Step 3: Elbow thresholding
        sorted_idx = np.argsort(-mean_abs_shap)
        sorted_vals = mean_abs_shap[sorted_idx]
        sorted_names = [mrmr_col_names[i] for i in sorted_idx]

        # Find elbow using second derivative
        if len(sorted_vals) >= 3:
            second_deriv = np.diff(sorted_vals, n=2)
            elbow_idx = int(np.argmax(np.abs(second_deriv))) + 1
        else:
            elbow_idx = len(sorted_vals) - 1

        # Retain features up to and including elbow
        selected_names = sorted_names[: elbow_idx + 1]

        # Fallback: keep at least 5 features
        if len(selected_names) < 5:
            selected_names = sorted_names[:min(10, len(sorted_names))]
            logger.warning(
                f"Elbow removed too many features, using top-{len(selected_names)} by SHAP"
            )

        # Step 4: Ensure L*, a*, b* coverage
        cie_axes = {"L": None, "A": None, "B": None}
        for name in selected_names:
            n_upper = name.upper()
            if cie_axes["L"] is None and any(k in n_upper for k in ["L*", "L_", " L ", "LAB_L"]):
                cie_axes["L"] = name
            if cie_axes["A"] is None and any(k in n_upper for k in ["A*", "A_", " A ", "LAB_A"]):
                cie_axes["A"] = name
            if cie_axes["B"] is None and any(k in n_upper for k in ["B*", "B_", " B ", "LAB_B"]):
                cie_axes["B"] = name

        # Also check first-letter matches for L, A, B columns
        for axis_key, col_prefix in [("L", "L"), ("A", "A"), ("B", "B")]:
            if cie_axes[axis_key] is None:
                candidates = [
                    (mrmr_col_names[i], mean_abs_shap[i])
                    for i in range(len(mrmr_col_names))
                    if mrmr_col_names[i].upper().startswith(col_prefix)
                ]
                if candidates:
                    best = max(candidates, key=lambda t: t[1])
                    if best[0] not in selected_names:
                        selected_names.append(best[0])
                        logger.info(
                            f"Added {axis_key}* axis feature '{best[0]}' for CIE coverage"
                        )
                    cie_axes[axis_key] = best[0]

        logger.info(f"CIE axis coverage: L*={cie_axes['L'] is not None}, "
                    f"a*={cie_axes['A'] is not None}, b*={cie_axes['B'] is not None}")

        self.selected_feature_names_ = selected_names

        # Map back to original col_names indices
        self.selected_indices_ = np.array([
            col_names.index(name)
            for name in selected_names
            if name in col_names
        ])

        logger.info(
            f"mRMR+SHAP: selected {len(self.selected_indices_)} features: {selected_names}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return X[:, self.selected_indices_]

    def get_feature_names(self) -> List[str]:
        assert self.selected_feature_names_ is not None, "Call fit() first"
        return self.selected_feature_names_

    def plot_shap_beeswarm(self, save_path: str, X: np.ndarray, col_names: List[str]) -> None:
        """Plot SHAP beeswarm for selected features."""
        try:
            import matplotlib.pyplot as plt
            if self.shap_values_mean_ is None:
                return
            names = [col_names[i] for i in range(len(self.shap_values_mean_))]
            vals = self.shap_values_mean_
            sorted_idx = np.argsort(-vals)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                range(len(sorted_idx)),
                vals[sorted_idx],
                color=["green" if names[i] in (self.selected_feature_names_ or []) else "gray"
                       for i in sorted_idx],
            )
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([names[i] for i in sorted_idx], fontsize=8)
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Colorimetric Feature SHAP Importance")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create SHAP beeswarm: {e}")

    def plot_mrmr_relevance(self, save_path: str) -> None:
        """Plot mRMR selection ranking."""
        try:
            import matplotlib.pyplot as plt
            if self.mrmr_features_ is None:
                return
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(range(len(self.mrmr_features_)), range(len(self.mrmr_features_), 0, -1))
            ax.set_yticks(range(len(self.mrmr_features_)))
            ax.set_yticklabels(self.mrmr_features_, fontsize=8)
            ax.set_xlabel("mRMR rank")
            ax.set_title("mRMR Feature Relevance Ranking")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create mRMR relevance plot: {e}")

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "mRMRSHAPSelector":
        return joblib.load(path)

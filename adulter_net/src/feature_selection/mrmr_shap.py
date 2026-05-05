"""mRMR + SHAP elbow selector for colorimetric stream."""
from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class mRMRSHAPSelector:
    """mRMR followed by SHAP elbow thresholding for colorimetric features."""

    def __init__(self, n_features: int = 20, elbow_threshold: float = 0.01,
                 random_state: int = 42):
        self.n_features = n_features
        self.elbow_threshold = elbow_threshold
        self.random_state = random_state
        self.selected_feature_names_: Optional[List[str]] = None
        self.selected_indices_: Optional[List[int]] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.all_col_names_: Optional[List[str]] = None

    def fit(self, X: np.ndarray, y: np.ndarray, col_names: List[str]) -> "mRMRSHAPSelector":
        self.all_col_names_ = col_names

        # Step 1: mRMR
        mrmr_features = self._run_mrmr(X, y, col_names)
        logger.info(f"mRMR selected {len(mrmr_features)} features: {mrmr_features}")

        if not mrmr_features:
            mrmr_features = col_names[:min(self.n_features, len(col_names))]

        mrmr_idx = [col_names.index(f) for f in mrmr_features if f in col_names]
        X_mrmr = X[:, mrmr_idx]
        mrmr_col_names = [col_names[i] for i in mrmr_idx]

        # Step 2: SHAP on mRMR subset
        shap_scores = self._compute_shap(X_mrmr, y, mrmr_col_names)

        # Step 3: Elbow thresholding
        sorted_order = np.argsort(shap_scores)[::-1]
        sorted_shap = shap_scores[sorted_order]

        elbow_idx = self._find_elbow(sorted_shap)
        retained_local = sorted_order[:max(elbow_idx + 1, 5)]

        # Step 4: Ensure L*, a*, b* are present
        retained_names = [mrmr_col_names[i] for i in retained_local]
        retained_names = self._ensure_cie_coverage(
            retained_names, mrmr_col_names, shap_scores, mrmr_idx
        )

        # Map back to global indices — include all original col_names as fallback
        name_to_global = {name: mrmr_idx[i] for i, name in enumerate(mrmr_col_names)}
        # Extend with any column added from all_col_names_ by _ensure_cie_coverage
        for i, name in enumerate(col_names):
            if name not in name_to_global:
                name_to_global[name] = i
        self.selected_indices_ = sorted([name_to_global[n] for n in retained_names if n in name_to_global])
        self.selected_feature_names_ = [col_names[i] for i in self.selected_indices_]
        logger.info(
            f"mRMR+SHAP selected {len(self.selected_feature_names_)} features: "
            f"{self.selected_feature_names_}"
        )
        return self

    def _run_mrmr(self, X, y, col_names):
        try:
            from mrmr import mrmr_classif
            df_X = pd.DataFrame(X, columns=col_names)
            s_y = pd.Series(y)
            selected = mrmr_classif(X=df_X, y=s_y, K=min(self.n_features, len(col_names)))
            return list(selected)
        except Exception as e:
            logger.warning(f"mRMR failed ({e}), falling back to correlation-based selection")
            return self._correlation_fallback(X, y, col_names)

    def _correlation_fallback(self, X, y, col_names):
        corrs = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
        corrs = np.nan_to_num(corrs)
        top = np.argsort(corrs)[::-1][:self.n_features]
        return [col_names[i] for i in top]

    def _compute_shap(self, X_mrmr, y, mrmr_col_names):
        try:
            import xgboost as xgb
            import shap
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, random_state=self.random_state,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0
            )
            model.fit(X_mrmr, y)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_mrmr)
            if isinstance(sv, list):
                scores = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
            else:
                scores = np.abs(sv).mean(0)
            self.shap_values_ = scores
            return scores
        except Exception as e:
            logger.warning(f"SHAP computation failed ({e}), using feature variance")
            scores = X_mrmr.var(axis=0)
            self.shap_values_ = scores
            return scores

    def _find_elbow(self, sorted_shap: np.ndarray) -> int:
        if len(sorted_shap) <= 2:
            return len(sorted_shap) - 1
        second_deriv = np.diff(np.diff(sorted_shap))
        if len(second_deriv) == 0:
            return len(sorted_shap) - 1
        elbow = int(np.argmax(np.abs(second_deriv))) + 1
        return max(elbow, 0)

    def _ensure_cie_coverage(self, retained_names, all_mrmr_names, shap_scores, mrmr_idx):
        retained = list(retained_names)
        cie_axes = {"l": False, "a": False, "b": False}

        def classify_col(name):
            nl = name.lower().strip()
            if "l*" in nl or nl == "l" or nl.startswith("l_") or nl.startswith("l "):
                return "l"
            if "a*" in nl or nl == "a" or nl.startswith("a_") or nl.startswith("a "):
                return "a"
            if "b*" in nl or nl == "b" or nl.startswith("b_") or nl.startswith("b "):
                return "b"
            return None

        for n in retained:
            ax = classify_col(n)
            if ax:
                cie_axes[ax] = True

        for axis, present in cie_axes.items():
            if not present:
                # Search mRMR subset first, then fall back to all original columns
                candidates = list(all_mrmr_names)
                if self.all_col_names_:
                    candidates = candidates + [c for c in self.all_col_names_ if c not in candidates]
                for cname in candidates:
                    if classify_col(cname) == axis and cname not in retained:
                        retained.append(cname)
                        logger.info(f"Added {cname} to ensure {axis}* CIE coverage")
                        break

        return retained

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return X[:, self.selected_indices_]

    def get_feature_names(self) -> List[str]:
        assert self.selected_feature_names_ is not None, "Call fit() first"
        return self.selected_feature_names_

    def plot_shap_beeswarm(self, save_path: str) -> None:
        try:
            import matplotlib.pyplot as plt
            if self.shap_values_ is None:
                return
            names = self.all_col_names_[:len(self.shap_values_)]
            sorted_idx = np.argsort(self.shap_values_)[::-1][:20]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(sorted_idx)), self.shap_values_[sorted_idx])
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([names[i] if i < len(names) else str(i) for i in sorted_idx])
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Colorimetric Feature SHAP Importance")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot SHAP beeswarm: {e}")

    def plot_mrmr_relevance(self, save_path: str) -> None:
        logger.info(f"mRMR relevance plot not implemented (placeholder): {save_path}")

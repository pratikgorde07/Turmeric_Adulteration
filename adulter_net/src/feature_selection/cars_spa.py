"""CARS + SPA feature selectors for FTIR stream."""
from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CARSSelector:
    """Competitive Adaptive Reweighted Sampling for FTIR wavenumber selection."""

    def __init__(self, n_iterations: int = 80, random_state: int = 42):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.selected_indices_: Optional[List[int]] = None
        self.rmsecv_history_: List[float] = []
        self.n_selected_history_: List[int] = []
        self.best_rmsecv_: float = float("inf")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARSSelector":
        try:
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.model_selection import cross_val_score
        except ImportError:
            return self._fit_fallback(X, y)

        np.random.seed(self.random_state)
        n_samples, n_wavenumbers = X.shape
        logger.info(f"CARS: fitting on {X.shape}, n_iterations={self.n_iterations}")

        active_idx = list(range(n_wavenumbers))
        best_indices = active_idx.copy()
        best_rmsecv = float("inf")

        rmsecv_list = []
        n_selected_list = []

        for k in range(1, self.n_iterations + 1):
            n_active = len(active_idx)
            if n_active <= 2:
                break

            # Exponentially decreasing sampling ratio
            ratio = (2.0 / n_active) * np.exp(
                -np.log(2.0 / n_active) / self.n_iterations * k
            )
            ratio = np.clip(ratio, 2.0 / n_active, 1.0)
            n_select = max(2, int(np.ceil(ratio * n_active)))

            # Bootstrap 70%
            boot_idx = np.random.choice(n_samples, int(0.7 * n_samples), replace=False)
            X_boot = X[boot_idx][:, active_idx]
            y_boot = y[boot_idx]

            n_comp = min(10, n_select, len(boot_idx) - 1)
            if n_comp < 1:
                n_comp = 1
            try:
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_boot, y_boot.astype(float))
                coefs = np.abs(pls.coef_.ravel())
            except Exception:
                break

            # Normalise weights
            coef_sum = coefs.sum()
            if coef_sum < 1e-12:
                break
            weights = coefs / coef_sum

            # Select top wavenumbers
            top_local = np.argsort(weights)[::-1][:n_select]
            active_idx = [active_idx[i] for i in top_local]

            # 5-fold RMSECV
            X_active = X[:, active_idx]
            n_comp_cv = min(10, len(active_idx), n_samples // 5 - 1)
            if n_comp_cv < 1:
                n_comp_cv = 1
            try:
                pls_cv = PLSRegression(n_components=n_comp_cv)
                scores = cross_val_score(
                    pls_cv, X_active, y.astype(float),
                    cv=5, scoring="neg_root_mean_squared_error"
                )
                rmsecv = float(-scores.mean())
            except Exception:
                rmsecv = float("inf")

            rmsecv_list.append(rmsecv)
            n_selected_list.append(len(active_idx))

            if rmsecv < best_rmsecv:
                best_rmsecv = rmsecv
                best_indices = active_idx.copy()

            logger.debug(f"CARS iter {k}: n_active={len(active_idx)}, RMSECV={rmsecv:.4f}")

        self.rmsecv_history_ = rmsecv_list
        self.n_selected_history_ = n_selected_list
        self.best_rmsecv_ = best_rmsecv
        self.selected_indices_ = sorted(best_indices)
        logger.info(
            f"CARS selected {len(self.selected_indices_)} wavenumbers, "
            f"best RMSECV={best_rmsecv:.4f}"
        )
        return self

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> "CARSSelector":
        """Fallback: select top-40 features by variance."""
        variances = X.var(axis=0)
        top_idx = np.argsort(variances)[::-1][:40]
        self.selected_indices_ = sorted(top_idx.tolist())
        self.best_rmsecv_ = float("nan")
        logger.warning(f"CARS fallback: selected {len(self.selected_indices_)} by variance")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return X[:, self.selected_indices_]

    def get_selected_wavenumbers(self) -> List[int]:
        assert self.selected_indices_ is not None
        return self.selected_indices_

    def get_wavenumber_labels(self, all_col_names: List[str]) -> List[str]:
        return [all_col_names[i] for i in self.selected_indices_]

    def plot_cars_path(self, save_path: str) -> None:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(self.rmsecv_history_, marker="o", markersize=3)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("RMSECV")
            ax1.set_title("CARS: RMSECV vs Iteration")
            ax2.plot(self.n_selected_history_, marker="o", markersize=3, color="orange")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("# Selected Wavenumbers")
            ax2.set_title("CARS: Selected Wavenumbers vs Iteration")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"CARS path plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"Could not plot CARS path: {e}")


class SPASelector:
    """Successive Projections Algorithm — applied after CARS."""

    def __init__(self, n_components: int = 40):
        self.n_components = n_components
        self.selected_local_indices_: Optional[List[int]] = None

    def fit(self, X_cars_selected: np.ndarray) -> "SPASelector":
        n_samples, n_feat = X_cars_selected.shape
        n_select = min(self.n_components, n_feat)
        logger.info(f"SPA: fitting on {X_cars_selected.shape}, n_components={n_select}")

        selected = [0]
        X_work = X_cars_selected.copy().astype(float)

        # Orthogonal projector residuals
        residuals = X_work.copy()

        for _ in range(n_select - 1):
            best_j = None
            best_norm = -1.0
            for j in range(n_feat):
                if j in selected:
                    continue
                x_j = residuals[:, j]
                norm = float(np.dot(x_j, x_j))
                if norm > best_norm:
                    best_norm = norm
                    best_j = j
            if best_j is None:
                break
            selected.append(best_j)

            # Project out the selected column
            x_sel = residuals[:, best_j].copy()
            norm_sq = np.dot(x_sel, x_sel)
            if norm_sq > 1e-12:
                for j in range(n_feat):
                    if j in selected:
                        continue
                    proj = np.dot(residuals[:, j], x_sel) / norm_sq
                    residuals[:, j] -= proj * x_sel

        self.selected_local_indices_ = sorted(selected)
        logger.info(f"SPA selected {len(self.selected_local_indices_)} wavenumbers")
        return self

    def transform(self, X_cars_selected: np.ndarray) -> np.ndarray:
        assert self.selected_local_indices_ is not None, "Call fit() first"
        return X_cars_selected[:, self.selected_local_indices_]

    def get_selected_local_indices(self) -> List[int]:
        assert self.selected_local_indices_ is not None
        return self.selected_local_indices_

    def get_selected_global_indices(self, cars_selector: CARSSelector) -> List[int]:
        cars_idx = cars_selector.get_selected_wavenumbers()
        return sorted([cars_idx[i] for i in self.selected_local_indices_])

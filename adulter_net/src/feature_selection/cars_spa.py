from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class CARSSelector:
    """Competitive Adaptive Reweighted Sampling (CARS) for FTIR wavenumber selection."""

    def __init__(self, n_iterations: int = 80, n_pls_components: int = 10, random_state: int = 42):
        self.n_iterations = n_iterations
        self.n_pls_components = n_pls_components
        self.random_state = random_state
        self.selected_indices_: Optional[np.ndarray] = None
        self.rmsecv_history_: List[float] = []
        self.n_selected_history_: List[int] = []
        self.best_iteration_: int = 0

    def _rmsecv(self, X: np.ndarray, y: np.ndarray, n_comp: int, n_cv: int = 5) -> float:
        """5-fold RMSECV for PLS regression."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_cv, shuffle=True, random_state=self.random_state)
        errors = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_v = X[train_idx], X[val_idx]
            y_tr, y_v = y[train_idx], y[val_idx]
            try:
                pls = PLSRegression(n_components=min(n_comp, X_tr.shape[1], X_tr.shape[0] - 1))
                pls.fit(X_tr, y_tr)
                pred = pls.predict(X_v).ravel()
                errors.extend((pred - y_v.ravel()) ** 2)
            except Exception:
                errors.append(1e6)
        return float(np.sqrt(np.mean(errors)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARSSelector":
        """Fit CARS on training FTIR data.

        X: (N, n_wavenumbers) — already SNV+MinMax scaled
        y: (N,) regression target (raw adulteration %)
        """
        from sklearn.cross_decomposition import PLSRegression

        rng = np.random.RandomState(self.random_state)
        n_samples, n_wn = X.shape
        active_indices = np.arange(n_wn)
        bootstrap_frac = 0.70

        best_rmsecv = np.inf
        best_indices = active_indices.copy()
        self.rmsecv_history_ = []
        self.n_selected_history_ = []

        logger.info(f"CARS: starting with {n_wn} wavenumbers, {self.n_iterations} iterations")

        for k in range(1, self.n_iterations + 1):
            n_active = len(active_indices)
            if n_active < 2:
                logger.info(f"CARS: stopping at iter {k}, only {n_active} wavenumbers left")
                break

            # Sampling ratio (exponentially decreasing)
            r_k = (2 / n_active) * np.exp(
                -np.log(2 / n_active) / self.n_iterations * k
            )
            r_k = float(np.clip(r_k, 2 / n_active, 1.0))

            # Bootstrap sample
            n_boot = max(int(bootstrap_frac * n_samples), 5)
            boot_idx = rng.choice(n_samples, n_boot, replace=False)
            X_boot = X[boot_idx][:, active_indices]
            y_boot = y[boot_idx]

            # Fit PLS on active wavenumbers
            n_comp = min(self.n_pls_components, n_active, n_boot - 1)
            try:
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_boot, y_boot)
                coefs = np.abs(pls.coef_.ravel()) if pls.coef_.ndim > 1 else np.abs(pls.coef_)
                if len(coefs) != n_active:
                    # PLS may return different shape depending on version
                    coefs = np.abs(pls.x_weights_[:, 0])
                    if len(coefs) != n_active:
                        coefs = np.ones(n_active)
            except Exception as e:
                logger.warning(f"CARS iter {k}: PLS failed ({e}), using uniform weights")
                coefs = np.ones(n_active)

            # Normalise weights
            coef_sum = coefs.sum()
            if coef_sum < 1e-10:
                coefs = np.ones(n_active) / n_active
            else:
                coefs = coefs / coef_sum

            # Competitive selection: keep top r_k fraction
            n_keep = max(2, int(round(r_k * n_active)))
            top_local = np.argsort(-coefs)[:n_keep]
            active_indices = active_indices[top_local]

            # Compute RMSECV on full training set with active wavenumbers
            n_comp_cv = min(self.n_pls_components, len(active_indices), n_samples - 1)
            rmsecv = self._rmsecv(X[:, active_indices], y, n_comp_cv)
            self.rmsecv_history_.append(rmsecv)
            self.n_selected_history_.append(len(active_indices))

            if rmsecv < best_rmsecv:
                best_rmsecv = rmsecv
                best_indices = active_indices.copy()
                self.best_iteration_ = k

            if k % 10 == 0:
                logger.info(
                    f"  CARS iter {k}: n_active={len(active_indices)}, "
                    f"RMSECV={rmsecv:.4f} (best={best_rmsecv:.4f} at iter {self.best_iteration_})"
                )

        self.selected_indices_ = np.sort(best_indices)
        logger.info(
            f"CARS: selected {len(self.selected_indices_)} wavenumbers "
            f"(best RMSECV={best_rmsecv:.4f} at iter {self.best_iteration_})"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return X[:, self.selected_indices_]

    def get_selected_wavenumbers(self) -> np.ndarray:
        assert self.selected_indices_ is not None, "Call fit() first"
        return self.selected_indices_

    def get_wavenumber_labels(self, all_col_names: List[str]) -> List[str]:
        """Return column names for selected wavenumbers."""
        assert self.selected_indices_ is not None
        return [all_col_names[i] for i in self.selected_indices_]

    def plot_cars_path(self, save_path: str) -> None:
        """Plot RMSECV and n_selected vs iteration."""
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            iters = list(range(1, len(self.rmsecv_history_) + 1))
            ax1.plot(iters, self.rmsecv_history_, "b-o", markersize=3)
            ax1.axvline(self.best_iteration_, color="r", linestyle="--", label=f"Best iter={self.best_iteration_}")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("RMSECV")
            ax1.set_title("CARS: RMSECV vs Iteration")
            ax1.legend()
            ax2.plot(iters, self.n_selected_history_, "g-o", markersize=3)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("N selected wavenumbers")
            ax2.set_title("CARS: N Selected vs Iteration")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"CARS path plot saved: {save_path}")
        except Exception as e:
            logger.warning(f"Could not create CARS plot: {e}")

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "CARSSelector":
        return joblib.load(path)


class SPASelector:
    """Successive Projections Algorithm (SPA) applied after CARS."""

    def __init__(self, n_components: int = 40):
        self.n_components = n_components
        self.selected_local_indices_: Optional[np.ndarray] = None

    def fit(self, X_cars_selected: np.ndarray) -> "SPASelector":
        """Fit SPA on CARS-selected wavenumber matrix.

        X_cars_selected: (N, n_cars) — columns are CARS-selected wavenumbers
        """
        n_samples, n_cols = X_cars_selected.shape
        n_select = min(self.n_components, n_cols)

        # Start with first wavenumber
        selected = [0]
        X_work = X_cars_selected.T.copy()  # (n_cols, n_samples)

        for k in range(1, n_select):
            residuals = []
            selected_set = set(selected)
            # Project each candidate onto the orthogonal complement of selected
            proj_basis = X_work[selected]  # (k, n_samples)
            # QR decomposition for orthogonal projection
            Q, _ = np.linalg.qr(proj_basis.T)  # Q: (n_samples, k)
            proj_matrix = Q @ Q.T  # (n_samples, n_samples)

            for j in range(n_cols):
                if j in selected_set:
                    continue
                x_j = X_work[j]  # (n_samples,)
                residual = x_j - proj_matrix @ x_j
                residuals.append((np.linalg.norm(residual), j))

            if not residuals:
                break

            _, best_j = max(residuals, key=lambda t: t[0])
            selected.append(best_j)

        self.selected_local_indices_ = np.array(sorted(selected[:n_select]))
        logger.info(
            f"SPA: selected {len(self.selected_local_indices_)} wavenumbers "
            f"(from {n_cols} CARS-selected)"
        )
        return self

    def transform(self, X_cars_selected: np.ndarray) -> np.ndarray:
        assert self.selected_local_indices_ is not None, "Call fit() first"
        return X_cars_selected[:, self.selected_local_indices_]

    def get_selected_local_indices(self) -> np.ndarray:
        assert self.selected_local_indices_ is not None, "Call fit() first"
        return self.selected_local_indices_

    def get_selected_global_indices(self, cars_selector: CARSSelector) -> np.ndarray:
        """Return global wavenumber indices (in original FTIR feature space)."""
        cars_indices = cars_selector.get_selected_wavenumbers()
        return cars_indices[self.selected_local_indices_]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SPASelector":
        return joblib.load(path)

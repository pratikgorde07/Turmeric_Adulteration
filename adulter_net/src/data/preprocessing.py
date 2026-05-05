from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


# ── Image features ──────────────────────────────────────────────────────────

def fit_image_scaler(X_train_img: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on image feature training data."""
    scaler = StandardScaler()
    scaler.fit(X_train_img)
    logger.info(f"Image StandardScaler fitted on shape {X_train_img.shape}")
    return scaler


def apply_image_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply fitted StandardScaler to image features."""
    out = scaler.transform(X)
    assert not np.isnan(out).any(), "NaN detected in scaled image features"
    return out


# ── FTIR features ────────────────────────────────────────────────────────────

def snv_transform(X_ftir: np.ndarray) -> np.ndarray:
    """Standard Normal Variate row-wise transform (no leakage — sample-level)."""
    row_means = X_ftir.mean(axis=1, keepdims=True)
    row_stds = X_ftir.std(axis=1, keepdims=True)
    # Avoid division by zero
    row_stds = np.where(row_stds < 1e-10, 1e-10, row_stds)
    return (X_ftir - row_means) / row_stds


def fit_ftir_scaler(X_train_ftir_snv: np.ndarray) -> MinMaxScaler:
    """Fit MinMaxScaler on SNV-transformed training FTIR data."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_ftir_snv)
    logger.info(f"FTIR MinMaxScaler fitted on shape {X_train_ftir_snv.shape}")
    return scaler


def apply_ftir_pipeline(X: np.ndarray, minmax_scaler: MinMaxScaler) -> np.ndarray:
    """Apply SNV then MinMaxScaler transform."""
    X_snv = snv_transform(X)
    out = minmax_scaler.transform(X_snv)
    # Clamp to [0, 1] for numerical safety
    out = np.clip(out, 0.0, 1.0)
    assert (out >= 0).all() and (out <= 1).all(), "FTIR output out of [0,1] after pipeline"
    return out


# ── Colorimetric features ────────────────────────────────────────────────────

def fit_color_scaler(X_train_color: np.ndarray) -> RobustScaler:
    """Fit RobustScaler on colorimetric training data."""
    scaler = RobustScaler()
    scaler.fit(X_train_color)
    logger.info(f"Color RobustScaler fitted on shape {X_train_color.shape}")
    return scaler


def apply_color_scaler(X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    """Transform colorimetric features with fitted RobustScaler."""
    return scaler.transform(X)


# ── Scaler persistence ───────────────────────────────────────────────────────

def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved: {path}")


def load_scaler(path: Path):
    scaler = joblib.load(path)
    logger.info(f"Scaler loaded: {path}")
    return scaler


def save_all_scalers(
    scaler_img, scaler_ftir, scaler_color, config, fold_idx: int
) -> None:
    """Save all three scalers for a given fold."""
    base = config.processed_dir / "scalers"
    save_scaler(scaler_img, base / f"scaler_image_fold{fold_idx}.pkl")
    save_scaler(scaler_ftir, base / f"scaler_ftir_minmax_fold{fold_idx}.pkl")
    save_scaler(scaler_color, base / f"scaler_color_fold{fold_idx}.pkl")


def load_all_scalers(config, fold_idx: int):
    """Load all three scalers for a given fold."""
    base = config.processed_dir / "scalers"
    scaler_img = load_scaler(base / f"scaler_image_fold{fold_idx}.pkl")
    scaler_ftir = load_scaler(base / f"scaler_ftir_minmax_fold{fold_idx}.pkl")
    scaler_color = load_scaler(base / f"scaler_color_fold{fold_idx}.pkl")
    return scaler_img, scaler_ftir, scaler_color


def check_distribution_shift(
    X_train: np.ndarray,
    X_val: np.ndarray,
    modality: str,
    n_sigma: float = 3.0,
) -> None:
    """Warn if val values fall outside [min_train - n_sigma*std, max_train + n_sigma*std]."""
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    train_std = X_train.std(axis=0)

    lower = train_min - n_sigma * train_std
    upper = train_max + n_sigma * train_std

    violations = ((X_val < lower) | (X_val > upper)).any(axis=0).sum()
    if violations > 0:
        logger.warning(
            f"[{modality}] Distribution shift: {violations} feature(s) in val "
            f"have values outside [{n_sigma}σ] of training range."
        )


def check_no_all_zeros(X: np.ndarray, split_name: str, modality: str) -> None:
    """Assert no column is all-zeros."""
    all_zero_cols = (X == 0).all(axis=0).sum()
    if all_zero_cols > 0:
        logger.warning(
            f"[{modality}] {split_name}: {all_zero_cols} column(s) are all-zeros after selection."
        )

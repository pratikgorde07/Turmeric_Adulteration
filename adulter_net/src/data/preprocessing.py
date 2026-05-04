"""Per-modality normalisation utilities."""
from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


# === Image features ===

def fit_image_scaler(X_train_img: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train_img)
    logger.info(f"Image StandardScaler fitted on {X_train_img.shape}")
    return scaler


def apply_image_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    out = scaler.transform(X)
    assert not np.isnan(out).any(), "NaN detected after image scaler transform"
    return out


# === FTIR ===

def snv_transform(X_ftir: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: row-wise mean subtraction and std division."""
    mean = X_ftir.mean(axis=1, keepdims=True)
    std = X_ftir.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1e-8, std)
    return (X_ftir - mean) / std


def fit_ftir_scaler(X_train_ftir_snv: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_ftir_snv)
    logger.info(f"FTIR MinMaxScaler fitted on SNV-transformed data {X_train_ftir_snv.shape}")
    return scaler


def apply_ftir_pipeline(X: np.ndarray, minmax_scaler: MinMaxScaler) -> np.ndarray:
    X_snv = snv_transform(X)
    X_mm = minmax_scaler.transform(X_snv)
    assert X_mm.min() >= -0.01 and X_mm.max() <= 1.01, (
        f"FTIR values out of [0,1]: min={X_mm.min():.4f}, max={X_mm.max():.4f}"
    )
    return X_mm


# === Colorimetric ===

def fit_color_scaler(X_train_color: np.ndarray) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(X_train_color)
    logger.info(f"Colorimetric RobustScaler fitted on {X_train_color.shape}")
    return scaler


def apply_color_scaler(X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(X)


# === Data quality ===

def audit_data_quality(df, modality_name: str, output_dir) -> dict:
    import os
    from pathlib import Path
    import pandas as pd

    report_lines = [f"=== Data Quality Report: {modality_name} ===\n"]
    nan_per_col = df.isnull().sum()
    inf_per_col = df.apply(lambda c: np.isinf(c).sum() if np.issubdtype(c.dtype, np.number) else 0)

    nan_total = int(nan_per_col.sum())
    inf_total = int(inf_per_col.sum())
    report_lines.append(f"Total NaN: {nan_total}")
    report_lines.append(f"Total Inf: {inf_total}")

    # Fill NaN with median
    for col in df.columns:
        if df[col].isnull().any() and np.issubdtype(df[col].dtype, np.number):
            med = df[col].median()
            df[col].fillna(med, inplace=True)
            logger.warning(f"[{modality_name}] Filled NaN in column '{col}' with median={med:.4f}")

    # Replace Inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            col_max = df[col].replace([np.inf, -np.inf], np.nan).max()
            replacement = col_max * 1.5 if not np.isnan(col_max) else 0.0
            df[col].replace([np.inf, -np.inf], replacement, inplace=True)
            logger.warning(f"[{modality_name}] Replaced Inf in column '{col}' with {replacement:.4f}")

    # Near-constant columns
    near_const = []
    for col in numeric_cols:
        mean_val = abs(df[col].mean())
        std_val = df[col].std()
        cv = std_val / mean_val if mean_val > 1e-10 else std_val
        if cv < 0.001:
            near_const.append(col)
    if near_const:
        report_lines.append(f"Near-constant columns (CV<0.001): {near_const}")
        logger.warning(f"[{modality_name}] Near-constant columns: {near_const}")

    result = {"nan_total": nan_total, "inf_total": inf_total, "near_constant": near_const}

    report_path = Path(output_dir) / "reports" / f"data_quality_{modality_name}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Data quality report saved: {report_path}")
    return result

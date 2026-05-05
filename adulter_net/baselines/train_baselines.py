"""Baseline model training and comparison."""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def run_baselines(config) -> pd.DataFrame:
    from src import set_all_seeds, get_logger
    from src.data.loader import (
        load_all_modalities, validate_sample_ids, validate_cross_file_alignment,
        align_modalities, get_feature_columns,
    )
    from src.data.preprocessing import (
        fit_image_scaler, apply_image_scaler,
        fit_ftir_scaler, apply_ftir_pipeline,
        fit_color_scaler, apply_color_scaler,
    )
    from src.data.splits import get_stratified_splits

    set_all_seeds(config.seed)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)
    logger_ = get_logger("baselines", str(log_dir / "baselines.log"))

    dfs = load_all_modalities(config)
    for modality, df in dfs.items():
        validate_sample_ids(df, config, modality)
    validate_cross_file_alignment(dfs, config)
    merged = align_modalities(dfs, config)
    merged = merged.sort_values(config.sample_id_col).reset_index(drop=True)

    img_cols = get_feature_columns(dfs["image"], config, "image")
    ftir_cols = get_feature_columns(dfs["ftir"], config, "ftir")
    color_cols = get_feature_columns(dfs["color"], config, "color")

    X_img = merged[img_cols].values.astype(np.float32)
    X_ftir = merged[ftir_cols].values.astype(np.float32)
    X_color = merged[color_cols].values.astype(np.float32)

    y_raw = merged[config.target_col].values.astype(int)
    y_class = np.array([config.class_labels.index(int(v)) for v in y_raw])

    splits = get_stratified_splits(y_class, config)

    classifiers = {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=config.seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=config.seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=500, random_state=config.seed),
    }

    try:
        import xgboost as xgb
        classifiers["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, random_state=config.seed, verbosity=0,
            eval_metric="mlogloss",
        )
    except ImportError:
        pass

    results = []

    for name, clf in classifiers.items():
        fold_accs, fold_f1s = [], []
        logger_.info(f"Training baseline: {name}")

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Preprocess
            sc_img = fit_image_scaler(X_img[train_idx])
            sc_ftir = fit_ftir_scaler(
                __import__("src.data.preprocessing", fromlist=["snv_transform"]).snv_transform(X_ftir[train_idx])
            )
            sc_color = fit_color_scaler(X_color[train_idx])

            X_img_tr = apply_image_scaler(X_img[train_idx], sc_img)
            X_img_va = apply_image_scaler(X_img[val_idx], sc_img)
            X_ftir_tr = apply_ftir_pipeline(X_ftir[train_idx], sc_ftir)
            X_ftir_va = apply_ftir_pipeline(X_ftir[val_idx], sc_ftir)
            X_color_tr = apply_color_scaler(X_color[train_idx], sc_color)
            X_color_va = apply_color_scaler(X_color[val_idx], sc_color)

            X_tr = np.concatenate([X_img_tr, X_ftir_tr, X_color_tr], axis=1)
            X_va = np.concatenate([X_img_va, X_ftir_va, X_color_va], axis=1)
            y_tr, y_va = y_class[train_idx], y_class[val_idx]

            try:
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_va)
                acc = accuracy_score(y_va, y_pred)
                f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
                fold_accs.append(acc)
                fold_f1s.append(f1)
                logger_.info(f"  Fold {fold_idx}: acc={acc:.4f}, f1={f1:.4f}")
            except Exception as e:
                logger_.warning(f"  Fold {fold_idx} failed: {e}")
                fold_accs.append(0.0)
                fold_f1s.append(0.0)

        results.append({
            "Model": name,
            "Acc_mean": float(np.mean(fold_accs)),
            "Acc_std": float(np.std(fold_accs)),
            "Macro_F1": float(np.mean(fold_f1s)),
        })
        logger_.info(
            f"  {name}: Acc={np.mean(fold_accs):.4f}±{np.std(fold_accs):.4f}, "
            f"F1={np.mean(fold_f1s):.4f}"
        )

    df_results = pd.DataFrame(results)
    out_path = Path(config.output_dir) / "reports" / "baseline_comparison.csv"
    df_results.to_csv(out_path, index=False)
    logger_.info(f"Baseline comparison saved: {out_path}")
    return df_results


if __name__ == "__main__":
    from config import Config
    config = Config()
    df = run_baselines(config)
    print(df.to_string(index=False))

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_logger, set_all_seeds

logger = get_logger(__name__)


def run_baselines(config, splits, all_features: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Train and cross-validate classical ML baselines."""
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    baselines: Dict = {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=500, random_state=42),
    }

    try:
        import xgboost as xgb
        baselines["XGBoost"] = xgb.XGBClassifier(n_estimators=300, random_state=42,
                                                    eval_metric="mlogloss", verbosity=0)
    except ImportError:
        logger.warning("XGBoost not available, skipping")

    results = []
    for name, clf in baselines.items():
        fold_accs, fold_f1s = [], []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr, X_v = all_features[train_idx], all_features[val_idx]
            y_tr, y_v = y[train_idx], y[val_idx]
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_v = sc.transform(X_v)
            try:
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_v)
                fold_accs.append(float((preds == y_v).mean()))
                fold_f1s.append(float(f1_score(y_v, preds, average="macro", zero_division=0)))
            except Exception as e:
                logger.warning(f"Baseline {name} fold {fold_idx} failed: {e}")
                fold_accs.append(0.0); fold_f1s.append(0.0)
        results.append({
            "Model": name, "Acc_mean": np.mean(fold_accs), "Acc_std": np.std(fold_accs),
            "Macro_F1": np.mean(fold_f1s),
        })
        logger.info(f"{name}: Acc={np.mean(fold_accs):.4f}±{np.std(fold_accs):.4f} F1={np.mean(fold_f1s):.4f}")

    df = pd.DataFrame(results)
    out_path = config.output_dir / "reports" / "baseline_comparison.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Baseline results saved: {out_path}")
    return df


if __name__ == "__main__":
    from config import config
    set_all_seeds(config.seed)
    logger.info("Run this after main CV to compare against AdulterNet.")
    logger.info("Requires splits and feature-selected arrays to be pre-computed.")

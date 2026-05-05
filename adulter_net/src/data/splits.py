from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


def create_stratified_folds(
    y: np.ndarray, config
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create stratified k-fold splits on class labels.

    Args:
        y: Integer class index array (0-6), shape (N,)
        config: Config dataclass

    Returns:
        List of (train_idx, val_idx) tuples, one per fold
    """
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )
    splits = list(skf.split(np.zeros(len(y)), y))
    logger.info(
        f"Created {config.n_folds} stratified folds. "
        f"Per-fold val size: ~{len(splits[0][1])}"
    )
    for i, (train_idx, val_idx) in enumerate(splits):
        # Validate class balance in each split
        train_classes, train_counts = np.unique(y[train_idx], return_counts=True)
        val_classes, val_counts = np.unique(y[val_idx], return_counts=True)
        logger.info(
            f"  Fold {i}: train={len(train_idx)} (classes={dict(zip(train_classes.tolist(), train_counts.tolist()))}), "
            f"val={len(val_idx)} (classes={dict(zip(val_classes.tolist(), val_counts.tolist()))})"
        )
    return splits


def save_splits(splits: List[Tuple[np.ndarray, np.ndarray]], config) -> None:
    """Save all fold splits to disk."""
    save_path = config.processed_dir / "splits" / "cv_splits.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(splits, f)
    logger.info(f"Splits saved to {save_path}")


def load_splits(config) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load pre-computed fold splits from disk."""
    save_path = config.processed_dir / "splits" / "cv_splits.pkl"
    with open(save_path, "rb") as f:
        splits = pickle.load(f)
    logger.info(f"Splits loaded from {save_path}")
    return splits

"""Stratified K-Fold split factory."""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_stratified_splits(y: np.ndarray, config) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) tuples for each fold."""
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    return splits

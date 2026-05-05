from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _dummy(n=70, n_img=100, n_ftir=60, n_color=15, n_cls=7):
    np.random.seed(42)
    X_img = np.random.randn(n, n_img).astype(np.float32)
    X_ftir = np.clip(np.random.randn(n, n_ftir).astype(np.float32) * 0.3 + 0.5, 0, 1)
    X_color = np.random.randn(n, n_color).astype(np.float32)
    y = np.repeat(np.arange(n_cls), n // n_cls + 1)[:n]
    col_img = [f"feat_{i}" for i in range(n_img)]
    col_ftir = [f"wn_{i*10}" for i in range(n_ftir)]
    col_color = ["L_star", "a_star", "b_star"] + [f"R_{i*10}nm" for i in range(n_color - 3)]
    return X_img, X_ftir, X_color, y, col_img, col_ftir, col_color


def test_shap_boruta_reduces_from_original():
    from src.feature_selection.shap_boruta import SHAPBorutaSelector
    X_img, _, _, y, _, _, _ = _dummy(n_img=100)
    sel = SHAPBorutaSelector(top_k=50, max_iter=5)
    sel.fit(X_img, y)
    X_sel = sel.transform(X_img)
    assert X_sel.shape[1] < 100, f"Expected < 100 features, got {X_sel.shape[1]}"
    assert X_sel.shape[1] >= 1


def test_cars_wavenumber_indices_in_range():
    from src.feature_selection.cars_spa import CARSSelector
    _, X_ftir, _, y, _, _, _ = _dummy()
    sel = CARSSelector(n_iterations=5)
    sel.fit(X_ftir, y.astype(float))
    idx = sel.get_selected_wavenumbers()
    assert all(0 <= i < 60 for i in idx), f"Index out of range [0,60): {idx}"


def test_spa_after_cars():
    from src.feature_selection.cars_spa import CARSSelector, SPASelector
    _, X_ftir, _, y, _, _, _ = _dummy()
    cars = CARSSelector(n_iterations=5)
    cars.fit(X_ftir, y.astype(float))
    X_cars = cars.transform(X_ftir)
    spa = SPASelector(n_components=10)
    spa.fit(X_cars)
    X_spa = spa.transform(X_cars)
    assert X_spa.shape[1] <= 10
    assert X_spa.shape[0] == 70


def test_mrmr_selects_features():
    from src.feature_selection.mrmr_shap import mRMRSHAPSelector
    _, _, X_color, y, _, _, col_color = _dummy()
    sel = mRMRSHAPSelector(n_mrmr=10)
    sel.fit(X_color, y, col_color)
    names = sel.get_feature_names()
    assert len(names) >= 1, "Must select at least 1 colorimetric feature"


def test_selector_persistence():
    from src.feature_selection.shap_boruta import SHAPBorutaSelector
    X_img, _, _, y, _, _, _ = _dummy(n_img=100)
    sel = SHAPBorutaSelector(top_k=50, max_iter=3)
    sel.fit(X_img, y)
    X_orig = sel.transform(X_img)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp = f.name
    sel.save(tmp)
    loaded = SHAPBorutaSelector.load(tmp)
    X_loaded = loaded.transform(X_img)
    np.testing.assert_array_equal(X_orig, X_loaded)


def test_cars_rmsecv_history_populated():
    from src.feature_selection.cars_spa import CARSSelector
    _, X_ftir, _, y, _, _, _ = _dummy()
    sel = CARSSelector(n_iterations=5)
    sel.fit(X_ftir, y.astype(float))
    assert len(sel.rmsecv_history_) > 0, "RMSECV history should be populated after fit"

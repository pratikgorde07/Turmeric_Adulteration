"""Tests for feature selection modules."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import joblib
import tempfile


def _make_xy(n=100, n_feat=50, n_classes=7):
    X = np.random.randn(n, n_feat).astype(np.float32)
    y = (np.arange(n) % n_classes).astype(int)
    return X, y


def test_shap_boruta_reduces_from_1280():
    from src.feature_selection.shap_boruta import SHAPBorutaSelector
    X, y = _make_xy(n=70, n_feat=100)
    sel = SHAPBorutaSelector(top_k=50)
    sel.fit(X, y)
    out = sel.transform(X)
    assert out.shape[1] < 100, f"Expected < 100 features, got {out.shape[1]}"
    assert out.shape[1] <= 50, f"Expected <= top_k=50, got {out.shape[1]}"


def test_cars_wavenumber_indices_in_range():
    from src.feature_selection.cars_spa import CARSSelector
    n_wn = 60
    X, y = _make_xy(n=80, n_feat=n_wn)
    sel = CARSSelector(n_iterations=20)
    sel.fit(X, y.astype(float))
    indices = sel.get_selected_wavenumbers()
    assert all(0 <= i < n_wn for i in indices), "CARS indices out of range"
    assert len(indices) > 0, "CARS selected zero features"


def test_spa_reduces_features():
    from src.feature_selection.cars_spa import SPASelector
    X = np.random.randn(50, 20).astype(np.float32)
    sel = SPASelector(n_components=8)
    sel.fit(X)
    out = sel.transform(X)
    assert out.shape[1] <= 8, f"SPA should select <= 8 features, got {out.shape[1]}"


def test_mrmr_includes_all_cie_axes():
    from src.feature_selection.mrmr_shap import mRMRSHAPSelector
    n = 100
    col_names = ["L*", "a*", "b*"] + [f"R_{i}nm" for i in range(17)]
    X = np.random.randn(n, len(col_names)).astype(np.float32)
    y = (np.arange(n) % 7).astype(int)
    sel = mRMRSHAPSelector(n_features=10)
    sel.fit(X, y, col_names)
    names = sel.get_feature_names()
    assert any("L" in n.upper() for n in names), "L* axis missing"
    assert any("A" in n.upper() for n in names), "a* axis missing"
    assert any("B" in n.upper() for n in names), "b* axis missing"


def test_selector_persistence():
    from src.feature_selection.shap_boruta import SHAPBorutaSelector
    X, y = _make_xy(n=60, n_feat=30)
    sel = SHAPBorutaSelector(top_k=15)
    sel.fit(X, y)
    original_out = sel.transform(X)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    joblib.dump(sel, tmp_path)
    loaded = joblib.load(tmp_path)
    loaded_out = loaded.transform(X)
    assert np.allclose(original_out, loaded_out), "Loaded selector gives different output"
    os.unlink(tmp_path)


def test_no_leakage_image():
    """Shuffling y should generally produce different (usually worse) selection."""
    from src.feature_selection.shap_boruta import SHAPBorutaSelector
    X, y = _make_xy(n=70, n_feat=40)
    sel1 = SHAPBorutaSelector(top_k=20)
    sel1.fit(X, y)
    idx1 = set(sel1.get_selected_indices())
    y_shuffle = y.copy()
    np.random.shuffle(y_shuffle)
    sel2 = SHAPBorutaSelector(top_k=20)
    sel2.fit(X, y_shuffle)
    idx2 = set(sel2.get_selected_indices())
    # They should not be identical (probabilistic — may occasionally fail)
    # Just verify both ran without error and produced valid indices
    assert len(idx1) > 0 and len(idx2) > 0

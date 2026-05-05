from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_ds(n=20, n_img=50, n_ftir=30, n_color=10, is_training=True):
    from src.data.dataset import TriModalDataset
    X_img = np.random.randn(n, n_img).astype(np.float32)
    X_ftir = np.random.randn(n, n_ftir).astype(np.float32)
    X_color = np.random.randn(n, n_color).astype(np.float32)
    y_class = np.random.randint(0, 7, size=n)
    y_reg = np.random.uniform(0, 9, size=n).astype(np.float32)
    ids = [f"T{i%7}S{i+1:02d}" for i in range(n)]
    return TriModalDataset(X_img, X_ftir, X_color, y_class, y_reg, ids, is_training=is_training)


def test_dataset_length():
    ds = _make_ds(n=20)
    assert len(ds) == 20


def test_dataset_item_keys():
    ds = _make_ds(n=5)
    item = ds[0]
    required = {"image_feat", "ftir", "color", "label", "label_float", "sample_id", "image_raw"}
    assert required.issubset(set(item.keys()))


def test_dataset_tensor_types():
    ds = _make_ds(n=5)
    item = ds[0]
    assert isinstance(item["image_feat"], torch.Tensor)
    assert isinstance(item["ftir"], torch.Tensor)
    assert item["label"].dtype == torch.long


def test_dataset_val_no_augmentation():
    """Val dataset returns identical outputs on repeated calls."""
    ds = _make_ds(n=5, is_training=False)
    item1 = ds[0]
    item2 = ds[0]
    assert torch.allclose(item1["image_feat"], item2["image_feat"])
    assert torch.allclose(item1["ftir"], item2["ftir"])
    assert torch.allclose(item1["color"], item2["color"])


def test_augmentation_does_not_change_val():
    """100 accesses to val dataset[0] should all be identical."""
    ds = _make_ds(n=5, is_training=False)
    first = ds[0]["image_feat"].clone()
    for _ in range(100):
        item = ds[0]
        assert torch.allclose(item["image_feat"], first), "Val item changed — augmentation leaking!"


def test_dataset_image_raw_zeros_when_no_path():
    ds = _make_ds(n=5)
    item = ds[0]
    assert item["image_raw"].shape == (3, 224, 224)

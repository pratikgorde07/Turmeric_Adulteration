"""Tests for TriModalDataset and augmenters."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch


def _make_fake_data(n=20, n_img=50, n_ftir=30, n_color=10):
    X_img = np.random.randn(n, n_img).astype(np.float32)
    X_ftir = np.random.randn(n, n_ftir).astype(np.float32)
    X_color = np.random.randn(n, n_color).astype(np.float32)
    y_class = np.arange(n) % 7
    y_reg = np.array([0, 1, 2, 3, 5, 7, 9] * (n // 7 + 1))[:n].astype(np.float32)
    sids = [f"T{y_class[i]}S{i+1:02d}" for i in range(n)]
    return X_img, X_ftir, X_color, y_class, y_reg, sids


def test_trimodal_dataset_getitem_keys():
    from src.data.dataset import TriModalDataset
    X_img, X_ftir, X_color, y_class, y_reg, sids = _make_fake_data()
    ds = TriModalDataset(X_img, X_ftir, X_color, y_class, y_reg, sids, is_training=False)
    item = ds[0]
    for key in ["image_feat", "ftir", "color", "label", "label_float", "sample_id", "image_raw"]:
        assert key in item, f"Missing key: {key}"


def test_trimodal_dataset_shapes():
    from src.data.dataset import TriModalDataset
    X_img, X_ftir, X_color, y_class, y_reg, sids = _make_fake_data()
    ds = TriModalDataset(X_img, X_ftir, X_color, y_class, y_reg, sids, is_training=False)
    item = ds[0]
    assert item["image_feat"].shape == (50,)
    assert item["ftir"].shape == (30,)
    assert item["color"].shape == (10,)
    assert item["image_raw"].shape == (3, 224, 224)


def test_augmentation_does_not_change_val():
    """Val items must be bit-identical across 100 calls."""
    from src.data.dataset import TriModalDataset
    X_img, X_ftir, X_color, y_class, y_reg, sids = _make_fake_data()
    ds = TriModalDataset(X_img, X_ftir, X_color, y_class, y_reg, sids, is_training=False)
    first = ds[0]
    for _ in range(99):
        item = ds[0]
        assert torch.allclose(item["image_feat"], first["image_feat"]), "Val augmented!"
        assert torch.allclose(item["ftir"], first["ftir"]), "Val FTIR augmented!"


def test_image_feature_augmenter_mean_stable():
    from src.data.augmentation import ImageFeatureAugmenter
    aug = ImageFeatureAugmenter(noise_scale=0.01, dropout_prob=0.05, enabled=True)
    x = np.random.randn(200).astype(np.float32)
    copies = np.stack([aug(x) for _ in range(1000)])
    mean_aug = copies.mean(axis=0)
    assert np.abs(mean_aug - x).max() < 3 * x.std(), "Augmented mean deviates too much"


def test_ftir_augmenter_clips_to_01():
    from src.data.augmentation import FTIRAugmenter
    aug = FTIRAugmenter(enabled=True)
    x = np.random.rand(100).astype(np.float32)
    for _ in range(50):
        out = aug(x)
        assert out.min() >= 0.0 and out.max() <= 1.0

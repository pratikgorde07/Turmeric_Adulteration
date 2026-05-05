from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_focal_perfect_prediction_low():
    from src.losses import FocalLossSmoothed
    loss_fn = FocalLossSmoothed(gamma=2.5, smoothing=0.05)
    logits = torch.zeros(4, 7)
    for i in range(4):
        logits[i, i] = 100.0
    targets = torch.arange(4)
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.5, f"Near-perfect prediction loss should be < 0.5, got {loss.item()}"


def test_focal_random_prediction_higher():
    from src.losses import FocalLossSmoothed
    loss_fn = FocalLossSmoothed(gamma=2.5, smoothing=0.05)
    torch.manual_seed(42)
    targets = torch.randint(0, 7, (16,))
    logits_bad = torch.randn(16, 7)
    loss_bad = loss_fn(logits_bad, targets)
    logits_good = torch.zeros(16, 7)
    for i in range(16):
        logits_good[i, targets[i]] = 100.0
    loss_good = loss_fn(logits_good, targets)
    assert loss_bad.item() > loss_good.item(), (
        f"Random loss ({loss_bad.item():.4f}) should be > perfect loss ({loss_good.item():.4f})"
    )


def test_supcon_same_class_nonnegative():
    from src.losses import SupConLoss
    loss_fn = SupConLoss(temperature=0.07)
    emb = F.normalize(torch.randn(8, 128), dim=-1)
    labels = torch.zeros(8, dtype=torch.long)
    loss = loss_fn(emb, labels)
    assert loss.item() >= 0, f"SupCon loss must be non-negative, got {loss.item()}"


def test_joint_loss_all_keys():
    from src.losses import AdulterNetLoss
    loss_fn = AdulterNetLoss()
    B = 8
    logits = torch.randn(B, 7)
    reg_pred = torch.rand(B, 1) * 9
    proj_emb = F.normalize(torch.randn(B, 128), dim=-1)
    label_int = torch.randint(0, 7, (B,))
    label_float = torch.rand(B) * 9
    result = loss_fn(logits, reg_pred, proj_emb, label_int, label_float)
    for key in ("total", "focal", "huber", "supcon"):
        assert key in result, f"Missing loss key: {key}"


def test_joint_loss_total_positive():
    from src.losses import AdulterNetLoss
    loss_fn = AdulterNetLoss()
    B = 8
    result = loss_fn(
        torch.randn(B, 7), torch.rand(B, 1) * 9,
        F.normalize(torch.randn(B, 128), dim=-1),
        torch.randint(0, 7, (B,)), torch.rand(B) * 9,
    )
    assert result["total"].item() >= 0, f"Total loss must be non-negative, got {result['total'].item()}"


def test_mixup_loss_nonnegative():
    from src.losses import MixupCriterion
    loss_fn = MixupCriterion()
    logits = torch.randn(4, 7)
    soft_labels = torch.zeros(4, 7)
    soft_labels[:, 1] = 0.5; soft_labels[:, 2] = 0.5
    loss = loss_fn(logits, soft_labels)
    assert loss.item() >= 0, f"Mixup loss must be non-negative, got {loss.item()}"

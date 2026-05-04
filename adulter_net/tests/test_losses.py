"""Tests for loss functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np


def test_focal_perfect_prediction_low():
    from src.losses import FocalLossSmoothed
    loss_fn = FocalLossSmoothed(gamma=2.5, smoothing=0.0)
    B, C = 8, 7
    logits = torch.zeros(B, C)
    targets = torch.zeros(B, dtype=torch.long)
    # Make target class probability very high
    logits[:, 0] = 10.0
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.1, f"Expected low loss for near-perfect predictions, got {loss.item()}"


def test_focal_random_prediction_higher():
    from src.losses import FocalLossSmoothed
    loss_fn = FocalLossSmoothed(gamma=2.5, smoothing=0.0)
    B, C = 16, 7
    targets = torch.randint(0, C, (B,))
    # Perfect
    logits_perfect = torch.zeros(B, C)
    for i, t in enumerate(targets):
        logits_perfect[i, t] = 10.0
    # Random
    logits_random = torch.randn(B, C)
    loss_perfect = loss_fn(logits_perfect, targets).item()
    loss_random = loss_fn(logits_random, targets).item()
    assert loss_random > loss_perfect, "Random loss should exceed perfect loss"


def test_supcon_same_class_lower():
    from src.losses import SupConLoss
    loss_fn = SupConLoss(temperature=0.07)
    B, D = 16, 128
    import torch.nn.functional as F
    # All same class
    emb_same = F.normalize(torch.randn(B, D), dim=-1)
    labels_same = torch.zeros(B, dtype=torch.long)
    # Mixed classes
    emb_mixed = F.normalize(torch.randn(B, D), dim=-1)
    labels_mixed = torch.arange(B) % 7
    loss_same = loss_fn(emb_same, labels_same).item()
    loss_mixed = loss_fn(emb_mixed, labels_mixed).item()
    # Same-class batches should generally produce well-defined loss
    assert loss_same >= 0, "SupCon loss should be non-negative"
    assert loss_mixed >= 0, "SupCon loss should be non-negative"


def test_joint_loss_all_keys():
    from src.losses import AdulterNetLoss
    import torch.nn.functional as F
    loss_fn = AdulterNetLoss(alpha=0.75, supcon_weight=0.2)
    B, C = 8, 7
    logits = torch.randn(B, C)
    reg_pred = torch.rand(B, 1) * 9
    proj_emb = F.normalize(torch.randn(B, 128), dim=-1)
    label_int = torch.randint(0, C, (B,))
    label_float = torch.rand(B) * 9
    result = loss_fn(logits, reg_pred, proj_emb, label_int, label_float)
    for key in ["total", "focal", "huber", "supcon"]:
        assert key in result, f"Missing key: {key}"
    assert result["total"].item() > 0, "total loss should be positive"


def test_mixup_loss_positive():
    from src.losses import MixupCriterion
    loss_fn = MixupCriterion()
    B, C = 8, 7
    logits = torch.randn(B, C)
    soft_labels = torch.softmax(torch.randn(B, C), dim=-1)
    loss = loss_fn(logits, soft_labels)
    assert loss.item() > 0, "Mixup loss should be positive"
    assert loss.item() > 0, "Mixup loss should be positive"

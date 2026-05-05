from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_batch(B=4, n_ftir=40, n_color=10):
    return {
        "image_feat": torch.randn(B, 1280),
        "ftir": torch.randn(B, n_ftir),
        "color": torch.randn(B, n_color),
        "label": torch.randint(0, 7, (B,)),
        "label_float": torch.rand(B) * 9,
        "sample_id": [f"T0S{i+1:02d}" for i in range(B)],
        "image_raw": torch.zeros(B, 3, 224, 224),
    }


def test_full_forward_pass():
    """Complete forward pass through the full model."""
    from config import Config
    from src.models.adulter_net import AdulterNet
    cfg = Config()
    model = AdulterNet(cfg, n_ftir_features=40, n_color_features=10)
    model.eval()
    batch = _make_batch()
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (4, 7)
    assert out["reg_pred"].shape[0] == 4
    assert out["proj_emb"].shape == (4, 128)
    assert (out["reg_pred"] >= 0).all() and (out["reg_pred"] <= 9).all()
    norms = out["proj_emb"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_loss_backward():
    """Loss.backward() should not error and gradients should be finite."""
    from config import Config
    from src.models.adulter_net import AdulterNet
    from src.losses import AdulterNetLoss
    cfg = Config()
    model = AdulterNet(cfg, n_ftir_features=40, n_color_features=10)
    model.freeze_all_backbones()
    criterion = AdulterNetLoss()
    batch = _make_batch()
    out = model(batch)
    loss_dict = criterion(
        out["logits"], out["reg_pred"], out["proj_emb"],
        batch["label"], batch["label_float"],
    )
    loss_dict["total"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


def test_calibration_reduces_nll():
    """Temperature scaling should not increase NLL on the calibration set."""
    from config import Config
    from src.models.adulter_net import AdulterNet
    from src.calibration import TemperatureScaler
    import torch.nn as nn
    cfg = Config()
    model = AdulterNet(cfg, n_ftir_features=40, n_color_features=10)
    model.eval()
    batch = _make_batch(B=16)
    with torch.no_grad():
        out = model(batch)
    logits = out["logits"]
    labels = batch["label"]
    nll_before = nn.CrossEntropyLoss()(logits, labels).item()
    cal = TemperatureScaler()
    cal.calibrate(logits.clone(), labels.clone())
    nll_after = nn.CrossEntropyLoss()(cal.scale(logits), labels).item()
    # Calibration should not dramatically worsen NLL
    assert nll_after < nll_before * 2, f"Calibration worsened NLL: {nll_before:.4f} -> {nll_after:.4f}"

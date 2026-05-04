"""Integration tests: end-to-end mini pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch


def _make_config():
    from config import Config
    cfg = Config()
    # Override for speed
    cfg.n_epochs = 3
    cfg.batch_size = 4
    cfg.n_folds = 2
    cfg.num_workers = 0
    cfg.use_amp = False
    cfg.use_wandb = False
    return cfg


def test_dataset_to_dataloader():
    from src.data.dataset import TriModalDataset, get_dataloaders
    config = _make_config()
    n = 20
    X_img = np.random.randn(n, 32).astype(np.float32)
    X_ftir = np.random.randn(n, 20).astype(np.float32)
    X_color = np.random.randn(n, 8).astype(np.float32)
    y_class = (np.arange(n) % 7).astype(np.int64)
    y_reg = np.zeros(n, dtype=np.float32)
    sids = [f"T{y_class[i]}S{i+1:02d}" for i in range(n)]

    train_idx = np.arange(14)
    val_idx = np.arange(14, 20)
    all_data = {
        "X_img": X_img, "X_ftir": X_ftir, "X_color": X_color,
        "y_class": y_class, "y_reg": y_reg, "sample_ids": sids,
    }
    train_loader, val_loader = get_dataloaders(train_idx, val_idx, all_data, config)
    batch = next(iter(train_loader))
    assert "image_feat" in batch
    assert "ftir" in batch
    assert "color" in batch
    assert batch["label"].dtype == torch.long


def test_model_forward_mini():
    from config import Config
    from src.models.adulter_net import AdulterNet
    config = _make_config()
    model = AdulterNet(config, n_img_features=32, n_ftir_features=10, n_color_features=6)
    batch = {
        "image_feat": torch.randn(4, 32),
        "ftir": torch.randn(4, 10),
        "color": torch.randn(4, 6),
        "label": torch.randint(0, 7, (4,)),
        "label_float": torch.randn(4),
        "image_raw": torch.zeros(4, 3, 224, 224),
        "sample_id": ["T0S01"] * 4,
    }
    out = model(batch)
    assert out["logits"].shape == (4, 7)
    assert out["reg_pred"].shape == (4, 1)


def test_loss_backward():
    from src.losses import AdulterNetLoss
    import torch.nn.functional as F
    loss_fn = AdulterNetLoss()
    B, C = 4, 7
    logits = torch.randn(B, C, requires_grad=True)
    reg = torch.rand(B, 1, requires_grad=True) * 9
    proj = F.normalize(torch.randn(B, 128), dim=-1)
    proj.requires_grad_(True)
    labels = torch.randint(0, C, (B,))
    label_f = torch.rand(B) * 9
    result = loss_fn(logits, reg, proj, labels, label_f)
    result["total"].backward()
    assert logits.grad is not None, "No gradient on logits"

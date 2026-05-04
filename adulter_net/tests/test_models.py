"""Tests for model components."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np


def _make_batch(B=8, n_img=1280, n_ftir=40, n_color=15, device="cpu"):
    return {
        "image_feat": torch.randn(B, n_img),
        "ftir": torch.randn(B, n_ftir),
        "color": torch.randn(B, n_color),
        "label": torch.randint(0, 7, (B,)),
        "label_float": torch.randn(B),
        "image_raw": torch.zeros(B, 3, 224, 224),
        "sample_id": [f"T0S{i:02d}" for i in range(B)],
    }


def test_visual_stream_feature_shape():
    from src.models.visual_stream import VisualStream
    model = VisualStream(img_feat_dim=1280, embed_dim=256, use_yolo=False)
    batch = _make_batch()
    out = model(batch)
    assert out.shape == (8, 256), f"Expected (8,256), got {out.shape}"


def test_spectral_transformer_shape():
    from src.models.spectral_stream import SpectralTransformer
    model = SpectralTransformer(n_ftir_features=40, out_dim=128)
    x = torch.randn(8, 40)
    out = model(x)
    assert out.shape == (8, 128), f"Expected (8,128), got {out.shape}"


def test_color_encoder_shape():
    from src.models.color_stream import ColorimetricEncoder
    model = ColorimetricEncoder(n_color_features=15, out_dim=64)
    x = torch.randn(8, 15)
    out = model(x)
    assert out.shape == (8, 64), f"Expected (8,64), got {out.shape}"


def test_cmt_fusion_shape():
    from src.models.cmt_fusion import CMTFusion
    model = CMTFusion(visual_dim=256, spectral_dim=128, color_dim=64, out_dim=512)
    v = torch.randn(8, 256)
    s = torch.randn(8, 128)
    c = torch.randn(8, 64)
    out = model(v, s, c)
    assert out.shape == (8, 512), f"Expected (8,512), got {out.shape}"


def test_full_forward_all_keys():
    from config import Config
    from src.models.adulter_net import AdulterNet
    config = Config()
    model = AdulterNet(config, n_img_features=50, n_ftir_features=10, n_color_features=8)
    batch = _make_batch(B=4, n_img=50, n_ftir=10, n_color=8)
    out = model(batch)
    for key in ["logits", "reg_pred", "proj_emb"]:
        assert key in out, f"Missing key: {key}"


def test_logits_shape():
    from config import Config
    from src.models.adulter_net import AdulterNet
    config = Config()
    model = AdulterNet(config, n_img_features=50, n_ftir_features=10, n_color_features=8)
    batch = _make_batch(B=4, n_img=50, n_ftir=10, n_color=8)
    out = model(batch)
    assert out["logits"].shape == (4, 7), f"Expected (4,7), got {out['logits'].shape}"


def test_reg_pred_in_range():
    from config import Config
    from src.models.adulter_net import AdulterNet
    config = Config()
    model = AdulterNet(config, n_img_features=50, n_ftir_features=10, n_color_features=8)
    for _ in range(10):
        batch = _make_batch(B=8, n_img=50, n_ftir=10, n_color=8)
        out = model(batch)
        assert out["reg_pred"].min() >= 0.0, "Regression below 0"
        assert out["reg_pred"].max() <= 9.0, "Regression above 9"


def test_freeze_unfreeze():
    from config import Config
    from src.models.adulter_net import AdulterNet
    config = Config()
    model = AdulterNet(config, n_img_features=50, n_ftir_features=10, n_color_features=8)
    model.freeze_all_backbones()
    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.unfreeze_visual_top_n(1)
    trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # After unfreezing, should have >= same as after freeze
    assert trainable_after_unfreeze >= trainable_after_freeze


def test_mc_dropout_nonzero_std():
    from config import Config
    from src.models.adulter_net import AdulterNet, mc_dropout_predict
    config = Config()
    model = AdulterNet(config, n_img_features=50, n_ftir_features=10, n_color_features=8)
    batch = _make_batch(B=4, n_img=50, n_ftir=10, n_color=8)
    result = mc_dropout_predict(model, batch, n_passes=30, device="cpu")
    assert result["std_logits"].mean().item() > 0, "MC Dropout std should be > 0"

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _cfg():
    from config import Config
    return Config()


def _batch(B=8, n_img=1280, n_ftir=40, n_color=15, device="cpu"):
    return {
        "image_feat": torch.randn(B, n_img).to(device),
        "ftir": torch.randn(B, n_ftir).to(device),
        "color": torch.randn(B, n_color).to(device),
        "label": torch.randint(0, 7, (B,)).to(device),
        "label_float": torch.rand(B).to(device) * 9,
        "sample_id": [f"T0S{i+1:02d}" for i in range(B)],
        "image_raw": torch.zeros(B, 3, 224, 224).to(device),
    }


def test_spectral_transformer_shape():
    from src.models.spectral_stream import SpectralTransformer
    model = SpectralTransformer(n_ftir_features=40, config=_cfg())
    out = model(torch.randn(8, 40))
    assert out.shape == (8, 128), f"Expected (8,128), got {out.shape}"


def test_color_encoder_shape():
    from src.models.color_stream import ColorimetricEncoder
    model = ColorimetricEncoder(n_color_features=15, config=_cfg())
    out = model(torch.randn(8, 15))
    assert out.shape == (8, 64), f"Expected (8,64), got {out.shape}"


def test_cmt_fusion_shape():
    from src.models.cmt_fusion import CMTFusion
    model = CMTFusion(_cfg())
    out = model(torch.randn(8, 256), torch.randn(8, 128), torch.randn(8, 64))
    assert out.shape == (8, 512), f"Expected (8,512), got {out.shape}"


def test_visual_stream_from_features():
    from src.models.visual_stream import EfficientNetVisualStream
    model = EfficientNetVisualStream(_cfg())
    out = model.forward_from_features(torch.randn(8, 1280))
    assert out.shape == (8, 256), f"Expected (8,256), got {out.shape}"


def test_full_forward_all_keys():
    from src.models.adulter_net import AdulterNet
    model = AdulterNet(_cfg(), n_ftir_features=40, n_color_features=15)
    out = model(_batch())
    for key in ("logits", "reg_pred", "proj_emb"):
        assert key in out, f"Missing key: {key}"


def test_logits_shape():
    from src.models.adulter_net import AdulterNet
    model = AdulterNet(_cfg(), n_ftir_features=40, n_color_features=15)
    out = model(_batch())
    assert out["logits"].shape == (8, 7), f"Expected (8,7), got {out['logits'].shape}"


def test_reg_pred_in_range():
    from src.models.adulter_net import AdulterNet
    model = AdulterNet(_cfg(), n_ftir_features=40, n_color_features=15)
    out = model(_batch())
    reg = out["reg_pred"]
    assert (reg >= 0).all() and (reg <= 9).all(), f"reg_pred out of [0,9]: {reg.min():.3f} - {reg.max():.3f}"


def test_freeze_unfreeze():
    from src.models.adulter_net import AdulterNet
    model = AdulterNet(_cfg(), n_ftir_features=40, n_color_features=15)
    model.freeze_all_backbones()
    backbone_trainable = sum(p.numel() for p in model.visual_stream.base_model.parameters() if p.requires_grad)
    assert backbone_trainable == 0, f"Expected 0 trainable backbone params, got {backbone_trainable}"


def test_mc_dropout_nonzero_std():
    from src.models.adulter_net import AdulterNet, mc_dropout_predict
    model = AdulterNet(_cfg(), n_ftir_features=40, n_color_features=15)
    result = mc_dropout_predict(model, _batch(), n_passes=30, device="cpu")
    assert result["std_logits"].max().item() > 0, "MC Dropout std should be > 0"


def test_proj_emb_l2_normalized():
    from src.models.adulter_net import AdulterNet
    model = AdulterNet(_cfg(), n_ftir_features=40, n_color_features=15)
    with torch.no_grad():
        out = model(_batch())
    norms = out["proj_emb"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), f"proj_emb not L2-normalized: {norms}"

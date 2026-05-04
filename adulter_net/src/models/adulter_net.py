"""Full AdulterNet model with dual heads and MC Dropout inference."""
from __future__ import annotations
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int = 512, n_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).clamp(0.0, 9.0)


class SupConProjector(nn.Module):
    def __init__(self, in_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.projector(x)
        return F.normalize(out, dim=-1)


class AdulterNet(nn.Module):
    def __init__(self, config, n_img_features: int = 1280,
                 n_ftir_features: int = 40, n_color_features: int = 20):
        super().__init__()
        from src.models.visual_stream import VisualStream
        from src.models.spectral_stream import SpectralTransformer
        from src.models.color_stream import ColorimetricEncoder
        from src.models.cmt_fusion import CMTFusion

        self.visual_stream = VisualStream(
            img_feat_dim=n_img_features,
            embed_dim=config.visual_embed_dim,
            dropout=config.dropout,
        )
        self.spectral_stream = SpectralTransformer(
            n_ftir_features=n_ftir_features,
            embed_dim=64,
            out_dim=config.spectral_embed_dim,
            n_heads=4,
            n_layers=config.n_transformer_layers,
            ffn_dim=config.ffn_dim // 4,
            dropout=0.1,
        )
        self.color_stream = ColorimetricEncoder(
            n_color_features=n_color_features,
            out_dim=config.color_embed_dim,
            dropout=0.2,
        )
        self.cmt_fusion = CMTFusion(
            visual_dim=config.visual_embed_dim,
            spectral_dim=config.spectral_embed_dim,
            color_dim=config.color_embed_dim,
            d_model=256,
            n_heads=config.n_attention_heads,
            n_layers=config.n_transformer_layers,
            ffn_dim=config.ffn_dim,
            dropout=0.1,
            out_dim=config.fused_dim,
        )
        self.classification_head = ClassificationHead(
            in_dim=config.fused_dim,
            n_classes=config.n_classes,
            dropout=0.3,
        )
        self.regression_head = RegressionHead(
            in_dim=config.fused_dim,
            dropout=0.2,
        )
        self.supcon_projector = SupConProjector(
            in_dim=config.fused_dim,
            proj_dim=128,
        )
        self.config = config

    def freeze_all_backbones(self) -> None:
        self.visual_stream.freeze_all()

    def unfreeze_visual_top_n(self, n: int) -> None:
        self.visual_stream.unfreeze_top_n_blocks(n)

    def count_parameters(self) -> None:
        total, trainable = self.get_num_parameters()
        logger.info(f"AdulterNet: total={total:,}, trainable={trainable:,}")
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            logger.info(f"  {name}: total={n:,}, trainable={t:,}")

    def get_num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(self, batch: dict, return_embeddings: bool = False) -> dict:
        v = self.visual_stream(batch)           # (B, 256)
        s = self.spectral_stream(batch["ftir"]) # (B, 128)
        c = self.color_stream(batch["color"])   # (B, 64)
        fused = self.cmt_fusion(v, s, c)        # (B, 512)

        logits = self.classification_head(fused)   # (B, n_classes)
        reg_pred = self.regression_head(fused)      # (B, 1)
        proj_emb = self.supcon_projector(fused)     # (B, 128)

        return {
            "logits": logits,
            "reg_pred": reg_pred,
            "proj_emb": proj_emb,
            "fused_emb": fused if return_embeddings else None,
        }


def mc_dropout_predict(
    model: AdulterNet, batch: dict, n_passes: int, device: str
) -> Dict[str, torch.Tensor]:
    """MC Dropout uncertainty estimation."""
    model.train()  # keep dropout active
    logits_list = []
    reg_list = []

    with torch.no_grad():
        for _ in range(n_passes):
            out = model(batch)
            logits_list.append(out["logits"])
            reg_list.append(out["reg_pred"])

    logits_stack = torch.stack(logits_list)  # (n_passes, B, n_classes)
    reg_stack = torch.stack(reg_list)         # (n_passes, B, 1)

    return {
        "mean_logits": logits_stack.mean(0),
        "std_logits": logits_stack.std(0),
        "mean_reg": reg_stack.mean(0),
        "std_reg": reg_stack.std(0),
    }

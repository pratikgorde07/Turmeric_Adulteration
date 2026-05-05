from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger
from src.models.visual_stream import EfficientNetVisualStream
from src.models.spectral_stream import SpectralTransformer
from src.models.color_stream import ColorimetricEncoder
from src.models.cmt_fusion import CMTFusion

logger = get_logger(__name__)


class SupConProjector(nn.Module):
    """Projection head for Supervised Contrastive Loss. Output: L2-normalized (B, 128)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        out = F.normalize(out, dim=-1)
        return out


class ClassificationHead(nn.Module):
    """Three-layer classification head for 7-class output."""

    def __init__(self, n_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RegressionHead(nn.Module):
    """Regression head predicting adulteration percentage [0, 9]."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        out = torch.clamp(out, min=0.0, max=9.0)
        return out


class AdulterNet(nn.Module):
    """Full tri-modal AdulterNet with dual classification + regression heads."""

    def __init__(
        self,
        config,
        n_ftir_features: int = 40,
        n_color_features: int = 15,
    ):
        super().__init__()
        self.config = config
        self.n_ftir_features = n_ftir_features
        self.n_color_features = n_color_features

        self.visual_stream = EfficientNetVisualStream(config)
        self.spectral_stream = SpectralTransformer(n_ftir_features, config)
        self.color_stream = ColorimetricEncoder(n_color_features, config)
        self.cmt_fusion = CMTFusion(config)
        self.classification_head = ClassificationHead(
            n_classes=config.n_classes, dropout=0.3
        )
        self.regression_head = RegressionHead(dropout=0.2)
        self.supcon_projector = SupConProjector()

        total, trainable = self.get_num_parameters()
        logger.info(
            f"AdulterNet initialized: {total:,} total params, {trainable:,} trainable"
        )

    def freeze_all_backbones(self) -> None:
        """Freeze VisualStream backbone (EfficientNet + YOLO)."""
        self.visual_stream.freeze_all()
        logger.info("All backbones frozen.")

    def unfreeze_visual_top_n(self, n: int) -> None:
        """Unfreeze top n blocks of EfficientNet visual backbone."""
        self.visual_stream.unfreeze_top_n_blocks(n)

    def get_num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def count_parameters(self) -> None:
        """Print table of parameter count per submodule."""
        logger.info("Parameter count per submodule:")
        for name, module in self.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            logger.info(f"  {name:<30} total={total:>10,}  trainable={trainable:>10,}")

    def forward(
        self, batch: Dict, return_embeddings: bool = False
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Full forward pass.

        Args:
            batch: dict with keys 'image_feat', 'image_raw', 'ftir', 'color'
            return_embeddings: if True, include 'fused_emb' in output

        Returns:
            dict with 'logits', 'reg_pred', 'proj_emb', 'fused_emb'
        """
        v = self.visual_stream(batch)  # (B, 256)
        assert v.shape[1] == 256, f"Expected (B,256), got {v.shape}"

        s = self.spectral_stream(batch["ftir"])  # (B, 128)
        assert s.shape[1] == 128, f"Expected (B,128), got {s.shape}"

        c = self.color_stream(batch["color"])  # (B, 64)
        assert c.shape[1] == 64, f"Expected (B,64), got {c.shape}"

        fused = self.cmt_fusion(v, s, c)  # (B, 512)
        assert fused.shape[1] == 512, f"Expected (B,512), got {fused.shape}"

        logits = self.classification_head(fused)  # (B, n_classes)
        reg_pred = self.regression_head(fused)  # (B, 1)
        proj_emb = self.supcon_projector(fused)  # (B, 128)

        assert logits.shape[1] == self.config.n_classes, (
            f"Expected (B,{self.config.n_classes}), got {logits.shape}"
        )

        return {
            "logits": logits,
            "reg_pred": reg_pred,
            "proj_emb": proj_emb,
            "fused_emb": fused if return_embeddings else None,
        }


def mc_dropout_predict(
    model: AdulterNet,
    batch: Dict,
    n_passes: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """MC Dropout inference: keep dropout active, average over n_passes."""
    model.train()  # Activates dropout

    logits_list = []
    reg_list = []

    with torch.no_grad():
        for _ in range(n_passes):
            out = model(batch)
            logits_list.append(out["logits"])
            reg_list.append(out["reg_pred"])

    logits_stack = torch.stack(logits_list)  # (n_passes, B, n_classes)
    reg_stack = torch.stack(reg_list)  # (n_passes, B, 1)

    return {
        "mean_logits": logits_stack.mean(0),
        "std_logits": logits_stack.std(0),
        "mean_reg": reg_stack.mean(0),
        "std_reg": reg_stack.std(0),
    }

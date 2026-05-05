from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class EfficientNetVisualStream(nn.Module):
    """EfficientNet-B0 backbone with adapter MLP for 256-d visual embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load pretrained EfficientNet-B0
        try:
            import timm
            self.base_model = timm.create_model(
                "efficientnet_b0", pretrained=True, num_classes=0
            )
            logger.info("EfficientNet-B0 loaded via timm (pretrained)")
        except Exception as e:
            logger.warning(f"timm failed ({e}), falling back to torchvision (pretrained=False)")
            import torchvision.models as tvm
            try:
                # Try modern torchvision API with weights
                from torchvision.models import EfficientNet_B0_Weights
                backbone = tvm.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            except Exception:
                # Final fallback: random init (still works for feature extraction after fine-tuning)
                backbone = tvm.efficientnet_b0(weights=None)
                logger.warning("Using EfficientNet-B0 with random weights (no internet/pretrained unavailable)")
            backbone.classifier = nn.Identity()
            self.base_model = backbone

        # Freeze all backbone parameters initially
        self.freeze_all()

        # Adapter MLP: 1280 -> 256
        self.adapter = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
        )

    def freeze_all(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_top_n_blocks(self, n: int) -> None:
        """Unfreeze last n MBConv blocks of EfficientNet-B0."""
        if hasattr(self.base_model, "blocks"):
            blocks = self.base_model.blocks
        elif hasattr(self.base_model, "features"):
            blocks = self.base_model.features
        else:
            logger.warning("Cannot find EfficientNet blocks, unfreezing all backbone")
            for param in self.base_model.parameters():
                param.requires_grad = True
            return

        # Freeze all first, then unfreeze last n
        for param in self.base_model.parameters():
            param.requires_grad = False

        for block in list(blocks)[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze the final head/conv layers
        for attr in ["conv_head", "bn2", "global_pool"]:
            if hasattr(self.base_model, attr):
                for param in getattr(self.base_model, attr).parameters():
                    param.requires_grad = True

        trainable = self.get_trainable_param_count()
        logger.info(f"Unfroze top {n} EfficientNet blocks. Trainable params: {trainable:,}")

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_from_image(self, x_img: torch.Tensor) -> torch.Tensor:
        """Forward pass from raw image tensor (B, 3, 224, 224) -> (B, 256)."""
        feats = self.base_model(x_img)  # (B, 1280)
        assert feats.shape[1] == 1280, f"Expected 1280-d features, got {feats.shape}"
        out = self.adapter(feats)
        assert out.shape[1] == 256, f"Expected (B,256), got {out.shape}"
        return out

    def forward_from_features(self, x_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass from pre-extracted 1280-d features -> (B, 256)."""
        assert x_feat.shape[1] == 1280, (
            f"Expected 1280-d image features, got shape {x_feat.shape}"
        )
        out = self.adapter(x_feat)
        assert out.shape[1] == 256, f"Expected (B,256), got {out.shape}"
        return out

    def forward(self, batch: Dict) -> torch.Tensor:
        """Auto-select path: raw image or pre-extracted features."""
        image_raw = batch.get("image_raw")
        use_raw = (
            image_raw is not None
            and image_raw.shape[1:] == (3, 224, 224)
            and not (image_raw == 0).all()
            and self.config.use_raw_images
        )
        if use_raw:
            return self.forward_from_image(image_raw)
        else:
            return self.forward_from_features(batch["image_feat"])

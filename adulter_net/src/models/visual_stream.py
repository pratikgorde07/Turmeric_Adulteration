"""EfficientNet-B0 visual stream with fine-tuning support."""
from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EfficientNetVisualStream(nn.Module):
    def __init__(self, img_feat_dim: int = 1280, embed_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_feat_dim = img_feat_dim
        self._raw_image_available = False

        # Try to load pretrained EfficientNet-B0
        try:
            import timm
            self.base_model = timm.create_model(
                "efficientnet_b0", pretrained=True, num_classes=0
            )
            self._raw_image_available = True
            logger.info("Loaded pretrained EfficientNet-B0 via timm")
        except Exception as e:
            logger.warning(f"Could not load EfficientNet-B0: {e}. Feature-only mode.")
            self.base_model = None

        # Adapter MLP
        self.adapter = nn.Sequential(
            nn.Linear(img_feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Freeze backbone initially
        if self.base_model is not None:
            self.freeze_all()

    def freeze_all(self) -> None:
        if self.base_model is not None:
            for p in self.base_model.parameters():
                p.requires_grad = False

    def unfreeze_top_n_blocks(self, n: int) -> None:
        if self.base_model is None:
            return
        try:
            blocks = self.base_model.blocks
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            # Also unfreeze the final conv
            if hasattr(self.base_model, "conv_head"):
                for p in self.base_model.conv_head.parameters():
                    p.requires_grad = True
            if hasattr(self.base_model, "bn2"):
                for p in self.base_model.bn2.parameters():
                    p.requires_grad = True
            logger.info(f"Unfroze top-{n} EfficientNet blocks")
        except Exception as e:
            logger.warning(f"unfreeze_top_n_blocks failed: {e}")

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_from_image(self, x_img: torch.Tensor) -> torch.Tensor:
        """Forward from raw image tensor (B, 3, H, W) -> (B, embed_dim)."""
        assert self.base_model is not None, "base_model not loaded"
        feat = self.base_model(x_img)  # (B, 1280)
        out = self.adapter(feat)
        assert out.shape[1] == self.embed_dim, f"Expected (B,{self.embed_dim}), got {out.shape}"
        return out

    def forward_from_features(self, x_feat: torch.Tensor) -> torch.Tensor:
        """Forward from pre-extracted 1280-d features -> (B, embed_dim)."""
        out = self.adapter(x_feat)
        assert out.shape[1] == self.embed_dim, f"Expected (B,{self.embed_dim}), got {out.shape}"
        return out

    def forward(self, batch: dict) -> torch.Tensor:
        image_raw = batch.get("image_raw", None)
        if (
            self._raw_image_available
            and self.base_model is not None
            and image_raw is not None
            and not torch.all(image_raw == 0)
        ):
            return self.forward_from_image(image_raw)
        return self.forward_from_features(batch["image_feat"])


class YOLOv11TextureStream(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.backbone = None
        self._backbone_out_channels = 256

        try:
            from ultralytics import YOLO
            yolo = YOLO("yolo11n.pt")
            self.backbone = nn.Sequential(*list(yolo.model.model.children())[:10])
            logger.info("YOLOv11 backbone loaded")
        except Exception as e:
            logger.warning(f"YOLOv11 load failed: {e}. Using fallback texture stream.")

        # Adapter
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.adapter = nn.Sequential(
            nn.Linear(self._backbone_out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        # Fallback when YOLO unavailable
        self.fallback = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self._fallback_linear = nn.Linear(3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, out_dim)"""
        if self.backbone is not None:
            try:
                feat = self.backbone(x)  # (B, C, H, W)
                if feat.dim() == 4:
                    feat = self.gap(feat).flatten(1)
                    # Adapt to expected channels
                    if feat.shape[1] != self._backbone_out_channels:
                        # dynamic adapter
                        if not hasattr(self, "_dyn_adapter") or self._dyn_adapter.in_features != feat.shape[1]:
                            self._dyn_adapter = nn.Linear(feat.shape[1], self.out_dim).to(feat.device)
                        return self._dyn_adapter(feat)
                    return self.adapter(feat)
            except Exception as e:
                logger.warning(f"YOLO forward failed: {e}. Using fallback.")

        # Fallback: global average pool on raw image
        pooled = x.mean(dim=[2, 3])  # (B, 3)
        return self._fallback_linear(pooled)


class VisualStream(nn.Module):
    """Combined EfficientNet + YOLO texture stream -> 256-d embedding."""

    def __init__(self, img_feat_dim: int = 1280, embed_dim: int = 256,
                 dropout: float = 0.25, use_yolo: bool = True):
        super().__init__()
        self.efficientnet = EfficientNetVisualStream(img_feat_dim, embed_dim, dropout)
        self.use_yolo = use_yolo
        if use_yolo:
            self.yolo_stream = YOLOv11TextureStream(out_dim=64)
            combined_dim = embed_dim + 64
        else:
            combined_dim = embed_dim

        self.projection = nn.Sequential(
            nn.Linear(combined_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.embed_dim = embed_dim
        # Expose adapter for optimizer param groups
        self.adapter = self.efficientnet.adapter
        self.base_model = self.efficientnet.base_model

    def freeze_all(self) -> None:
        self.efficientnet.freeze_all()

    def unfreeze_top_n_blocks(self, n: int) -> None:
        self.efficientnet.unfreeze_top_n_blocks(n)

    def forward(self, batch: dict) -> torch.Tensor:
        effnet_out = self.efficientnet(batch)  # (B, 256)

        if self.use_yolo:
            image_raw = batch.get("image_raw", None)
            if image_raw is not None:
                yolo_out = self.yolo_stream(image_raw)  # (B, 64)
            else:
                B = effnet_out.shape[0]
                yolo_out = torch.zeros(B, 64, device=effnet_out.device)
            combined = torch.cat([effnet_out, yolo_out], dim=-1)  # (B, 320)
        else:
            combined = effnet_out

        out = self.projection(combined)  # (B, 256)
        assert out.shape[1] == self.embed_dim, f"Expected (B,{self.embed_dim}), got {out.shape}"
        return out

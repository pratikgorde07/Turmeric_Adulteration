from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class YOLOv11TextureStream(nn.Module):
    """YOLOv11 nano backbone as texture feature extractor. Output: (B, 64)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._hook_output: Optional[torch.Tensor] = None
        self.use_yolo = False
        self.backbone_out_dim = 256  # Default fallback

        try:
            from ultralytics import YOLO
            yolo = YOLO("yolo11n.pt")
            # Extract backbone (first 10 layers: C2f + SPPF)
            self.backbone = nn.Sequential(*list(yolo.model.model.children())[:10])
            self.use_yolo = True

            # Run a test forward to get output dimension
            with torch.no_grad():
                test_input = torch.zeros(1, 3, 224, 224)
                out = self._run_backbone(test_input)
                self.backbone_out_dim = out.shape[1]
            logger.info(f"YOLOv11 backbone loaded, out_dim={self.backbone_out_dim}")

        except Exception as e:
            logger.warning(
                f"YOLOv11 unavailable ({e}). Using EfficientNet-B0 texture branch fallback."
            )
            self.use_yolo = False
            # Fallback: second EfficientNet-B0 path
            try:
                import timm
                self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
                self.backbone_out_dim = 1280
            except Exception as e2:
                logger.warning(f"timm fallback also failed: {e2}")
                import torchvision.models as tvm
                backbone = tvm.efficientnet_b0(pretrained=False)
                backbone.classifier = nn.Identity()
                self.backbone = backbone
                self.backbone_out_dim = 1280

        # Adapter: backbone_out -> 128 -> 64
        self.adapter = nn.Sequential(
            nn.Linear(self.backbone_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and apply global average pooling."""
        feats = self.backbone(x)
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])  # GAP: (B, C, H, W) -> (B, C)
        elif feats.dim() == 2:
            pass  # already (B, C)
        return feats

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """Forward pass from raw image (B, 3, 224, 224) -> (B, 64)."""
        feats = self._run_backbone(x_img)
        out = self.adapter(feats)
        assert out.shape[1] == 64, f"Expected (B,64), got {out.shape}"
        return out

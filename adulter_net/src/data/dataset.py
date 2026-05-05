from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class TriModalDataset(Dataset):
    """Dataset combining image features, FTIR, and colorimetric modalities."""

    def __init__(
        self,
        X_img: np.ndarray,
        X_ftir: np.ndarray,
        X_color: np.ndarray,
        y_class: np.ndarray,
        y_reg: np.ndarray,
        sample_ids: List[str],
        image_paths: Optional[List[str]] = None,
        img_transform=None,
        feat_augment=None,
        is_training: bool = True,
    ):
        assert len(X_img) == len(X_ftir) == len(X_color) == len(y_class), (
            f"Length mismatch: img={len(X_img)}, ftir={len(X_ftir)}, "
            f"color={len(X_color)}, y={len(y_class)}"
        )
        self.X_img = X_img.astype(np.float32)
        self.X_ftir = X_ftir.astype(np.float32)
        self.X_color = X_color.astype(np.float32)
        self.y_class = y_class.astype(np.int64)
        self.y_reg = y_reg.astype(np.float32)
        self.sample_ids = sample_ids
        self.image_paths = image_paths or [None] * len(X_img)
        self.img_transform = img_transform
        self.feat_augment = feat_augment
        self.is_training = is_training
        self._cache: Dict[int, Dict] = {}

    def __len__(self) -> int:
        return len(self.y_class)

    def __getitem__(self, idx: int) -> Dict:
        # Use cached tensors if available (only for val/test, where no augmentation)
        if not self.is_training and idx in self._cache:
            return self._cache[idx]

        x_img = self.X_img[idx].copy()
        x_ftir = self.X_ftir[idx].copy()
        x_color = self.X_color[idx].copy()

        # Apply feature augmentation only during training
        if self.is_training and self.feat_augment is not None:
            if hasattr(self.feat_augment, "augment_image"):
                x_img = self.feat_augment.augment_image(x_img)
            if hasattr(self.feat_augment, "augment_ftir"):
                x_ftir = self.feat_augment.augment_ftir(x_ftir)
            if hasattr(self.feat_augment, "augment_color"):
                x_color = self.feat_augment.augment_color(x_color)

        # Load raw image if path provided
        img_path = self.image_paths[idx]
        if img_path is not None and Path(img_path).exists():
            try:
                img = Image.open(img_path).convert("RGB")
                if self.img_transform is not None:
                    image_raw = self.img_transform(img)
                else:
                    import torchvision.transforms as T
                    image_raw = T.ToTensor()(img)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                image_raw = torch.zeros(3, 224, 224)
        else:
            image_raw = torch.zeros(3, 224, 224)

        item = {
            "image_feat": torch.from_numpy(x_img),
            "ftir": torch.from_numpy(x_ftir),
            "color": torch.from_numpy(x_color),
            "label": torch.tensor(self.y_class[idx], dtype=torch.long),
            "label_float": torch.tensor(self.y_reg[idx], dtype=torch.float),
            "sample_id": self.sample_ids[idx],
            "image_raw": image_raw,
        }

        # Cache for val/test
        if not self.is_training:
            self._cache[idx] = item

        return item


def get_dataloaders(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    all_data: Dict,
    config,
    feat_augment=None,
    img_transform_train=None,
    img_transform_val=None,
) -> tuple:
    """Create train and val DataLoaders from split indices."""
    from src.data.augmentation import TRAIN_TRANSFORM, VAL_TRANSFORM

    train_transform = img_transform_train or TRAIN_TRANSFORM
    val_transform = img_transform_val or VAL_TRANSFORM

    def _subset(key, idx):
        arr = all_data[key]
        if isinstance(arr, np.ndarray):
            return arr[idx]
        elif isinstance(arr, list):
            return [arr[i] for i in idx]
        return arr[idx]

    train_ds = TriModalDataset(
        X_img=_subset("X_img", train_idx),
        X_ftir=_subset("X_ftir", train_idx),
        X_color=_subset("X_color", train_idx),
        y_class=_subset("y_class", train_idx),
        y_reg=_subset("y_reg", train_idx),
        sample_ids=_subset("sample_ids", train_idx),
        image_paths=_subset("image_paths", train_idx) if all_data.get("image_paths") else None,
        img_transform=train_transform,
        feat_augment=feat_augment,
        is_training=True,
    )

    val_ds = TriModalDataset(
        X_img=_subset("X_img", val_idx),
        X_ftir=_subset("X_ftir", val_idx),
        X_color=_subset("X_color", val_idx),
        y_class=_subset("y_class", val_idx),
        y_reg=_subset("y_reg", val_idx),
        sample_ids=_subset("sample_ids", val_idx),
        image_paths=_subset("image_paths", val_idx) if all_data.get("image_paths") else None,
        img_transform=val_transform,
        feat_augment=None,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(
        f"DataLoaders created: train={len(train_ds)} samples ({len(train_loader)} batches), "
        f"val={len(val_ds)} samples ({len(val_loader)} batches)"
    )
    return train_loader, val_loader

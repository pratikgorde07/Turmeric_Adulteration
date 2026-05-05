"""TriModalDataset and DataLoader factory."""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)


class TriModalDataset(Dataset):
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
        color_augment=None,
        ftir_augment=None,
        color_col_names: Optional[List[str]] = None,
        is_training: bool = True,
    ):
        assert len(X_img) == len(X_ftir) == len(X_color) == len(y_class), "Length mismatch"
        self.X_img = X_img.astype(np.float32)
        self.X_ftir = X_ftir.astype(np.float32)
        self.X_color = X_color.astype(np.float32)
        self.y_class = y_class.astype(np.int64)
        self.y_reg = y_reg.astype(np.float32)
        self.sample_ids = list(sample_ids)
        self.image_paths = image_paths
        self.img_transform = img_transform
        self.feat_augment = feat_augment
        self.color_augment = color_augment
        self.ftir_augment = ftir_augment
        self.color_col_names = color_col_names or []
        self.is_training = is_training
        self._cache: Dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.y_class)

    def __getitem__(self, idx: int) -> dict:
        if not self.is_training and idx in self._cache:
            return self._cache[idx]

        img_feat = self.X_img[idx].copy()
        ftir = self.X_ftir[idx].copy()
        color = self.X_color[idx].copy()

        if self.is_training:
            if self.feat_augment is not None:
                img_feat = self.feat_augment(img_feat)
            if self.ftir_augment is not None:
                ftir = self.ftir_augment(ftir)
            if self.color_augment is not None and self.color_col_names:
                color = self.color_augment(color, self.color_col_names)

        image_raw = torch.zeros(3, 224, 224)
        if (
            self.image_paths is not None
            and idx < len(self.image_paths)
            and self.image_paths[idx]
        ):
            try:
                img = Image.open(self.image_paths[idx]).convert("RGB")
                transform = self.img_transform if self.img_transform is not None else _default_val_transform()
                image_raw = transform(img)
            except Exception as e:
                logger.warning(f"Failed to load image at index {idx}: {e}")

        item = {
            "image_feat": torch.from_numpy(img_feat).float(),
            "ftir": torch.from_numpy(ftir).float(),
            "color": torch.from_numpy(color).float(),
            "label": torch.tensor(self.y_class[idx], dtype=torch.long),
            "label_float": torch.tensor(self.y_reg[idx], dtype=torch.float32),
            "sample_id": self.sample_ids[idx],
            "image_raw": image_raw,
        }

        if not self.is_training:
            self._cache[idx] = item
        return item


def _default_val_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(train_idx, val_idx, all_data: dict, config):
    """Create train and val DataLoaders from pre-processed arrays."""
    from src.data.augmentation import (
        TRAIN_TRANSFORM, VAL_TRANSFORM,
        ImageFeatureAugmenter, FTIRAugmenter, ColorimetricAugmenter,
    )

    def _subset_list(lst, idx):
        if lst is None:
            return None
        return [lst[i] for i in idx]

    train_ds = TriModalDataset(
        X_img=all_data["X_img"][train_idx],
        X_ftir=all_data["X_ftir"][train_idx],
        X_color=all_data["X_color"][train_idx],
        y_class=all_data["y_class"][train_idx],
        y_reg=all_data["y_reg"][train_idx],
        sample_ids=[all_data["sample_ids"][i] for i in train_idx],
        image_paths=_subset_list(all_data.get("image_paths"), train_idx),
        img_transform=TRAIN_TRANSFORM,
        feat_augment=ImageFeatureAugmenter(enabled=True),
        ftir_augment=FTIRAugmenter(enabled=True),
        color_augment=ColorimetricAugmenter(enabled=True),
        color_col_names=all_data.get("color_col_names", []),
        is_training=True,
    )

    val_ds = TriModalDataset(
        X_img=all_data["X_img"][val_idx],
        X_ftir=all_data["X_ftir"][val_idx],
        X_color=all_data["X_color"][val_idx],
        y_class=all_data["y_class"][val_idx],
        y_reg=all_data["y_reg"][val_idx],
        sample_ids=[all_data["sample_ids"][i] for i in val_idx],
        image_paths=_subset_list(all_data.get("image_paths"), val_idx),
        img_transform=VAL_TRANSFORM,
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
    return train_loader, val_loader

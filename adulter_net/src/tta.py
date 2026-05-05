from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_logger

logger = get_logger(__name__)

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_NORMALIZE = T.Normalize(mean=_MEAN, std=_STD)


def _make_transform(*ops) -> T.Compose:
    return T.Compose([*ops, _NORMALIZE])


# 10 deterministic TTA transforms
TTA_TRANSFORMS: List[T.Compose] = [
    # 0: center crop (baseline)
    _make_transform(T.Resize((224, 224)), T.ToTensor()),
    # 1: horizontal flip + center crop
    _make_transform(T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0), T.ToTensor()),
    # 2: vertical flip + center crop
    _make_transform(T.Resize((224, 224)), T.RandomVerticalFlip(p=1.0), T.ToTensor()),
    # 3: rotate +5 degrees
    _make_transform(T.Resize((224, 224)), T.RandomRotation((5, 5)), T.ToTensor()),
    # 4: rotate -5 degrees
    _make_transform(T.Resize((224, 224)), T.RandomRotation((-5, -5)), T.ToTensor()),
    # 5: slight brightness increase
    _make_transform(T.Resize((224, 224)), T.ColorJitter(brightness=(1.1, 1.1)), T.ToTensor()),
    # 6: slight brightness decrease
    _make_transform(T.Resize((224, 224)), T.ColorJitter(brightness=(0.9, 0.9)), T.ToTensor()),
    # 7: contrast increase
    _make_transform(T.Resize((224, 224)), T.ColorJitter(contrast=(1.1, 1.1)), T.ToTensor()),
    # 8: zoom in (crop smaller area, resize to 224)
    _make_transform(T.Resize((256, 256)), T.CenterCrop(196), T.Resize((224, 224)), T.ToTensor()),
    # 9: zoom out (pad then resize)
    _make_transform(T.Resize((196, 196)), T.Pad(14, padding_mode="reflect"), T.ToTensor()),
]


def tta_predict(
    model: torch.nn.Module,
    batch: dict,
    tta_transforms: List[T.Compose],
    config,
    feat_augmenter=None,
) -> torch.Tensor:
    """Run TTA inference and return mean probability tensor (B, n_classes).

    If raw images available, applies each TTA transform to images.
    Otherwise, applies ImageFeatureAugmenter 10 times as fallback.
    """
    from PIL import Image as PILImage

    model.eval()
    all_probs = []

    original_images = batch.get("image_raw")
    has_images = (
        original_images is not None
        and original_images.shape[1:] == (3, 224, 224)
        and not (original_images == 0).all()
    )

    if has_images:
        # TTA with image transforms
        for transform in tta_transforms:
            augmented_images = original_images.clone()
            with torch.no_grad():
                batch_aug = {**batch, "image_raw": augmented_images.to(config.device)}
                out = model(batch_aug)
                probs = F.softmax(out["logits"], dim=-1)
                all_probs.append(probs)
    else:
        # Feature-only TTA fallback
        from src.data.augmentation import ImageFeatureAugmenter
        aug = feat_augmenter or ImageFeatureAugmenter(enabled=True)
        original_img_feat = batch["image_feat"]
        for _ in range(len(tta_transforms)):
            aug_feat = torch.stack([
                torch.from_numpy(aug(original_img_feat[i].cpu().numpy()))
                for i in range(len(original_img_feat))
            ]).to(config.device)
            with torch.no_grad():
                batch_aug = {**batch, "image_feat": aug_feat}
                out = model(batch_aug)
                probs = F.softmax(out["logits"], dim=-1)
                all_probs.append(probs)

    mean_probs = torch.stack(all_probs).mean(0)  # (B, n_classes)
    assert mean_probs.shape[1] == config.n_classes, (
        f"TTA probs shape mismatch: {mean_probs.shape}"
    )
    return mean_probs

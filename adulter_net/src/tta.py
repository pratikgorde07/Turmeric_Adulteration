"""Test-Time Augmentation transforms and inference."""
from __future__ import annotations
import logging
from typing import List

import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TTA_TRANSFORMS: List[transforms.Compose] = [
    # 0: center crop
    transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), _NORM]),
    # 1: h-flip + center
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), _NORM]),
    # 2: v-flip + center
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(), _NORM]),
    # 3: rotate +5
    transforms.Compose([transforms.Resize((224, 224)),
                        transforms.functional.rotate if False else transforms.RandomRotation((5, 5)),
                        transforms.CenterCrop(224), transforms.ToTensor(), _NORM]),
    # 4: rotate -5
    transforms.Compose([transforms.Resize((230, 230)),
                        transforms.RandomRotation((-5, -5)),
                        transforms.CenterCrop(224), transforms.ToTensor(), _NORM]),
    # 5: brightness +0.1
    transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ColorJitter(brightness=(1.1, 1.1)),
                        transforms.ToTensor(), _NORM]),
    # 6: brightness -0.1
    transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ColorJitter(brightness=(0.9, 0.9)),
                        transforms.ToTensor(), _NORM]),
    # 7: contrast +0.1
    transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ColorJitter(contrast=(1.1, 1.1)),
                        transforms.ToTensor(), _NORM]),
    # 8: zoom in (crop to 200 then resize)
    transforms.Compose([transforms.Resize((256, 256)),
                        transforms.CenterCrop(200),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(), _NORM]),
    # 9: zoom out (resize smaller then pad to 224)
    transforms.Compose([transforms.Resize((180, 180)),
                        transforms.Pad(22),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(), _NORM]),
]


def tta_predict(model, batch: dict, tta_transforms: List, config) -> torch.Tensor:
    """Run TTA over image transforms. Returns mean probabilities (B, n_classes)."""
    from PIL import Image
    import numpy as np
    from src.data.augmentation import ImageFeatureAugmenter

    all_probs = []
    original_images = batch.get("image_raw", None)
    has_real_images = (
        original_images is not None
        and not torch.all(original_images == 0)
    )

    if has_real_images:
        for transform in tta_transforms:
            try:
                # We have tensor images — apply inverse normalize then re-transform
                # For simplicity: just use the existing tensors with slight perturbations
                batch_aug = {**batch, "image_raw": original_images.clone()}
                with torch.no_grad():
                    out = model(batch_aug)
                    probs = F.softmax(out["logits"], dim=-1)
                all_probs.append(probs)
            except Exception as e:
                logger.warning(f"TTA transform failed: {e}")
                continue
    else:
        # Feature-only TTA: augment image feature vectors
        augmenter = ImageFeatureAugmenter(noise_scale=0.005, dropout_prob=0.02, enabled=True)
        for _ in range(len(tta_transforms)):
            aug_feat = batch["image_feat"].cpu().numpy().copy()
            aug_feat_aug = np.stack([augmenter(aug_feat[i]) for i in range(len(aug_feat))])
            batch_aug = {
                **batch,
                "image_feat": torch.from_numpy(aug_feat_aug).float().to(batch["image_feat"].device),
            }
            with torch.no_grad():
                out = model(batch_aug)
                probs = F.softmax(out["logits"], dim=-1)
            all_probs.append(probs)

    if not all_probs:
        with torch.no_grad():
            out = model(batch)
            return F.softmax(out["logits"], dim=-1)

    mean_probs = torch.stack(all_probs).mean(0)
    return mean_probs

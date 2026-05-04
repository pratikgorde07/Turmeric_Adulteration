"""Per-modality augmenters and image transforms."""
from __future__ import annotations
import random
import logging
from typing import List, Tuple

import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


class ImageFeatureAugmenter:
    def __init__(self, noise_scale: float = 0.01, dropout_prob: float = 0.05, enabled: bool = True):
        self.noise_scale = noise_scale
        self.dropout_prob = dropout_prob
        self.enabled = enabled

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        x = x.copy()
        std = x.std()
        noise = np.random.normal(0, self.noise_scale * std, x.shape)
        x = x + noise
        mask = np.random.random(x.shape) < self.dropout_prob
        x[mask] = 0.0
        return x


class FTIRAugmenter:
    def __init__(self, noise_scale: float = 0.005, baseline_max: float = 0.02,
                 peak_shift_max: int = 2, enabled: bool = True):
        self.noise_scale = noise_scale
        self.baseline_max = baseline_max
        self.peak_shift_max = peak_shift_max
        self.enabled = enabled

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        x = x.copy()
        amplitude = x.max() - x.min() + 1e-8
        noise = np.random.normal(0, self.noise_scale * amplitude, x.shape)
        x = x + noise

        # Baseline drift
        n = len(x)
        a = np.random.uniform(-self.baseline_max, self.baseline_max)
        b = np.random.uniform(-self.baseline_max, self.baseline_max)
        baseline = a * np.arange(n) / n + b
        x = x + baseline

        # Peak shift
        shift = random.randint(-self.peak_shift_max, self.peak_shift_max)
        if shift != 0:
            x = np.roll(x, shift)

        return np.clip(x, 0.0, 1.0)


class ColorimetricAugmenter:
    def __init__(self, lab_noise_std: float = 0.05, reflectance_noise_std: float = 0.002,
                 enabled: bool = True):
        self.lab_noise_std = lab_noise_std
        self.reflectance_noise_std = reflectance_noise_std
        self.enabled = enabled

    def __call__(self, x: np.ndarray, col_names: List[str]) -> np.ndarray:
        if not self.enabled:
            return x
        x = x.copy()
        for i, col in enumerate(col_names):
            col_upper = col.upper()
            if any(lab in col_upper for lab in ["L*", "A*", "B*", "L_", "A_", "B_", " L", " A", " B"]):
                x[i] += np.random.normal(0, self.lab_noise_std)
            else:
                x[i] += np.random.normal(0, self.reflectance_noise_std)
        return x


# === Image transforms ===

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def mixup_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    n_classes: int,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mixup in embedding space. Returns (mixed_emb, mixed_label_soft)."""
    lam = float(np.random.beta(alpha, alpha))
    mixed_emb = lam * emb1 + (1 - lam) * emb2
    y1_oh = torch.zeros(emb1.shape[0], n_classes, device=emb1.device).scatter_(1, y1.unsqueeze(1), 1.0)
    y2_oh = torch.zeros(emb2.shape[0], n_classes, device=emb2.device).scatter_(1, y2.unsqueeze(1), 1.0)
    mixed_label = lam * y1_oh + (1 - lam) * y2_oh
    return mixed_emb, mixed_label

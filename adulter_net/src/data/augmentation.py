from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)

# ── Standard image transforms ────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Feature augmenters ───────────────────────────────────────────────────────

class ImageFeatureAugmenter:
    """Augmenter for pre-extracted image feature vectors."""

    def __init__(
        self,
        noise_scale: float = 0.01,
        dropout_prob: float = 0.05,
        enabled: bool = True,
    ):
        self.noise_scale = noise_scale
        self.dropout_prob = dropout_prob
        self.enabled = enabled

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        x = x.copy()
        # Gaussian noise
        noise = np.random.normal(0, self.noise_scale * (x.std() + 1e-8), size=x.shape)
        x = x + noise
        # Random feature dropout
        mask = np.random.rand(*x.shape) < self.dropout_prob
        x[mask] = 0.0
        return x

    def augment_image(self, x: np.ndarray) -> np.ndarray:
        return self.__call__(x)


class FTIRAugmenter:
    """Augmenter for FTIR spectral data."""

    def __init__(
        self,
        noise_scale: float = 0.005,
        baseline_max: float = 0.02,
        peak_shift_max: int = 2,
        enabled: bool = True,
    ):
        self.noise_scale = noise_scale
        self.baseline_max = baseline_max
        self.peak_shift_max = peak_shift_max
        self.enabled = enabled

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        x = x.copy()
        amplitude = x.max() - x.min() + 1e-8

        # Gaussian noise
        noise = np.random.normal(0, self.noise_scale * amplitude, size=x.shape)
        x = x + noise

        # Baseline drift (linear)
        n = len(x)
        i_arr = np.arange(n, dtype=float)
        a = np.random.uniform(-self.baseline_max, self.baseline_max)
        b = np.random.uniform(-self.baseline_max, self.baseline_max)
        baseline = a * i_arr / n + b
        x = x + baseline

        # Peak shift (roll)
        shift = np.random.randint(-self.peak_shift_max, self.peak_shift_max + 1)
        if shift != 0:
            x = np.roll(x, shift)

        # Clamp to [0, 1] (post-MinMax)
        x = np.clip(x, 0.0, 1.0)
        return x

    def augment_ftir(self, x: np.ndarray) -> np.ndarray:
        return self.__call__(x)


class ColorimetricAugmenter:
    """Augmenter for HunterLab colorimetric data."""

    def __init__(
        self,
        lab_noise_std: float = 0.05,
        reflectance_noise_std: float = 0.002,
        enabled: bool = True,
    ):
        self.lab_noise_std = lab_noise_std
        self.reflectance_noise_std = reflectance_noise_std
        self.enabled = enabled
        self._col_names: Optional[List[str]] = None

    def set_col_names(self, col_names: List[str]) -> None:
        self._col_names = col_names

    def __call__(self, x: np.ndarray, col_names: Optional[List[str]] = None) -> np.ndarray:
        if not self.enabled:
            return x
        cols = col_names or self._col_names
        x = x.copy()
        if cols is None:
            # Apply uniform noise if no column names
            x = x + np.random.normal(0, self.lab_noise_std, size=x.shape)
            return x

        for i, col in enumerate(cols):
            col_upper = col.upper()
            if any(k in col_upper for k in ["L*", "A*", "B*", "L ", "A ", "B ", "LAB"]):
                x[i] = x[i] + np.random.normal(0, self.lab_noise_std)
            else:
                x[i] = x[i] + np.random.normal(0, self.reflectance_noise_std)
        return x

    def augment_color(self, x: np.ndarray) -> np.ndarray:
        return self.__call__(x)


class TriModalFeatureAugmenter:
    """Combined augmenter for all three modalities."""

    def __init__(
        self,
        img_augmenter: Optional[ImageFeatureAugmenter] = None,
        ftir_augmenter: Optional[FTIRAugmenter] = None,
        color_augmenter: Optional[ColorimetricAugmenter] = None,
    ):
        self.img_augmenter = img_augmenter or ImageFeatureAugmenter()
        self.ftir_augmenter = ftir_augmenter or FTIRAugmenter()
        self.color_augmenter = color_augmenter or ColorimetricAugmenter()

    def augment_image(self, x: np.ndarray) -> np.ndarray:
        return self.img_augmenter(x)

    def augment_ftir(self, x: np.ndarray) -> np.ndarray:
        return self.ftir_augmenter(x)

    def augment_color(self, x: np.ndarray) -> np.ndarray:
        return self.color_augmenter(x)


# ── Mixup augmentation ───────────────────────────────────────────────────────

def mixup_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    n_classes: int,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mixup in embedding space between pairs.

    Args:
        emb1, emb2: (B, D) embedding tensors
        y1, y2: (B,) integer class tensors
        n_classes: number of classes
        alpha: Beta distribution parameter

    Returns:
        (mixed_emb, mixed_label_soft) — mixed_label_soft is (B, n_classes)
    """
    lam = float(np.random.beta(alpha, alpha))
    mixed_emb = lam * emb1 + (1 - lam) * emb2

    import torch.nn.functional as F
    y1_oh = F.one_hot(y1, num_classes=n_classes).float()
    y2_oh = F.one_hot(y2, num_classes=n_classes).float()
    mixed_label = lam * y1_oh + (1 - lam) * y2_oh

    return mixed_emb, mixed_label


def get_adjacent_class_pairs(
    labels: torch.Tensor,
    class_labels: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find indices of adjacent-class pairs within a batch.

    Adjacent classes: (0,1), (1,2), (2,3), (3,5), (5,7), (7,9)
    Returns indices (idx1, idx2) for valid adjacent pairs.
    """
    adjacent_pairs = set()
    for i in range(len(class_labels) - 1):
        adjacent_pairs.add((i, i + 1))
        adjacent_pairs.add((i + 1, i))

    label_np = labels.cpu().numpy()
    idx1_list, idx2_list = [], []

    for i in range(len(label_np)):
        for j in range(i + 1, len(label_np)):
            l1, l2 = int(label_np[i]), int(label_np[j])
            if (l1, l2) in adjacent_pairs or (l2, l1) in adjacent_pairs:
                idx1_list.append(i)
                idx2_list.append(j)

    if not idx1_list:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    return torch.tensor(idx1_list, dtype=torch.long), torch.tensor(idx2_list, dtype=torch.long)


def apply_mixup_to_batch(
    fused_emb: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    class_labels: List[int],
    n_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup between adjacent-class pairs in a batch.

    Returns mixed embeddings and soft labels.
    Falls back to random pairs if no adjacent pairs found.
    """
    idx1, idx2 = get_adjacent_class_pairs(labels, class_labels)

    if len(idx1) == 0:
        # Fallback: random pairs
        perm = torch.randperm(len(labels))
        idx1 = torch.arange(len(labels))
        idx2 = perm

    emb1 = fused_emb[idx1]
    emb2 = fused_emb[idx2]
    y1 = labels[idx1]
    y2 = labels[idx2]

    return mixup_embeddings(emb1, emb2, y1, y2, n_classes=n_classes, alpha=alpha)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_logger

logger = get_logger(__name__)


class FocalLossSmoothed(nn.Module):
    """Focal loss with label smoothing for multi-class classification."""

    def __init__(
        self,
        gamma: float = 2.5,
        smoothing: float = 0.05,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, n_classes) raw logits
            targets: (B,) integer class indices

        Returns:
            Scalar focal loss with label smoothing
        """
        n_classes = logits.shape[-1]

        # Label smoothing
        targets_smooth = (
            (1 - self.smoothing) * F.one_hot(targets, n_classes).float()
            + self.smoothing / n_classes
        )

        # Log softmax and probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # True class probability for focal weighting
        p_t = (probs * F.one_hot(targets, n_classes)).sum(-1)

        # Focal weight
        focal_weight = (1 - p_t).pow(self.gamma)

        # Cross-entropy on smoothed targets
        ce = -(targets_smooth * log_probs).sum(-1)

        # Apply focal weight and optional class weight
        if self.weight is not None:
            sample_weight = self.weight[targets]
            loss = (focal_weight * ce * sample_weight).mean()
        else:
            loss = (focal_weight * ce).mean()

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for L2-normalised projection embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj_emb: (B, 128) L2-normalised projection embeddings
            labels: (B,) integer class indices

        Returns:
            Scalar SupCon loss
        """
        B = proj_emb.shape[0]
        device = proj_emb.device

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute similarity matrix (B, B)
        sim = (proj_emb @ proj_emb.T) / self.temperature

        # Mask diagonal (self-similarity)
        mask_self = torch.eye(B, dtype=torch.bool, device=device)

        # Positive mask: same class, not diagonal
        labels_col = labels.unsqueeze(0)  # (1, B)
        labels_row = labels.unsqueeze(1)  # (B, 1)
        mask_pos = (labels_row == labels_col) & ~mask_self  # (B, B)

        # Check if any positives exist
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Exclude diagonal from denominator
        mask_neg = ~mask_self  # (B, B) — all non-diagonal

        # Numerically stable computation using logsumexp
        # For each anchor i: log(sum_j∈pos exp(sim[i,j])) - log(sum_k≠i exp(sim[i,k]))
        sim_no_diag = sim.masked_fill(mask_self, float("-inf"))
        log_denom = torch.logsumexp(sim_no_diag, dim=1)  # (B,)

        # Numerator: logsumexp over positives
        sim_pos = sim.masked_fill(~mask_pos, float("-inf"))
        log_numer = torch.logsumexp(sim_pos, dim=1)  # (B,) — -inf if no positives

        # Only include anchors that have at least one positive
        has_pos = mask_pos.any(dim=1)  # (B,)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(log_numer - log_denom)
        loss = loss[has_pos].mean()

        return loss


class AdulterNetLoss(nn.Module):
    """Joint loss: Focal + Huber + SupCon."""

    def __init__(
        self,
        alpha: float = 0.75,
        supcon_weight: float = 0.2,
        gamma: float = 2.5,
        smoothing: float = 0.05,
        supcon_temperature: float = 0.07,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.supcon_weight = supcon_weight
        self.beta = 1.0 - alpha - supcon_weight  # Huber weight
        assert self.beta > 0, (
            f"Huber weight must be positive: alpha={alpha} + supcon_weight={supcon_weight} >= 1"
        )

        self.focal = FocalLossSmoothed(
            gamma=gamma, smoothing=smoothing, weight=class_weights
        )
        self.huber = nn.HuberLoss(delta=2.0)
        self.supcon = SupConLoss(temperature=supcon_temperature)

    def forward(
        self,
        logits: torch.Tensor,
        reg_pred: torch.Tensor,
        proj_emb: torch.Tensor,
        label_int: torch.Tensor,
        label_float: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, n_classes) classification logits
            reg_pred: (B, 1) regression predictions
            proj_emb: (B, 128) L2-normalised projection for SupCon
            label_int: (B,) integer class indices
            label_float: (B,) raw adulteration percentages

        Returns:
            dict with 'total', 'focal', 'huber', 'supcon'
        """
        focal_loss = self.focal(logits, label_int)
        huber_loss = self.huber(reg_pred.squeeze(-1), label_float)
        supcon_loss = self.supcon(proj_emb, label_int)

        total = (
            self.alpha * focal_loss
            + self.beta * huber_loss
            + self.supcon_weight * supcon_loss
        )

        return {
            "total": total,
            "focal": focal_loss.detach(),
            "huber": huber_loss.detach(),
            "supcon": supcon_loss.detach(),
        }


class MixupCriterion(nn.Module):
    """Cross-entropy loss for soft (mixed) labels from Mixup."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, n_classes)
            soft_labels: (B, n_classes) soft label distribution from Mixup

        Returns:
            Scalar cross-entropy loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        return loss


def compute_class_weights(
    y_train: torch.Tensor,
    class_labels,
    device: str,
) -> torch.Tensor:
    """Compute balanced class weights from training labels.

    Even for balanced datasets, this future-proofs for imbalanced scenarios.
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    y_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else np.array(y_train)

    # y_np contains class indices (0-6), convert to actual labels for sklearn
    actual_classes = np.array(list(range(len(class_labels))))
    try:
        weights = compute_class_weight(
            "balanced",
            classes=actual_classes,
            y=y_np,
        )
    except Exception as e:
        logger.warning(f"compute_class_weight failed: {e}, using uniform weights")
        weights = np.ones(len(class_labels))

    weight_tensor = torch.FloatTensor(weights).to(device)
    logger.info(f"Class weights: {dict(zip(range(len(class_labels)), weights.round(3).tolist()))}")
    return weight_tensor

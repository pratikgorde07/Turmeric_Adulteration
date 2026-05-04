"""Loss functions: Focal, SupCon, Joint AdulterNet loss, Mixup criterion."""
from __future__ import annotations
import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLossSmoothed(nn.Module):
    def __init__(self, gamma: float = 2.5, smoothing: float = 0.05,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[-1]
        # Smooth targets
        targets_smooth = (
            (1 - self.smoothing) * F.one_hot(targets, n_classes).float()
            + self.smoothing / n_classes
        )
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # True class probability
        p_t = (probs * F.one_hot(targets, n_classes).float()).sum(-1)
        focal_weight = (1 - p_t).pow(self.gamma)
        # CE with smooth targets
        ce = -(targets_smooth * log_probs).sum(-1)
        if self.weight is not None:
            sample_weight = self.weight[targets]
            loss = (focal_weight * ce * sample_weight).mean()
        else:
            loss = (focal_weight * ce).mean()
        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        proj_emb: (B, 128) L2-normalised
        labels:   (B,) class indices
        """
        B = proj_emb.shape[0]
        if B <= 1:
            return proj_emb.sum() * 0.0

        # Similarity matrix
        sim = torch.mm(proj_emb, proj_emb.T) / self.temperature  # (B, B)

        # Mask for positives (same class, different sample)
        labels_col = labels.view(-1, 1)
        mask_pos = (labels_col == labels_col.T).float()
        mask_diag = torch.eye(B, device=proj_emb.device)
        mask_pos = mask_pos - mask_diag  # exclude self

        # Logsumexp over all non-self pairs (denominator)
        sim_masked = sim - mask_diag * 1e9  # mask diagonal
        log_denom = torch.logsumexp(sim_masked, dim=1)  # (B,)

        # Loss per anchor: only anchors with at least one positive
        n_pos = mask_pos.sum(dim=1)
        valid = n_pos > 0

        if not valid.any():
            return proj_emb.sum() * 0.0

        # Numerator: sum over positives
        log_numerator = (mask_pos * sim).sum(dim=1)
        loss_per_anchor = -(log_numerator - n_pos * log_denom)
        loss_per_anchor = loss_per_anchor[valid] / n_pos[valid]
        return loss_per_anchor.mean()


class MixupCriterion(nn.Module):
    """Cross-entropy with soft (mixed) labels."""

    def forward(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_labels * log_probs).sum(-1).mean()
        return loss


class AdulterNetLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, supcon_weight: float = 0.2,
                 focal_gamma: float = 2.5, label_smoothing: float = 0.05,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.supcon_weight = supcon_weight
        self.focal = FocalLossSmoothed(gamma=focal_gamma, smoothing=label_smoothing,
                                       weight=class_weights)
        self.huber = nn.HuberLoss(delta=2.0)
        self.supcon = SupConLoss()

    def forward(
        self,
        logits: torch.Tensor,
        reg_pred: torch.Tensor,
        proj_emb: torch.Tensor,
        label_int: torch.Tensor,
        label_float: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        focal = self.focal(logits, label_int)
        huber = self.huber(reg_pred.squeeze(), label_float)
        supcon = self.supcon(proj_emb, label_int)

        huber_weight = max(0.0, 1.0 - self.alpha - self.supcon_weight)
        total = self.alpha * focal + huber_weight * huber + self.supcon_weight * supcon

        return {
            "total": total,
            "focal": focal.detach(),
            "huber": huber.detach(),
            "supcon": supcon.detach(),
        }

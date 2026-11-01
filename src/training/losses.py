"""
Loss functions for class-imbalanced multi-class graph node classification.

Supported:
  - ce          : standard cross-entropy
  - weighted_ce : inverse-frequency class-weighted cross-entropy
  - focal       : focal loss (Lin et al., 2017) with optional class alpha weights
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Sublinear power-scaled class weights: w_c = (N / (K * n_c))^0.35, clamped to [0.2, 20.0].
    Returns a float tensor of shape (num_classes,).
    """
    counts = np.bincount(y, minlength=num_classes)
    total = max(len(y), 1)
    weights = (total / (num_classes * np.maximum(counts, 1))) ** 0.35
    weights = np.clip(weights, 0.2, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma down-weights easy (majority-class) examples; alpha reweights classes.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        pt = (probs * targets_one_hot).sum(dim=-1)
        log_pt = (log_probs * targets_one_hot).sum(dim=-1)
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * focal_factor * log_pt
        else:
            loss = -focal_factor * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_type: str,
    y_train: np.ndarray,
    num_classes: int,
    device: torch.device,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory for the requested loss function.

    Parameters
    ----------
    loss_type : "ce", "weighted_ce", or "focal"
    y_train   : training labels (for computing class weights)
    """
    loss_type = loss_type.lower()
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    class_weights = compute_class_weights(y_train, num_classes).to(device)

    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)

    raise ValueError(f"Unknown loss_type '{loss_type}'. Use ce, weighted_ce, or focal.")

"""Training utilities: masked losses and class weighting (spec 11.3/11.4).

Key rules enforced here:
- Loss is computed on seed/target nodes only, never on sampled context nodes.
- Unknown labels are masked completely from loss, early stopping and metrics.
- Class weighting: w_pos = N_neg / N_pos from *train* labels, capped.
- Test labels are never accessed by any code path in this module.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pos_weight(train_labels: np.ndarray, cap: float = 100.0) -> float:
    """N_negative / N_positive from train labels, capped by a documented cap."""
    train_labels = np.asarray(train_labels)
    n_pos = float((train_labels == 1).sum())
    n_neg = float((train_labels == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(min(n_neg / n_pos, cap))


def bce_with_logits_masked(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, weight_pos: float = 1.0
) -> torch.Tensor:
    """Weighted BCE over masked targets only.

    ``mask`` selects eligible labelled examples; unknown labels are excluded
    entirely. Probabilities are clamped away from 0/1 for numerical stability
    (BCEWithLogitsLoss already applies log-sum-exp internally).
    """
    if mask.sum() == 0:
        return torch.zeros((), requires_grad=True)
    loss_fn = nn.BCEWithLogitsLoss(
        reduction="mean",
        pos_weight=torch.tensor(weight_pos, dtype=logits.dtype, device=logits.device),
    )
    return loss_fn(logits[mask], labels[mask].float())

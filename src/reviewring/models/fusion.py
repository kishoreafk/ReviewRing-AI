"""Fusion modules (spec 11.1): fixed uniform mean, global learned mixture,
and the adaptive context-aware gate (P1, the main proposal).

The gate concatenates the three 64-d expert representations, availability
flags and selected past-only context counts, then produces three gate logits.
Unavailable experts are masked before the softmax; weights are nonnegative and
sum to one over available experts. If no expert is available the scorer must
return ``insufficient_data`` rather than softmax over an empty set.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_NAMES = ("text", "graph", "time")


def masked_softmax(logits: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    """Softmax over available experts only.

    Rows with no available expert produce NaN - callers must treat those as
    ``insufficient_data`` before calling this function.
    """
    masked = logits.masked_fill(availability <= 0, float("-inf"))
    return F.softmax(masked, dim=-1)


class FixedFusion(nn.Module):
    """Uniform mean of available expert probabilities (B5). No parameters."""

    def forward(self, expert_probs: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        masked = expert_probs * availability
        denom = availability.sum(dim=-1).clamp(min=1.0)
        return masked.sum(dim=-1) / denom


class GlobalFusion(nn.Module):
    """Learned global mixture weights over experts (B6).

    Weights are global parameters, not per-example; availability is applied by
    renormalising over the available subset.
    """

    def __init__(self, n_experts: int = 3):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(n_experts))

    def forward(self, expert_probs: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.raw_weights, dim=0)
        w = w * availability
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (expert_probs * w).sum(dim=-1)


class AdaptiveGate(nn.Module):
    """Per-example gate (P1): MLP over expert states + availability + context.

    Context counts are past-only inputs available to every expert; no labels,
    simulation flags or lifetime counts are admitted here (spec 11.2).
    """

    def __init__(self, repr_dim: int = 64, n_context: int = 4, hidden: int = 64, n_experts: int = 3):
        super().__init__()
        in_dim = n_experts * repr_dim + n_experts + n_context
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_experts),
        )

    def forward(
        self,
        expert_reprs: torch.Tensor,  # [B, K, H]
        availability: torch.Tensor,  # [B, K]
        context: torch.Tensor,  # [B, n_context]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gate_weights [B, K], gate_logits [B, K])."""
        flat = torch.cat(
            [
                expert_reprs.flatten(start_dim=1),
                availability.float(),
                context.float(),
            ],
            dim=-1,
        )
        logits = self.net(flat)
        weights = masked_softmax(logits, availability)
        return weights, logits

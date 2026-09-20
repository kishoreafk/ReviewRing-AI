"""Label-masked metrics (spec 15.2) and ring-recovery metrics (spec 15.3).

All classification metrics operate only on eligible labelled examples passed
by the caller; unknown labels must already be filtered out. If a partition
contains one class, threshold-free metrics return ``None`` and the report
marks them undefined instead of inventing a value. Precision@K uses the actual
denominator when K exceeds the number of examples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _check(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {scores.shape}")
    return y_true, scores


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """AP via sklearn's ``average_precision_score`` (labelled 'AP' in reports)."""
    y, s = _check(y_true, scores)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y, s = _check(y_true, scores)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    """Share of positives among the K highest scores (actual denominator)."""
    y, s = _check(y_true, scores)
    if len(y) == 0:
        return None
    k = min(k, len(y))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / k)


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    y, s = _check(y_true, scores)
    total_pos = float((y == 1).sum())
    if total_pos == 0:
        return None
    k = min(k, len(y))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / total_pos)


def f1_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    y, s = _check(y_true, scores)
    pred = (s >= threshold).astype(np.int64)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Validation-set threshold sweep; returns (threshold, f1)."""
    y, s = _check(y_true, scores)
    if len(np.unique(y)) < 2:
        return 0.5, float("nan")
    candidates = np.unique(s)
    if len(candidates) > 512:
        candidates = np.quantile(s, np.linspace(0, 1, 512))
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        f = f1_at_threshold(y, s, float(t))
        if f is not None and f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y, p = _check(y_true, probabilities)
    return float(np.mean((p - y) ** 2))


def reliability_bins(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability diagram data: mean predicted vs observed per score bin."""
    y, p = _check(y_true, probabilities)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            out.append({"bin": i, "lo": float(lo), "hi": float(hi), "count": 0,
                        "mean_pred": None, "observed": None})
        else:
            out.append({
                "bin": i, "lo": float(lo), "hi": float(hi), "count": int(mask.sum()),
                "mean_pred": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
            })
    return out


# ---------------------------------------------------------------------------
# Ring recovery metrics (spec 15.3)
# ---------------------------------------------------------------------------

def iou(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def match_campaigns(
    predicted: list[set], truth: list[set], threshold: float = 0.5
) -> dict:
    """One-to-one maximum-weight IoU matching between predicted rings and
    known campaign memberships. A match counts only above ``threshold``.

    Returns campaign precision/recall, matched pairs, fragmentation (one truth
    split into many predictions), and merging (one prediction joining several
    truths), evaluated on the raw IoU matrix before thresholding.
    """
    n_p, n_t = len(predicted), len(truth)
    if n_p == 0 or n_t == 0:
        return {
            "n_predicted": n_p, "n_truth": n_t, "matches": [],
            "campaign_precision": None, "campaign_recall": None,
            "member_recovery": None, "mean_iou": None,
            "fragmentation": None, "merging": None,
        }
    iou_mat = np.zeros((n_p, n_t), dtype=np.float64)
    for i, p in enumerate(predicted):
        for j, t in enumerate(truth):
            iou_mat[i, j] = iou(p, t)
    # one-to-one MAXIMUM-WEIGHT matching (spec 15.3), via the Hungarian method
    from scipy.optimize import linear_sum_assignment

    rows_sel, cols_sel = linear_sum_assignment(-iou_mat)
    matches = [
        {
            "predicted": int(p_i),
            "truth": int(t_j),
            "iou": float(iou_mat[p_i, t_j]),
            "matched_above_threshold": bool(iou_mat[p_i, t_j] >= threshold),
        }
        for p_i, t_j in zip(rows_sel, cols_sel)
        if iou_mat[p_i, t_j] > 0
    ]
    good = [m for m in matches if m["matched_above_threshold"]]
    campaign_precision = len(good) / n_p if n_p else None
    campaign_recall = len(good) / n_t if n_t else None
    # member recovery: shared members over true members, over matched pairs
    # (spec 15.3 member recovery, computed on matched pairs only)
    recovery_values = []
    for m in good:
        p_set, t_set = predicted[m["predicted"]], truth[m["truth"]]
        if t_set:
            recovery_values.append(len(p_set & t_set) / len(t_set))
    member_recovery = float(np.mean(recovery_values)) if recovery_values else 0.0
    frag_counts: dict[int, int] = {}
    merge_counts: dict[int, int] = {}
    for m in matches:
        frag_counts[m["truth"]] = frag_counts.get(m["truth"], 0) + 1
        merge_counts[m["predicted"]] = merge_counts.get(m["predicted"], 0) + 1
    fragmentation = (
        float(np.mean([c for c in frag_counts.values() if c > 1])) if any(c > 1 for c in frag_counts.values()) else 0.0
    )
    merging = (
        float(np.mean([c for c in merge_counts.values() if c > 1])) if any(c > 1 for c in merge_counts.values()) else 0.0
    )
    return {
        "n_predicted": n_p,
        "n_truth": n_t,
        "matches": matches,
        "campaign_precision": campaign_precision,
        "campaign_recall": campaign_recall,
        "member_recovery": member_recovery,
        "mean_iou": float(np.mean([m["iou"] for m in good])) if good else None,
        "fragmentation": fragmentation,
        "merging": merging,
    }


@dataclass
class ClassificationReport:
    partition: str
    n_examples: int
    prevalence: float
    ap: float | None
    auc: float | None
    precision_at_k: dict
    recall_at_k: dict
    f1: float | None
    threshold: float
    brier: float | None

    def to_dict(self) -> dict:
        return self.__dict__

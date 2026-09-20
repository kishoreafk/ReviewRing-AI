"""Past-only numerical history features (timing expert inputs).

Every feature for event ``i`` at time ``t_i`` is computed from events strictly
before ``t_i`` (or a documented batch/tie policy). A new account is not a
suspicious account: missing history produces *both* a zero value and a
missingness indicator so the model can distinguish "new" from "observed zero".

The computation is a single chronological sweep with bounded per-reviewer and
per-target deques, so complexity is O(N) in events, not quadratic.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

HISTORY_SCHEMA_VERSION = "1.0.0"

FEATURE_NAMES = [
    # reviewer velocity over windows
    "rev_count_1d", "rev_count_7d", "rev_count_30d",
    # gaps and rhythm
    "time_since_prev_s", "interarrival_std_s", "interarrival_mean_s",
    # breadth of behaviour
    "prior_unique_targets", "prior_reviews_31d",
    # target-side context
    "target_volume_prior", "target_reviewer_volume_prior", "time_since_target_prev_s",
    # rating behaviour
    "rating_minus_reviewer_prior_mean", "reviewer_prior_rating_count",
    "target_prior_mean_rating",
    # burst intensity on the current target within 24h
    "target_burst_24h", "reviewer_burst_24h",
]

MISSING_BY_DEFAULT = {
    "time_since_prev_s", "interarrival_std_s", "interarrival_mean_s",
    "time_since_target_prev_s", "rating_minus_reviewer_prior_mean",
    "reviewer_prior_rating_count", "target_prior_mean_rating",
}


@dataclass
class HistoryResult:
    values: np.ndarray  # [N, F] float32, raw (unlogged) counts
    missing: np.ndarray  # [N, F] bool
    names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.names)


class _ReviewerState:
    __slots__ = ("times", "targets", "ratings", "gaps")

    def __init__(self):
        self.times: deque[float] = deque()
        self.targets: deque[str] = deque()
        self.ratings: deque[float] = deque()
        self.gaps: deque[float] = deque()


class _TargetState:
    __slots__ = ("times", "reviewers", "ratings")

    def __init__(self):
        self.times: deque[float] = deque()
        self.reviewers: deque[str] = deque()
        self.ratings: deque[float] = deque()


def _expire(deq: deque, cutoff: float) -> None:
    while deq and deq[0] < cutoff:
        deq.popleft()


def compute_history_features(
    reviews: pd.DataFrame,
    history_days: tuple[int, ...] = (1, 7, 30),
) -> HistoryResult:
    """Sweep events chronologically, emitting past-only features per event.

    Ties: events sharing the exact timestamp are processed as one batch; all of
    them observe the same prior state (nothing within the batch sees another
    batch member as history). This implements the documented equal-timestamp
    policy without letting file order leak future information.
    """
    n = len(reviews)
    ts = pd.to_datetime(reviews["timestamp"], utc=True)
    epoch = (ts.astype("int64") / 1_000_000_000).to_numpy()  # seconds
    reviewer = reviews["reviewer_id"].to_numpy()
    target = reviews["target_id"].to_numpy()
    rating = reviews["rating"].to_numpy(dtype=np.float64)

    feature_names = list(FEATURE_NAMES)
    n_feat = len(feature_names)
    values = np.zeros((n, n_feat), dtype=np.float32)
    missing = np.ones((n, n_feat), dtype=bool)

    reviewer_state: dict[str, _ReviewerState] = defaultdict(_ReviewerState)
    target_state: dict[str, _TargetState] = defaultdict(_TargetState)

    order = np.argsort(epoch, kind="stable")
    day = 86400.0
    d1, d7, d30 = history_days[0] * day, history_days[1] * day, history_days[2] * day

    i = 0
    while i < n:
        j = i
        group = [order[i]]
        while j + 1 < n and epoch[order[j + 1]] == epoch[order[i]]:
            j += 1
            group.append(order[j])
        # group holds all indices sharing this timestamp; compute features from
        # current state, then apply updates afterwards
        for idx in group:
            rs = reviewer_state[reviewer[idx]]
            tgt = target_state[target[idx]]
            _expire(rs.times, epoch[idx] - d30)
            _expire(tgt.times, epoch[idx] - d30)

            vals = values[idx]
            miss = missing[idx]

            for w_days, w_sec, out_col in ((1, d1, 0), (7, d7, 1), (30, d30, 2)):
                cutoff = epoch[idx] - w_sec
                cnt = sum(1 for t in rs.times if t >= cutoff)
                vals[out_col] = cnt
                miss[out_col] = False  # zero is an observed count here

            if rs.gaps:
                arr = np.asarray(rs.gaps, dtype=np.float64)
                if len(arr):
                    vals[3] = arr[-1]
                    miss[3] = False
                    if len(arr) >= 2:
                        vals[4] = float(np.std(arr))
                        vals[5] = float(np.mean(arr))
                        miss[4] = False
                        miss[5] = False
            vals[6] = len(set(rs.targets))
            vals[7] = len(rs.times)
            miss[6] = False
            miss[7] = False

            # target-side context
            vals[8] = len(tgt.times)
            miss[8] = False
            vals[9] = len(set(tgt.reviewers))
            miss[9] = False
            if tgt.times:
                vals[10] = epoch[idx] - tgt.times[-1]
                miss[10] = False
            if rs.ratings:
                vals[11] = rating[idx] - float(np.mean(rs.ratings))
                vals[12] = len(rs.ratings)
                miss[11] = False
                miss[12] = False
            if tgt.ratings:
                vals[13] = float(np.mean(tgt.ratings))
                miss[13] = False

            cutoff24 = epoch[idx] - day
            vals[14] = sum(1 for t in tgt.times if t >= cutoff24)
            vals[15] = sum(1 for t in rs.times if t >= cutoff24)
            miss[14] = False
            miss[15] = False

        # apply updates for the whole tie batch after all features are read
        for idx in group:
            rs = reviewer_state[reviewer[idx]]
            tgt = target_state[target[idx]]
            if rs.times:
                rs.gaps.append(epoch[idx] - rs.times[-1])
            rs.times.append(epoch[idx])
            rs.targets.append(target[idx])
            rs.ratings.append(rating[idx])
            tgt.times.append(epoch[idx])
            tgt.reviewers.append(reviewer[idx])
            tgt.ratings.append(rating[idx])
            # bounded memory: keep at most the 30-day window plus slack
            _expire(rs.times, epoch[idx] - 31 * day)
            _expire(tgt.times, epoch[idx] - 31 * day)
            if len(rs.targets) > 512:
                rs.targets.popleft()
            if len(rs.ratings) > 512:
                rs.ratings.popleft()
            if len(tgt.reviewers) > 512:
                tgt.reviewers.popleft()
            if len(tgt.ratings) > 512:
                tgt.ratings.popleft()
            if len(rs.gaps) > 512:
                rs.gaps.popleft()
        i = j + 1

    return HistoryResult(values=values, missing=missing, names=feature_names)


def transform_history(
    values: np.ndarray,
    missing: np.ndarray,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """log1p counts, robust-scale, impute missing with train medians.

    Scaler statistics are returned so they can be frozen and reused at
    inference time. When ``means``/``stds`` are None the statistics are fitted
    on ``fit_mask`` rows only (the train partition); passing ``fit_mask=None``
    fits on all rows and is only appropriate for tests.
    """
    x = values.astype(np.float32).copy()
    # counts and seconds are long-tailed: log1p everything except differences
    log_cols = [0, 1, 2, 6, 7, 8, 9, 12, 14, 15]
    for c in log_cols:
        x[:, c] = np.log1p(np.maximum(x[:, c], 0.0))
    # seconds columns: log1p after clamping negatives (impossible but safe)
    for c in (3, 4, 5, 10):
        x[:, c] = np.log1p(np.maximum(x[:, c], 0.0))

    if means is None or stds is None:
        fit_rows = fit_mask if fit_mask is not None else np.arange(len(x))
        x_fit = x[fit_rows]
        visible = ~missing[fit_rows]
        means = np.zeros(x.shape[1], dtype=np.float32)
        stds = np.ones(x.shape[1], dtype=np.float32)
        for c in range(x.shape[1]):
            col_visible = visible[:, c]
            if col_visible.sum() > 1:
                means[c] = np.median(x_fit[col_visible, c])
                q75, q25 = np.percentile(x_fit[col_visible, c], [75, 25])
                iqr = q75 - q25
                stds[c] = iqr / 1.349 if iqr > 0 else max(float(np.std(x_fit[col_visible, c])), 1e-6)
    x_scaled = (x - means) / np.maximum(stds, 1e-6)
    x_imputed = np.where(missing, 0.0, x_scaled)
    return x_imputed.astype(np.float32), missing, means, stds

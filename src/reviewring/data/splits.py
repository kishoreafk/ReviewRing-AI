"""Split protocols with saved assignments (spec section 9).

Track A: fixed stratified random split of eligible labelled nodes into
train/validation/calibration/test. Transductive setting: test-node features and
edges stay in the graph while test labels are hidden from training.

Track B: chronological split with campaign disjointness. Time boundaries are
chosen first; every simulated campaign (and its rewritten variants) is kept
entirely within one partition. The saved assignments are the single source of
truth for every later stage; nothing recomputes membership on the fly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reviewring.data.schema import SPLIT_ROLES

SPLITS_SCHEMA_VERSION = "1.0.0"


def stratified_split(
    labels: np.ndarray,
    eligible_mask: np.ndarray,
    fractions: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Return a splits frame with one role per eligible node.

    Non-eligible nodes receive role ``excluded`` and must never enter loss,
    calibration or metrics. Stratification is per class within the eligible
    set, preserving prevalence in every partition.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    roles = np.full(len(labels), "excluded", dtype=object)
    eligible_idx = np.where(eligible_mask)[0]
    if len(eligible_idx) == 0:
        raise ValueError("no eligible labelled nodes")
    for cls in np.unique(labels[eligible_idx]):
        cls_idx = eligible_idx[labels[eligible_idx] == cls]
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        n = len(cls_idx)
        bounds = np.cumsum([int(n * f) for f in fractions])
        bounds[-1] = n  # last partition absorbs rounding
        start = 0
        for role, end in zip(SPLIT_ROLES, bounds):
            roles[cls_idx[start:end]] = role
            start = end
    return _splits_frame(node_idx=np.arange(len(labels)), roles=roles, seed=seed, protocol="stratified_random")


def chronological_split(
    timestamps: pd.Series,
    fractions: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
    review_ids: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assign roles by event time without looking at outcomes or labels.

    Returns the splits frame plus a manifest with the actual cutoff timestamps
    (not just fractions), as required by the reporting rules. ``timestamps``
    and ``review_ids`` must share the same index as the reviews table row
    positions so that ``node_idx`` maps directly into the reviews frame.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    ts = pd.to_datetime(timestamps, utc=True).sort_values()
    n = len(ts)
    bounds = np.cumsum([int(n * f) for f in fractions])
    bounds[-1] = n
    ordered_roles = np.empty(n, dtype=object)
    start = 0
    cutoffs = {}
    for role, end in zip(SPLIT_ROLES, bounds):
        ordered_roles[start:end] = role
        cutoffs[role] = (ts.iloc[start].isoformat(), ts.iloc[end - 1].isoformat())
        start = end
    order_index = ts.index.to_numpy()
    roles = np.empty(n, dtype=object)
    roles[order_index] = ordered_roles
    node_key = None
    if review_ids is not None:
        node_key = np.asarray(review_ids, dtype=object)
    frame = _splits_frame(
        node_idx=np.arange(n),
        roles=roles,
        seed=None,
        protocol="chronological",
        node_key=node_key,
    )
    return frame, {"cutoffs": cutoffs, "fractions": list(fractions)}


def assign_campaign_partitions(
    campaigns: pd.DataFrame,
    split_frame: pd.DataFrame,
    review_id_col: str = "member_review_ids",
) -> pd.DataFrame:
    """Give every campaign the partition of its members (majority vote).

    Campaigns must not span partitions. If a campaign's reviews fall in more
    than one partition the campaign is assigned to the partition of its *latest*
    member event so that late-arriving manipulators are evaluated on future
    behaviour, and a warning column ``spanning`` is set for the audit.
    """
    role_by_review = dict(zip(split_frame["node_key"], split_frame["role"]))
    records = []
    for _, row in campaigns.iterrows():
        member_ids = row[review_id_col]
        roles = [role_by_review.get(r) for r in member_ids if r in role_by_review]
        if not roles:
            partition = "train"
        else:
            order = {r: i for i, r in enumerate(SPLIT_ROLES)}
            partition = max(set(roles), key=lambda r: order.get(r, -1))
        records.append(
            {
                "campaign_id": row["campaign_id"],
                "partition": partition,
                "spanning": len(set(roles)) > 1,
            }
        )
    return pd.DataFrame(records)


def _splits_frame(
    node_idx: np.ndarray,
    roles: np.ndarray,
    seed: int | None,
    protocol: str,
    node_key: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "node_idx": node_idx.astype(np.int64),
            "role": roles,
        }
    )
    if node_key is not None:
        frame["node_key"] = node_key
    frame["protocol"] = protocol
    frame["seed"] = -1 if seed is None else seed
    frame["schema_version"] = SPLITS_SCHEMA_VERSION
    return frame


def assert_no_campaign_leakage(
    assignments: pd.DataFrame, split_frame: pd.DataFrame, member_map: dict[str, list[str]]
) -> None:
    """Fail if any campaign's members span supervised partitions.

    ``member_map`` maps campaign_id -> list of review_ids. Paraphrase families
    count as campaigns here: one family per entry.
    """
    role_by_review = dict(zip(split_frame["node_key"], split_frame["role"]))
    for campaign_id, members in member_map.items():
        roles = {role_by_review.get(r) for r in members if r in role_by_review}
        supervised = roles & set(SPLIT_ROLES)
        if len(supervised) > 1:
            raise ValueError(
                f"Campaign {campaign_id} spans partitions {sorted(supervised)}; "
                "campaign/variant leakage detected."
            )


def role_indices(split_frame: pd.DataFrame, role: str) -> np.ndarray:
    """Node indices for one role, sorted for determinism."""
    sel = split_frame[split_frame["role"] == role]
    return np.sort(sel["node_idx"].to_numpy().astype(np.int64))

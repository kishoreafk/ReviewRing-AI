"""Candidate ring discovery (spec 13.2/13.3).

The model scores reviews; rings are *groups of accounts supported by event
evidence*. Discovery consumes review scores plus observed relationship data
only - never evaluator labels or campaign ids. Steps:

1. Take reviews above a threshold (or a bounded top-K) as candidates.
2. Retrieve authors and their past target histories.
3. Build a reviewer projection graph on repeated shared targets + time overlap.
4. Connected components; require >= min_accounts and >= min_targets.
5. Deduplicate by member-set Jaccard; compute transparent priority scores.

Known limitation (documented, not a bug): the minimum group size and target
count intentionally miss single-target or two-account manipulation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RINGS_SCHEMA_VERSION = "1.0.0"


@dataclass
class RingCandidate:
    candidate_id: str
    members: list[str]  # reviewer ids
    review_ids: list[str]
    targets: list[str]
    mean_top_score: float
    adjusted_overlap: float
    synchrony: float
    priority: float
    size_ref_mean: float  # size-matched shuffled reference for the overlap
    components_merged: bool = False

    def to_dict(self) -> dict:
        # return a copy: callers annotate lengths and must not mutate the
        # candidate itself (members/review_ids stay lists on the object)
        return dict(self.__dict__)


def _jaccard(a: set, b: set) -> float:
    if not a | b:
        return 0.0
    return len(a & b) / len(a | b)


def discover_rings(
    scores: pd.DataFrame,  # review_id, score
    reviews: pd.DataFrame,  # canonical reviews
    threshold: float | None = None,
    top_k: int | None = 300,
    min_accounts: int = 3,
    min_targets: int = 2,
    top_m_reviews: int = 5,
    priority_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    min_shared_targets: int = 2,
    bridge_percentile: float = 50.0,
    seed: int = 42,
) -> list[RingCandidate]:
    """Score-driven candidate discovery using observed relationships only."""
    if threshold is None and top_k is None:
        raise ValueError("either threshold or top_k must be provided")
    rng = np.random.default_rng(seed)

    sc = scores.sort_values("score", ascending=False)
    if threshold is not None:
        selected = sc[sc["score"] >= threshold]
        if len(selected) > 4000:  # bounded work
            selected = selected.head(4000)
    else:
        selected = sc.head(top_k)

    selected_ids = set(selected["review_id"])
    rev = reviews.set_index("review_id")
    cand_reviews = rev.loc[rev.index.intersection(selected_ids)]

    # bridge inclusion: keep moderately scored reviews whose authors appear in
    # the candidate set, so thresholding does not erase every bridge (spec 13.2)
    candidate_authors = set(cand_reviews["reviewer_id"])
    if bridge_percentile is not None and len(scores) > 0:
        cutoff = np.percentile(scores["score"], bridge_percentile)
        eligible = scores["score"] >= cutoff
        bridge_ids = set(
            scores.loc[eligible, "review_id"]
        ) & set(reviews.loc[reviews["reviewer_id"].isin(candidate_authors), "review_id"]) - selected_ids
    else:
        bridge_ids = set()

    use_ids = selected_ids | bridge_ids
    events = reviews[reviews["review_id"].isin(use_ids)]
    if len(events) == 0:
        return []

    # reviewer -> set of targets (full observed history within the cohort)
    reviewer_targets: dict[str, set[str]] = {}
    reviewer_reviews: dict[str, list[str]] = {}
    for reviewer, target, rid in zip(events["reviewer_id"], events["target_id"], events["review_id"]):
        reviewer_targets.setdefault(reviewer, set()).add(target)
        reviewer_reviews.setdefault(reviewer, []).append(rid)

    reviewers = sorted(reviewer_targets)
    if len(reviewers) < min_accounts:
        return []

    # projection: edge if repeated shared targets
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(reviewers)
    target_index: dict[str, list[str]] = {}
    for reviewer in reviewers:
        for target in reviewer_targets[reviewer]:
            target_index.setdefault(target, []).append(reviewer)
    for target, rvs in target_index.items():
        if len(rvs) < 2:
            continue
        for i in range(len(rvs)):
            for j in range(i + 1, len(rvs)):
                a, b = rvs[i], rvs[j]
                shared = reviewer_targets[a] & reviewer_targets[b]
                if len(shared) >= min_shared_targets:
                    if g.has_edge(a, b):
                        g[a][b]["weight"] += 1
                    else:
                        g.add_edge(a, b, weight=len(shared))

    # connected components with evidence-based strengthening for giants
    components = [c for c in nx.connected_components(g) if len(c) >= min_accounts]
    if components and max(len(c) for c in components) > 0.6 * len(reviewers):
        # giant component: keep only edges with above-median weight and retry
        weights = [d["weight"] for _, _, d in g.edges(data=True)]
        if weights:
            med = np.median(weights)
            strong = nx.Graph()
            strong.add_nodes_from(g.nodes)
            for a, b, d in g.edges(data=True):
                if d["weight"] > med:
                    strong.add_edge(a, b, weight=d["weight"])
            components = [c for c in nx.connected_components(strong) if len(c) >= min_accounts]
        logger.info("giant component detected; strengthened to %d components", len(components))

    scores_by_review = dict(zip(scores["review_id"], scores["score"]))

    candidates: list[RingCandidate] = []
    for comp in components:
        comp_reviews = [rid for r in comp for rid in reviewer_reviews[r]]
        comp_targets = sorted(set().union(*[reviewer_targets[r] for r in comp]))
        if len(comp_targets) < min_targets:
            continue
        # mean top-m member review score
        member_scores = sorted((scores_by_review.get(rid, 0.0) for rid in comp_reviews), reverse=True)
        mean_top = float(np.mean(member_scores[:top_m_reviews]))
        # adjusted overlap: mean pairwise jaccard of members' target sets
        pairwise = []
        cl = sorted(comp)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                pairwise.append(_jaccard(reviewer_targets[cl[i]], reviewer_targets[cl[j]]))
        raw_overlap = float(np.mean(pairwise)) if pairwise else 0.0
        # size-matched shuffled reference: reassign targets across members
        all_t = [t for r in cl for t in reviewer_targets[r]]
        shuffled = rng.permutation(all_t)
        sizes = [len(reviewer_targets[r]) for r in cl]
        shuffled_sets = []
        pos = 0
        for s in sizes:
            shuffled_sets.append(set(shuffled[pos : pos + s].tolist()))
            pos += s
        ref_pairs = []
        for i in range(len(shuffled_sets)):
            for j in range(i + 1, len(shuffled_sets)):
                ref_pairs.append(_jaccard(shuffled_sets[i], shuffled_sets[j]))
        ref_overlap = float(np.mean(ref_pairs)) if ref_pairs else 0.0
        adjusted = max(raw_overlap - ref_overlap, 0.0)
        # synchrony: share of member events within the tightest 7-day window
        times = pd.to_datetime(events[events["review_id"].isin(comp_reviews)]["timestamp"], utc=True)
        sync = 0.0
        if len(times) >= 2:
            t_sorted = times.sort_values().astype("int64").to_numpy() // 86400 // 10**9
            counts = np.bincount(t_sorted - t_sorted.min())
            window = 7
            best = max(counts[i : i + window].sum() for i in range(max(len(counts) - window, 1)))
            sync = float(best / len(times))
        priority = (
            priority_weights[0] * mean_top
            + priority_weights[1] * min(adjusted * 3.0, 1.0)
            + priority_weights[2] * sync
        )
        candidates.append(
            RingCandidate(
                candidate_id=f"ring-{len(candidates):04d}",
                members=cl,
                review_ids=sorted(comp_reviews),
                targets=comp_targets,
                mean_top_score=mean_top,
                adjusted_overlap=adjusted,
                synchrony=sync,
                priority=float(priority),
                size_ref_mean=ref_overlap,
            )
        )

    # deduplicate by member-set Jaccard, keep highest priority
    candidates.sort(key=lambda c: -c.priority)
    kept: list[RingCandidate] = []
    for cand in candidates:
        member_set = set(cand.members)
        duplicate = False
        for existing in kept:
            if _jaccard(member_set, set(existing.members)) >= 0.7:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def candidate_priority_table(candidates: list[RingCandidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        d = c.to_dict()
        d["members"] = len(c.members)
        d["targets"] = len(c.targets)
        d["review_ids"] = len(c.review_ids)
        rows.append(d)
    cols = ["candidate_id", "priority", "mean_top_score", "adjusted_overlap", "synchrony",
            "size_ref_mean", "members", "targets", "review_ids"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)

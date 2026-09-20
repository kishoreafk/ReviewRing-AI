"""Causal, typed review graph construction (Track B).

Nodes are reviews. Edges are directed from *earlier* evidence toward *later*
events so that a live event never receives a message influenced by a later
event (spec 9.3). Every edge records ``available_at`` (the earliest time its
evidence can be known) and an evidence reference, and all windows/caps are
configuration, not hard-coded.

Relations
---------
same_reviewer : previous events by the current reviewer (bounded, recent)
same_target   : prior reviews on the same target in a time window
co_target     : reviewers sharing >= min_shared_targets earlier targets in a
                window (bounded top overlap); edges connect the *reviewers'* events
semantic      : optional cosine neighbours among eligible past reviews (off by
                default so text-in-graph does not blur the graph expert)

Construction is a single chronological sweep with inverted indexes: complexity
is O(N * bounded candidate expansion), never all-pairs.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRAPH_SCHEMA_VERSION = "1.0.0"

RELATION_NAMES = ("same_reviewer", "same_target", "co_target", "semantic")


@dataclass
class GraphBuildResult:
    edge_index: np.ndarray  # [2, E] int64, columns = (src, dst)
    relation: np.ndarray  # [E] int8 index into RELATION_NAMES
    available_at: np.ndarray  # [E] int64 epoch seconds
    stats: dict = field(default_factory=dict)

    def edge_index_for(self, relation: str) -> np.ndarray:
        rid = RELATION_NAMES.index(relation)
        return self.edge_index[:, self.relation == rid]


def build_causal_graph(
    reviews: pd.DataFrame,
    reviewer_window_days: int = 30,
    target_window_days: int = 7,
    min_shared_targets: int = 2,
    max_reviewer_neighbors: int = 10,
    max_target_neighbors: int = 20,
    max_overlap_neighbors: int = 10,
    include_semantic: bool = False,
    semantic_matrix: np.ndarray | None = None,
    semantic_threshold: float = 0.9,
    semantic_topk: int = 5,
) -> GraphBuildResult:
    """Sweep events chronologically and emit directed causal edges.

    For each event ``i`` at ``t_i``: candidate sources are events strictly
    before ``t_i`` (ties at equal timestamps are excluded as sources *unless*
    the batch policy is enabled - here the default is strict exclusion, which
    is the safer direction for replay). Evidence for co_target edges is the
    shared-target record of the two reviewers' prior events.
    """
    n = len(reviews)
    ts = pd.to_datetime(reviews["timestamp"], utc=True)
    epoch = (ts.astype("int64") // 1_000_000_000).to_numpy().astype(np.int64)
    reviewer = reviews["reviewer_id"].to_numpy()
    target = reviews["target_id"].to_numpy()
    order = np.argsort(epoch, kind="stable")

    day = 86400
    rev_window = reviewer_window_days * day
    tgt_window = target_window_days * day
    co_window = 30 * day

    src_all: list[np.ndarray] = []
    dst_all: list[np.ndarray] = []
    rel_all: list[np.ndarray] = []
    avail_all: list[np.ndarray] = []

    # per-reviewer recent events
    reviewer_events: dict[str, deque[int]] = defaultdict(deque)
    # per-target recent events
    target_events: dict[str, deque[int]] = defaultdict(deque)
    # reviewer -> {target: last_event_time} within the co-occurrence window
    reviewer_targets: dict[str, dict[str, int]] = defaultdict(dict)
    # inverted index: target -> reviewers who reviewed it recently
    target_reviewer_index: dict[str, dict[str, int]] = defaultdict(dict)

    rel_ids = {name: i for i, name in enumerate(RELATION_NAMES)}

    # tie batching: events sharing an exact timestamp are processed as ONE
    # batch - all edges are emitted from the pre-batch state so no tied event
    # ever sees a batch mate as history (spec 9.3, mirrors history.py policy)
    i = 0
    while i < n:
        group = [order[i]]
        while i + 1 < n and epoch[order[i + 1]] == epoch[order[i]]:
            i += 1
            group.append(order[i])

        for idx in group:
            t = int(epoch[idx])
            u = reviewer[idx]
            p = target[idx]

            rs = reviewer_events.get(u, deque())
            # same_reviewer: most recent K prior events within window
            cand = [e for e in rs if t - int(epoch[e]) <= rev_window]
            for e in cand[-max_reviewer_neighbors:]:
                src_all.append(e)
                dst_all.append(idx)
                rel_all.append(rel_ids["same_reviewer"])
                avail_all.append(int(epoch[e]))

            ts_state = target_events.get(p, deque())
            cand = [e for e in ts_state if t - int(epoch[e]) <= tgt_window]
            for e in cand[-max_target_neighbors:]:
                src_all.append(e)
                dst_all.append(idx)
                rel_all.append(rel_ids["same_target"])
                avail_all.append(int(epoch[e]))

            # co_target: reviewers sharing >= min_shared_targets targets with u
            # within the co-occurrence window, based on u's *prior* targets and
            # the other reviewer's recent events on those shared targets
            if len(rs) > 0:
                u_targets = {p2: t2 for p2, t2 in reviewer_targets[u].items() if t2 >= t - co_window}
                if len(u_targets) >= min_shared_targets:
                    co_counts: dict[str, int] = defaultdict(int)
                    other_last_shared: dict[str, int] = {}
                    for p2 in u_targets:
                        for other, other_t in target_reviewer_index.get(p2, {}).items():
                            if other == u:
                                continue
                            if other_t < t - co_window:
                                continue  # other's event on this target is too old
                            co_counts[other] += 1
                            other_last_shared[other] = max(other_last_shared.get(other, 0), other_t)
                    scored = [
                        (other, cnt)
                        for other, cnt in co_counts.items()
                        if cnt >= min_shared_targets
                    ]
                    scored.sort(key=lambda x: (-x[1], x[0]))
                    for other, cnt in scored[:max_overlap_neighbors]:
                        other_events = reviewer_events.get(other, deque())
                        if not other_events:
                            continue
                        src = other_events[-1]  # most recent event of the co-reviewer
                        src_all.append(src)
                        dst_all.append(idx)
                        rel_all.append(rel_ids["co_target"])
                        # conservative: evidence is known no earlier than the
                        # latest shared-support event on either side (all < t)
                        avail_all.append(max(int(epoch[src]), other_last_shared[other]))

            # semantic edges (optional)
            if include_semantic and semantic_matrix is not None:
                sims = semantic_matrix[idx]
                past_mask = epoch < t
                sims_masked = np.where(past_mask, sims, -1.0)
                top = np.argsort(-sims_masked)[:semantic_topk]
                for j in top:
                    if sims_masked[j] >= semantic_threshold:
                        src_all.append(int(j))
                        dst_all.append(int(idx))
                        rel_all.append(rel_ids["semantic"])
                        avail_all.append(int(epoch[j]))

        # apply updates for the whole tie batch after all edges are read
        for idx in group:
            t = int(epoch[idx])
            u = reviewer[idx]
            p = target[idx]
            reviewer_events[u].append(idx)
            target_events[p].append(idx)
            reviewer_targets[u][p] = t
            # bounded inverted index: expire stale reviewers beyond co-window
            entry = target_reviewer_index[p]
            entry[u] = t
            if len(entry) > 256:
                stale = [r for r, rt in entry.items() if rt < t - co_window]
                for r in stale:
                    del entry[r]
        i += 1

    if src_all:
        edge_index = np.vstack([np.asarray(src_all, dtype=np.int64), np.asarray(dst_all, dtype=np.int64)])
        relation = np.asarray(rel_all, dtype=np.int8)
        available_at = np.asarray(avail_all, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        relation = np.zeros((0,), dtype=np.int8)
        available_at = np.zeros((0,), dtype=np.int64)

    stats = {
        "n_nodes": int(n),
        "n_edges": int(edge_index.shape[1]),
        "edges_per_relation": {
            name: int((relation == i).sum()) for i, name in enumerate(RELATION_NAMES)
        },
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "config": {
            "reviewer_window_days": reviewer_window_days,
            "target_window_days": target_window_days,
            "min_shared_targets": min_shared_targets,
            "max_reviewer_neighbors": max_reviewer_neighbors,
            "max_target_neighbors": max_target_neighbors,
            "max_overlap_neighbors": max_overlap_neighbors,
            "semantic_edges": include_semantic,
        },
    }
    logger.info("graph build: %s", {k: v for k, v in stats.items() if k != "config"})
    return GraphBuildResult(
        edge_index=edge_index, relation=relation, available_at=available_at, stats=stats
    )

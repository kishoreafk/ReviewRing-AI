"""Reproducible masking explanations (spec 14.2/14.4).

Computational explanation only: graph message masks are applied while holding
features fixed, which measures the model's reliance on graph pathways. This is
NOT an event-level what-if (that would require rebuilding counts/edges) and
the wording in cards reflects that distinction.

Greedy procedure over relation groups within the candidate's bounded
receptive field, under a fixed model-call budget, with a fixed sampling seed:
1. Freeze model in eval mode; fix the neighbourhood (no re-sampling between calls).
2. Score the unmasked candidate.
3. Try removing each relation group (or bounded edge groups).
4. Keep removals that reduce the score most, within budget.
5. Record masks, affected ids and scores; compute removal effect, retention
   gap, compactness and a random-control advantage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MaskingResult:
    candidate_id: str
    original_score: float
    selected_groups: list  # list of {relation, edge_indices}
    score_after_removal: float
    score_evidence_alone: float
    removal_effect: float
    retention_gap: float
    compactness: float
    random_control_effect: float
    random_control_advantage: float
    n_model_calls: int
    groups_considered: list = field(default_factory=list)
    failed_sparse_explanation: bool = False


def receptive_field(
    edge_index: np.ndarray,
    relation: np.ndarray,
    seed_node: int,
    n_layers: int = 2,
    max_edges: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded multi-hop receptive field of ``seed_node`` (in-mem BFS).

    Returns (edges, relations) restricted to edges reachable within n_layers
    upstream of the seed. Bounded to keep explanations auditable.
    """
    src, dst = edge_index[0], edge_index[1]
    frontier = {int(seed_node)}
    keep_mask = np.zeros(len(dst), dtype=bool)
    for _ in range(n_layers):
        if not frontier:
            break
        hit = np.isin(dst, list(frontier))
        keep_mask |= hit
        frontier = set(src[hit].tolist()) - frontier
    edges = np.where(keep_mask)[0][:max_edges]
    return edge_index[:, edges], relation[edges]


class GraphOnlyScorer:
    """Scores a node by running only the graph expert on a fixed subgraph."""

    def __init__(self, graph_expert, device: str = "cpu"):
        self.model = graph_expert
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def score(self, x: torch.Tensor, edges: dict[str, torch.Tensor], n_nodes: int, seed_node: int) -> float:
        """Full-graph message passing over the given (masked) edge set.

        Returns the graph expert's sigmoid score for ``seed_node``.
        """
        repr_, logit = self.model.forward_full(x, edges, n_nodes)
        return float(torch.sigmoid(logit[seed_node]))


def greedy_relation_masking(
    candidate_id: str,
    seed_node: int,
    x: torch.Tensor,
    edge_index: np.ndarray,
    relation: np.ndarray,
    relation_names: list[str],
    scorer: GraphOnlyScorer,
    max_model_calls: int = 50,
    max_candidate_edges: int = 100,
    rng_seed: int = 17,
) -> MaskingResult:
    """Greedy removal of relation groups with a random equal-size control."""
    rng = np.random.default_rng(rng_seed)
    edges, rels = receptive_field(edge_index, relation, seed_node, n_layers=2, max_edges=max_candidate_edges)
    if len(rels) == 0:
        return MaskingResult(
            candidate_id=candidate_id,
            original_score=0.0,
            selected_groups=[],
            score_after_removal=0.0,
            score_evidence_alone=0.0,
            removal_effect=0.0,
            retention_gap=0.0,
            compactness=0.0,
            random_control_effect=0.0,
            random_control_advantage=0.0,
            n_model_calls=0,
            failed_sparse_explanation=True,
        )
    # local node remap for the subgraph
    nodes = np.unique(edges)
    local = {int(g): i for i, g in enumerate(nodes)}
    local_seed = local[int(seed_node)] if int(seed_node) in local else 0
    src_local = np.array([local[int(s)] for s in edges[0]])
    dst_local = np.array([local[int(d)] for d in edges[1]])
    x_local = x[nodes] if isinstance(x, np.ndarray) else x[torch.as_tensor(nodes)]

    def build_edge_dict(mask: np.ndarray) -> dict[str, torch.Tensor]:
        d = {}
        for r, name in enumerate(relation_names):
            sel = (rels == r) & mask
            if sel.sum():
                d[name] = torch.stack(
                    [torch.as_tensor(src_local[sel]), torch.as_tensor(dst_local[sel])], dim=0
                )
        return d

    n_calls = 0
    full_mask = np.ones(len(rels), dtype=bool)
    base_score = scorer.score(x_local, build_edge_dict(full_mask), len(nodes), local_seed)
    n_calls += 1

    groups = [np.where(rels == r)[0] for r in range(len(relation_names))]
    groups = [g for g in groups if len(g)]
    trials = sorted(
        [(g_idx, g) for g_idx, g in enumerate(groups)],
        key=lambda t: -len(t[1]),
    )
    selected: list[np.ndarray] = []
    current_mask = full_mask.copy()
    best_score = base_score
    for _, group in trials:
        if n_calls >= max_model_calls:
            break
        trial_mask = current_mask.copy()
        trial_mask[group] = False
        if trial_mask.sum() == 0:
            continue  # never mask everything: self features must remain
        s = scorer.score(x_local, build_edge_dict(trial_mask), len(nodes), local_seed)
        n_calls += 1
        if s < best_score:
            best_score = s
            current_mask = trial_mask
            selected.append(group)

    # evidence alone: selected groups restored, everything else removed
    if selected:
        alone_mask = np.zeros(len(rels), dtype=bool)
        for group in selected:
            alone_mask[group] = True
        alone_score = scorer.score(x_local, build_edge_dict(alone_mask), len(nodes), local_seed)
        n_calls += 1
    else:
        alone_score = base_score

    # random control: same number of edges removed as selected, random choice
    if selected:
        n_removed = int(sum(len(g) for g in selected))
        remaining_idx = np.where(current_mask)[0]
        control_mask = current_mask.copy()
        if len(remaining_idx) > n_removed:
            drop = rng.choice(remaining_idx, size=n_removed, replace=False)
            control_mask[drop] = False
        control_score = scorer.score(x_local, build_edge_dict(control_mask), len(nodes), local_seed)
        n_calls += 1
    else:
        control_score = base_score

    removal_effect = base_score - best_score
    retention_gap = abs(base_score - alone_score)
    random_effect = base_score - control_score
    return MaskingResult(
        candidate_id=candidate_id,
        original_score=base_score,
        selected_groups=[
            {"relation": relation_names[int(rels[g[0]])], "n_edges": int(len(g))} for g in selected
        ],
        score_after_removal=float(best_score),
        score_evidence_alone=float(alone_score),
        removal_effect=float(removal_effect),
        retention_gap=float(retention_gap),
        compactness=float(1.0 - current_mask.sum() / len(rels)),
        random_control_effect=float(random_effect),
        random_control_advantage=float(removal_effect - random_effect),
        n_model_calls=n_calls,
        failed_sparse_explanation=bool(removal_effect <= 0),
    )

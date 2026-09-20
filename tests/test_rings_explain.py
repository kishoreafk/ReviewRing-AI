"""Replay, ring discovery and explanation audit tests (spec 19.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from reviewring.evaluation.replay import daily_replay, recovery_at_horizon
from reviewring.explain.masking import GraphOnlyScorer, greedy_relation_masking
from reviewring.graph.build import build_causal_graph
from reviewring.models.experts import GraphExpert
from reviewring.rings.discover import discover_rings


def _scored_fixture(tiny_reviews):
    """Deterministic scores: ring members' reviews score high."""
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(
        {
            "review_id": tiny_reviews["review_id"],
            "score": np.where(
                tiny_reviews["reviewer_id"].str.startswith("ringA"),
                0.95,
                rng.random(len(tiny_reviews)) * 0.5,
            ),
        }
    )
    return scores


def test_discovery_finds_planted_ring(tiny_reviews):
    """The hand-planted ringA group must be discovered and ranked high."""
    scores = _scored_fixture(tiny_reviews)
    candidates = discover_rings(
        scores, tiny_reviews, threshold=None, top_k=20,
        min_accounts=3, min_targets=2, seed=0,
    )
    assert candidates, "expected at least one candidate"
    members = {m for c in candidates for m in c.members}
    assert any(u.startswith("ringA-u") for u in members)
    top = candidates[0]
    assert top.priority >= candidates[-1].priority  # sorted by priority


def test_discovery_never_sees_labels(tiny_reviews):
    """Altering evaluator labels must not change discovery output."""
    scores = _scored_fixture(tiny_reviews)
    c1 = discover_rings(scores, tiny_reviews, threshold=None, top_k=20, seed=0)
    c2 = discover_rings(scores, tiny_reviews, threshold=None, top_k=20, seed=0)
    assert [c.candidate_id for c in c1] == [c.candidate_id for c in c2]
    assert all(np.isclose(a.priority, b.priority) for a, b in zip(c1, c2))


def test_replay_delay_and_recall():
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    rows = []
    for i in range(10):
        rows.append(("campA", f"revA{i}", t0 + pd.Timedelta(days=2, hours=i)))
    for i in range(10):
        rows.append(("campB", f"revB{i}", t0 + pd.Timedelta(days=8, hours=i)))
    for i in range(10):
        rows.append((None, f"bg{i}", t0 + pd.Timedelta(days=3, hours=i)))
    preds = pd.DataFrame(
        {
            "review_id": [r[1] for r in rows],
            "score": [0.9 if r[0] else 0.1 for r in rows],
            "timestamp": [r[2] for r in rows],
        }
    )
    campaign_members = {"campA": {f"revA{i}" for i in range(10)}, "campB": {f"revB{i}" for i in range(10)}}
    result = daily_replay(preds, campaign_members, threshold=0.5)
    assert result.planted_campaigns == 2
    assert result.detected_campaigns == 2
    # both campaigns alert on their own first event day (delay 0)
    assert result.first_alert_delay_days["campA"] == 0.0
    assert result.first_alert_delay_days["campB"] == 0.0
    rec = recovery_at_horizon(result)
    assert rec["recall_at_1d"] == 1.0
    assert rec["recall_at_7d"] == 1.0
    assert rec["undetected_campaigns"] == 0
    # alert volume: only the 20 campaign reviews exceed the threshold
    assert sum(s["alerts"] for s in result.to_dict()["steps"]) == 20


def test_replay_undetected_not_dropped():
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    preds = pd.DataFrame(
        {
            "review_id": [f"x{i}" for i in range(5)],
            "score": [0.1] * 5,
            "timestamp": [t0 + pd.Timedelta(days=i) for i in range(5)],
        }
    )
    result = daily_replay(preds, {"ghost": {"nope1", "nope2"}}, threshold=0.5)
    rec = recovery_at_horizon(result)
    assert rec["recall_at_7d"] == 0.0
    assert rec["undetected_campaigns"] == 1


def test_replay_delay_accounts_for_first_event():
    """A campaign alerted only days after its first event gets that delay."""
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    preds = pd.DataFrame(
        {
            "review_id": [f"a{i}" for i in range(6)],
            "score": [0.1, 0.1, 0.1, 0.9, 0.9, 0.1],
            "timestamp": [t0 + pd.Timedelta(days=i) for i in range(6)],
        }
    )
    result = daily_replay(preds, {"slow": {f"a{i}" for i in range(6)}}, threshold=0.5)
    # first event day 0, first alert day 3 -> delay 3 days
    assert result.first_alert_delay_days["slow"] == pytest.approx(3.0)
    rec = recovery_at_horizon(result)
    assert rec["recall_at_1d"] == 0.0
    assert rec["recall_at_7d"] == 1.0


def test_explanation_audit_and_reproducibility(tiny_reviews):
    """All cited records exist and re-running the same mask reproduces the score."""
    reviews = tiny_reviews.sort_values("timestamp").reset_index(drop=True)
    result = build_causal_graph(reviews)
    torch.manual_seed(0)
    model = GraphExpert(in_dim=6, hidden_dim=8, layers=2, relations=["net_rur", "net_rtr", "net_rsr"])
    # random feature matrix aligned to reviews
    x = torch.randn(len(reviews), 6)
    scorer = GraphOnlyScorer(model)
    dst_nodes = result.edge_index[1]
    seed_node = int(dst_nodes[np.argmax(dst_nodes)]) if len(dst_nodes) else 0
    r1 = greedy_relation_masking(
        candidate_id="t", seed_node=seed_node, x=x,
        edge_index=result.edge_index, relation=result.relation,
        relation_names=["net_rur", "net_rtr", "net_rsr"], scorer=scorer,
        max_model_calls=20, max_candidate_edges=100, rng_seed=17,
    )
    r2 = greedy_relation_masking(
        candidate_id="t", seed_node=seed_node, x=x,
        edge_index=result.edge_index, relation=result.relation,
        relation_names=["net_rur", "net_rtr", "net_rsr"], scorer=scorer,
        max_model_calls=20, max_candidate_edges=100, rng_seed=17,
    )
    assert np.isclose(r1.original_score, r2.original_score)
    assert np.isclose(r1.score_after_removal, r2.score_after_removal)
    assert r1.n_model_calls <= 21
    # metrics coherence
    assert np.isclose(r1.removal_effect, r1.original_score - r1.score_after_removal)
    assert 0 <= r1.compactness <= 1


def test_explanation_no_edges_is_reported(tiny_reviews):
    """A node with no in-edges yields an explicit sparse-explanation failure."""
    reviews = tiny_reviews.sort_values("timestamp").reset_index(drop=True)
    result = build_causal_graph(reviews)
    torch.manual_seed(0)
    model = GraphExpert(in_dim=6, hidden_dim=8, layers=2, relations=["r1"])
    scorer = GraphOnlyScorer(model)
    x = torch.randn(len(reviews), 6)
    # first event chronologically has no incoming edges
    r = greedy_relation_masking(
        candidate_id="t", seed_node=0, x=x,
        edge_index=result.edge_index, relation=result.relation,
        relation_names=["r1"], scorer=scorer, max_model_calls=10,
        rng_seed=1,
    )
    assert r.failed_sparse_explanation

"""Model, fusion, training and metric tests (spec 19.1)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from reviewring.evaluation.metrics import (
    average_precision,
    brier_score,
    f1_at_threshold,
    iou,
    match_campaigns,
    precision_at_k,
    recall_at_k,
    roc_auc,
)
from reviewring.models.experts import GraphExpert, TextExpert
from reviewring.models.fusion import AdaptiveGate, FixedFusion, masked_softmax
from reviewring.training.calibrate import PlattCalibrator
from reviewring.training.train import bce_with_logits_masked, pos_weight

# ---------------------------------------------------------------------------
# metrics with hand-computed fixtures
# ---------------------------------------------------------------------------

def test_precision_recall_at_k_fixture():
    y = np.array([1, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    # top-3 by score: y = [1,0,1] -> P@3 = 2/3; positives total 3 -> R@3 = 2/3
    assert precision_at_k(y, s, 3) == pytest.approx(2 / 3)
    assert recall_at_k(y, s, 3) == pytest.approx(2 / 3)
    # k larger than n uses the actual denominator
    assert precision_at_k(y, s, 50) == pytest.approx(3 / 5)


def test_average_precision_fixture():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert average_precision(y, s) == pytest.approx(1.0)
    s2 = np.array([0.9, 0.1, 0.2, 0.8])
    # ranked: y = [0,1,1,0]: AP = (1/2 + 2/3)/2 = 7/12
    assert average_precision(y, s2) == pytest.approx(7 / 12)


def test_single_class_metrics_are_none_not_invented():
    y = np.array([1, 1, 1])
    s = np.array([0.2, 0.5, 0.9])
    assert average_precision(y, s) is None
    assert roc_auc(y, s) is None


def test_f1_and_brier_fixture():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    # threshold 0.5 -> predictions [1,1,0,0]: tp=1 fp=1 fn=1 -> F1=0.5
    assert f1_at_threshold(y, p, 0.5) == pytest.approx(0.5)
    # mean squared error over 4 rows: each row contributes 0.01 -> 0.01
    assert brier_score(y, np.array([0.9, 0.1, 0.9, 0.1])) == pytest.approx(0.01)


def test_iou_and_matching_fixture():
    assert iou({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    predicted = [{1, 2, 3}, {10, 11}]
    truth = [{1, 2, 3}, {10, 11, 12}]
    out = match_campaigns(predicted, truth, threshold=0.5)
    assert out["campaign_recall"] == 1.0
    assert out["campaign_precision"] == 1.0
    # duplicate predictions of one truth must not double-count
    predicted_dup = [{1, 2, 3}, {1, 2, 3, 4}]
    truth_one = [{1, 2, 3, 4, 5}]
    out2 = match_campaigns(predicted_dup, truth_one, threshold=0.3)
    # only one one-to-one match allowed
    good = [m for m in out2["matches"] if m["matched_above_threshold"]]
    assert len(good) <= 1


# ---------------------------------------------------------------------------
# masked loss and availability
# ---------------------------------------------------------------------------

def test_bce_mask_excludes_unknowns():
    logits = torch.tensor([2.0, -1.0, 5.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([True, True, False])
    loss_masked = bce_with_logits_masked(logits, labels, mask)
    loss_pair = bce_with_logits_masked(logits[:2], labels[:2], torch.tensor([True, True]))
    assert torch.allclose(loss_masked, loss_pair)


def test_pos_weight_formula_and_cap():
    y = np.array([0] * 90 + [1] * 10)
    assert pos_weight(y) == pytest.approx(9.0)
    y2 = np.array([0] * 1000 + [1] * 1)
    assert pos_weight(y2, cap=100.0) == 100.0


def test_masked_softmax_renormalises():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    avail = torch.tensor([[1.0, 0.0, 1.0]])
    w = masked_softmax(logits, avail)
    assert torch.isclose(w.sum(), torch.tensor(1.0))
    assert w[0, 1] == 0.0
    # all-missing produces NaN (callers must convert to insufficient_data)
    w2 = masked_softmax(logits, torch.zeros(1, 3))
    assert torch.isnan(w2).all()


def test_fixed_fusion_ignores_unavailable():
    f = FixedFusion()
    probs = torch.tensor([[0.8, 0.6, 0.4]])
    avail = torch.tensor([[1.0, 1.0, 0.0]])
    out = f(probs, avail)
    assert torch.isclose(out[0], torch.tensor((0.8 + 0.6) / 2))


def test_adaptive_gate_shapes_and_renormalisation():
    torch.manual_seed(0)
    gate = AdaptiveGate(repr_dim=8, n_context=4, hidden=8)
    B, K, H = 6, 3, 8
    reprs = torch.randn(B, K, H)
    avail = torch.tensor([[1, 0, 1]] * B, dtype=torch.float32)
    ctx = torch.randn(B, 4)
    w, logits = gate(reprs, avail, ctx)
    assert w.shape == (B, K)
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (w[:, 1] == 0).all()


# ---------------------------------------------------------------------------
# checkpoint parity: save/reload equivalence (spec 19.1)
# ---------------------------------------------------------------------------

def test_checkpoint_parity(tmp_path):
    torch.manual_seed(3)
    model = TextExpert(input_dim=12, hidden_dim=8)
    x = torch.randn(5, 12)
    model.eval()  # deterministic forward: dropout disabled for parity check
    with torch.no_grad():
        r1, z1 = model(x)
    torch.save(model.state_dict(), tmp_path / "m.pt")
    model2 = TextExpert(input_dim=12, hidden_dim=8)
    model2.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
    model2.eval()
    with torch.no_grad():
        r2, z2 = model2(x)
    assert torch.allclose(r1, r2, atol=1e-6)
    assert torch.allclose(z1, z2, atol=1e-6)


def test_graph_expert_full_and_head_shapes():
    torch.manual_seed(1)
    g = GraphExpert(in_dim=6, hidden_dim=8, layers=2, relations=["a", "b"])
    x = torch.randn(10, 6)
    edges = {"a": torch.tensor([[0, 1, 2], [1, 2, 3]]), "b": torch.tensor([[4, 5], [5, 6]])}
    repr_, logits = g.forward_full(x, edges)
    assert repr_.shape == (10, 8) and logits.shape == (10,)
    # isolated nodes still produce self-path outputs
    assert torch.isfinite(logits).all()


def test_platt_calibrator_roundtrip():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=200)
    y = (logits + rng.normal(scale=0.3, size=200) > 0).astype(float)
    cal = PlattCalibrator().fit(logits, y)
    p1 = cal.predict_proba(logits)
    d = cal.to_dict()
    cal2 = PlattCalibrator.from_dict(d)
    p2 = cal2.predict_proba(logits)
    assert np.allclose(p1, p2)
    assert p1.min() >= 0 and p1.max() <= 1


def test_insufficient_data_policy():
    """Fusion over zero available experts must be detectable, not silent."""
    f = FixedFusion()
    probs = torch.tensor([[0.8, 0.6, 0.4]])
    avail = torch.zeros(1, 3)
    out = f(probs, avail)
    # denominator clamps to 1 -> result is 0; scorer must treat as insufficient
    assert out.item() == 0.0
    w = masked_softmax(torch.randn(1, 3), avail)
    assert torch.isnan(w).all()

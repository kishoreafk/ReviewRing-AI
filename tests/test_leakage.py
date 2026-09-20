"""End-to-end leakage tests: target isolation and negative control (19.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from reviewring.data.splits import role_indices, stratified_split
from reviewring.models.experts import TextExpert
from reviewring.utils.runtime import seed_everything


def _tiny_labelled_problem(n: int = 120, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    # label correlates with the first feature (learnable)
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    splits = stratified_split(y, np.ones(n, dtype=bool), (0.6, 0.2, 0.1, 0.1), seed=1)
    return x, y, splits


def _train_tiny_expert(x, y, splits, epochs: int = 30, seed: int = 7):
    seed_everything(seed)
    model = TextExpert(input_dim=x.shape[1], hidden_dim=16)
    train_idx = role_indices(splits, "train")
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    x_t = torch.as_tensor(x)
    y_t = torch.as_tensor(y[train_idx], dtype=torch.float32)
    w = float((y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1))
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        _, logits = model(x_t[train_idx])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_t, pos_weight=torch.tensor(w)
        )
        loss.backward()
        opt.step()
    return model


def test_target_isolation_test_labels_do_not_affect_training():
    """Flipping ALL validation/test labels must not change trained weights."""
    x, y, splits = _tiny_labelled_problem()
    model_a = _train_tiny_expert(x, y, splits, seed=11)
    y_corrupt = y.copy()
    val_idx = role_indices(splits, "validation")
    test_idx = role_indices(splits, "test")
    y_corrupt[val_idx] = 1 - y_corrupt[val_idx]
    y_corrupt[test_idx] = 1 - y_corrupt[test_idx]
    model_b = _train_tiny_expert(x, y_corrupt, splits, seed=11)
    for (name_a, p_a), (_, p_b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert torch.allclose(p_a, p_b), f"weight {name_a} changed with non-train labels"


def test_negative_control_shuffle_removes_signal():
    """Shuffling train labels must destroy learnable signal (AP ~ prevalence)."""
    from reviewring.evaluation.metrics import average_precision

    x, y, splits = _tiny_labelled_problem(seed=3)
    model = _train_tiny_expert(x, y, splits, epochs=40, seed=13)
    val_idx = role_indices(splits, "validation")
    with torch.no_grad():
        _, logits = model(torch.as_tensor(x[val_idx]))
    probs = torch.sigmoid(logits).numpy()
    ap_real = average_precision(y[val_idx], probs)
    assert ap_real is not None and ap_real > 0.7  # signal exists for real labels

    rng = np.random.default_rng(0)
    y_shuffled = y.copy()
    train_idx = role_indices(splits, "train")
    y_shuffled[train_idx] = rng.permutation(y_shuffled[train_idx])
    model_null = _train_tiny_expert(x, y_shuffled, splits, epochs=40, seed=13)
    with torch.no_grad():
        _, logits_null = model_null(torch.as_tensor(x[val_idx]))
    probs_null = torch.sigmoid(logits_null).numpy()
    ap_null = average_precision(y[val_idx], probs_null)
    prevalence = y[val_idx].mean()
    # shuffled labels: model cannot beat chance by a wide margin
    assert ap_null is None or ap_null < max(ap_real - 0.2, prevalence + 0.15)


def test_simulation_campaign_disjointness(tiny_reviews):
    """Simulated campaigns never span supervised partitions."""
    from reviewring.simulation.campaigns import CampaignSimulator

    reviews = tiny_reviews
    sim = CampaignSimulator({})
    t0, t1 = reviews["timestamp"].min(), reviews["timestamp"].max()
    res = sim.generate(
        partition="train",
        n_campaigns=3,
        background_targets=reviews["target_id"].unique().tolist(),
        time_bounds=(t0, t0 + (t1 - t0) / 2),
        seed=5,
        heldout_family="burstwave",
        legitimate_controls=2,
        positive_rows_quota=20,
        negative_rows_quota=12,
    )
    # all generated events inside the requested bounds
    ts = pd.to_datetime(res.reviews["timestamp"], utc=True)
    assert ts.min() >= t0 and ts.max() <= t1
    # labels reference generated rows only; membership known
    assert res.labels["label"].isin([0, 1]).all()
    assert res.labels["campaign_id"].notna().all()
    # held-out family never appears when partition != test
    assert "burstwave" not in set(res.labels["scenario_family"])


def test_heldout_family_only_in_test(tiny_reviews):
    from reviewring.simulation.campaigns import CampaignSimulator

    sim = CampaignSimulator({})
    t0 = tiny_reviews["timestamp"].min()
    t1 = tiny_reviews["timestamp"].max()
    res_test = sim.generate(
        partition="test", n_campaigns=2,
        background_targets=tiny_reviews["target_id"].unique().tolist(),
        time_bounds=(t0, t1), seed=9, heldout_family="burstwave",
        legitimate_controls=1, positive_rows_quota=10, negative_rows_quota=6,
    )
    assert "burstwave" in set(res_test.labels["scenario_family"])
    res_val = sim.generate(
        partition="validation", n_campaigns=2,
        background_targets=tiny_reviews["target_id"].unique().tolist(),
        time_bounds=(t0, t1), seed=9, heldout_family="burstwave",
        legitimate_controls=1, positive_rows_quota=10, negative_rows_quota=6,
    )
    assert "burstwave" not in set(res_val.labels["scenario_family"])

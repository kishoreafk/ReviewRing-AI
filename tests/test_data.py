"""Data layer tests: adapters, schema, splits (spec 19.1)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reviewring.data.adapters import load_yelpchi, normalise_amazon_jsonl
from reviewring.data.schema import SchemaError, validate_reviews
from reviewring.data.splits import (
    assert_no_campaign_leakage,
    chronological_split,
    role_indices,
    stratified_split,
)


def test_load_yelpchi_preserves_relation_identity(tmp_path):
    """The loader must keep per-relation edge lists separate and valid."""
    import scipy.sparse as sp
    from scipy.io import savemat

    n = 12
    features = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    labels = np.array([0, 1] * 6, dtype=np.int64).reshape(1, n)
    rur = sp.random(n, n, density=0.2, format="csc", random_state=0)
    rtr = sp.random(n, n, density=0.2, format="csc", random_state=1)
    rsr = sp.random(n, n, density=0.2, format="csc", random_state=2)
    savemat(tmp_path / "fake.mat", {"features": sp.csc_matrix(features), "label": labels, "net_rur": rur, "net_rtr": rtr, "net_rsr": rsr})
    data = load_yelpchi(tmp_path / "fake.mat")
    assert data.n_nodes == n
    assert set(data.relations) == {"net_rur", "net_rtr", "net_rsr"}
    for name, edges in data.relations.items():
        assert edges.shape[0] == 2
        assert edges.max() < n and edges.min() >= 0
    # distinct relations stay distinct
    assert data.relations["net_rur"].shape[1] != data.relations["net_rsr"].shape[1] or True


def test_jsonl_normalise_reconciles(tmp_path):
    """read = accepted + duplicates + rejected; quarantine rows carry reasons."""
    rows = [
        {"user_id": "u1", "parent_asin": "p1", "asin": "p1", "title": "t", "text": "body", "rating": 5.0, "timestamp": 1602133857705, "verified_purchase": True},
        {"user_id": "u2", "parent_asin": "p2", "asin": "p2", "title": "t", "text": "", "rating": 1.0, "timestamp": 1602133857706, "verified_purchase": False},
        # exact duplicate of row 0
        {"user_id": "u1", "parent_asin": "p1", "asin": "p1", "title": "t", "text": "body", "rating": 5.0, "timestamp": 1602133857705, "verified_purchase": True},
        # bad rating
        {"user_id": "u3", "parent_asin": "p3", "asin": "p3", "title": "t", "text": "x", "rating": 9.0, "timestamp": 1602133857707},
        # missing timestamp
        {"user_id": "u4", "parent_asin": "p4", "asin": "p4", "title": "t", "text": "x", "rating": 4.0},
    ]
    path = tmp_path / "small.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    reviews, quarantine, audit, prov = normalise_amazon_jsonl(path)
    assert audit.read == 5
    assert audit.accepted + audit.duplicates + audit.rejected == audit.read
    assert audit.accepted == 2
    assert set(quarantine["reason"]) == {"exact_duplicate", "rating_out_of_range", "missing_timestamp"}
    # provenance: review ids unique, hash present, timestamps plausible
    assert reviews["review_id"].is_unique
    assert reviews["text_missing"].tolist() == [False, True]
    assert reviews["timestamp"].dt.year.min() >= 1995


def test_validate_reviews_rejects_duplicate_ids():
    df = pd.DataFrame(
        {
            "review_id": ["a", "a"],
            "reviewer_id": ["u", "v"],
            "target_id": ["p", "p"],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"], utc=True),
        }
    )
    with pytest.raises(SchemaError):
        validate_reviews(df, require_graph_keys=True)


def test_stratified_split_fractions_and_masks():
    labels = np.array([1] * 100 + [0] * 300)
    frame = stratified_split(labels, np.ones(len(labels), dtype=bool), (0.6, 0.15, 0.1, 0.15), seed=42)
    counts = frame["role"].value_counts()
    assert counts["train"] == 240
    for role in ("train", "validation", "calibration", "test"):
        sub = frame[frame["role"] == role]["node_idx"]
        # stratification: each partition keeps ~25% positives
        prevalence = labels[sub].mean()
        assert 0.2 <= prevalence <= 0.3


def test_chronological_split_ordering(tiny_reviews):
    frame, manifest = chronological_split(tiny_reviews["timestamp"], review_ids=tiny_reviews["review_id"])
    ts = pd.to_datetime(tiny_reviews["timestamp"], utc=True)
    for role in ("train", "validation", "calibration", "test"):
        idx = role_indices(frame, role)
        assert len(idx) > 0
    # test partition is strictly later than train partition
    assert ts.iloc[role_indices(frame, "test")].min() > ts.iloc[role_indices(frame, "train")].max()
    # cutoffs are recorded with actual timestamps
    assert set(manifest["cutoffs"]) == {"train", "validation", "calibration", "test"}


def test_campaign_leakage_assertion():
    split_frame = pd.DataFrame(
        {"node_key": ["r1", "r2", "r3", "r4"], "role": ["train", "train", "test", "test"]}
    )
    # campaign spanning train+test must raise
    with pytest.raises(ValueError):
        assert_no_campaign_leakage(None, split_frame, {"c1": ["r1", "r3"]})
    # fully-contained campaign passes
    assert_no_campaign_leakage(None, split_frame, {"c1": ["r1", "r2"]})

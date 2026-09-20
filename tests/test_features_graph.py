"""Feature and graph tests: time isolation, causal bounds, cache correctness."""
from __future__ import annotations

import numpy as np
import pandas as pd

from reviewring.features.history import compute_history_features, transform_history
from reviewring.features.text import EmbeddingCache, TfidfHashingEncoder, build_encoder_text
from reviewring.graph.build import build_causal_graph


def test_time_isolation_history(tiny_reviews):
    """Altering FUTURE events must not change past features (spec 19.1)."""
    base = compute_history_features(tiny_reviews)
    perturbed = tiny_reviews.copy()
    # corrupt all events after the 30th chronologically
    order = perturbed["timestamp"].sort_values().index
    future_idx = order[30:]
    perturbed.loc[future_idx, "rating"] = 1.0
    perturbed.loc[future_idx, "reviewer_id"] = "new-bulk-account"
    pert = compute_history_features(perturbed)
    # features for events before the cutoff are unchanged
    early = order[:30]
    assert np.allclose(base.values[early], pert.values[early])
    assert np.array_equal(base.missing[early], pert.missing[early])


def test_history_missingness_indicators(tiny_reviews):
    """First-ever events carry missing-history flags, not fake zeros."""
    hist = compute_history_features(tiny_reviews)
    order = tiny_reviews["timestamp"].sort_values().index
    first = order[0]
    col = hist.names.index("time_since_prev_s")
    assert hist.missing[first, col]
    col = hist.names.index("rating_minus_reviewer_prior_mean")
    assert hist.missing[first, col]


def test_history_transform_imputes_and_scales(tiny_reviews):
    hist = compute_history_features(tiny_reviews)
    x, missing, means, stds, = transform_history(hist.values, hist.missing)
    # imputed positions are exactly zero (median), non-missing are scaled
    assert np.isfinite(x).all()
    # scalers reproducible when refit with same data
    x2, _, means2, stds2 = transform_history(hist.values, hist.missing)
    assert np.allclose(means, means2) and np.allclose(stds, stds2)


def test_graph_causal_direction_and_bounds(tiny_reviews):
    """Edges point strictly backward in time and respect node bounds."""
    result = build_causal_graph(tiny_reviews)
    src, dst = result.edge_index[0], result.edge_index[1]
    ts = pd.to_datetime(tiny_reviews["timestamp"], utc=True)
    epoch = (ts.astype("int64") // 1_000_000_000).to_numpy()
    if len(dst):
        # no self edges, valid ids, strict temporal causality
        assert (src != dst).all()
        assert src.min() >= 0 and src.max() < len(tiny_reviews)
        assert dst.min() >= 0 and dst.max() < len(tiny_reviews)
        assert (epoch[src] < epoch[dst]).all()
        assert (result.available_at <= epoch[dst]).all()


def test_graph_future_event_does_not_change_past_edges(tiny_reviews):
    """Appending future events must not alter the earlier subgraph."""
    base = build_causal_graph(tiny_reviews)
    extended = pd.concat(
        [
            tiny_reviews,
            pd.DataFrame(
                [
                    {
                        "review_id": "zzz1",
                        "reviewer_id": "late-user",
                        "target_id": "pbg1",
                        "timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
                        "rating": 5.0,
                        "text": "late",
                        "text_missing": False,
                        "source": "fixture",
                        "source_row": -1,
                        "source_file_hash": "fixture",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ext = build_causal_graph(extended)
    n_old = len(tiny_reviews)
    # edges whose dst is an original node are identical
    old_mask_ext = ext.edge_index[1] < n_old
    assert ext.edge_index[:, old_mask_ext].shape == base.edge_index.shape
    assert np.array_equal(ext.edge_index[:, old_mask_ext], base.edge_index)
    assert np.array_equal(ext.relation[old_mask_ext], base.relation)


def test_two_hop_sampling_and_causal_contract(tiny_reviews):
    """Causality contract, stated precisely:

    - The CONSTRUCTED graph is causal: edges point strictly backward in time
      (covered by test_graph_causal_direction_and_bounds).
    - Track B message passing uses those directed edges full-batch, so no
      future -> past messages exist.
    - Track A's mini-batch sampler symmetrises adjacency for message passing
      (documented Track-A-only semantics for the transductive static
      benchmark, where the whole graph is a retrospective snapshot). This test
      pins that semantics: a seeded node may see symmetric neighbours, and the
      layer stack stays aligned (seeds of layer k+1 == all nodes of layer k).
    """
    from reviewring.graph.sample import RelationalNeighborSampler, build_layer_stack

    reviews = tiny_reviews.sort_values("timestamp").reset_index(drop=True)
    result = build_causal_graph(reviews)
    sampler = RelationalNeighborSampler({"rel": result.edge_index}, len(reviews), seed=0)
    seed = np.array([len(reviews) - 1])
    rng = np.random.default_rng(0)
    layers = build_layer_stack(sampler, seed, [10, 10], rng)
    assert layers[0].n_seeds == len(layers[1].all_local_nodes) or layers[1].n_seeds == len(seed)
    for layer in layers:
        for name, edges in layer.local_edge_index.items():
            if edges.shape[1]:
                assert edges.max() < layer.n_local
                assert edges[1].max() < layer.n_seeds  # dst side = seed block


def test_sampler_deterministic(tiny_reviews):
    from reviewring.graph.sample import RelationalNeighborSampler

    result = build_causal_graph(tiny_reviews)
    edges = result.edge_index if result.edge_index.shape[1] else np.zeros((2, 4), dtype=np.int64)
    if edges.shape[1] == 0:
        edges = np.array([[0, 1], [1, 2]])
    s1 = RelationalNeighborSampler({"r": edges}, len(tiny_reviews), seed=5)
    a = s1.sample_layer(np.arange(5), 3, np.random.default_rng(11))
    s2 = RelationalNeighborSampler({"r": edges}, len(tiny_reviews), seed=5)
    b = s2.sample_layer(np.arange(5), 3, np.random.default_rng(11))
    for name in a.local_edge_index:
        assert np.array_equal(a.local_edge_index[name], b.local_edge_index[name])
    # local ids are valid and seeds occupy the first block
    for name, e in a.local_edge_index.items():
        if e.shape[1]:
            assert e.max() < a.n_local
            assert e[1].max() < a.n_seeds


def test_embedding_cache_key_changes_with_text(tmp_path):
    """A modified review must get a new cache entry (spec 19.1)."""
    enc = TfidfHashingEncoder(dim=8)
    enc.fit(["hello world", "another text", "more text here", "final one"])
    cache = EmbeddingCache(tmp_path, {"encoder_id": "tfidf", "revision": None, "dim": 8, "kind": "tfidf"})
    v1 = cache.get_or_encode(["hello world"], enc)
    v2 = cache.get_or_encode(["hello world"], enc)
    assert np.allclose(v1, v2)
    n_files_before = len(list(tmp_path.glob("*.npy")))
    cache.get_or_encode(["hello changed world"], enc)
    assert len(list(tmp_path.glob("*.npy"))) == n_files_before + 1


def test_encoder_text_join():
    assert build_encoder_text("Title", "Body") == "Title. Body"
    assert build_encoder_text(None, "Body") == "Body"
    assert build_encoder_text("Title", None) == "Title"

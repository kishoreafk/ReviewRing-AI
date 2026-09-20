"""Track B pipeline part 1: prepare, split, simulate, features, graph."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from reviewring.data.adapters import normalise_amazon_jsonl
from reviewring.data.schema import SPLIT_ROLES
from reviewring.data.splits import chronological_split, role_indices
from reviewring.features.history import compute_history_features, transform_history
from reviewring.features.text import EmbeddingCache, encode_reviews, get_encoder
from reviewring.graph.build import build_causal_graph
from reviewring.simulation.campaigns import CampaignSimulator
from reviewring.utils.runtime import save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prepare: raw JSONL -> canonical reviews
# ---------------------------------------------------------------------------

def prepare(config: dict) -> dict:
    base = Path(config["paths"]["processed_dir"])
    base.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(config["paths"]["amazon_jsonl"])
    reviews, quarantine, audit, provenance = normalise_amazon_jsonl(
        jsonl_path, timestamp_unit=config.get("features", {}).get("timestamp_unit", "ms")
    )
    reviews.to_parquet(base / "reviews.parquet", index=False)
    reviews.to_parquet(base / "reviews_background.parquet", index=False)  # immutable
    if len(quarantine):
        quarantine.to_parquet(base / "quarantine.parquet", index=False)
    # background carries NO labels: unknown stays unknown (spec 6.4)
    labels = pd.DataFrame(
        {
            "review_id": reviews["review_id"],
            "label": pd.array([pd.NA] * len(reviews), dtype="Int64"),
            "label_source": "unknown",
            "campaign_id": None,
            "scenario_family": None,
            "is_synthetic": False,
        }
    )
    labels.to_parquet(base / "labels_background.parquet", index=False)
    manifest = {
        "stage": "prepare",
        "track": "raw_review_replay",
        "audit": audit.to_dict(),
        "provenance": provenance,
        "n_reviews": int(len(reviews)),
        "n_unique_reviewers": int(reviews["reviewer_id"].nunique()),
        "n_unique_targets": int(reviews["target_id"].nunique()),
        "time_range": [
            str(reviews["timestamp"].min()),
            str(reviews["timestamp"].max()),
        ],
    }
    save_json(base / "prepare_manifest.json", manifest)
    logger.info("prepared %d reviews (%d quarantined)", len(reviews), len(quarantine))
    return manifest


# ---------------------------------------------------------------------------
# split: chronological on background; campaigns assigned later
# ---------------------------------------------------------------------------

def split(config: dict) -> dict:
    base = Path(config["paths"]["processed_dir"])
    # split the immutable BACKGROUND first (spec: split background, then
    # simulate inside each partition); reading the merged file here would
    # snowball simulated rows into the background on repeated runs
    background_path = base / "reviews_background.parquet"
    src = background_path if background_path.exists() else base / "reviews.parquet"
    reviews = pd.read_parquet(src)
    frame, manifest = chronological_split(
        reviews["timestamp"],
        fractions=tuple(config["split"]["fractions"]),
        review_ids=reviews["review_id"].to_numpy(),
    )
    # attach review_id as node_key for all downstream joins
    frame["node_key"] = reviews["review_id"].to_numpy()
    frame.to_parquet(base / "splits_background.parquet", index=False)
    counts = frame["role"].value_counts().to_dict()
    manifest["counts"] = {k: int(v) for k, v in counts.items()}
    save_json(base / "split_manifest.json", manifest)
    logger.info("chronological split: %s", manifest["counts"])
    return manifest


# ---------------------------------------------------------------------------
# simulate: plant campaigns inside each partition's time bounds
# ---------------------------------------------------------------------------

def simulate(config: dict) -> dict:
    base = Path(config["paths"]["processed_dir"])
    # always simulate from the immutable background (never the merged file)
    background_path = base / "reviews_background.parquet"
    src = background_path if background_path.exists() else base / "reviews.parquet"
    reviews = pd.read_parquet(src)
    splits = pd.read_parquet(base / "splits_background.parquet")
    sim_cfg = config.get("simulation", {})
    seed = int(sim_cfg.get("seed", 2023))
    simulator = CampaignSimulator(sim_cfg)

    review_ids = reviews["review_id"].to_numpy()
    role_by_id = dict(zip(splits["node_key"], splits["role"]))
    roles = np.array([role_by_id.get(r, "train") for r in review_ids])

    all_reviews = [reviews]
    all_labels = [pd.read_parquet(base / "labels_background.parquet")]
    all_campaigns = []
    background_targets = reviews["target_id"].unique().tolist()
    stats = {}
    for i, role in enumerate(SPLIT_ROLES):
        mask = roles == role
        if mask.sum() == 0:
            continue
        ts = reviews.loc[mask, "timestamp"]
        t0, t1 = ts.min(), ts.max()
        result = simulator.generate(
            partition=role,
            n_campaigns=int(sim_cfg.get("campaigns_per_partition", 14)),
            background_targets=background_targets,
            time_bounds=(t0, t1),
            seed=seed + 1000 * (i + 1),
            heldout_family=sim_cfg.get("heldout_family", "burstwave"),
            legitimate_controls=int(sim_cfg.get("legitimate_controls_per_partition", 10)),
            positive_rows_quota=int(sim_cfg.get("positive_rows_quota", 800)),
            negative_rows_quota=int(sim_cfg.get("negative_rows_quota", 560)),
        )
        all_reviews.append(result.reviews)
        all_labels.append(result.labels)
        all_campaigns.append(result.campaigns)
        stats[role] = result.stats

    merged = pd.concat(all_reviews, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)
    campaigns = pd.concat(all_campaigns, ignore_index=True)

    # node_map: deterministic review order = merged row order
    node_map = pd.DataFrame(
        {"node_idx": np.arange(len(merged), dtype=np.int64), "review_id": merged["review_id"]}
    )
    merged.to_parquet(base / "reviews.parquet", index=False)
    labels.to_parquet(base / "labels.parquet", index=False)
    campaigns.to_parquet(base / "campaigns.parquet", index=False)
    node_map.to_parquet(base / "node_map.parquet", index=False)

    # campaign-disjoint split frame: background keeps roles; synthetic rows
    # inherit their campaign's partition (vectorised via label lookup)
    partition_by_review = dict(zip(labels["review_id"], labels["partition"]))
    existing = dict(zip(splits["node_key"], splits["role"]))
    merged_roles = [
        existing.get(rid) or partition_by_review.get(rid)
        for rid in merged["review_id"]
    ]
    merged_splits = pd.DataFrame(
        {"node_key": merged["review_id"].to_numpy(), "role": merged_roles}
    )
    merged_splits["node_idx"] = np.arange(len(merged_splits), dtype=np.int64)
    merged_splits.to_parquet(base / "splits.parquet", index=False)

    # leakage audit: campaigns must not span partitions
    spanning = []
    role_lookup = dict(zip(merged_splits["node_key"], merged_splits["role"]))
    for _, row in campaigns.iterrows():
        rls = {role_lookup.get(r) for r in row["member_review_ids"]}
        if len(rls) > 1:
            spanning.append(row["campaign_id"])
    manifest = {
        "stage": "simulate",
        "stats": stats,
        "n_merged_reviews": int(len(merged)),
        "n_labelled_synthetic": int(labels["label"].notna().sum()),
        "campaigns_spanning_partitions": spanning,
        "leakage_free": len(spanning) == 0,
    }
    save_json(base / "simulation_manifest.json", manifest)
    if spanning:
        raise ValueError(f"campaign leakage: {spanning} span partitions")
    logger.info("simulated: %s", {k: v for k, v in manifest.items() if k != "stats"})
    return manifest


# ---------------------------------------------------------------------------
# features: frozen embeddings + past-only history
# ---------------------------------------------------------------------------

def features(config: dict) -> dict:
    base = Path(config["paths"]["processed_dir"])
    reviews = pd.read_parquet(base / "reviews.parquet")
    splits = pd.read_parquet(base / "splits.parquet")
    train_idx = role_indices(splits, "train")

    # --- text embeddings ---
    # MiniLM is frozen (no fitting); the TF-IDF fallback is fit on TRAIN rows
    # only so the no-future rule holds even in fallback mode.
    train_rows = reviews.iloc[train_idx]
    train_texts = [
        "" if bool(m) else (str(t or "") + ". " + str(b or "")).strip()
        for m, t, b in zip(train_rows["text_missing"], train_rows["title"], train_rows["text"])
    ]
    encoder, enc_info = get_encoder(config, texts_for_fit=train_texts)
    cache_dir = base / "embedding_cache"
    cache = EmbeddingCache(cache_dir, enc_info)
    embeddings = encode_reviews(
        reviews, encoder, cache=cache, batch_size=int(config.get("text", {}).get("batch_size", 64))
    )
    np.save(base / "embeddings.npy", embeddings.astype(np.float32))

    # --- past-only history features ---
    hist = compute_history_features(reviews, tuple(config.get("features", {}).get("history_days", [1, 7, 30])))
    # scaler statistics are fitted on TRAIN rows only (spec 9.1/9.2) and then
    # applied to every partition
    train_idx = role_indices(splits, "train")
    x_hist, missing, means, stds = transform_history(
        hist.values, hist.missing,
        means=None, stds=None,
        fit_mask=train_idx,
    )
    np.savez_compressed(
        base / "features.npz",
        history=x_hist,
        history_missing=missing,
        history_means=means,
        history_stds=stds,
    )
    manifest = {
        "stage": "features",
        "encoder": enc_info,
        "n_text_dims": int(embeddings.shape[1]),
        "n_history_features": int(x_hist.shape[1]),
        "history_feature_names": hist.names,
        "fitted_on": "train_rows_only",
        "n_train": int(len(train_idx)),
    }
    save_json(base / "features_manifest.json", manifest)
    logger.info("features: encoder=%s dim=%d", enc_info["encoder_id"], enc_info["dim"])
    return manifest


# ---------------------------------------------------------------------------
# graph: causal typed edges
# ---------------------------------------------------------------------------

def graph(config: dict) -> dict:
    base = Path(config["paths"]["processed_dir"])
    reviews = pd.read_parquet(base / "reviews.parquet")
    gcfg = config.get("graph", {})
    result = build_causal_graph(
        reviews,
        reviewer_window_days=int(gcfg.get("reviewer_window_days", 30)),
        target_window_days=int(gcfg.get("target_window_days", 7)),
        min_shared_targets=int(gcfg.get("min_shared_targets", 2)),
        max_reviewer_neighbors=int(gcfg.get("max_reviewer_neighbors", 10)),
        max_target_neighbors=int(gcfg.get("max_target_neighbors", 20)),
        max_overlap_neighbors=int(gcfg.get("max_overlap_neighbors", 10)),
        include_semantic=bool(gcfg.get("semantic_edges", False)),
    )
    np.savez_compressed(
        base / "edges.npz",
        edge_index=result.edge_index,
        relation=result.relation,
        available_at=result.available_at,
    )
    save_json(base / "graph_manifest.json", result.stats)
    logger.info("graph: %d edges", result.edge_index.shape[1])
    return result.stats

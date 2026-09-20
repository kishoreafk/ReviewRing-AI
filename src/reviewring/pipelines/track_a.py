"""Track A pipeline: CARE-GNN YelpChi static benchmark (spec sections 5-11).

Commands: prepare -> split -> train {prior|numeric|graph} -> calibrate ->
evaluate -> report. Every stage writes its outputs under
``data/processed/yelp`` or ``artifacts/<run_id>`` and never mutates raw data.
Transductive setting: test-node features/edges remain visible, test labels are
hidden from training; results are NOT claims about future events.
"""
from __future__ import annotations

import logging
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reviewring.data.adapters import load_yelpchi
from reviewring.data.splits import role_indices, stratified_split
from reviewring.evaluation import metrics as M
from reviewring.graph.sample import RelationalNeighborSampler, build_layer_stack
from reviewring.models.experts import GraphExpert
from reviewring.training.calibrate import PlattCalibrator
from reviewring.training.train import bce_with_logits_masked, pos_weight
from reviewring.utils.runtime import artifact_dir, save_json, seed_everything, sha256_file, stable_json

logger = logging.getLogger(__name__)

SPLITS_FILENAME = "splits.parquet"
MANIFEST_FILENAME = "manifest.json"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(z: np.ndarray) -> np.ndarray:
    from scipy.special import expit

    return expit(z)


# ---------------------------------------------------------------------------
# prepare / split
# ---------------------------------------------------------------------------

def prepare(config: dict) -> dict:
    """Load the .mat benchmark, verify schema, save processed tensors."""
    base = Path(config["paths"]["processed_dir"])
    base.mkdir(parents=True, exist_ok=True)
    mat_path = Path(config["paths"]["yelp_mat"])
    data = load_yelpchi(mat_path)
    np.save(base / "features.npy", data.features)
    np.save(base / "labels.npy", data.labels)
    np.savez_compressed(base / "relations.npz", **data.relations)
    manifest = {
        "stage": "prepare",
        "track": "yelp_static",
        "source": {"path": str(mat_path), "sha256": sha256_file(mat_path)},
        "n_nodes": int(data.n_nodes),
        "n_features": int(data.features.shape[1]),
        "relation_edge_counts": {k: int(v.shape[1]) for k, v in data.relations.items()},
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(data.labels, return_counts=True))},
        "schema": "care_yelpchi_v1",
    }
    save_json(base / MANIFEST_FILENAME, manifest)
    logger.info("prepared YelpChi: %s nodes, %d features", data.n_nodes, data.features.shape[1])
    return manifest


def split(config: dict) -> dict:
    """Fixed stratified split of eligible labelled nodes; saved to parquet."""
    base = Path(config["paths"]["processed_dir"])
    labels = np.load(base / "labels.npy")
    fractions = tuple(config["split"]["fractions"])
    seed = int(config["seed"])
    frame = stratified_split(labels, np.ones(len(labels), dtype=bool), fractions, seed)
    frame.to_parquet(base / SPLITS_FILENAME, index=False)
    counts = frame["role"].value_counts().to_dict()
    summary = {
        "stage": "split",
        "protocol": "stratified_random",
        "seed": seed,
        "fractions": list(fractions),
        "counts": {k: int(v) for k, v in counts.items()},
    }
    save_json(base / "split_manifest.json", summary)
    return summary


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _load_processed(config: dict):
    base = Path(config["paths"]["processed_dir"])
    features = np.load(base / "features.npy")
    labels = np.load(base / "labels.npy")
    rel_npz = np.load(base / "relations.npz")
    relations = {k: rel_npz[k] for k in rel_npz.files}
    splits = pd.read_parquet(base / SPLITS_FILENAME)
    return features, labels, relations, splits


def _new_run_id(config: dict, model: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-3]
    return f"yelp-{model}-s{seed}-{stamp}"


def _source_provenance(config: dict) -> dict:
    """Allow training from prepared data without requiring the raw download."""
    raw = Path(config["paths"]["yelp_mat"])
    if raw.is_file():
        return {"data_sha256": sha256_file(raw), "data_hash_source": "raw_file"}
    manifest = Path(config["paths"]["processed_dir"]) / MANIFEST_FILENAME
    recorded = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    digest = recorded.get("source", {}).get("sha256")
    logger.warning("raw Yelp file unavailable; using source hash from preparation manifest")
    return {"data_sha256": digest, "data_hash_source": "prepare_manifest" if digest else "unavailable"}


def _classification_metrics(y: np.ndarray, probs: np.ndarray, config: dict, prefix: str) -> dict:
    ev = config.get("evaluation", {})
    out = {
        f"{prefix}ap": M.average_precision(y, probs),
        f"{prefix}auc": M.roc_auc(y, probs),
        f"{prefix}brier": M.brier_score(y, probs),
        f"{prefix}prevalence": float((y == 1).mean()),
        f"{prefix}n": int(len(y)),
        f"{prefix}f1_at_0.5": M.f1_at_threshold(y, probs, 0.5),
    }
    for k in ev.get("precision_at_k", [50, 100]):
        out[f"{prefix}precision_at_{k}"] = M.precision_at_k(y, probs, int(k))
    for k in ev.get("recall_at_k", [50, 100]):
        out[f"{prefix}recall_at_{k}"] = M.recall_at_k(y, probs, int(k))
    return out


def train(config: dict, model: str, seed: int | None = None) -> dict:
    """Train one Track A model, calibrate on the calibration partition, and
    evaluate ONCE on test. Returns the run summary."""
    seed = int(seed if seed is not None else config["seed"])
    seed_everything(seed)
    features, labels, relations, splits = _load_processed(config)
    train_idx = role_indices(splits, "train")
    val_idx = role_indices(splits, "validation")
    calib_idx = role_indices(splits, "calibration")
    test_idx = role_indices(splits, "test")

    artifacts = Path(config["paths"].get("artifacts_dir", "artifacts"))
    run_id = _new_run_id(config, model, seed)
    run_dir = artifact_dir(artifacts, run_id)
    y = labels.astype(np.float64)
    w_pos = pos_weight(y[train_idx], cap=float(config["training"].get("pos_weight_cap", 100.0)))
    graph_ctx = None

    if model == "prior":
        prevalence = float(y[train_idx].mean())
        val_probs = np.full(len(val_idx), prevalence)
        test_probs = np.full(len(test_idx), prevalence)
        bundle = {"model_type": "prior", "prevalence": prevalence}
    elif model == "numeric":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(features[train_idx])
        clf = LogisticRegression(max_iter=2000, class_weight={0: 1.0, 1: w_pos}, random_state=seed)
        clf.fit(scaler.transform(features[train_idx]), y[train_idx])
        val_probs = clf.predict_proba(scaler.transform(features[val_idx]))[:, 1]
        test_probs = clf.predict_proba(scaler.transform(features[test_idx]))[:, 1]
        bundle = {
            "model_type": "numeric",
            "coef": clf.coef_.tolist(),
            "intercept": float(clf.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
    elif model == "graph":
        model_obj, sampler, val_probs, test_probs = _train_graph(
            config, features, relations, train_idx, val_idx, test_idx, y, seed, run_dir
        )
        graph_ctx = (model_obj, sampler)
        bundle = {
            "model_type": "graph",
            "relation_names": list(relations.keys()),
            "in_dim": int(features.shape[1]),
            "hidden_dim": int(config["model"]["hidden_dim"]),
            "layers": int(config["model"]["layers"]),
            "dropout": float(config["model"]["dropout"]),
            "fanout": list(config["training"].get("fanout_per_relation", [10, 5])),
            "seed": seed,
            "checkpoint": "graph_model.pt",
        }
    else:
        raise ValueError(f"unknown Track A model '{model}' (expected prior|numeric|graph)")

    # validation metrics (checkpoint/selection view)
    val_metrics = _classification_metrics(y[val_idx], val_probs, config, prefix="validation_")

    # Platt calibration fit on the calibration partition only
    if model == "prior":
        calibrator = PlattCalibrator(score_kind="proxy")
    else:
        if model == "numeric":
            cal_logits = _logit(clf.predict_proba(scaler.transform(features[calib_idx]))[:, 1])
        else:
            model_obj, sampler = graph_ctx
            cal_logits = graph_predict_logits(model_obj, sampler, bundle, features, calib_idx)
        calibrator = PlattCalibrator(score_kind="proxy").fit(cal_logits, y[calib_idx])

    # frozen test evaluation (calibrated)
    if model == "prior":
        test_probs_cal = test_probs
    else:
        if model == "numeric":
            test_logits = _logit(clf.predict_proba(scaler.transform(features[test_idx]))[:, 1])
        else:
            model_obj, sampler = graph_ctx
            test_logits = graph_predict_logits(model_obj, sampler, bundle, features, test_idx)
        test_probs_cal = calibrator.predict_proba(test_logits)

    test_metrics = _classification_metrics(y[test_idx], test_probs_cal, config, prefix="test_")

    summary = {
        "run_id": run_id,
        "track": "yelp_static",
        "model": model,
        "seed": seed,
        "pos_weight": w_pos,
        "metrics": {**val_metrics, **test_metrics},
        "calibration": calibrator.to_dict(),
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "bundle.json", bundle)
    save_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "model": model,
            "seed": seed,
            "config_resolved": config,
            **_source_provenance(config),
            "processed_features_sha256": sha256_file(Path(config["paths"]["processed_dir"]) / "features.npy"),
            "split_hash": stable_json(role_indices(splits, "train")[:1000].tolist()),
            "torch": _torch_version(),
            "python": platform.python_version(),
            "protocol_note": "transductive static benchmark; not future-event prediction",
        },
    )
    pd.DataFrame({"node_idx": test_idx, "score": test_probs_cal, "label": labels[test_idx]}).to_parquet(
        run_dir / "test_predictions.parquet", index=False
    )
    logger.info(
        "run %s: val AP %s | test AP %s",
        run_id,
        f"{val_metrics['validation_ap']:.4f}" if val_metrics["validation_ap"] else "n/a",
        f"{test_metrics['test_ap']:.4f}" if test_metrics["test_ap"] else "n/a",
    )
    return summary


# ---------------------------------------------------------------------------
# graph model helpers
# ---------------------------------------------------------------------------

def _graph_forward(model, sampler, feat_t, fanouts, seed_nodes, rng, chunk: int = 1024) -> np.ndarray:
    """Deterministic chunked forward pass; returns logits for seed nodes."""
    outs = []
    with torch.no_grad():
        for start in range(0, len(seed_nodes), chunk):
            seeds = seed_nodes[start : start + chunk]
            layers = build_layer_stack(sampler, seeds, fanouts, rng)
            x = feat_t[layers[0].all_local_nodes]
            for conv, adj in zip(model.convs, layers):
                edges = {r: torch.as_tensor(e, dtype=torch.long) for r, e in adj.local_edge_index.items()}
                x = conv(x, edges, adj.n_seeds)
            outs.append(model.head(x).squeeze(-1).detach().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0)


def _train_graph(config, features, relations, train_idx, val_idx, test_idx, y, seed, run_dir):
    device = "cpu"
    relation_names = list(relations.keys())
    n_nodes = features.shape[0]
    sampler = RelationalNeighborSampler(relations, n_nodes, seed=seed)
    fanouts = list(config["training"].get("fanout_per_relation", [10, 5]))
    if len(fanouts) != int(config["model"]["layers"]) or any(int(f) < 1 for f in fanouts):
        raise ValueError("fanout_per_relation must contain one positive fanout per model layer")
    batch_size = int(config["training"].get("batch_size", 512))
    feat_t = torch.as_tensor(features, dtype=torch.float32)

    model = GraphExpert(
        in_dim=features.shape[1],
        hidden_dim=int(config["model"]["hidden_dim"]),
        layers=int(config["model"]["layers"]),
        relations=relation_names,
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    w_pos = pos_weight(y[train_idx], cap=float(config["training"].get("pos_weight_cap", 100.0)))
    epochs = int(config["training"].get("max_epochs", 60))
    patience = int(config["training"].get("patience", 8))
    lr = float(config["training"].get("learning_rate", 1e-3))
    wd = float(config["training"].get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng_train = np.random.default_rng(seed)
    best_ap, best_state, best_epoch = -1.0, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng_train.permutation(len(train_idx))
        losses = []
        rng = np.random.default_rng(seed * 7919 + epoch)
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[order[start : start + batch_size]]
            optimizer.zero_grad()
            # per-epoch deterministic training sampling
            layers = build_layer_stack(sampler, batch, fanouts, rng)
            x = feat_t[layers[0].all_local_nodes]
            for conv, adj in zip(model.convs, layers):
                edges = {r: torch.as_tensor(e, dtype=torch.long) for r, e in adj.local_edge_index.items()}
                x = conv(x, edges, adj.n_seeds)
            logits = model.head(x).squeeze(-1)
            labels_batch = torch.as_tensor(y[batch], dtype=torch.float32)
            mask = torch.ones(len(batch), dtype=torch.bool)
            loss = bce_with_logits_masked(logits, labels_batch, mask, weight_pos=w_pos)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_logits = _graph_forward(model, sampler, feat_t, fanouts, val_idx, np.random.default_rng(seed + 104729))
        val_probs = _expit(val_logits)
        ap = M.average_precision(y[val_idx], val_probs)
        ap = ap if ap is not None else 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            logger.info("early stop at epoch %d (best %d, val AP %.4f)", epoch, best_epoch, best_ap)
            break
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("epoch %d loss %.4f val AP %.4f", epoch, float(np.mean(losses)), ap)

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), Path(run_dir) / "graph_model.pt")
    save_json(Path(run_dir) / "train_history.json", history)

    model.eval()
    with torch.no_grad():
        val_logits = _graph_forward(model, sampler, feat_t, fanouts, val_idx, np.random.default_rng(seed + 104729))
        test_logits = _graph_forward(model, sampler, feat_t, fanouts, test_idx, np.random.default_rng(seed + 104729))
    return model, sampler, _expit(val_logits), _expit(test_logits)


def graph_predict_logits(model, sampler, bundle, features, targets) -> np.ndarray:
    """Inference helper reusing an in-memory trained model (eval sampling is
    deterministic via the fixed seed offset)."""
    feat_t = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = _graph_forward(
            model, sampler, feat_t, bundle["fanout"], np.asarray(targets),
            np.random.default_rng(bundle["seed"] + 104729),
        )
    return logits


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None

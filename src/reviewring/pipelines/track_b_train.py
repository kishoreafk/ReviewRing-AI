"""Track B pipeline part 2: experts, fusion, calibration, evaluation.

Training protocol (spec 11.3):
1. Train each expert independently on the same eligible labelled train rows
   (synthetic manipulative=1, legitimate control=0, background=masked).
2. Keep each expert's best validation checkpoint.
3. Freeze experts. Fusion models (global mixture / adaptive gate) train on
   frozen expert outputs with early stopping on validation AP; the adaptive
   gate is additionally compared against fixed fusion on validation (the spec
   warns gates can overfit or collapse to one expert - we measure it).
4. Calibrate on the separate calibration partition only.
5. Test is evaluated once, after thresholds are frozen.

Gate inputs: expert 64-d states, availability flags, and past-only context
counts. No labels, simulation flags or provenance columns are admitted.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reviewring.data.splits import role_indices
from reviewring.evaluation import metrics as M
from reviewring.graph.build import RELATION_NAMES
from reviewring.models.experts import GraphExpert, TextExpert, TimeExpert
from reviewring.models.fusion import EXPERT_NAMES, AdaptiveGate, FixedFusion, GlobalFusion
from reviewring.training.calibrate import PlattCalibrator
from reviewring.utils.runtime import artifact_dir, save_json, seed_everything, sha256_file, stable_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data bundle
# ---------------------------------------------------------------------------

class TrackBData:
    """In-memory bundle joined by node_idx (row order of reviews.parquet)."""

    def __init__(self, config: dict):
        base = Path(config["paths"]["processed_dir"])
        self.base = base
        self.reviews = pd.read_parquet(base / "reviews.parquet")
        self.labels = pd.read_parquet(base / "labels.parquet")
        self.splits = pd.read_parquet(base / "splits.parquet")
        self.campaigns = pd.read_parquet(base / "campaigns.parquet")
        self.embeddings = np.load(base / "embeddings.npy")
        feats = np.load(base / "features.npz")
        self.history = feats["history"]
        self.history_missing = feats["history_missing"]
        edges = np.load(base / "edges.npz")
        self.edge_index = edges["edge_index"]
        self.relation = edges["relation"]

        # mapping integrity: every aligned artifact must have one row per review
        n = len(self.reviews)
        if self.embeddings.shape[0] != n or self.history.shape[0] != n:
            raise RuntimeError(
                f"artefact row mismatch: reviews={n}, embeddings={self.embeddings.shape[0]}, "
                f"history={self.history.shape[0]}. Re-run 'features' and 'graph' after "
                "re-simulating (data changed since the last feature build)."
            )

        self.train_idx = role_indices(self.splits, "train")
        self.val_idx = role_indices(self.splits, "validation")
        self.calib_idx = role_indices(self.splits, "calibration")
        self.test_idx = role_indices(self.splits, "test")

        lab_by_id = self.labels.set_index("review_id")["label"]
        self.y = lab_by_id.reindex(self.reviews["review_id"]).to_numpy(dtype=float)
        self.y[np.isnan(self.y)] = -1.0  # -1 = unknown, masked everywhere

        n = len(self.reviews)
        self.avail_text = ~self.reviews["text_missing"].fillna(True).to_numpy(dtype=bool)
        deg_in = (
            np.bincount(self.edge_index[1], minlength=n)
            if self.edge_index.shape[1] else np.zeros(n, dtype=np.int64)
        )
        self.avail_graph = deg_in > 0
        self.avail_time = ~self.history_missing.all(axis=1)
        self.availability = np.stack(
            [self.avail_text, self.avail_graph, self.avail_time], axis=1
        ).astype(np.float32)

        # past-only gate context (scaled history: velocity, breadth, gaps)
        ctx_cols = [1, 2, 8, 3]
        self.context = self.history[:, ctx_cols].astype(np.float32)

        self.labelled = self.y >= 0
        self.edge_index_by_relation = {
            name: self.edge_index[:, self.relation == i] for i, name in enumerate(RELATION_NAMES)
        }

    def labelled_rows(self, idx: np.ndarray) -> np.ndarray:
        return idx[self.labelled[idx]]


# ---------------------------------------------------------------------------
# expert training
# ---------------------------------------------------------------------------

def _pos_weight(y: np.ndarray, rows: np.ndarray, cap: float = 100.0) -> float:
    yp = y[rows]
    n_pos = max(float((yp == 1).sum()), 1.0)
    n_neg = float((yp == 0).sum())
    return float(min(n_neg / n_pos, cap))


def _train_text_expert(data, config, seed, device):
    model = TextExpert(data.embeddings.shape[1], int(config["model"]["hidden_dim"]), float(config["model"]["dropout"]))
    return _train_pointwise(model, data, "text", config, seed, device)


def _train_time_expert(data, config, seed, device):
    model = TimeExpert(data.history.shape[1], int(config["model"]["hidden_dim"]), float(config["model"]["dropout"]))
    return _train_pointwise(model, data, "time", config, seed, device)


def _train_pointwise(model, data, kind, config, seed, device):
    model = model.to(device)
    train_rows = data.labelled_rows(data.train_idx)
    val_rows = data.labelled_rows(data.val_idx)
    lr = float(config["training"].get("learning_rate", 1e-3))
    wd = float(config["training"].get("weight_decay", 1e-4))
    epochs = int(config["training"].get("max_epochs", 80))
    patience = int(config["training"].get("patience", 10))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w_pos = _pos_weight(data.y, train_rows)
    x_all = data.embeddings if kind == "text" else data.history
    x_t = torch.as_tensor(x_all, dtype=torch.float32)
    m_t = torch.as_tensor(data.history_missing.astype(np.float32)) if kind == "time" else None
    y_t = torch.as_tensor(data.y[train_rows], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best_ap, best_state, best_epoch = -1.0, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_rows))
        losses = []
        for start in range(0, len(train_rows), 256):
            rows = train_rows[order[start : start + 256]]
            opt.zero_grad()
            _, logits = model(x_t[rows]) if m_t is None else model(x_t[rows], m_t[rows])
            target = y_t[order[start : start + 256]]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=torch.tensor(w_pos)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vr = val_rows
            _, vlogits = model(x_t[vr]) if m_t is None else model(x_t[vr], m_t[vr])
        probs = torch.sigmoid(vlogits).cpu().numpy()
        ap = M.average_precision(data.y[vr], probs) or 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ap, history


def _train_graph_expert(data, config, seed, device):
    model = GraphExpert(
        in_dim=data.history.shape[1],
        hidden_dim=int(config["model"]["hidden_dim"]),
        layers=int(config["model"]["layers"]),
        relations=list(RELATION_NAMES),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    train_rows = data.labelled_rows(data.train_idx)
    val_rows = data.labelled_rows(data.val_idx)
    lr = float(config["training"].get("learning_rate", 1e-3))
    wd = float(config["training"].get("weight_decay", 1e-4))
    epochs = int(config["training"].get("max_epochs", 80))
    patience = int(config["training"].get("patience", 10))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x_all = torch.as_tensor(data.history, dtype=torch.float32)
    edges = {k: torch.as_tensor(v, dtype=torch.long) for k, v in data.edge_index_by_relation.items()}
    n = x_all.shape[0]
    y_t = torch.as_tensor(data.y[train_rows], dtype=torch.float32)
    w_pos = _pos_weight(data.y, train_rows)
    best_ap, best_state, best_epoch = -1.0, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        _, logits_all = model.forward_full(x_all, edges, n)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_all[train_rows], y_t, pos_weight=torch.tensor(w_pos)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            _, val_logits_all = model.forward_full(x_all, edges, n)
        probs = torch.sigmoid(val_logits_all[val_rows]).cpu().numpy()
        ap = M.average_precision(data.y[val_rows], probs) or 0.0
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ap, history


class ExpertBundle:
    """Frozen experts producing (repr, logit) per expert for arbitrary rows."""

    def __init__(self, text_model, graph_model, time_model, data: TrackBData):
        self.text = text_model
        self.graph = graph_model
        self.time = time_model
        self.data = data

    @property
    def experts_present(self) -> list[str]:
        return [k for k, m in zip(EXPERT_NAMES, [self.text, self.graph, self.time]) if m is not None]

    def outputs(self, rows: np.ndarray) -> dict:
        rows = np.asarray(rows)
        out = {}
        with torch.no_grad():
            if self.text is not None:
                x = torch.as_tensor(self.data.embeddings[rows], dtype=torch.float32)
                r, z = self.text(x)
                out["text"] = (r.numpy(), z.numpy())
            if self.time is not None:
                x = torch.as_tensor(self.data.history[rows], dtype=torch.float32)
                m = torch.as_tensor(self.data.history_missing[rows].astype(np.float32))
                r, z = self.time(x, m)
                out["time"] = (r.numpy(), z.numpy())
            if self.graph is not None:
                x_all = torch.as_tensor(self.data.history, dtype=torch.float32)
                edges = {
                    k: torch.as_tensor(v, dtype=torch.long)
                    for k, v in self.data.edge_index_by_relation.items()
                }
                r_all, z_all = self.graph.forward_full(x_all, edges, x_all.shape[0])
                out["graph"] = (r_all[rows].numpy(), z_all[rows].numpy())
        return out

    def availability(self, rows: np.ndarray) -> np.ndarray:
        idx_map = {name: i for i, name in enumerate(EXPERT_NAMES)}
        return self.data.availability[np.asarray(rows)][:, [idx_map[k] for k in self.experts_present]]


def _stack(outputs: dict, order: list[str]):
    reprs = np.stack([outputs[k][0] for k in order], axis=1)
    logits = np.stack([outputs[k][1] for k in order], axis=1)
    return torch.as_tensor(reprs), torch.as_tensor(logits)


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def _train_fusion(fusion, bundle: ExpertBundle, data: TrackBData, include_context: bool, config):
    train_rows = data.labelled_rows(data.train_idx)
    val_rows = data.labelled_rows(data.val_idx)
    order = bundle.experts_present
    out_tr = bundle.outputs(train_rows)
    out_va = bundle.outputs(val_rows)
    _, logits_tr = _stack(out_tr, order)
    _, logits_va = _stack(out_va, order)
    avail_tr = torch.as_tensor(bundle.availability(train_rows))
    avail_va = torch.as_tensor(bundle.availability(val_rows))
    probs_tr = torch.sigmoid(logits_tr)
    probs_va = torch.sigmoid(logits_va)
    ctx_tr = torch.as_tensor(data.context[train_rows])
    ctx_va = torch.as_tensor(data.context[val_rows])
    y_tr = torch.as_tensor(data.y[train_rows], dtype=torch.float32)
    y_va = data.y[val_rows]

    if isinstance(fusion, AdaptiveGate):
        reprs_tr, _ = _stack(out_tr, order)
        reprs_va, _ = _stack(out_va, order)
    lr = float(config["training"].get("learning_rate", 1e-3))
    opt = torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=1e-4)
    best_ap, best_state, best_epoch = -1.0, None, 0
    history = []
    for epoch in range(300):
        fusion.train()
        opt.zero_grad()
        mixed = _mix(fusion, probs_tr, reprs_tr if isinstance(fusion, AdaptiveGate) else None,
                     avail_tr, ctx_tr if include_context else None)
        loss = torch.nn.functional.binary_cross_entropy(mixed.clamp(1e-6, 1 - 1e-6), y_tr)
        loss.backward()
        opt.step()
        fusion.eval()
        with torch.no_grad():
            mixed_va = _mix(fusion, probs_va, reprs_va if isinstance(fusion, AdaptiveGate) else None,
                            avail_va, ctx_va if include_context else None)
        ap = M.average_precision(y_va, mixed_va.numpy()) or 0.0
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_ap": ap})
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().clone() for k, v in fusion.state_dict().items()}
        if epoch - best_epoch >= 20:
            break
    if best_state is not None:
        fusion.load_state_dict(best_state)
    fusion.eval()
    return fusion, best_ap, history


def _mix(fusion, expert_probs, expert_reprs, availability, context):
    if isinstance(fusion, FixedFusion):
        return fusion(expert_probs, availability)
    if isinstance(fusion, GlobalFusion):
        return fusion(expert_probs, availability)
    ctx = context if context is not None else torch.zeros(expert_probs.shape[0], 4)
    weights, _ = fusion(expert_reprs, availability, ctx)
    return (expert_probs * weights).sum(dim=-1)


def _fusion_predict(fusion, bundle: ExpertBundle, rows: np.ndarray, include_context: bool, availability_override: np.ndarray | None = None):
    order = bundle.experts_present
    out = bundle.outputs(rows)
    reprs, logits = _stack(out, order)
    avail = torch.as_tensor(bundle.availability(rows) if availability_override is None else availability_override)
    ctx = torch.as_tensor(bundle.data.context[np.asarray(rows)])
    with torch.no_grad():
        expert_probs = torch.sigmoid(logits)
        # ablation semantics: the no-context gate must also SCORE without
        # context, matching its training condition
        mixed = _mix(fusion, expert_probs, reprs, avail, ctx if include_context else None)
        weights = None
        if isinstance(fusion, AdaptiveGate):
            weights, _ = fusion(reprs, avail, ctx if include_context else torch.zeros_like(ctx))
    return mixed.numpy(), weights.numpy() if weights is not None else None


# ---------------------------------------------------------------------------
# main train entry
# ---------------------------------------------------------------------------

FUSION_MODELS = {"fixed_fusion", "global_fusion", "adaptive_fusion", "adaptive_fusion_noctx"}
VALID_MODELS = {"text", "time", "graph", *FUSION_MODELS}


def train(config: dict, model: str, seed: int | None = None) -> dict:
    if model not in VALID_MODELS:
        raise ValueError(f"unknown Track B model '{model}' (expected one of {sorted(VALID_MODELS)})")
    seed = int(seed if seed is not None else config["seed"])
    seed_everything(seed)
    data = TrackBData(config)
    device = "cpu"
    artifacts = Path(config["paths"].get("artifacts_dir", "artifacts"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-3]
    run_id = f"amazon-{model}-s{seed}-{stamp}"
    run_dir = artifact_dir(artifacts, run_id)

    text_model = graph_model = time_model = None
    expert_aps = {}
    histories = {}
    need_text = model in {"text", *FUSION_MODELS}
    need_graph = model in {"graph", *FUSION_MODELS}
    need_time = model in {"time", *FUSION_MODELS}

    if need_text:
        text_model, ap, hist = _train_text_expert(data, config, seed, device)
        expert_aps["text"], histories["text"] = ap, hist
    if need_graph:
        graph_model, ap, hist = _train_graph_expert(data, config, seed, device)
        expert_aps["graph"], histories["graph"] = ap, hist
    if need_time:
        time_model, ap, hist = _train_time_expert(data, config, seed, device)
        expert_aps["time"], histories["time"] = ap, hist

    fusion = None
    include_context = True
    gate_weights_test = None
    if model == "text":
        bundle = ExpertBundle(text_model, None, None, data)
    elif model == "graph":
        bundle = ExpertBundle(None, graph_model, None, data)
    elif model == "time":
        bundle = ExpertBundle(None, None, time_model, data)
    else:
        bundle = ExpertBundle(text_model, graph_model, time_model, data)
        if model == "fixed_fusion":
            fusion = FixedFusion()
        elif model == "global_fusion":
            fusion = GlobalFusion(len(bundle.experts_present))
            fusion, ap, hist = _train_fusion(fusion, bundle, data, True, config)
            histories["fusion"] = hist
        elif model in ("adaptive_fusion", "adaptive_fusion_noctx"):
            include_context = model == "adaptive_fusion"
            n_ctx = 4  # zeros are supplied when the ablation disables context
            fusion = AdaptiveGate(
                repr_dim=int(config["model"]["hidden_dim"]),
                n_context=n_ctx,
                hidden=64,
                n_experts=len(bundle.experts_present),
            )
            fusion, ap, hist = _train_fusion(fusion, bundle, data, include_context, config)
            histories["fusion"] = hist

    # ---- predictions on all partitions (background included, for replay) ----
    insufficient_rows = 0

    def score_rows(rows: np.ndarray):
        nonlocal insufficient_rows
        if fusion is not None:
            probs, weights = _fusion_predict(fusion, bundle, rows, include_context)
            # spec 11.2: rows with zero available experts are insufficient_data,
            # not silent zeros; count them and mark NaN (excluded downstream)
            avail = bundle.availability(rows)
            dead = avail.sum(axis=1) == 0
            if dead.any():
                insufficient_rows += int(dead.sum())
                probs = probs.copy()
                probs[dead] = np.nan
            return probs, weights
        out = bundle.outputs(rows)
        _, logits = _stack(out, bundle.experts_present)
        return torch.sigmoid(logits).mean(dim=1).numpy(), None

    partitions = {}
    for role, idx in (("train", data.train_idx), ("validation", data.val_idx),
                      ("calibration", data.calib_idx), ("test", data.test_idx)):
        probs, weights = score_rows(idx)
        partitions[role] = (idx, probs, weights)
        if role == "test":
            gate_weights_test = weights

    # ---- calibration on the separate calibration partition ----
    calib_idx, calib_probs, _ = partitions["calibration"]
    calib_labelled = data.labelled_rows(calib_idx)
    pos_in_calib = calib_labelled[data.y[calib_labelled] == 1]
    if len(pos_in_calib) >= 3:
        cal_logits = np.log(np.clip(calib_probs, 1e-6, 1 - 1e-6) / (1 - np.clip(calib_probs, 1e-6, 1 - 1e-6)))
        # fit only on labelled calibration rows
        cal_label_logits = cal_logits[np.isin(calib_idx, calib_labelled)]
        calibrator = PlattCalibrator(score_kind="synthetic").fit(
            cal_label_logits, data.y[calib_labelled]
        )
    else:
        calibrator = PlattCalibrator(score_kind="synthetic")
        logger.warning("too few labelled calibration rows; calibration skipped")

    # ---- metrics ----
    val_idx, val_probs, _ = partitions["validation"]
    val_labelled = data.labelled_rows(val_idx)
    pos_of = {node: i for i, node in enumerate(val_idx)}
    val_probs_labelled = np.array([val_probs[pos_of[r]] for r in val_labelled])
    val_metrics = _classification_metrics(data.y[val_labelled], val_probs_labelled, config, "validation_")

    test_idx, test_probs, _ = partitions["test"]
    test_labelled = data.labelled_rows(test_idx)
    pos_of_t = {node: i for i, node in enumerate(test_idx)}
    # calibrate
    test_logits_all = np.log(np.clip(test_probs, 1e-6, 1 - 1e-6) / (1 - np.clip(test_probs, 1e-6, 1 - 1e-6)))
    test_probs_cal_all = calibrator.predict_proba(test_logits_all)
    test_probs_cal_labelled = np.array([test_probs_cal_all[pos_of_t[r]] for r in test_labelled])
    test_metrics = _classification_metrics(data.y[test_labelled], test_probs_cal_labelled, config, "test_")

    # validation threshold + gate weight distribution (selection view).
    # The threshold is chosen on CALIBRATED validation probabilities so that
    # the replay/ring stages can apply it to calibrated test scores directly.
    val_logits_all = np.log(np.clip(val_probs, 1e-6, 1 - 1e-6) / (1 - np.clip(val_probs, 1e-6, 1 - 1e-6)))
    val_probs_cal_all = calibrator.predict_proba(val_logits_all)
    val_probs_cal_labelled = np.array([val_probs_cal_all[pos_of[r]] for r in val_labelled])
    if len(val_labelled) and len(np.unique(data.y[val_labelled])) > 1:
        threshold, _ = M.best_f1_threshold(data.y[val_labelled], val_probs_cal_labelled)
    else:
        threshold = 0.5
    gate_stats = None
    if gate_weights_test is not None:
        gate_stats = {
            "mean": gate_weights_test.mean(axis=0).tolist(),
            "std": gate_weights_test.std(axis=0).tolist(),
            "expert_order": bundle.experts_present,
            "collapse_max_mean_weight": float(gate_weights_test.mean(axis=0).max()),
        }

    # ---- missing-modality stress tests (spec 12.3 mode 4) ----
    # Computational stress: availability flags are overridden for a random
    # subset of test rows; features are unchanged. Compares how gracefully
    # each fusion strategy degrades when an expert cannot contribute.
    stress = {}
    if fusion is not None and len(bundle.experts_present) > 1:
        rng_stress = np.random.default_rng(seed + 77)
        pos_of_test = {node: i for i, node in enumerate(test_idx)}
        for modality in set(bundle.experts_present) & {"text", "time"}:
            for frac in (0.3, 0.6):
                override = data.availability[test_idx].copy()
                drop = rng_stress.random(len(test_idx)) < frac
                override[drop, EXPERT_NAMES.index(modality)] = 0.0
                probs_s, _ = _fusion_predict(fusion, bundle, test_idx, include_context, availability_override=override)
                probs_s_labelled = np.array([probs_s[pos_of_test[r]] for r in test_labelled])
                key = f"drop_{modality}_{int(frac * 100)}"
                stress[key] = {
                    "ap": M.average_precision(data.y[test_labelled], probs_s_labelled),
                    "auc": M.roc_auc(data.y[test_labelled], probs_s_labelled),
                }
        stress["baseline_ap"] = test_metrics["test_ap"]
        stress["note"] = "availability overridden on a random test subset; features unchanged"

    summary = {
        "run_id": run_id,
        "track": "raw_review_replay",
        "model": model,
        "seed": seed,
        "expert_validation_aps": expert_aps,
        "metrics": {**val_metrics, **test_metrics},
        "frozen_threshold": float(threshold),
        "calibration": calibrator.to_dict(),
        "gate_stats": gate_stats,
        "experts_present": bundle.experts_present,
        "stress": stress,
        "insufficient_data_rows": int(insufficient_rows),
    }

    # ---- artifacts ----
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "train_history.json", histories)
    torch.save(
        {
            "text": text_model.state_dict() if text_model is not None else None,
            "graph": graph_model.state_dict() if graph_model is not None else None,
            "time": time_model.state_dict() if time_model is not None else None,
            "fusion": fusion.state_dict() if fusion is not None else None,
        },
        run_dir / "model_bundle.pt",
    )
    pd.DataFrame(
        {
            "review_id": data.reviews["review_id"].iloc[test_idx].to_numpy(),
            "node_idx": test_idx,
            "score": test_probs_cal_all,
            "score_raw": test_probs,
            "timestamp": data.reviews["timestamp"].iloc[test_idx].to_numpy(),
            "label": data.y[test_idx],
        }
    ).to_parquet(run_dir / "test_predictions.parquet", index=False)
    save_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "model": model,
            "seed": seed,
            "config_resolved": config,
            "data_hash": sha256_file(data.base / "reviews.parquet"),
            "split_hash": stable_json(data.splits[["node_key", "role"]].to_dict(orient="records")[:20000]),
            "encoder": (Path(data.base / "features_manifest.json")).exists()
            and __import__("json").loads((data.base / "features_manifest.json").read_text())["encoder"]
            or None,
            "torch": torch.__version__,
            "label_policy": "unknown background masked from loss/metrics; "
            "synthetic labels describe the simulator only",
        },
    )
    logger.info(
        "run %s: val AP %s | test AP %s",
        run_id,
        f"{val_metrics['validation_ap']:.4f}" if val_metrics["validation_ap"] is not None else "n/a",
        f"{test_metrics['test_ap']:.4f}" if test_metrics["test_ap"] is not None else "n/a",
    )
    return summary


def _classification_metrics(y, probs, config, prefix):
    out = {
        f"{prefix}ap": M.average_precision(y, probs),
        f"{prefix}auc": M.roc_auc(y, probs),
        f"{prefix}brier": M.brier_score(y, probs) if len(y) else None,
        f"{prefix}prevalence": float((y == 1).mean()) if len(y) else None,
        f"{prefix}n": int(len(y)),
        f"{prefix}f1_at_0.5": M.f1_at_threshold(y, probs, 0.5) if len(y) else None,
    }
    for k in config.get("evaluation", {}).get("precision_at_k", [50, 100]):
        out[f"{prefix}precision_at_{k}"] = M.precision_at_k(y, probs, int(k))
    for k in config.get("evaluation", {}).get("recall_at_k", [50, 100]):
        out[f"{prefix}recall_at_{k}"] = M.recall_at_k(y, probs, int(k))
    return out

"""Track B pipeline part 3: replay, ring discovery, explanations, reports.

These stages consume a saved run's test predictions plus evaluator-only
campaign information. Discovery never reads campaign ids; evaluation reads
them only after scores are frozen.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reviewring.evaluation import metrics as M
from reviewring.evaluation.replay import daily_replay, recovery_at_horizon
from reviewring.explain.cards import build_evidence_card
from reviewring.explain.masking import GraphOnlyScorer, greedy_relation_masking
from reviewring.graph.build import RELATION_NAMES
from reviewring.models.experts import GraphExpert
from reviewring.pipelines.track_b_train import TrackBData
from reviewring.rings.discover import candidate_priority_table, discover_rings
from reviewring.utils.runtime import save_json

logger = logging.getLogger(__name__)


def _load_run(config: dict, run_id: str) -> tuple[Path, dict]:
    run_dir = Path(config["paths"].get("artifacts_dir", "artifacts")) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run {run_id} not found under {run_dir}")
    summary = json.loads((run_dir / "summary.json").read_text())
    return run_dir, summary


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def replay(config: dict, run_id: str) -> dict:
    run_dir, summary = _load_run(config, run_id)
    base = Path(config["paths"]["processed_dir"])
    campaigns = pd.read_parquet(base / "campaigns.parquet")
    preds = pd.read_parquet(run_dir / "test_predictions.parquet")
    threshold = float(summary["frozen_threshold"])

    test_campaigns = campaigns[
        (campaigns["partition"] == "test") & (campaigns["is_manipulative"])
    ]
    campaign_members = {
        row["campaign_id"]: set(row["member_review_ids"])
        for _, row in test_campaigns.iterrows()
    }
    result = daily_replay(preds[["review_id", "score", "timestamp"]], campaign_members, threshold)
    out = result.to_dict()
    out.update(recovery_at_horizon(result))
    save_json(run_dir / "replay.json", out)
    logger.info(
        "replay: %d/%d planted campaigns detected (%s)",
        result.detected_campaigns,
        result.planted_campaigns,
        out.get("detection_recall"),
    )
    return out


# ---------------------------------------------------------------------------
# ring discovery + evaluation
# ---------------------------------------------------------------------------

def rings(config: dict, run_id: str) -> dict:
    run_dir, summary = _load_run(config, run_id)
    data = TrackBData(config)
    preds = pd.read_parquet(run_dir / "test_predictions.parquet")

    rcfg = config.get("rings", {})
    scores = pd.DataFrame(
        {"review_id": preds["review_id"], "score": preds["score"]}
    )
    test_node_ids = set(data.reviews["review_id"].iloc[data.test_idx])
    scores = scores[scores["review_id"].isin(test_node_ids)]

    top_k_budget = int(rcfg.get("top_k_review_budget", 300))
    candidates = discover_rings(
        scores,
        data.reviews,
        threshold=None,  # bounded top-K avoids threshold sensitivity
        top_k=top_k_budget,
        min_accounts=int(rcfg.get("min_accounts", 3)),
        min_targets=int(rcfg.get("min_targets", 2)),
        top_m_reviews=int(rcfg.get("top_m_reviews", 5)),
        priority_weights=tuple(rcfg.get("priority_weights", [0.5, 0.3, 0.2])),
        seed=int(config["seed"]),
    )
    table = candidate_priority_table(candidates)
    table.to_parquet(run_dir / "rings.parquet", index=False)

    # evaluate membership recovery against planted test campaigns (evaluator only)
    campaigns = pd.read_parquet(Path(config["paths"]["processed_dir"]) / "campaigns.parquet")
    test_truth = campaigns[(campaigns["partition"] == "test") & (campaigns["is_manipulative"])]
    predicted_sets = [set(c.members) for c in candidates]
    # campaign membership is by REVIEW id; convert to reviewer ids per campaign
    rev_by_review = dict(zip(data.reviews["review_id"], data.reviews["reviewer_id"]))
    truth_reviewer_sets = []
    for _, row in test_truth.iterrows():
        reviewers = {rev_by_review.get(r) for r in row["member_review_ids"] if r in rev_by_review}
        truth_reviewer_sets.append(reviewers - {None})
    match = M.match_campaigns(predicted_sets, truth_reviewer_sets, threshold=0.5)

    out = {
        "n_candidates": len(candidates),
        "priority_table": table.to_dict(orient="records"),
        "campaign_matching": {k: v for k, v in match.items() if k != "matches"},
        "matches": match["matches"],
        "n_test_truth_campaigns": int(len(test_truth)),
    }
    save_json(run_dir / "rings_eval.json", out)
    logger.info(
        "rings: %d candidates, campaign recall %s",
        len(candidates),
        match.get("campaign_recall"),
    )
    return out


# ---------------------------------------------------------------------------
# explanations
# ---------------------------------------------------------------------------

def explain(config: dict, run_id: str, candidate_id: str | None = None, seed: int = 17) -> dict:
    run_dir, summary = _load_run(config, run_id)
    data = TrackBData(config)
    rings_path = run_dir / "rings.parquet"
    if not rings_path.exists():
        raise FileNotFoundError("run rings first: reviewring rings --run <run_id>")
    table = pd.read_parquet(rings_path)
    if candidate_id is None:
        candidate_id = table.sort_values("priority", ascending=False).iloc[0]["candidate_id"]
    row = table[table["candidate_id"] == candidate_id]
    if not len(row):
        raise ValueError(f"candidate {candidate_id} not in run rings")

    # re-derive the candidate via deterministic discovery (same seed/threshold
    # policy as the rings stage), so masked edges map to real review records
    preds = pd.read_parquet(run_dir / "test_predictions.parquet")
    test_node_ids = set(data.reviews["review_id"].iloc[data.test_idx])
    scores = pd.DataFrame({"review_id": preds["review_id"], "score": preds["score"]})
    scores = scores[scores["review_id"].isin(test_node_ids)]
    rcfg = config.get("rings", {})
    top_k_budget = int(rcfg.get("top_k_review_budget", 300))
    candidates = discover_rings(
        scores,
        data.reviews,
        threshold=None,
        top_k=top_k_budget,
        min_accounts=int(rcfg.get("min_accounts", 3)),
        min_targets=int(rcfg.get("min_targets", 2)),
        seed=int(config["seed"]),
    )
    match = [c for c in candidates if c.candidate_id == candidate_id]
    if not match:
        raise ValueError(f"candidate {candidate_id} not reproduced by deterministic discovery")
    candidate = match[0]

    # seed node = candidate member review with the highest score
    member_review_rows = data.reviews[data.reviews["review_id"].isin(candidate.review_ids)]
    member_rows = data.reviews.index[data.reviews["review_id"].isin(candidate.review_ids)].to_numpy()
    score_by_review = dict(zip(preds["review_id"], preds["score"]))
    seed_node = int(member_rows[np.argmax([score_by_review.get(r, 0) for r in member_review_rows["review_id"]])])

    # graph expert for scoring
    bundle = torch_load_bundle(run_dir)
    model = GraphExpert(
        in_dim=data.history.shape[1],
        hidden_dim=bundle["hidden_dim"],
        layers=bundle["layers"],
        relations=list(RELATION_NAMES),
        dropout=0.0,
    )
    model.load_state_dict(bundle["graph_state"])
    model.eval()
    scorer = GraphOnlyScorer(model)
    x = torch.as_tensor(data.history, dtype=torch.float32)

    result = greedy_relation_masking(
        candidate_id=candidate_id,
        seed_node=seed_node,
        x=x,
        edge_index=data.edge_index,
        relation=data.relation,
        relation_names=list(RELATION_NAMES),
        scorer=scorer,
        max_model_calls=int(config.get("explanations", {}).get("max_model_calls", 50)),
        max_candidate_edges=int(config.get("explanations", {}).get("max_candidate_edges", 100)),
        rng_seed=seed,
    )
    card = build_evidence_card(
        candidate,
        data.reviews,
        masking_result=result,
        model_version=run_id,
        as_of=str(data.reviews["timestamp"].iloc[data.test_idx].max()),
        label_status="synthetic (controlled campaign)",
    )
    cards_path = run_dir / "evidence.jsonl"
    existing = cards_path.read_text().strip().splitlines() if cards_path.exists() else []
    with open(cards_path, "w", encoding="utf-8") as handle:
        for line in existing:
            parsed = json.loads(line)
            if parsed["candidate_id"] != candidate_id:
                handle.write(line + "\n")
        handle.write(json.dumps(card, default=str) + "\n")
    out = {
        "candidate_id": candidate_id,
        "card": card,
        "masking": {
            "original_score": result.original_score,
            "score_after_removal": result.score_after_removal,
            "removal_effect": result.removal_effect,
            "retention_gap": result.retention_gap,
            "compactness": result.compactness,
            "random_control_advantage": result.random_control_advantage,
            "n_model_calls": result.n_model_calls,
        },
    }
    save_json(run_dir / f"explain_{candidate_id}.json", out)
    logger.info("explained %s: removal effect %.4f", candidate_id, result.removal_effect)
    return out


def torch_load_bundle(run_dir: Path) -> dict:
    import torch

    payload = torch.load(Path(run_dir) / "model_bundle.pt", map_location="cpu", weights_only=False)
    graph_sd = payload.get("graph")
    if graph_sd is None:
        raise ValueError("run has no graph expert; cannot explain graph pathways")
    hidden_dim = graph_sd["convs.0.self_lin.bias"].shape[0]
    layers = len({k.split(".")[1] for k in graph_sd if k.startswith("convs.")})
    return {"graph_state": graph_sd, "hidden_dim": hidden_dim, "layers": layers}


# ---------------------------------------------------------------------------
# reports and figures
# ---------------------------------------------------------------------------

def report(config: dict, run_id: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, summary = _load_run(config, run_id)
    preds = pd.read_parquet(run_dir / "test_predictions.parquet")
    # join by review_id (mapping-integrity rule): predictions carry their IDs,
    # so reports never depend on node_idx alignment with later data rebuilds
    labels_by_id = pd.Series(
        pd.read_parquet(Path(config["paths"]["processed_dir"]) / "labels.parquet")["label"].to_numpy(dtype=float),
        index=pd.read_parquet(Path(config["paths"]["processed_dir"]) / "labels.parquet")["review_id"],
    )
    preds["label_known"] = preds["review_id"].map(labels_by_id)
    labelled = preds[preds["label_known"].notna()]
    y = labelled["label_known"].to_numpy()
    p = labelled["score"].to_numpy()

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    paths = {}

    # PR curve
    if len(y) and len(np.unique(y)) > 1:
        order = np.argsort(-p)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        precision = tp / (np.arange(len(y_sorted)) + 1)
        recall = tp / max(y_sorted.sum(), 1)
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.plot(recall, precision, lw=2)
        ap = M.average_precision(y, p)
        ax.set_title(f"Precision-Recall (test AP={ap:.3f})" if ap is not None else "Precision-Recall")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.savefig(figures / "pr_curve.png", dpi=150)
        plt.close(fig)
        paths["pr_curve"] = str(figures / "pr_curve.png")

    # training curves
    hist_path = run_dir / "train_history.json"
    if hist_path.exists():
        histories = json.loads(hist_path.read_text())
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        for name, h in histories.items():
            if isinstance(h, list) and h:
                ax.plot([r["epoch"] for r in h], [r["validation_ap"] for r in h], label=name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation AP")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.set_title("Validation AP by stage")
        fig.savefig(figures / "val_ap.png", dpi=150)
        plt.close(fig)
        paths["val_ap"] = str(figures / "val_ap.png")

    # gate weights
    if summary.get("gate_stats"):
        means = summary["gate_stats"]["mean"]
        names = summary["gate_stats"]["expert_order"]
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.bar(names, means)
        ax.set_ylim(0, 1)
        ax.set_title("Mean gate weights (test)")
        fig.savefig(figures / "gate_weights.png", dpi=150)
        plt.close(fig)
        paths["gate_weights"] = str(figures / "gate_weights.png")

    # replay recovery
    replay_path = run_dir / "replay.json"
    if replay_path.exists():
        rep = json.loads(replay_path.read_text())
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        days = [s["day"] for s in rep["steps"]]
        alerts = [s["alerts"] for s in rep["steps"]]
        ax.plot(days, alerts)
        ax.set_title("Alerts per replay day")
        ax.set_xlabel("day")
        ax.set_ylabel("alerts")
        plt.xticks(rotation=45, fontsize=7)
        fig.savefig(figures / "alerts_per_day.png", dpi=150)
        plt.close(fig)
        paths["alerts_per_day"] = str(figures / "alerts_per_day.png")

    # markdown report
    md = [f"# Run report: {run_id}", ""]
    md.append(f"- Model: `{summary['model']}` (seed {summary['seed']})")
    md.append(f"- Experts: {summary.get('experts_present')}")
    md.append(f"- Frozen threshold: {summary.get('frozen_threshold'):.3f}")
    md.append("")
    md.append("## Test metrics (labelled synthetic rows only)")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    for k, v in summary["metrics"].items():
        if k.startswith("test_") and v is not None:
            md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    md.append("")
    md.append(
        "> Label source: synthetic controlled campaigns (known membership). "
        "These numbers describe the simulator task, not real-world fraud detection."
    )
    if (run_dir / "replay.json").exists():
        rep = json.loads((run_dir / "replay.json").read_text())
        md += ["", "## Replay (daily resolution)"]
        md.append(
            f"- Planted campaigns detected: {rep['detected_campaigns']}/{rep['planted_campaigns']}"
        )
        for h in ("recall_at_1d", "recall_at_7d", "recall_at_30d", "mean_delay_detected_only"):
            if rep.get(h) is not None:
                md.append(f"- {h}: {rep[h]:.2f}" if isinstance(rep[h], float) else f"- {h}: {rep[h]}")
    if (run_dir / "rings_eval.json").exists():
        rr = json.loads((run_dir / "rings_eval.json").read_text())
        md += ["", "## Ring recovery"]
        cm = rr.get("campaign_matching", {})
        md.append(f"- Candidates: {rr.get('n_candidates')}")
        for k in ("campaign_precision", "campaign_recall", "member_recovery", "fragmentation", "merging"):
            if cm.get(k) is not None:
                md.append(f"- {k}: {cm[k]:.3f}" if isinstance(cm[k], float) else f"- {k}: {cm[k]}")
    explain_files = sorted(run_dir.glob("explain_*.json"))
    if explain_files:
        ex = json.loads(explain_files[0].read_text())
        md += ["", "## Explanation (masked graph messages)"]
        m = ex["masking"]
        for k, v in m.items():
            md.append(f"- {k}: {v}")
    md += ["", "## Boundary"]
    md.append(
        "- Track B results are measured on a controlled simulator with known "
        "campaign membership; unmatched alerts in unlabelled background are "
        "unverified, not false positives. No real-world fraud claim is made."
    )
    (run_dir / "report.md").write_text("\n".join(md))
    logger.info("report written to %s", run_dir / "report.md")
    return {"report": str(run_dir / "report.md"), "figures": paths}

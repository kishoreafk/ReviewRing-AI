"""Report generation for Track A runs (markdown + figures)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from reviewring.utils.runtime import save_json

logger = logging.getLogger(__name__)


def track_a_report(config: dict, run_id: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reviewring.evaluation import metrics as M

    run_dir = Path(config["paths"].get("artifacts_dir", "artifacts")) / run_id
    summary = json.loads((run_dir / "summary.json").read_text())
    preds = pd.read_parquet(run_dir / "test_predictions.parquet")
    y = preds["label"].to_numpy()
    p = preds["score"].to_numpy()

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    paths = {}
    if len(np.unique(y)) > 1:
        order = np.argsort(-p)
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        precision = tp / (np.arange(len(y_sorted)) + 1)
        recall = tp / max(y_sorted.sum(), 1)
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.plot(recall, precision, lw=2)
        ap = M.average_precision(y, p)
        ax.set_title(f"YelpChi test Precision-Recall (AP={ap:.3f})" if ap is not None else "Precision-Recall")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.savefig(figures / "pr_curve.png", dpi=150)
        plt.close(fig)
        paths["pr_curve"] = str(figures / "pr_curve.png")

    hist_path = run_dir / "train_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.plot([r["epoch"] for r in history], [r["validation_ap"] for r in history])
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation AP")
        ax.set_title("Validation AP during training")
        fig.savefig(figures / "val_ap.png", dpi=150)
        plt.close(fig)
        paths["val_ap"] = str(figures / "val_ap.png")

    md = [f"# Track A run report: {run_id}", ""]
    md.append(f"- Model: `{summary['model']}` (seed {summary['seed']})")
    md.append("- Protocol: stratified random split, transductive; labels are Yelp filter proxy labels")
    md.append(f"- Calibration: {summary.get('calibration')}")
    md += ["", "## Validation metrics", "", "| metric | value |", "|---|---|"]
    for k, v in summary["metrics"].items():
        if k.startswith("validation_") and v is not None:
            md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    md += ["", "## Test metrics (frozen evaluation)", "", "| metric | value |", "|---|---|"]
    for k, v in summary["metrics"].items():
        if k.startswith("test_") and v is not None:
            md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    md += ["", "## Boundary"]
    md.append(
        "- Track A matrices are preprocessed benchmark features; no review text, "
        "timestamps or identities exist at this level. Results are benchmark "
        "proxy-label classification, not text/timeline fraud claims."
    )
    (run_dir / "report.md").write_text("\n".join(md))
    save_json(run_dir / "report_paths.json", paths)
    logger.info("report written to %s", run_dir / "report.md")
    return {"report": str(run_dir / "report.md"), "figures": paths}

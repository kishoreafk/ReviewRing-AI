"""Diagnostic: why doesn't the adaptive gate beat fixed fusion? (investigation only).

Reads saved run artifacts (test predictions, summaries, fusion histories) and
prints expert quality/redundancy, a global-mixture upper bound, per-example
headroom probes and val-vs-test gaps. Investigation only: it never trains and
never writes into run directories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Run directory parent (default: artifacts). For isolated "
        "experiments pass <experiment>/artifacts.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed family to compare (default: 42).",
    )
    return p.parse_args(argv)


def load_preds(art: Path, run_glob: str) -> pd.DataFrame:
    # Latest timestamped run, matching collect_results.py selection semantics.
    paths = sorted(art.glob(run_glob + "/test_predictions.parquet"))
    assert paths, run_glob
    df = pd.read_parquet(paths[-1])
    return df.set_index("node_idx")


def main(argv=None) -> int:
    args = parse_args(argv)
    art = args.artifacts_dir
    seed = args.seed

    # One seed family keeps the expert bundle comparable across fusion variants.
    required = {
        "text": f"amazon-text-s{seed}-*",
        "time": f"amazon-time-s{seed}-*",
        "graph": f"amazon-graph-s{seed}-*",
        "fixed": f"amazon-fixed_fusion-s{seed}-*",
        "adapt": f"amazon-adaptive_fusion-s{seed}-*",
    }
    missing = [
        name for name, glob in required.items()
        if not sorted(art.glob(glob + "/test_predictions.parquet"))
    ]
    if missing:
        print(
            f"SKIP: seed-{seed} family incomplete under {art} "
            f"(missing: {', '.join(missing)}). "
            "Run a full (non-smoke) amazon experiment first."
        )
        return 0

    text = load_preds(art, required["text"])
    time = load_preds(art, required["time"])
    graph = load_preds(art, required["graph"])
    fixed = load_preds(art, required["fixed"])
    adapt = load_preds(art, required["adapt"])

    y = text["label"].to_numpy()
    t = text["score_raw"].to_numpy()
    m = time["score_raw"].to_numpy()
    g = graph["score_raw"].to_numpy()
    lab = np.isfinite(t) & np.isfinite(m) & np.isfinite(g) & np.isin(y, [0, 1])
    y, t, m, g = y[lab], t[lab], m[lab], g[lab]
    print(f"labelled test rows: {len(y)}  prevalence: {y.mean():.4f}")

    def ap(scores):
        return average_precision_score(y, scores)

    def ap2(scores, yy):
        return average_precision_score(yy, scores)

    print("\n== expert quality and redundancy ==")
    print(f"AP  text={ap(t):.4f}  time={ap(m):.4f}  graph={ap(g):.4f}")
    print(f"Spearman text~time={spearmanr(t, m).statistic:.3f}  text~graph={spearmanr(t, g).statistic:.3f}  time~graph={spearmanr(m, g).statistic:.3f}")
    mean3 = (t + m + g) / 3
    print(f"AP uniform mean (fixed fusion, recomputed) = {ap(mean3):.4f}")
    print(f"AP fixed_fusion run score_raw              = {ap(fixed['score_raw'].to_numpy()[lab]):.4f}")

    print("\n== best GLOBAL mixture (grid over simplex; diagnostic upper bound for global weights) ==")
    best = (-1, None)
    for wt in range(11):
        for wg in range(11 - wt):
            wtt, wgg = wt / 10, wg / 10
            s = wtt * t + wgg * g + (1 - wtt - wgg) * m
            a = ap(s)
            if a > best[0]:
                best = (a, (wtt, wgg, 1 - wtt - wgg))
    print(f"best global mixture AP = {best[0]:.4f}  weights (text,graph,time) = {best[1]}")

    print("\n== per-example headroom probes (optimistic bounds) ==")
    print(f"AP row-wise max of expert probs  = {ap(np.maximum(np.maximum(t, m), g)):.4f}")
    print(f"AP rank-avg (rank-mean of 3)     = {ap((t + m + g).rank().to_numpy() if False else (pd.Series(t).rank() + pd.Series(m).rank() + pd.Series(g).rank()).to_numpy()):.4f}")

    # where do the two strong experts disagree, and who is right?
    r_m, r_g = pd.Series(m).rank(pct=True).to_numpy(), pd.Series(g).rank(pct=True).to_numpy()
    disagree = np.abs(r_m - r_g) > 0.2
    print(f"\nrows where time & graph ranks differ >0.2: {disagree.mean():.1%}")
    yd = y[disagree]
    print(f"  prevalence there={yd.mean():.4f} vs overall={y.mean():.4f}")
    print(f"  AP time={ap2(m[disagree], yd):.4f}, graph={ap2(g[disagree], yd):.4f}, mean={ap2(((m+g)/2)[disagree], yd):.4f}")
    print(f"  corr(adaptive score, fixed score) on test = {np.corrcoef(adapt['score_raw'].to_numpy()[lab], fixed['score_raw'].to_numpy()[lab])[0,1]:.4f}")

    print("\n== fusion training history (overfit check) ==")
    hist_dirs = sorted(art.glob(f"amazon-adaptive_fusion*-s{seed}-*"))
    if not hist_dirs:
        print("  (no adaptive fusion runs found)")
    for run_dir in hist_dirs:
        hist_path = run_dir / "train_history.json"
        if not hist_path.exists():
            print(f"  {run_dir.name}: no train_history.json, skipped")
            continue
        hist = json.loads(hist_path.read_text()).get("fusion")
        if not hist:
            print(f"  {run_dir.name}: no fusion history, skipped")
            continue
        losses = [h["train_loss"] for h in hist]
        vaps = [h["validation_ap"] for h in hist]
        best_ep = int(np.argmax(vaps))
        print(f"{run_dir.name}:")
        print(f"  epochs run={len(hist)}  best epoch={best_ep}  best val AP={max(vaps):.4f}")
        print(f"  train_loss first/last={losses[0]:.4f}/{losses[-1]:.4f}  val_ap last={vaps[-1]:.4f}")
        print(f"  val_ap trajectory (every 10): {[round(v,4) for v in vaps[::10]]}")

    print("\n== val vs test AP gap, fusion variants (from summaries) ==")
    fusion_runs = sorted(art.glob("amazon-*fusion-*"))
    if not fusion_runs:
        print("  (no fusion runs found)")
    for run in fusion_runs:
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text())
        print(f"{s['model']:22s} val={s['metrics']['validation_ap']:.4f}  test={s['metrics']['test_ap']:.4f}  val-test={s['metrics']['validation_ap']-s['metrics']['test_ap']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnostic: why doesn't the adaptive gate beat fixed fusion? (investigation only)"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

ART = Path("artifacts")

def load_preds(run_glob):
    paths = sorted(ART.glob(run_glob + "/test_predictions.parquet"))
    assert paths, run_glob
    df = pd.read_parquet(paths[0])
    return df.set_index("node_idx")

# seed-42 family (same expert bundle for all fusion variants)
text = load_preds("amazon-text-s42-*")
time = load_preds("amazon-time-s42-*")
graph = load_preds("amazon-graph-s42-*")
fixed = load_preds("amazon-fixed_fusion-s42-*")
adapt = load_preds("amazon-adaptive_fusion-s42-*")
glob = load_preds("amazon-global_fusion-s42-*")

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

print("\n== fusion training history (seed 42, overfit check) ==")
for run in ["amazon-adaptive_fusion-s42-20260916T2006475778", "amazon-adaptive_fusion_noctx-s42-20260916T2008304614"]:
    hist = json.loads((ART / run / "train_history.json").read_text())["fusion"]
    losses = [h["train_loss"] for h in hist]
    vaps = [h["validation_ap"] for h in hist]
    best_ep = int(np.argmax(vaps))
    print(f"{run}:")
    print(f"  epochs run={len(hist)}  best epoch={best_ep}  best val AP={max(vaps):.4f}")
    print(f"  train_loss first/last={losses[0]:.4f}/{losses[-1]:.4f}  val_ap last={vaps[-1]:.4f}")
    print(f"  val_ap trajectory (every 10): {[round(v,4) for v in vaps[::10]]}")

print("\n== val vs test AP gap, fusion variants (seed 42, from summaries) ==")
for run in sorted(ART.glob("amazon-*fusion-s42-*")):
    s = json.loads((run / "summary.json").read_text())
    print(f"{s['model']:22s} val={s['metrics']['validation_ap']:.4f}  test={s['metrics']['test_ap']:.4f}  val-test={s['metrics']['validation_ap']-s['metrics']['test_ap']:+.4f}")


"""Collect all run summaries into docs/RESULTS.md with seed statistics."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def collect(track_filter: str) -> dict:
    """run_id -> summary for one track."""
    out = {}
    for path in sorted(glob.glob("artifacts/*/summary.json")):
        s = json.loads(Path(path).read_text())
        if s.get("track") == track_filter:
            out[s["run_id"]] = s
    return out


def mean_std(runs: list[dict], key: str) -> tuple[float | None, float | None]:
    vals = [r["metrics"].get(key) for r in runs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    # sample std requires >= 2 seeds; single-seed rows report no std
    std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else None
    return float(np.mean(vals)), std


def fmt(v: float | None, digits: int = 4) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def fmt_pair(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{digits}f} (single seed)"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def track_a_table(runs: dict) -> str:
    by_model: dict[str, list] = {}
    for s in runs.values():
        by_model.setdefault(s["model"], []).append(s)
    lines = [
        "| Model | Seeds | Test AP (mean ± std) | Test ROC-AUC | P@100 | R@100 |",
        "|---|---|---|---|---|---|",
    ]
    for model in ("prior", "numeric", "graph"):
        group = by_model.get(model, [])
        ap_m, ap_s = mean_std(group, "test_ap")
        auc_m, _ = mean_std(group, "test_auc")
        p100, _ = mean_std(group, "test_precision_at_100")
        r100, _ = mean_std(group, "test_recall_at_100")
        lines.append(
            f"| {model} | {len(group)} | {fmt_pair(ap_m, ap_s)} | {fmt(auc_m)} | {fmt(p100)} | {fmt(r100)} |"
        )
    return "\n".join(lines)


def track_b_table(runs: dict) -> str:
    by_model: dict[str, list] = {}
    for s in runs.values():
        by_model.setdefault(s["model"], []).append(s)
    order = ["text", "time", "graph", "fixed_fusion", "global_fusion", "adaptive_fusion", "adaptive_fusion_noctx"]
    lines = [
        "| Model | Seeds | Test AP (mean ± std) | Val AP | drop_text_60 AP | drop_time_60 AP |",
        "|---|---|---|---|---|---|",
    ]
    for model in order:
        group = by_model.get(model, [])
        if not group:
            continue
        ap_m, ap_s = mean_std(group, "test_ap")
        vap_m, _ = mean_std(group, "validation_ap")
        dtext = [g["stress"]["drop_text_60"]["ap"] for g in group if g.get("stress", {}).get("drop_text_60")]
        dtime = [g["stress"]["drop_time_60"]["ap"] for g in group if g.get("stress", {}).get("drop_time_60")]
        dtext_s = fmt(float(np.mean(dtext))) if dtext else "n/a"
        dtime_s = fmt(float(np.mean(dtime))) if dtime else "n/a"
        lines.append(
            f"| {model} | {len(group)} | {fmt_pair(ap_m, ap_s)} | {fmt(vap_m)} | {dtext_s} | {dtime_s} |"
        )
    return "\n".join(lines)


def main() -> None:
    a = collect("yelp_static")
    b = collect("raw_review_replay")

    md = ["# ReviewRing AI - measured results", ""]
    md.append("All numbers come from saved runs under `artifacts/` (summary.json per run).")
    md.append("Mean ± std over seeds [17, 42, 73] where three seeds exist; single-seed ablations are labelled as such.")
    md.append("")

    md.append("## Track A: YelpChi static benchmark (proxy labels, transductive)")
    md.append("")
    md.append(track_a_table(a))
    md.append("")
    md.append("> Labels are Yelp filter-status proxy labels on preprocessed features. This is NOT text/timeline fraud detection and NOT future-event prediction.")

    md += ["", "## Track B: Amazon replay prototype (controlled synthetic campaigns)"]
    md.append("")
    md.append(track_b_table(b))

    # principal comparison paired deltas
    fixed = [s for s in b.values() if s["model"] == "fixed_fusion"]
    adaptive = [s for s in b.values() if s["model"] == "adaptive_fusion"]
    if fixed and adaptive:
        by_seed_fixed = {s["seed"]: s["metrics"]["test_ap"] for s in fixed}
        deltas = []
        for s in adaptive:
            if s["seed"] in by_seed_fixed:
                deltas.append(s["metrics"]["test_ap"] - by_seed_fixed[s["seed"]])
        if deltas:
            md.append("")
            md.append(
                f"**Principal comparison (adaptive - fixed, paired by seed):** "
                f"{float(np.mean(deltas)):+.4f} ± {float(np.std(deltas)):.4f} AP over seeds "
                f"{[s['seed'] for s in adaptive if s['seed'] in by_seed_fixed]}."
            )

    # main run extras
    main = sorted(
        [s for s in b.values() if s["model"] == "adaptive_fusion" and s["seed"] == 42],
        key=lambda s: s["run_id"],
    )
    if main:
        run_id = main[-1]["run_id"]
        md += ["", f"## Main run details (`{run_id}`)", ""]
        rep_dir = Path("artifacts") / run_id
        if (rep_dir / "replay.json").exists():
            rep = json.loads((rep_dir / "replay.json").read_text())
            md.append(f"- Daily replay: planted campaigns detected {rep['detected_campaigns']}/{rep['planted_campaigns']}")
            for k in ("recall_at_1d", "recall_at_7d", "recall_at_30d", "mean_delay_detected_only"):
                if rep.get(k) is not None:
                    md.append(f"- {k}: {rep[k]:.2f}")
        if (rep_dir / "rings_eval.json").exists():
            rr = json.loads((rep_dir / "rings_eval.json").read_text())
            cm = rr["campaign_matching"]
            md.append(f"- Ring candidates: {rr['n_candidates']} (top-{300} review budget, predeclared)")
            md.append(f"- Campaign precision/recall: {fmt(cm.get('campaign_precision'))} / {fmt(cm.get('campaign_recall'))}")
            md.append(f"- Matched-member mean IoU: {fmt(cm.get('mean_iou'))}")
            md.append(f"- Fragmentation/merging: {fmt(cm.get('fragmentation'), 2)} / {fmt(cm.get('merging'), 2)}")

    md += [
        "",
        "## Boundaries",
        "- Track B numbers describe the controlled simulator (known campaign membership), not real-world fraud performance.",
        "- Unmatched alerts in unlabelled Amazon background are unverified, not false positives.",
        "- Gate weights, stress tests and per-run figures are inside each run directory (`report.md`, `figures/`).",
    ]
    Path("docs/RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

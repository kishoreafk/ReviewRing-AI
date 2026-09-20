"""Collect the latest run per model/seed, without counting reruns as new seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def latest_runs(runs: list[dict]) -> dict:
    """Select one run per track/model/seed using the timestamped run ID.

    Run IDs sort chronologically within each model/seed. Do not use filesystem
    modification times: copying artifacts into Colab changes those timestamps.
    """
    selected = {}
    for summary in sorted(runs, key=lambda s: s["run_id"]):
        key = (summary["track"], summary["model"], summary["seed"])
        selected[key] = summary
    return {s["run_id"]: s for s in selected.values()}


def collect(track_filter: str, artifacts_dir: str | Path = "artifacts") -> dict:
    """Return the latest run_id -> summary for each model/seed in one track."""
    summaries = []
    for path in sorted(Path(artifacts_dir).glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("track") == track_filter:
            summaries.append(s)
    return latest_runs(summaries)


def finite_number(value: object) -> float | None:
    """Undefined metrics are absent, never zeros or NaNs in an aggregate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def metric_values(runs: list[dict], *keys: str) -> list[float]:
    values = []
    for run in runs:
        value = run
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        number = finite_number(value)
        if number is not None:
            values.append(number)
    return values


def values_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    # Sample SD across independent seeds; a single seed has no estimated SD.
    std = float(np.std(values, ddof=1)) if len(values) >= 2 else None
    return float(np.mean(values)), std


def mean_std(runs: list[dict], key: str) -> tuple[float | None, float | None]:
    return values_mean_std(metric_values(runs, "metrics", key))


def fmt(v: float | None, digits: int = 4) -> str:
    return "—" if finite_number(v) is None else f"{v:.{digits}f}"


def fmt_pair(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{digits}f} (single seed)"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def metric_cell(group: list[dict], *keys: str, with_std: bool = False) -> str:
    values = metric_values(group, *keys)
    mean, std = values_mean_std(values)
    result = fmt_pair(mean, std) if with_std else fmt(mean)
    if values and len(values) < len(group):
        result += f" (n={len(values)}/{len(group)} valid seeds)"
    return result


def seed_cell(group: list[dict]) -> str:
    seeds = sorted({s["seed"] for s in group})
    return f"{len(seeds)} ({', '.join(map(str, seeds))})" if seeds else "0"


def by_model(runs: dict) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for summary in latest_runs(list(runs.values())).values():
        groups.setdefault(summary["model"], []).append(summary)
    return groups


def track_a_table(runs: dict) -> str:
    groups = by_model(runs)
    lines = [
        "| Model | Seeds | Test AP (mean ± sample SD) | Test ROC-AUC | P@100 | R@100 |",
        "|---|---|---|---|---|---|",
    ]
    for model in ("prior", "numeric", "graph"):
        group = groups.get(model, [])
        lines.append(
            f"| {model} | {seed_cell(group)} | {metric_cell(group, 'metrics', 'test_ap', with_std=True)} "
            f"| {metric_cell(group, 'metrics', 'test_auc')} | {metric_cell(group, 'metrics', 'test_precision_at_100')} "
            f"| {metric_cell(group, 'metrics', 'test_recall_at_100')} |"
        )
    return "\n".join(lines)


def track_b_table(runs: dict) -> str:
    groups = by_model(runs)
    order = ["text", "time", "graph", "fixed_fusion", "global_fusion", "adaptive_fusion", "adaptive_fusion_noctx"]
    lines = [
        "| Model | Seeds | Test AP (mean ± sample SD) | Val AP | drop_text_60 AP | drop_time_60 AP |",
        "|---|---|---|---|---|---|",
    ]
    for model in order:
        group = groups.get(model, [])
        if not group:
            continue
        lines.append(
            f"| {model} | {seed_cell(group)} | {metric_cell(group, 'metrics', 'test_ap', with_std=True)} "
            f"| {metric_cell(group, 'metrics', 'validation_ap')} | {metric_cell(group, 'stress', 'drop_text_60', 'ap')} "
            f"| {metric_cell(group, 'stress', 'drop_time_60', 'ap')} |"
        )
    return "\n".join(lines)


def paired_ap_deltas(runs: dict) -> tuple[list[int], list[float]]:
    groups = by_model(runs)
    fixed = {s["seed"]: s for s in groups.get("fixed_fusion", [])}
    adaptive = {s["seed"]: s for s in groups.get("adaptive_fusion", [])}
    seeds, deltas = [], []
    for seed in sorted(fixed.keys() & adaptive.keys()):
        fixed_ap = metric_values([fixed[seed]], "metrics", "test_ap")
        adaptive_ap = metric_values([adaptive[seed]], "metrics", "test_ap")
        if fixed_ap and adaptive_ap:
            seeds.append(seed)
            deltas.append(adaptive_ap[0] - fixed_ap[0])
    return seeds, deltas


def build_report(artifacts_dir: str | Path = "artifacts") -> str:
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")
    a = collect("yelp_static", artifacts_dir)
    b = collect("raw_review_replay", artifacts_dir)

    md = ["# ReviewRing AI - measured results", ""]
    md.append(f"All numbers come from saved runs under `{artifacts_dir.as_posix()}` (summary.json per run).")
    md.append("Selection: keep the lexicographically latest timestamped run ID for each (track, model, seed). "
              "Repeated runs of one seed count once; selection never uses test performance.")
    md.append("Use an isolated artifacts directory for each experiment: latest-per-seed selection alone does not "
              "prove historical runs share the same code, data, split, or settings.")
    md.append("The Seeds column lists distinct selected seeds. Mean ± sample standard deviation uses finite metrics only; "
              "partial coverage is labelled with n, missing metrics are —, and single-seed results have no estimated SD.")
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
    seeds, deltas = paired_ap_deltas(b)
    if deltas:
        mean, std = values_mean_std(deltas)
        spread = f" ± {std:.4f}" if std is not None else " (single paired seed; SD unavailable)"
        md += ["", f"**Principal comparison (adaptive - fixed, paired by seed):** "
               f"{mean:+.4f}{spread} AP over {len(seeds)} distinct paired seeds {seeds}."]

    # main run extras
    main = sorted(
        [s for s in b.values() if s["model"] == "adaptive_fusion" and s["seed"] == 42],
        key=lambda s: s["run_id"],
    )
    if main:
        run_id = main[-1]["run_id"]
        md += ["", f"## Main run details (`{run_id}`)", ""]
        rep_dir = artifacts_dir / run_id
        if (rep_dir / "replay.json").exists():
            rep = json.loads((rep_dir / "replay.json").read_text(encoding="utf-8"))
            md.append(f"- Daily replay: planted campaigns detected {rep['detected_campaigns']}/{rep['planted_campaigns']}")
            for k in ("recall_at_1d", "recall_at_7d", "recall_at_30d", "mean_delay_detected_only"):
                if finite_number(rep.get(k)) is not None:
                    md.append(f"- {k}: {rep[k]:.2f}")
        if (rep_dir / "rings_eval.json").exists():
            rr = json.loads((rep_dir / "rings_eval.json").read_text(encoding="utf-8"))
            cm = rr["campaign_matching"]
            md.append(f"- Ring candidates: {rr['n_candidates']} (review budget recorded in the run configuration)")
            md.append(f"- Campaign precision/recall: {fmt(cm.get('campaign_precision'))} / {fmt(cm.get('campaign_recall'))}")
            md.append(f"- Matched-member mean IoU: {fmt(cm.get('mean_iou'))}")
            md.append(f"- Fragmentation/merging: {fmt(cm.get('fragmentation'), 2)} / {fmt(cm.get('merging'), 2)}")

    md += ["", "## Selected runs", ""]
    md += [f"- `{run_id}`" for run_id in sorted([*a, *b])]
    if not a and not b:
        md.append("No completed training summaries found.")
    md += [
        "",
        "## Boundaries",
        "",
        "- Track B numbers describe the controlled simulator (known campaign membership), not real-world fraud performance.",
        "- Unmatched alerts in unlabelled Amazon background are unverified, not false positives.",
        "- Gate weights, stress tests and per-run figures are inside each run directory (`report.md`, `figures/`).",
    ]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"),
                        help="Directory containing run subdirectories (default: artifacts)")
    parser.add_argument("--output", type=Path, default=Path("docs/RESULTS.md"),
                        help="Markdown output path (default: docs/RESULTS.md)")
    args = parser.parse_args(argv)
    report = build_report(args.artifacts_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()

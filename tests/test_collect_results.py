"""Regression tests for repeated seeds and isolated Colab experiment reports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "collect_results", Path(__file__).resolve().parents[1] / "scripts" / "collect_results.py"
)
collector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collector)


def write_run(root, model, seed, ap, stamp="20260920T1200000000", track="raw_review_replay", **extras):
    run_id = f"amazon-{model}-s{seed}-{stamp}"
    summary = dict(run_id=run_id, model=model, seed=seed, track=track, metrics={"test_ap": ap}, **extras)
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def test_latest_seed_selection_ignores_old_reruns_and_other_tracks(tmp_path):
    old = write_run(tmp_path, "graph", 17, 0.99, stamp="20260916T2300000000")
    new = write_run(tmp_path, "graph", 17, 0.6)
    second = write_run(tmp_path, "graph", 42, 0.8)
    write_run(tmp_path, "numeric", 42, 0.4, track="yelp_static")
    runs = collector.collect("raw_review_replay", tmp_path)
    assert set(runs) == {new["run_id"], second["run_id"]}
    assert old["run_id"] not in runs
    mean, std = collector.mean_std(list(runs.values()), "test_ap")
    assert mean == pytest.approx(0.7)
    assert std == pytest.approx(np.sqrt(0.02))
    assert "| graph | 2 (17, 42) | 0.7000 ± 0.1414 |" in collector.track_b_table(runs)


def test_paired_deltas_use_latest_run_once_for_each_common_finite_seed(tmp_path):
    summaries = [
        write_run(tmp_path, "fixed_fusion", 17, 0.1, stamp="20260916T1200000000"),
        write_run(tmp_path, "fixed_fusion", 17, 0.7),
        write_run(tmp_path, "adaptive_fusion", 17, 1.0, stamp="20260916T1200000000"),
        write_run(tmp_path, "adaptive_fusion", 17, 0.8),
        write_run(tmp_path, "fixed_fusion", 42, 0.7),
        write_run(tmp_path, "adaptive_fusion", 42, 0.9),
        write_run(tmp_path, "fixed_fusion", 73, 0.7),
        write_run(tmp_path, "adaptive_fusion", 73, float("nan")),
        write_run(tmp_path, "adaptive_fusion", 99, 0.9),
    ]
    seeds, deltas = collector.paired_ap_deltas({s["run_id"]: s for s in summaries})
    assert seeds == [17, 42]
    assert deltas == pytest.approx([0.1, 0.2])
    report = collector.build_report(tmp_path)
    assert "+0.1500 ± 0.0707 AP over 2 distinct paired seeds [17, 42]" in report


def test_single_seed_reruns_do_not_produce_standard_deviation(tmp_path):
    old = write_run(tmp_path, "graph", 42, 0.5, stamp="20260916T1200000000")
    new = write_run(tmp_path, "graph", 42, 0.8)
    table = collector.track_b_table({s["run_id"]: s for s in (old, new)})
    assert "| graph | 1 (42) | 0.8000 (single seed) |" in table
    assert "±" not in table.splitlines()[-1]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf"), "undefined", True])
def test_nonfinite_and_undefined_metrics_are_omitted(value):
    group = [{"metrics": {"test_ap": 0.0}}, {"metrics": {"test_ap": value}}, {"metrics": None}]
    assert collector.mean_std(group, "test_ap") == (0.0, None)
    assert collector.metric_cell(group, "metrics", "test_ap", with_std=True) == (
        "0.0000 (single seed) (n=1/3 valid seeds)"
    )
    assert collector.metric_cell([{"stress": {"drop_text_60": {"ap": value}}}], "stress", "drop_text_60", "ap") == "—"


def test_cli_isolated_root_output_and_main_run_extras(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_run(tmp_path / "artifacts", "adaptive_fusion", 42, 0.01)
    custom = tmp_path / "experiment" / "artifacts"
    main_run = write_run(custom, "adaptive_fusion", 42, 0.9)
    write_run(custom, "fixed_fusion", 42, 0.8)
    run_dir = custom / main_run["run_id"]
    (run_dir / "replay.json").write_text(json.dumps({"detected_campaigns": 2, "planted_campaigns": 3}), encoding="utf-8")
    (run_dir / "rings_eval.json").write_text(json.dumps({
        "n_candidates": 7, "campaign_matching": {"campaign_precision": 0.5, "campaign_recall": None}
    }), encoding="utf-8")
    output = tmp_path / "new reports" / "results.md"
    collector.main(["--artifacts-dir", str(custom), "--output", str(output)])
    report = output.read_text(encoding="utf-8")
    assert "0.9000 (single seed)" in report
    assert "0.0100" not in report
    assert "Daily replay: planted campaigns detected 2/3" in report
    assert "Ring candidates: 7" in report
    assert "+0.1000 (single paired seed; SD unavailable)" in report
    assert "Repeated runs of one seed count once" in report
    assert not (tmp_path / "docs" / "RESULTS.md").exists()
    assert capsys.readouterr().out == report


def test_missing_artifacts_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Artifacts directory does not exist"):
        collector.build_report(tmp_path / "missing")

"""Colab evaluation regressions: empty discovery and immutable run labels."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reviewring.pipelines import track_b_eval
from reviewring.rings.discover import candidate_priority_table


@pytest.fixture
def saved_run(tmp_path):
    run_dir = tmp_path / "artifacts" / "amazon-test"
    run_dir.mkdir(parents=True)
    summary = {
        "model": "adaptive_fusion", "seed": 42, "frozen_threshold": 0.5,
        "metrics": {"test_ap": 1.0},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    config = {
        "seed": 42,
        "paths": {"artifacts_dir": str(run_dir.parent), "processed_dir": str(tmp_path / "processed")},
        "rings": {"top_k_review_budget": 7},
    }
    return config, run_dir


def test_empty_discovery_skips_explanation_and_report_completes(saved_run):
    config, run_dir = saved_run
    candidate_priority_table([]).to_parquet(run_dir / "rings.parquet", index=False)
    pd.DataFrame({"review_id": ["a", "b"], "label": [0, 1], "score": [0.1, 0.9]}).to_parquet(
        run_dir / "test_predictions.parquet", index=False,
    )
    result = track_b_eval.explain(config, run_dir.name)
    assert result["status"] == "no_candidates"
    assert json.loads((run_dir / "explain_status.json").read_text())["status"] == "no_candidates"
    report = track_b_eval.report(config, run_dir.name)
    assert "No ring candidates" in Path(report["report"]).read_text()
    # An explicit requested candidate still gets a meaningful error.
    with pytest.raises(ValueError, match="not in run rings"):
        track_b_eval.explain(config, run_dir.name, candidate_id="ring-missing")


def test_report_uses_saved_labels_and_finite_scores(saved_run, monkeypatch):
    config, run_dir = saved_run
    # No processed data exists: a completed run's report is self-contained.
    pd.DataFrame({
        "review_id": ["a", "b", "c", "d", "e", "f"],
        "label": [0, 1, 1, -1, 0, np.nan],
        "score": [0.1, 0.9, np.nan, 0.99, np.inf, 0.8],
    }).to_parquet(run_dir / "test_predictions.parquet", index=False)
    seen = []

    def checked_ap(labels, scores):
        seen.append((labels.copy(), scores.copy()))
        return 1.0

    monkeypatch.setattr(track_b_eval.M, "average_precision", checked_ap)
    result = track_b_eval.report(config, run_dir.name)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], [0, 1])
    np.testing.assert_allclose(seen[0][1], [0.1, 0.9])
    assert Path(result["figures"]["pr_curve"]).exists()
    text = Path(result["report"]).read_text()
    assert "Report labels: saved predictions" in text
    assert "Labelled reviews with finite scores: 2" in text


def test_report_handles_no_usable_scores(saved_run):
    config, run_dir = saved_run
    pd.DataFrame({"review_id": ["a", "b"], "label": [0, 1], "score": [np.nan, np.nan]}).to_parquet(
        run_dir / "test_predictions.parquet", index=False,
    )
    result = track_b_eval.report(config, run_dir.name)
    assert "pr_curve" not in result["figures"]
    assert "Labelled reviews with finite scores: 0" in Path(result["report"]).read_text()


def test_legacy_report_joins_labels_by_id(saved_run, monkeypatch, caplog):
    config, run_dir = saved_run
    processed = Path(config["paths"]["processed_dir"])
    processed.mkdir()
    pd.DataFrame({"review_id": ["b", "a", "c"], "label": [1, 0, -1]}).to_parquet(
        processed / "labels.parquet", index=False,
    )
    pd.DataFrame({"review_id": ["a", "b", "c"], "score": [0.1, 0.9, 0.99]}).to_parquet(
        run_dir / "test_predictions.parquet", index=False,
    )

    def checked_ap(labels, scores):
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(scores, [0.1, 0.9])
        return 1.0

    monkeypatch.setattr(track_b_eval.M, "average_precision", checked_ap)
    result = track_b_eval.report(config, run_dir.name)
    assert "legacy run fallback" in Path(result["report"]).read_text()
    assert "no saved labels" in caplog.text


def test_ring_evaluation_records_budget_and_excludes_missing_scores(saved_run, monkeypatch):
    config, run_dir = saved_run
    reviews = pd.DataFrame({"review_id": ["a", "b"], "reviewer_id": ["u", "v"]})
    monkeypatch.setattr(track_b_eval, "TrackBData", lambda _: SimpleNamespace(reviews=reviews, test_idx=[0, 1]))
    pd.DataFrame({"review_id": ["a", "b"], "score": [0.8, np.nan]}).to_parquet(
        run_dir / "test_predictions.parquet", index=False,
    )
    processed = Path(config["paths"]["processed_dir"])
    processed.mkdir()
    pd.DataFrame({
        "partition": ["test"], "is_manipulative": [True], "member_review_ids": [["a"]],
    }).to_parquet(processed / "campaigns.parquet", index=False)

    def discover(scores, reviews, **kwargs):
        assert scores["review_id"].tolist() == ["a"]
        assert kwargs["top_k"] == 7
        return []

    monkeypatch.setattr(track_b_eval, "discover_rings", discover)
    result = track_b_eval.rings(config, run_dir.name)
    assert result["n_candidates"] == 0
    assert result["top_k_review_budget"] == 7
    assert json.loads((run_dir / "rings_eval.json").read_text())["top_k_review_budget"] == 7

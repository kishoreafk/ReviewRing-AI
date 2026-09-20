"""ReviewRing AI investigation interface (spec 18.1).

A thin presentation layer: loads versioned artifacts from a saved run and
calls library functions. No training, no data mutation, no direct label
access for discovery. Every claim shown is traceable to stored review IDs.

Run:  streamlit run app/streamlit_app.py -- --run <run_id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="ReviewRing AI", page_icon="🔍", layout="wide")

ROOT = Path(__file__).resolve().parents[1]


def _load_run(run_id: str) -> Path:
    run_dir = ROOT / "artifacts" / run_id
    if not run_dir.exists():
        st.error(f"Run {run_id} not found under artifacts/. Run training first.")
        st.stop()
    return run_dir


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "amazon" / "reviews.parquet")


@st.cache_data(show_spinner=False)
def load_rings(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "artifacts" / run_id / "rings.parquet")


@st.cache_data(show_spinner=False)
def load_predictions(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "artifacts" / run_id / "test_predictions.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id under artifacts/")
    args, _ = parser.parse_known_args()

    st.title("🔍 ReviewRing AI - Investigation interface")
    st.caption(
        "Candidate ring triage with evidence cards. Scores are model outputs on a "
        "controlled simulator; they are NOT confirmations of fraud. Shared activity "
        "can have legitimate explanations."
    )

    run_ids = sorted(
        [p.name for p in (ROOT / "artifacts").glob("amazon-*") if p.is_dir()], reverse=True
    )
    if not run_ids:
        st.warning("No Track B runs found. Train one: `python -m reviewring.cli --config configs/amazon_replay.yaml train --model adaptive_fusion`")
        st.stop()
    default = args.run if args.run in run_ids else run_ids[0]
    run_id = st.sidebar.selectbox("Saved run", run_ids, index=run_ids.index(default))
    run_dir = _load_run(run_id)
    summary = _read_json(run_dir / "summary.json") or {}

    st.sidebar.metric("Test AP (labelled)", f"{summary.get('metrics', {}).get('test_ap', float('nan')):.3f}")
    threshold = summary.get("frozen_threshold", 0.5)
    st.sidebar.caption(f"Frozen threshold: {threshold:.3f} (selected on validation)")

    tab_queue, tab_detail, tab_eval = st.tabs(
        ["🗂 Investigation queue", "🔎 Candidate detail", "📊 Evaluation"]
    )

    with tab_queue:
        budget = st.slider("Case budget (groups per session)", 3, 15, 8)
        try:
            rings = load_rings(run_id)
        except Exception:
            st.info("No rings for this run yet: `reviewring ... rings --run <run_id>`")
            st.stop()
        st.dataframe(rings.head(budget), use_container_width=True)
        st.caption(
            "priority = 0.5*mean(top-m scores) + 0.3*min(3*adjusted_overlap,1) + 0.2*synchrony; "
            "not a calibrated probability. Size-matched reference overlap is shown for context."
        )

    with tab_detail:
        if not len(rings):
            st.stop()
        candidate_id = st.selectbox("Candidate", rings["candidate_id"], index=0)
        card = None
        explain_file = run_dir / f"explain_{candidate_id}.json"
        if explain_file.exists():
            card = _read_json(explain_file)["card"]
        row = rings[rings["candidate_id"] == candidate_id].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Priority", f"{row['priority']:.3f}")
        c2.metric("Members", int(row["members"]))
        c3.metric("Targets", int(row["targets"]))
        c4.metric("Cited reviews", int(row["review_ids"]))

        st.subheader("Score components")
        st.write(
            {
                "mean top review score": round(float(row["mean_top_score"]), 3),
                "adjusted overlap": round(float(row["adjusted_overlap"]), 3),
                "synchrony": round(float(row["synchrony"]), 3),
                "size-matched reference": round(float(row["size_ref_mean"]), 3),
            }
        )

        # member events with citations
        reviews = load_reviews()
        pred = load_predictions(run_id).set_index("review_id")["score"]
        if card is not None:
            cited = card["cited_review_ids"]
        else:
            cited = []
        if cited:
            ev = reviews[reviews["review_id"].isin(cited)].sort_values("timestamp")
            ev = ev[["review_id", "reviewer_id", "target_id", "timestamp", "rating", "text"]]
            ev["model_score"] = ev["review_id"].map(pred)
            st.subheader("Supporting reviews (traceable IDs)")
            st.dataframe(ev, use_container_width=True)

        st.subheader("What-if (mask graph evidence)")
        if card and card.get("masking_test"):
            mt = card["masking_test"]
            colA, colB = st.columns(2)
            colA.metric("Score with graph evidence", f"{mt['original_score']:.3f}")
            colB.metric("Score after masking", f"{mt['score_after_removal']:.3f}")
            st.write(
                f"Removal effect: **{mt['removal_effect']:+.3f}** | retention gap: "
                f"{mt['retention_gap']:.3f} | compactness: {mt['compactness']:.2f} | "
                f"random-control advantage: {mt['random_control_advantage']:+.3f}"
            )
            st.caption(
                "Computational explanation: graph messages masked, features held fixed. "
                "This measures model-score dependence on graph pathways, not causation."
            )
        else:
            st.info("Run `reviewring ... explain --run <run> --candidate <id>` to compute.")

        if card:
            st.subheader("Evidence card (plain language)")
            st.info(card["summary"])

    with tab_eval:
        st.subheader("Run summary")
        st.json({k: v for k, v in summary.items() if k != "metrics"} if summary else {})
        report_file = run_dir / "report.md"
        if report_file.exists():
            st.markdown(report_file.read_text())
        fig_dir = run_dir / "figures"
        if fig_dir.exists():
            for fig in sorted(fig_dir.glob("*.png")):
                st.image(str(fig), caption=fig.stem, width=480)


if __name__ == "__main__":
    main()

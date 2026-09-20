"""Grounded evidence cards (spec 14.1): every claim cites verified fields.

Cards contain candidate id, model version, inference cutoff, members, targets,
review ids, relation types, time spans, counts, score components, missing
context, and a plain-language summary generated *only* from verified fields.
The wording never claims confirmed fraud; shared activity can have legitimate
explanations, and masking results are described as model-score dependence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_evidence_card(
    candidate,  # RingCandidate
    reviews: pd.DataFrame,
    masking_result=None,  # MaskingResult | None
    model_version: str = "unknown",
    as_of: str | None = None,
    label_status: str = "unverified",
) -> dict:
    """Assemble one structured card; all fields traceable to actual records."""
    member_set = set(candidate.members)
    events = reviews[reviews["reviewer_id"].isin(member_set)].sort_values("timestamp")
    events_cited = events[events["review_id"].isin(candidate.review_ids)]
    targets = sorted(set(candidate.targets))
    window = None
    if len(events_cited):
        ts = pd.to_datetime(events_cited["timestamp"], utc=True)
        window = {"start": ts.min().isoformat(), "end": ts.max().isoformat()}
    card = {
        "candidate_id": candidate.candidate_id,
        "model_version": model_version,
        "as_of": as_of,
        "label_status": label_status,
        "members": candidate.members,
        "n_members": len(candidate.members),
        "targets": targets,
        "n_targets": len(targets),
        "cited_review_ids": sorted(candidate.review_ids),
        "n_cited_reviews": len(candidate.review_ids),
        "relation_types": ["same_reviewer", "same_target", "co_target"],
        "observed_window": window,
        "score_components": {
            "mean_top_review_score": candidate.mean_top_score,
            "adjusted_overlap": candidate.adjusted_overlap,
            "synchrony": candidate.synchrony,
            "priority": candidate.priority,
            "size_matched_reference_overlap": candidate.size_ref_mean,
        },
        "priority_definition": (
            "0.5*mean(top-m review scores) + 0.3*min(3*adjusted_overlap,1) + 0.2*synchrony; "
            "not a calibrated fraud probability"
        ),
        "masking_test": None,
        "missing_context": _missing_context(events_cited),
        "summary": "",
    }
    if masking_result is not None:
        card["masking_test"] = {
            "mode": "computational_mask (graph messages masked, features held fixed)",
            "original_score": masking_result.original_score,
            "score_after_removal": masking_result.score_after_removal,
            "removal_effect": masking_result.removal_effect,
            "retention_gap": masking_result.retention_gap,
            "compactness": masking_result.compactness,
            "random_control_advantage": masking_result.random_control_advantage,
            "groups": masking_result.selected_groups,
            "n_model_calls": masking_result.n_model_calls,
            "sparse_explanation": masking_result.failed_sparse_explanation,
        }
    card["summary"] = _summary(card)
    return card


def _missing_context(events: pd.DataFrame) -> list[str]:
    missing = []
    if len(events) == 0:
        missing.append("no cited events found in the provided snapshot")
        return missing
    if events["text_missing"].any():
        missing.append(f"{int(events['text_missing'].sum())} cited reviews have no text")
    if events["verified_purchase"].isna().all():
        missing.append("verified-purchase status unknown for cited reviews")
    return missing


def _summary(card: dict) -> str:
    """Deterministic template summary; no invented facts."""
    parts = []
    parts.append(
        f"These {card['n_members']} accounts reviewed {card['n_targets']} common "
        f"target(s) between {card['observed_window']['start'][:10]} and "
        f"{card['observed_window']['end'][:10]}." if card["observed_window"] else
        f"These {card['n_members']} accounts reviewed {card['n_targets']} common target(s)."
    )
    sc = card["score_components"]
    parts.append(
        f"The model assigned a mean top review score of {sc['mean_top_review_score']:.2f} "
        f"and an adjusted shared-target overlap of {sc['adjusted_overlap']:.2f} "
        f"(size-matched reference {sc['size_matched_reference_overlap']:.2f})."
    )
    if card["masking_test"] and not card["masking_test"]["sparse_explanation"]:
        mt = card["masking_test"]
        parts.append(
            f"Masking the highlighted graph connections ({', '.join(g['relation'] for g in mt['groups']) or 'no groups'}) "
            f"lowered the model score from {mt['original_score']:.2f} to {mt['score_after_removal']:.2f}; "
            f"an equal-size random mask changed it by {mt['random_control_advantage']:+.2f} relative to the selected mask. "
            "This shows model-score dependence on graph pathways, not causation."
        )
    elif card["masking_test"]:
        parts.append(
            "Masking the highlighted connections did not lower the model score; "
            "no sparse graph explanation exists for this candidate."
        )
    parts.append(
        "Shared activity can also have legitimate explanations. "
        f"Label status: {card['label_status']}. This is not a confirmation of fraud."
    )
    return " ".join(parts)


def write_cards(cards: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, default=str) + "\n")

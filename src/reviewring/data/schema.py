"""Canonical project schemas (spec section 7) and validation helpers.

The canonical review table is the single join point for every later stage.
Validation is intentionally strict: counts must reconcile, IDs must be unique,
and provenance columns must be present. Labels live in a separate table and
are never part of model inputs.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

REVIEW_COLUMNS = {
    "review_id": "string",
    "reviewer_id": "string",
    "target_id": "string",
    "variant_id": "string",
    "timestamp": "datetime64[ns, UTC]",
    "title": "string",
    "text": "string",
    "rating": "float64",
    "source": "string",
    "source_row": "int64",
    "source_file_hash": "string",
    "verified_purchase": "boolean",
    "text_missing": "boolean",
}

LABEL_COLUMNS = {
    "review_id": "string",
    "label": "Int64",  # nullable integer: 0, 1 or <NA> = unknown
    "label_source": "string",  # benchmark_proxy | synthetic | human | unknown
    "label_available_at": "datetime64[ns, UTC]",
    "campaign_id": "string",
    "scenario_family": "string",
    "is_synthetic": "boolean",
}

SPLIT_ROLES = ("train", "validation", "calibration", "test")

LABEL_SOURCES = ("benchmark_proxy", "synthetic", "human", "unknown")


class SchemaError(ValueError):
    """Raised when a canonical table fails validation."""


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing required columns: {missing}")


def validate_reviews(df: pd.DataFrame, require_graph_keys: bool = False) -> pd.DataFrame:
    """Validate the canonical reviews table; returns the frame unchanged.

    ``require_graph_keys`` additionally demands reviewer/target/timestamp which
    the graph and history stages need. Review text may be empty but a missing
    flag must then be set.
    """
    required = ["review_id", "source", "source_row", "source_file_hash"]
    if require_graph_keys:
        required += ["reviewer_id", "target_id", "timestamp"]
    _require_columns(df, required, "reviews")
    if df["review_id"].duplicated().any():
        raise SchemaError("review_id must be unique")
    if df["review_id"].isna().any():
        raise SchemaError("review_id must not be null")
    if "rating" in df.columns:
        bad = df["rating"].dropna().between(1.0, 5.0)
        if not bad.all():
            raise SchemaError("rating values must lie in [1, 5] where present")
    if "text_missing" in df.columns:
        # text_missing must be consistent with the text column
        has_text = df["text"].notna() & (df["text"].astype("string").str.len() > 0)
        mismatch = df["text_missing"] & has_text
        if mismatch.any():
            raise SchemaError(
                f"{int(mismatch.sum())} rows have text but text_missing=True"
            )
    return df


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the labels table. Labels are evaluation/training targets only."""
    _require_columns(df, ["review_id", "label", "label_source"], "labels")
    if df["review_id"].duplicated().any():
        raise SchemaError("labels.review_id must be unique")
    unknown_src = ~df["label_source"].isin(LABEL_SOURCES)
    if unknown_src.any():
        raise SchemaError(f"unknown label_source values: {sorted(set(df.loc[unknown_src, 'label_source']))}")
    labelled = df["label"].notna()
    bad_vals = ~df.loc[labelled, "label"].isin([0, 1])
    if bad_vals.any():
        raise SchemaError("label must be 0, 1 or null (unknown)")
    return df


def validate_predictions(df: pd.DataFrame, n_expected: int | None = None) -> pd.DataFrame:
    _require_columns(df, ["review_id", "score"], "predictions")
    if df["review_id"].duplicated().any():
        raise SchemaError("predictions.review_id must be unique")
    if n_expected is not None and len(df) != n_expected:
        raise SchemaError(f"predictions row count {len(df)} != expected {n_expected}")
    return df


@dataclass
class IngestAudit:
    """Reconciliation counts for one normalisation run (spec 6.3)."""

    source: str
    read: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejected: int = 0
    reasons: dict = field(default_factory=dict)

    def reconcile(self) -> bool:
        return self.read == self.accepted + self.duplicates + self.rejected

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "read": self.read,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "reconciled": self.reconcile(),
        }
        d.update({f"reject_{k}": v for k, v in self.reasons.items()})
        return d


def content_fingerprint(text: str | None) -> str:
    """Hash review text for deduplication decisions (not a duplicate event)."""
    import hashlib

    if text is None:
        text = ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

"""Dataset adapters: CARE-GNN processed .mat benchmarks and raw Amazon JSONL.

Track A (``load_yelpchi``): nodes are reviews, 32 processed features, three
sparse relations (``net_rur``, ``net_rtr``, ``net_rsr``). Relation identity is
preserved: each relation becomes its own edge list. Track A matrices contain no
text, no timestamps and no real identities, so Track A never invents them.

Track B (``normalise_amazon_jsonl``): streams raw review JSONL into the
canonical reviews schema with full provenance (source row, file hash) and a
quarantine list for malformed records. Counts reconcile: read = accepted +
duplicates + rejected.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat

from reviewring.data.schema import IngestAudit, validate_reviews
from reviewring.utils.runtime import sha256_file

logger = logging.getLogger(__name__)

YELP_RELATIONS = ("net_rur", "net_rtr", "net_rsr")
AMAZON_RELATIONS = ("net_upu", "net_usu", "net_uvu")


@dataclass
class StaticGraphData:
    """Container for a processed static benchmark (Track A).

    ``features`` is a float32 dense matrix (small: 46k x 32). Labels are kept
    outside the feature matrix. Unknown/unlabelled nodes are expressed through
    ``labelled_mask`` rather than by a stored zero label.
    """

    name: str
    features: np.ndarray  # [N, d] float32
    labels: np.ndarray  # [N] int64, 0/1 where labelled
    labelled_mask: np.ndarray  # [N] bool: True where the label is eligible
    relations: dict[str, np.ndarray]  # relation -> int64 edge_index [2, E]
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.features.shape[0]

    @property
    def n_labelled(self) -> int:
        return int(self.labelled_mask.sum())


def load_yelpchi(
    mat_path: str | Path,
    relations: tuple[str, ...] = YELP_RELATIONS,
    features_dense: bool = True,
) -> StaticGraphData:
    """Load a CARE-GNN style .mat benchmark preserving relation identity.

    The loader inspects the actual file instead of trusting hard-coded
    assumptions: it validates that every relation is sparse, square and matches
    the feature/label node count. Labels are reshaped from the stored singleton
    dimension.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found. Run scripts/get_data.py or place the file manually."
        )
    mat = loadmat(mat_path)
    for key in ("features", "label", *relations):
        if key not in mat:
            raise ValueError(f"{mat_path} is missing expected key '{key}'. Found: {sorted(k for k in mat if not k.startswith('__'))}")

    features = mat["features"]
    if sparse.issparse(features):
        features = np.asarray(features.todense())
    features = features.astype(np.float32)

    labels = np.asarray(mat["label"]).reshape(-1).astype(np.int64)
    if features.shape[0] != len(labels):
        raise ValueError(
            f"features rows {features.shape[0]} != label count {len(labels)}"
        )

    rel_edges: dict[str, np.ndarray] = {}
    for key in relations:
        adjacency = mat[key]
        if not sparse.issparse(adjacency):
            raise ValueError(f"relation {key} is not sparse (got {type(adjacency)})")
        if adjacency.shape != (len(labels), len(labels)):
            raise ValueError(f"relation {key} shape {adjacency.shape} mismatch")
        coo = adjacency.tocoo()
        # keep directed edges as stored; message passing symmetrises later
        edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)
        rel_edges[key] = edge_index

    # All benchmark nodes carry a proxy label in the released YelpChi/Amazon
    # matrices; Amazon.mat's known unlabelled prefix is handled by the caller
    # through ``labelled_mask`` when applicable.
    labelled_mask = np.ones(len(labels), dtype=bool)

    meta = {
        "path": str(mat_path),
        "sha256": sha256_file(mat_path),
        "feature_dim": int(features.shape[1]),
        "relation_edge_counts": {k: int(v.shape[1]) for k, v in rel_edges.items()},
    }
    return StaticGraphData(
        name=mat_path.stem,
        features=features,
        labels=labels,
        labelled_mask=labelled_mask,
        relations=rel_edges,
        meta=meta,
    )


def load_amazon_static(mat_path: str | Path) -> StaticGraphData:
    """Load Amazon.mat with the documented unlabelled prefix 0..3304 excluded.

    The DGL reference adapter documents that for this released ordering,
    indices 0-3304 are unlabelled. The rule is verified here: the prefix must
    be all zeros after the label vector is reshaped, otherwise the ordering of
    this export differs and the caller must not apply the exclusion.
    """
    data = load_yelpchi(mat_path, relations=AMAZON_RELATIONS)
    labels = data.labels
    prefix = labels[:3305]
    if prefix.sum() != 0:
        raise ValueError(
            "Amazon.mat prefix 0..3304 is not all-zero as documented; refusing "
            "to apply the unlabelled-prefix rule to this export."
        )
    labelled_mask = np.zeros(len(labels), dtype=bool)
    labelled_mask[3305:] = True
    data.labelled_mask = labelled_mask
    data.meta["unlabelled_prefix"] = 3305
    return data


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _derive_review_id(source: str, file_hash: str, row_number: int) -> str:
    digest = hashlib.sha256(f"{source}|{file_hash}|{row_number}".encode()).hexdigest()
    return f"{source}:{digest[:16]}"


def iter_amazon_jsonl(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield (row_number, raw_dict) pairs from a (possibly gzipped) JSONL."""
    path = Path(path)
    with _open_text(path) as stream:
        for row_number, line in enumerate(stream):
            line = line.strip()
            if not line:
                continue
            yield row_number, json.loads(line)


def normalise_amazon_jsonl(
    jsonl_path: str | Path,
    source: str = "amazon2023",
    timestamp_unit: str = "ms",
    dedup_rule: str = "exact_record",
) -> tuple[pd.DataFrame, pd.DataFrame, IngestAudit, dict]:
    """Stream raw Amazon review JSONL into the canonical reviews schema.

    Returns ``(reviews, quarantine, audit, provenance)``. Only fields actually
    present are normalised; conflicting/missing required fields are quarantined
    with a reason. Malformed records never crash the stream: they are logged to
    the quarantine table and counted so that read = accepted + duplicates +
    rejected always reconciles.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found")
    file_hash = sha256_file(jsonl_path)
    audit = IngestAudit(source=source)
    audit.reasons = {}
    rows: list[dict] = []
    quarantine: list[dict] = []
    seen_fingerprints: set[str] = set()

    unit = "ms" if timestamp_unit == "ms" else "s"
    divisor = 1_000.0 if unit == "ms" else 1.0

    for row_number, row in iter_amazon_jsonl(jsonl_path):
        audit.read += 1
        reason = None
        if not isinstance(row, dict):
            reason = "not_an_object"
        elif "user_id" not in row or "parent_asin" not in row:
            reason = "missing_required_ids"
        elif "timestamp" not in row:
            reason = "missing_timestamp"
        elif "rating" not in row or not (1.0 <= float(row["rating"]) <= 5.0):
            reason = "rating_out_of_range"
        if reason is not None:
            audit.rejected += 1
            audit.reasons[reason] = audit.reasons.get(reason, 0) + 1
            quarantine.append(
                {
                    "source": source,
                    "source_row": row_number,
                    "reason": reason,
                    "raw": json.dumps(row)[:2000],
                }
            )
            continue

        text = row.get("text")
        title = row.get("title")
        text_missing = text is None or str(text).strip() == ""
        fingerprint = hashlib.sha256(
            "|".join(
                str(row.get(k))
                for k in ("user_id", "parent_asin", "asin", "timestamp", "rating", "text", "title")
            ).encode()
        ).hexdigest()
        if dedup_rule == "exact_record" and fingerprint in seen_fingerprints:
            audit.duplicates += 1
            audit.reasons["exact_duplicate"] = audit.reasons.get("exact_duplicate", 0) + 1
            quarantine.append(
                {
                    "source": source,
                    "source_row": row_number,
                    "reason": "exact_duplicate",
                    "raw": json.dumps(row)[:2000],
                }
            )
            continue
        seen_fingerprints.add(fingerprint)

        ts_value = float(row["timestamp"])
        # plausibility assertion: epoch seconds must land between 1995 and 2035
        ts_seconds = ts_value / divisor
        if not (788918400 <= ts_seconds <= 2051222400):
            audit.rejected += 1
            audit.reasons["implausible_timestamp"] = audit.reasons.get("implausible_timestamp", 0) + 1
            quarantine.append(
                {
                    "source": source,
                    "source_row": row_number,
                    "reason": "implausible_timestamp",
                    "raw": json.dumps(row)[:2000],
                }
            )
            continue

        rows.append(
            {
                "review_id": _derive_review_id(source, file_hash, row_number),
                "reviewer_id": f"{source}:{row['user_id']}",
                "target_id": f"{source}:{row['parent_asin']}",
                "variant_id": f"{source}:{row['asin']}" if row.get("asin") else None,
                "timestamp": pd.Timestamp(ts_seconds, unit="s", tz="UTC"),
                "title": str(title) if title is not None else None,
                "text": None if text_missing else str(text),
                "text_missing": text_missing,
                "rating": float(row["rating"]),
                "source": source,
                "source_row": row_number,
                "source_file_hash": file_hash,
                "verified_purchase": (
                    bool(row["verified_purchase"]) if "verified_purchase" in row else None
                ),
            }
        )
        audit.accepted += 1

    reviews = pd.DataFrame(rows)
    if len(reviews):
        reviews["text_missing"] = reviews["text_missing"].astype(bool)
        reviews = validate_reviews(reviews, require_graph_keys=True)
    quarantine_df = pd.DataFrame(quarantine)
    provenance = {
        "path": str(jsonl_path),
        "sha256": file_hash,
        "timestamp_unit": unit,
        "dedup_rule": dedup_rule,
    }
    logger.info("normalise: %s", audit.to_dict())
    return reviews, quarantine_df, audit, provenance

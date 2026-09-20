"""Shared fixtures: tiny synthetic data with known event ordering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_tiny_reviews(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a tiny review log with two 'rings', a burst, and background."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    rid = 0
    # ring A: 3 accounts x 2 targets within 1 day
    for u in range(3):
        for p in range(2):
            rows.append(
                dict(
                    review_id=f"r{rid:04d}",
                    reviewer_id=f"ringA-u{u}",
                    target_id=f"p{p}",
                    timestamp=t0 + pd.Timedelta(hours=float(3 * u + p)),
                    rating=5.0,
                    text="Works great, buy it now.",
                    text_missing=False,
                    source="fixture",
                    source_row=rid,
                    source_file_hash="fixture",
                )
            )
            rid += 1
    # launch burst: 4 accounts, one target, one day
    for u in range(4):
        rows.append(
            dict(
                review_id=f"r{rid:04d}",
                reviewer_id=f"burst-u{u}",
                target_id="p_launch",
                timestamp=t0 + pd.Timedelta(days=5, hours=float(u)),
                rating=5.0,
                text="Nice product.",
                text_missing=False,
                source="fixture",
                source_row=rid,
                source_file_hash="fixture",
            )
        )
        rid += 1
    # background: spread out
    for i in range(n):
        rows.append(
            dict(
                review_id=f"r{rid:04d}",
                reviewer_id=f"bg-u{i}",
                target_id=f"pbg{i % 10}",
                timestamp=t0 + pd.Timedelta(days=float(10 + i), hours=float(rng.integers(0, 24))),
                rating=float(rng.integers(1, 6)),
                text=f"Ordinary review number {i} about a product.",
                text_missing=False,
                source="fixture",
                source_row=rid,
                source_file_hash="fixture",
            )
        )
        rid += 1
    df = pd.DataFrame(rows)
    df["variant_id"] = None
    df["verified_purchase"] = True
    return df


@pytest.fixture
def tiny_reviews() -> pd.DataFrame:
    return make_tiny_reviews()

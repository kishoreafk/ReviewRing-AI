"""Daily replay: score after each day's batch; measure planted recovery.

Semantics (spec 9.3): each replay day scores events of that day using context
available up to the end of the previous day. Results are daily-resolution;
they are never described as instant event-time detection. Planted-campaign
recall, member recovery and alert volume are the reported quantities -
unmatched alerts in unlabelled background are unverified, not false positives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReplayStep:
    day: str
    n_scored: int
    alerts: int
    planted_scored: int
    planted_alerted: int


@dataclass
class ReplayResult:
    steps: list = field(default_factory=list)
    planted_campaigns: int = 0
    detected_campaigns: int = 0
    first_alert_delay_days: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "steps": [s.__dict__ for s in self.steps],
            "planted_campaigns": self.planted_campaigns,
            "detected_campaigns": self.detected_campaigns,
            "first_alert_delay_days": self.first_alert_delay_days,
            "detection_recall": (
                self.detected_campaigns / self.planted_campaigns if self.planted_campaigns else None
            ),
        }


def daily_replay(
    predictions: pd.DataFrame,
    campaign_members: dict[str, set],
    threshold: float,
    max_days: int | None = None,
) -> ReplayResult:
    """Aggregate per-day alerting and campaign detection timing.

    Evaluation only: ``campaign_members`` never influences any score, it only
    measures recovery after scores exist.
    """
    result = ReplayResult(planted_campaigns=len(campaign_members))
    df = predictions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["day"] = df["timestamp"].dt.floor("D")
    first_event = {}
    for campaign_id, members in campaign_members.items():
        member_ts = df[df["review_id"].isin(members)]["day"]
        if len(member_ts):
            first_event[campaign_id] = member_ts.min()
    detected = set()
    days = sorted(df["day"].unique())
    if max_days:
        days = days[:max_days]
    for day in days:
        day_rows = df[df["day"] == day]
        alerted = day_rows[day_rows["score"] >= threshold]
        step = ReplayStep(
            day=str(day.date()),
            n_scored=int(len(day_rows)),
            alerts=int(len(alerted)),
            planted_scored=0,
            planted_alerted=0,
        )
        for campaign_id, members in campaign_members.items():
            member_rows = day_rows[day_rows["review_id"].isin(members)]
            if len(member_rows):
                step.planted_scored += int(len(member_rows))
                hit = int(member_rows["score"].ge(threshold).sum())
                step.planted_alerted += hit
                if hit and campaign_id not in detected:
                    detected.add(campaign_id)
                    delay_days = (day - first_event[campaign_id]).total_seconds() / 86400
                    result.first_alert_delay_days[campaign_id] = float(max(delay_days, 0.0))
        result.steps.append(step)
    result.detected_campaigns = len(detected)
    return result


def recovery_at_horizon(result: ReplayResult, horizons: tuple[int, ...] = (1, 7, 30)) -> dict:
    """Campaign recall within N days of each campaign's first event.

    Undetected campaigns count as misses at every horizon (never silently
    dropped from the denominator).
    """
    out = {}
    delays = result.first_alert_delay_days
    for h in horizons:
        within = sum(1 for d in delays.values() if d <= h)
        out[f"recall_at_{h}d"] = (within / result.planted_campaigns) if result.planted_campaigns else None
    out["undetected_campaigns"] = result.planted_campaigns - len(delays)
    out["mean_delay_detected_only"] = float(np.mean(list(delays.values()))) if delays else None
    return out

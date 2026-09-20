"""Controlled manipulation campaigns and legitimate controls (spec 12.2).

Design principles enforced here:
- Known membership: every planted review carries a campaign id in the
  *evaluator-only* labels table, never in model inputs.
- Scenario families: ``template_wave`` and ``paraphrase_wave`` are trainable;
  ``burstwave`` (multi-peak timing, varied text) is held out for test only.
- No simulator shortcuts: manipulative groups and legitimate controls use the
  same text template pools, length distributions and metadata patterns. Both
  positive and negative sentiment appear on both sides, and legitimate
  all-5-star "fan waves" exist so rating polarity alone cannot decide.
- Legitimate controls: single-target launch bursts (synchrony without
  multi-target overlap), genuine repeat customers (overlap without
  synchrony), and isolated ordinary reviews.

The generator only *adds* rows to the event log; background rows are never
relabeled. Manipulation = the CONJUNCTION of multi-account + multi-target
shared activity + tight timing + similar text; controls carry only some of
these signals, which is what makes the task learnable but non-trivial.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MANIPULATIVE_FAMILIES = ("template_wave", "paraphrase_wave", "burstwave")

_POS_TEMPLATES = [
    "Worked as described, no complaints.",
    "Good quality for the price. Arrived on time.",
    "Exactly what I needed, packaging was fine.",
    "Second time buying this and it holds up well.",
    "Does the job. Would buy again.",
    "Arrived quickly and matches the listing.",
    "Solid value, my family likes it.",
    "Simple to use and works every time.",
    "Great little box, always something fun inside.",
    "Perfect gift, recipient was happy.",
]

_NEG_TEMPLATES = [
    "Stopped working after a few weeks, disappointed.",
    "Not as described, the size runs small.",
    "Packaging arrived damaged and the item leaked.",
    "Quality has gone down since I last ordered.",
    "Does not match the photos, returning it.",
    "Mediocre at best, expected more for the price.",
    "Worked once then failed, support was unhelpful.",
    "Cheap materials, would not recommend.",
    "Box arrived late and half the items were missing.",
    "Two shipments in a row had problems.",
]

_PARAPHRASE_RULES = [
    ("no complaints", "nothing to complain about"),
    ("arrived on time", "showed up on schedule"),
    ("exactly what I needed", "precisely what I was after"),
    ("would buy again", "will purchase again"),
    ("matches the listing", "is as listed"),
    ("disappointed", "let down"),
    ("runs small", "is smaller than expected"),
    ("would not recommend", "can't recommend"),
    ("very ", "quite "),
    ("really ", "genuinely "),
    ("good ", "decent "),
    ("Great ", "Fine "),
    ("arrived quickly", "came fast"),
    ("holds up well", "lasts"),
]


def _paraphrase(text: str, rng: np.random.Generator) -> str:
    for src, dst in _PARAPHRASE_RULES:
        if src in text and rng.random() < 0.6:
            text = text.replace(src, dst, 1)
    return text


def _pick_text(style: str, sentiment: str, member_i: int, rng: np.random.Generator) -> str:
    pool = _POS_TEMPLATES if sentiment == "positive" else _NEG_TEMPLATES
    base = pool[member_i % len(pool)]
    if style == "template":
        return base
    if style == "paraphrase":
        return _paraphrase(base, rng)
    # varied: light human-like variation
    variants = [
        base,
        base + " Overall fine.",
        f"Bought this recently. {base[0].lower()}{base[1:]}" if base else base,
        _paraphrase(base, rng),
        base + " Not much else to say.",
    ]
    return variants[(member_i + int(rng.integers(0, 3))) % len(variants)]


@dataclass
class SimulationResult:
    reviews: pd.DataFrame
    labels: pd.DataFrame  # evaluator-only
    campaigns: pd.DataFrame  # evaluator-only design manifest
    stats: dict = field(default_factory=dict)


class CampaignSimulator:
    """Plants manipulative campaigns and legitimate controls into partitions."""

    def __init__(self, config: dict):
        self.cfg = config
        self.source = "sim"

    # ------------------------------------------------------------------
    def _row(self, rid: str, reviewer: str, target: str, ts: pd.Timestamp, rating: float, text: str) -> dict:
        return {
            "review_id": rid,
            "reviewer_id": reviewer,
            "target_id": target,
            "variant_id": None,
            "timestamp": ts,
            "title": "Review",
            "text": text,
            "text_missing": False,
            "rating": float(rating),
            "source": self.source,
            "source_row": -1,
            "source_file_hash": "simulated",
            # verified flag is deterministic on the reviewer, not the label;
            # manipulators and controls both get a realistic mix
            "verified_purchase": bool(int(hashlib.md5(reviewer.encode()).hexdigest()[:2], 16) % 5 != 0),
        }

    def _rid(self, campaign_id: str, reviewer: str, target: str, ts: pd.Timestamp) -> str:
        digest = hashlib.sha256(f"{campaign_id}|{reviewer}|{target}|{ts.isoformat()}".encode()).hexdigest()
        return f"{self.source}:{digest[:16]}"

    def _targets(self, rng: np.random.Generator, n_targets: int, background_targets: list[str]) -> list[str]:
        chosen = list(
            rng.choice(background_targets, size=min(n_targets, len(background_targets)), replace=False)
        )
        while len(chosen) < n_targets:
            suffix = hashlib.md5(f"{rng.random()}".encode()).hexdigest()[:8]
            chosen.append(f"{self.source}:p{suffix}")
        return chosen

    # ------------------------------------------------------------------
    def _manipulative_campaign(
        self, campaign_id: str, family: str, rng: np.random.Generator,
        background_targets: list[str], t0: pd.Timestamp, t1: pd.Timestamp, span_days: int,
    ) -> dict:
        accounts = int(rng.choice(self.cfg.get("accounts_per_group", [5, 10, 20])))
        n_targets = int(rng.choice(self.cfg.get("targets_per_group", [2, 4, 8])))
        # weight durations toward spreading: real campaigns rarely finish in a day
        duration = int(rng.choice([1, 7, 30], p=[0.25, 0.35, 0.40]))
        sentiment = str(rng.choice(["positive", "negative"]))
        shared_fraction = float(rng.choice([0.3, 0.5, 0.8]))
        style = {"template_wave": "template", "paraphrase_wave": "paraphrase", "burstwave": "varied"}[family]
        start = t0 + pd.Timedelta(days=float(rng.integers(0, span_days)), hours=float(rng.integers(0, 24)))
        targets = self._targets(rng, n_targets, background_targets)
        n_shared = max(1, int(round(n_targets * shared_fraction)))
        # most campaigns mix in ordinary activity (harder, more realistic)
        mix_moderate = rng.random() < 0.6

        member_review_ids: list[str] = []
        rows: list[dict] = []
        for i in range(accounts):
            u = f"{self.source}:{campaign_id}-u{i:03d}"
            own = targets[:n_shared]
            if n_targets > n_shared:
                extra = list(rng.choice(targets, size=n_targets - n_shared, replace=False))
                own = list(dict.fromkeys(own + extra))
            for target in own:
                if family == "burstwave":
                    # multi-peak timing: 2-3 tight waves separated by quiet gaps
                    wave = int(rng.integers(0, max(duration // 3, 1) + 1))
                    ts = start + pd.Timedelta(
                        days=float(wave * 3), hours=float(rng.integers(0, 6)), minutes=float(rng.integers(0, 60))
                    )
                else:
                    ts = start + pd.Timedelta(
                        hours=float(rng.integers(0, max(duration * 24, 1))), minutes=float(rng.integers(0, 60))
                    )
                ts = min(ts, t1 - pd.Timedelta(minutes=1))
                rating = 5.0 if sentiment == "positive" else 1.0
                if mix_moderate and rng.random() < 0.3:
                    # ordinary activity interleaved with campaign behaviour
                    target = str(rng.choice(background_targets))
                    rating = float(rng.integers(1, 6))
                text = _pick_text(style, sentiment, i, rng)
                rid = self._rid(campaign_id, u, target, ts)
                rows.append(self._row(rid, u, target, ts, rating, text))
                member_review_ids.append(rid)
        return {
            "campaign_id": campaign_id, "family": family, "is_manipulative": True,
            "n_accounts": accounts, "n_targets": n_targets, "duration_days": duration,
            "sentiment": sentiment, "start": start.isoformat(),
            "member_review_ids": member_review_ids, "rows": rows,
        }

    def _legitimate_control(
        self, control_id: str, kind: str, rng: np.random.Generator,
        background_targets: list[str], t0: pd.Timestamp, t1: pd.Timestamp, span_days: int,
    ) -> dict:
        """Legitimate groups that share SOME coordination-like signals."""
        accounts = int(rng.integers(4, 16))
        member_review_ids: list[str] = []
        rows: list[dict] = []
        if kind == "launch_burst":
            # single shared target, tight window: synchrony WITHOUT multi-target
            # design; ~25% of accounts also buy 1-2 other items in the same
            # sale window, so per-account velocity is not a class shortcut
            target = self._targets(rng, 1, background_targets)[0]
            extra_pool = self._targets(rng, 3, background_targets)
            style = str(rng.choice(["varied", "template"]))  # some bursts use templates
            sentiment = str(rng.choice(["positive", "negative"]))
            start = t0 + pd.Timedelta(days=float(rng.integers(0, span_days)))
            fan_wave = bool(rng.random() < 0.4)  # enthusiastic genuine all-5 group
            for i in range(accounts):
                u = f"{self.source}:{control_id}-u{i:03d}"
                own_targets = [(target, 5.0 if fan_wave else float(rng.integers(2, 6)))]
                if rng.random() < 0.25:
                    for extra in list(rng.choice(extra_pool, size=int(rng.integers(1, 3)), replace=False)):
                        own_targets.append((str(extra), float(rng.integers(1, 6))))
                for target_i, rating_i in own_targets:
                    ts = start + pd.Timedelta(hours=float(rng.integers(0, 48)))
                    ts = min(ts, t1 - pd.Timedelta(minutes=1))
                    text = _pick_text(style, sentiment, i, rng) if style == "template" else _pick_text("varied", sentiment, i, rng)
                    rid = self._rid(control_id, u, target_i, ts)
                    rows.append(self._row(rid, u, target_i, ts, rating_i, text))
                    member_review_ids.append(rid)
        elif kind == "binge_shopper":
            # velocity WITHOUT coordination: accounts post several reviews of
            # unrelated targets within 1-3 days. Matches the manipulators'
            # per-account timing profile, so the timing expert cannot win on
            # velocity alone (spec 10.5: legitimate coordination controls).
            for i in range(accounts):
                u = f"{self.source}:{control_id}-u{i:03d}"
                n_reviews = int(rng.integers(3, 9))
                own = self._targets(rng, n_reviews, background_targets)
                start = t0 + pd.Timedelta(days=float(rng.integers(0, span_days)))
                for target in own:
                    ts = start + pd.Timedelta(hours=float(rng.integers(0, 72)))
                    ts = min(ts, t1 - pd.Timedelta(minutes=1))
                    rating = float(rng.integers(1, 6))
                    text = _pick_text("varied", "positive" if rating >= 3 else "negative", i, rng)
                    rid = self._rid(control_id, u, target, ts)
                    rows.append(self._row(rid, u, target, ts, rating, text))
                    member_review_ids.append(rid)
        elif kind == "repeat_customer":
            # overlap WITHOUT synchrony: same reviewers, wide time spread
            n_targets = int(rng.integers(2, 6))
            targets = self._targets(rng, n_targets, background_targets)
            for i in range(accounts):
                u = f"{self.source}:{control_id}-u{i:03d}"
                for target in list(rng.choice(targets, size=int(rng.integers(1, n_targets + 1)), replace=False)):
                    ts = t0 + pd.Timedelta(days=float(rng.integers(0, span_days)), hours=float(rng.integers(0, 24)))
                    ts = min(ts, t1 - pd.Timedelta(minutes=1))
                    rating = float(rng.integers(1, 6))
                    text = _pick_text("varied", "positive" if rating >= 3 else "negative", i, rng)
                    rid = self._rid(control_id, u, target, ts)
                    rows.append(self._row(rid, u, target, ts, rating, text))
                    member_review_ids.append(rid)
        else:  # isolated ordinary reviews
            for i in range(accounts):
                u = f"{self.source}:{control_id}-u{i:03d}"
                target = self._targets(rng, 1, background_targets)[0]
                ts = t0 + pd.Timedelta(days=float(rng.integers(0, span_days)))
                rating = float(rng.integers(1, 6))
                text = _pick_text("varied", "positive" if rating >= 3 else "negative", i, rng)
                rid = self._rid(control_id, u, target, ts)
                rows.append(self._row(rid, u, target, ts, rating, text))
                member_review_ids.append(rid)
        return {
            "campaign_id": control_id, "family": f"legit_{kind}", "is_manipulative": False,
            "n_accounts": accounts, "n_targets": 0, "duration_days": 0,
            "sentiment": "mixed", "start": "", "member_review_ids": member_review_ids,
            "rows": rows,
        }

    # ------------------------------------------------------------------
    def generate(
        self,
        partition: str,
        n_campaigns: int,
        background_targets: list[str],
        time_bounds: tuple[pd.Timestamp, pd.Timestamp],
        seed: int,
        heldout_family: str | None = "burstwave",
        legitimate_controls: int = 10,
        positive_rows_quota: int = 800,
        negative_rows_quota: int = 560,
    ) -> SimulationResult:
        """Generate campaigns until per-partition labelled-volume quotas are met.

        Quota-based generation keeps prevalence comparable across partitions
        (campaign sizes are random, so fixed campaign counts would make test
        prevalence swing wildly). ``n_campaigns`` is retained as the *minimum*
        number of manipulative campaigns per family quota.
        """
        rng = np.random.default_rng(seed)
        t0, t1 = pd.Timestamp(time_bounds[0]), pd.Timestamp(time_bounds[1])
        span_days = max((t1 - t0).days - 1, 1)

        rows: list[dict] = []
        label_rows: list[dict] = []
        campaign_rows: list[dict] = []

        def emit(design: dict) -> None:
            rows.extend(design["rows"])
            is_man = design["is_manipulative"]
            for r in design["rows"]:
                label_rows.append({
                    "review_id": r["review_id"],
                    "label": 1 if is_man else 0,
                    "label_source": "synthetic",
                    "campaign_id": design["campaign_id"],
                    "scenario_family": design["family"],
                    "is_synthetic": True,
                    "partition": partition,
                })
            campaign_rows.append({k: v for k, v in design.items() if k != "rows"})

        # manipulative campaigns from trainable families until positive quota
        families = ["template_wave", "paraphrase_wave"]
        idx = 0
        pos_rows = 0
        while pos_rows < positive_rows_quota or idx < n_campaigns:
            family = families[idx % len(families)]
            design = self._manipulative_campaign(
                f"sim-{partition}-c{idx:03d}", family, rng, background_targets, t0, t1, span_days
            )
            pos_rows += len(design["rows"])
            emit(design)
            idx += 1
        # held-out family ONLY in test (spec 9.2 step 6), fixed modest volume
        heldout_rows = 0
        if partition == "test" and heldout_family:
            for _ in range(max(n_campaigns // 3, 3)):
                design = self._manipulative_campaign(
                    f"sim-{partition}-c{idx:03d}", heldout_family, rng, background_targets, t0, t1, span_days
                )
                heldout_rows += len(design["rows"])
                emit(design)
                idx += 1
        # legitimate controls until negative quota
        kinds = ["launch_burst", "repeat_customer", "isolated", "binge_shopper"]
        neg_rows = 0
        c = 0
        while neg_rows < negative_rows_quota or c < legitimate_controls:
            kind = kinds[c % len(kinds)]
            design = self._legitimate_control(
                f"sim-{partition}-leg{c:03d}", kind, rng, background_targets, t0, t1, span_days
            )
            neg_rows += len(design["rows"])
            emit(design)
            c += 1

        reviews = pd.DataFrame(rows)
        labels = pd.DataFrame(label_rows)
        campaigns = pd.DataFrame(campaign_rows)
        campaigns["partition"] = partition  # evaluator-only design metadata
        stats = {
            "partition": partition,
            "n_manipulative_campaigns": int(len(campaigns[campaigns["is_manipulative"]])) if len(campaigns) else 0,
            "n_legitimate_controls": int(len(campaigns[~campaigns["is_manipulative"]])) if len(campaigns) else 0,
            "n_positive_rows": int(pos_rows + heldout_rows),
            "n_negative_rows": int(neg_rows),
            "n_reviews": len(reviews),
            "seed": seed,
        }
        return SimulationResult(reviews=reviews, labels=labels, campaigns=campaigns, stats=stats)

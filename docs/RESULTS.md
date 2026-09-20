# ReviewRing AI - measured results

All numbers come from saved runs under `artifacts/` (summary.json per run).
Mean ± std over seeds [17, 42, 73] where three seeds exist; single-seed ablations are labelled as such.

## Track A: YelpChi static benchmark (proxy labels, transductive)

| Model | Seeds | Test AP (mean ± std) | Test ROC-AUC | P@100 | R@100 |
|---|---|---|---|---|---|
| prior | 3 | 0.1454 ± 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| numeric | 3 | 0.4053 ± 0.0000 | 0.7803 | 0.6700 | 0.0668 |
| graph | 3 | 0.7814 ± 0.0056 | 0.9391 | 0.9767 | 0.0974 |

> Labels are Yelp filter-status proxy labels on preprocessed features. This is NOT text/timeline fraud detection and NOT future-event prediction.

## Track B: Amazon replay prototype (controlled synthetic campaigns)

| Model | Seeds | Test AP (mean ± std) | Val AP | drop_text_60 AP | drop_time_60 AP |
|---|---|---|---|---|---|
| text | 3 | 0.7595 ± 0.0039 | 0.8103 | n/a | n/a |
| time | 3 | 0.9283 ± 0.0010 | 0.8482 | n/a | n/a |
| graph | 3 | 0.9379 ± 0.0115 | 0.8585 | n/a | n/a |
| fixed_fusion | 3 | 0.9497 ± 0.0065 | 0.9240 | 0.9426 | 0.9015 |
| global_fusion | 1 | 0.9547 (single seed) | 0.9236 | 0.9464 | 0.9322 |
| adaptive_fusion | 3 | 0.9475 ± 0.0094 | 0.9308 | 0.9429 | 0.9051 |
| adaptive_fusion_noctx | 1 | 0.9563 (single seed) | 0.9213 | 0.9474 | 0.9356 |

**Principal comparison (adaptive - fixed, paired by seed):** -0.0022 ± 0.0024 AP over seeds [17, 42, 73].

## Main run details (`amazon-adaptive_fusion-s42-20260916T2006475778`)

- Daily replay: planted campaigns detected 24/24
- recall_at_1d: 0.96
- recall_at_7d: 1.00
- recall_at_30d: 1.00
- mean_delay_detected_only: 0.29
- Ring candidates: 14 (top-300 review budget, predeclared)
- Campaign precision/recall: 0.9286 / 0.5417
- Matched-member mean IoU: 0.7769
- Fragmentation/merging: 0.00 / 0.00

## Boundaries
- Track B numbers describe the controlled simulator (known campaign membership), not real-world fraud performance.
- Unmatched alerts in unlabelled Amazon background are unverified, not false positives.
- Gate weights, stress tests and per-run figures are inside each run directory (`report.md`, `figures/`).

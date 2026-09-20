# Run report: amazon-adaptive_fusion-s42-20260916T2006475778

- Model: `adaptive_fusion` (seed 42)
- Experts: ['text', 'graph', 'time']
- Frozen threshold: 0.526

## Test metrics (labelled synthetic rows only)

| metric | value |
|---|---|
| test_ap | 0.9537 |
| test_auc | 0.9331 |
| test_brier | 0.1033 |
| test_f1_at_0.5 | 0.8589 |
| test_n | 1612 |
| test_precision_at_100 | 1.0000 |
| test_precision_at_50 | 1.0000 |
| test_prevalence | 0.5738 |
| test_recall_at_100 | 0.1081 |
| test_recall_at_50 | 0.0541 |

> Label source: synthetic controlled campaigns (known membership). These numbers describe the simulator task, not real-world fraud detection.

## Replay (daily resolution)
- Planted campaigns detected: 24/24
- recall_at_1d: 0.96
- recall_at_7d: 1.00
- recall_at_30d: 1.00
- mean_delay_detected_only: 0.29

## Ring recovery
- Candidates: 14
- campaign_precision: 0.929
- campaign_recall: 0.542
- member_recovery: 0.777
- fragmentation: 0.000
- merging: 0.000

## Explanation (masked graph messages)
- compactness: 0.9299999999999999
- n_model_calls: 5
- original_score: 0.8339856266975403
- random_control_advantage: 0.0
- removal_effect: 0.6677833795547485
- retention_gap: 0.12631946802139282
- score_after_removal: 0.16620224714279175

## Boundary
- Track B results are measured on a controlled simulator with known campaign membership; unmatched alerts in unlabelled background are unverified, not false positives. No real-world fraud claim is made.
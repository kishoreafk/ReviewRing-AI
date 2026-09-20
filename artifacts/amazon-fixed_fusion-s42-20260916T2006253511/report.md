# Run report: amazon-fixed_fusion-s42-20260916T2006253511

- Model: `fixed_fusion` (seed 42)
- Experts: ['text', 'graph', 'time']
- Frozen threshold: 0.593

## Test metrics (labelled synthetic rows only)

| metric | value |
|---|---|
| test_ap | 0.9547 |
| test_auc | 0.9359 |
| test_brier | 0.1039 |
| test_f1_at_0.5 | 0.8575 |
| test_n | 1612 |
| test_precision_at_100 | 1.0000 |
| test_precision_at_50 | 1.0000 |
| test_prevalence | 0.5738 |
| test_recall_at_100 | 0.1081 |
| test_recall_at_50 | 0.0541 |

> Label source: synthetic controlled campaigns (known membership). These numbers describe the simulator task, not real-world fraud detection.

## Boundary
- Track B results are measured on a controlled simulator with known campaign membership; unmatched alerts in unlabelled background are unverified, not false positives. No real-world fraud claim is made.
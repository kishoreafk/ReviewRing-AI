# Run report: amazon-time-s42-20260916T2004337660

- Model: `time` (seed 42)
- Experts: ['time']
- Frozen threshold: 0.517

## Test metrics (labelled synthetic rows only)

| metric | value |
|---|---|
| test_ap | 0.9287 |
| test_auc | 0.8888 |
| test_brier | 0.1328 |
| test_f1_at_0.5 | 0.7990 |
| test_n | 1612 |
| test_precision_at_100 | 1.0000 |
| test_precision_at_50 | 1.0000 |
| test_prevalence | 0.5738 |
| test_recall_at_100 | 0.1081 |
| test_recall_at_50 | 0.0541 |

> Label source: synthetic controlled campaigns (known membership). These numbers describe the simulator task, not real-world fraud detection.

## Boundary
- Track B results are measured on a controlled simulator with known campaign membership; unmatched alerts in unlabelled background are unverified, not false positives. No real-world fraud claim is made.
# Run report: amazon-time-s73-20260916T2004554192

- Model: `time` (seed 73)
- Experts: ['time']
- Frozen threshold: 0.511

## Test metrics (labelled synthetic rows only)

| metric | value |
|---|---|
| test_ap | 0.9290 |
| test_auc | 0.8900 |
| test_brier | 0.1335 |
| test_f1_at_0.5 | 0.8017 |
| test_n | 1612 |
| test_precision_at_100 | 1.0000 |
| test_precision_at_50 | 1.0000 |
| test_prevalence | 0.5738 |
| test_recall_at_100 | 0.1081 |
| test_recall_at_50 | 0.0541 |

> Label source: synthetic controlled campaigns (known membership). These numbers describe the simulator task, not real-world fraud detection.

## Boundary
- Track B results are measured on a controlled simulator with known campaign membership; unmatched alerts in unlabelled background are unverified, not false positives. No real-world fraud claim is made.
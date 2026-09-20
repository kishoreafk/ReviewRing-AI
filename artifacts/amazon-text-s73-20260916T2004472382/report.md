# Run report: amazon-text-s73-20260916T2004472382

- Model: `text` (seed 73)
- Experts: ['text']
- Frozen threshold: 0.333

## Test metrics (labelled synthetic rows only)

| metric | value |
|---|---|
| test_ap | 0.7638 |
| test_auc | 0.7509 |
| test_brier | 0.1932 |
| test_f1_at_0.5 | 0.7659 |
| test_n | 1612 |
| test_precision_at_100 | 0.8900 |
| test_precision_at_50 | 1.0000 |
| test_prevalence | 0.5738 |
| test_recall_at_100 | 0.0962 |
| test_recall_at_50 | 0.0541 |

> Label source: synthetic controlled campaigns (known membership). These numbers describe the simulator task, not real-world fraud detection.

## Boundary
- Track B results are measured on a controlled simulator with known campaign membership; unmatched alerts in unlabelled background are unverified, not false positives. No real-world fraud claim is made.
# Run report: amazon-graph-s73-20260916T2005338994

- Model: `graph` (seed 73)
- Experts: ['graph']
- Frozen threshold: 0.451

## Test metrics (labelled synthetic rows only)

| metric | value |
|---|---|
| test_ap | 0.9453 |
| test_auc | 0.9191 |
| test_brier | 0.1181 |
| test_f1_at_0.5 | 0.8360 |
| test_n | 1612 |
| test_precision_at_100 | 1.0000 |
| test_precision_at_50 | 1.0000 |
| test_prevalence | 0.5738 |
| test_recall_at_100 | 0.1081 |
| test_recall_at_50 | 0.0541 |

> Label source: synthetic controlled campaigns (known membership). These numbers describe the simulator task, not real-world fraud detection.

## Boundary
- Track B results are measured on a controlled simulator with known campaign membership; unmatched alerts in unlabelled background are unverified, not false positives. No real-world fraud claim is made.
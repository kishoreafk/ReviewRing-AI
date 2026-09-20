# Track A run report: yelp-graph-s42-20260916T2018432034

- Model: `graph` (seed 42)
- Protocol: stratified random split, transductive; labels are Yelp filter proxy labels
- Calibration: {'coef': 0.8963226224550968, 'fitted': True, 'intercept': -1.5335717184148163, 'score_kind': 'proxy'}

## Validation metrics

| metric | value |
|---|---|
| validation_ap | 0.7885 |
| validation_auc | 0.9416 |
| validation_brier | 0.0913 |
| validation_f1_at_0.5 | 0.6639 |
| validation_n | 6892 |
| validation_precision_at_100 | 0.9800 |
| validation_precision_at_50 | 1.0000 |
| validation_precision_at_500 | 0.8880 |
| validation_prevalence | 0.1452 |
| validation_recall_at_100 | 0.0979 |
| validation_recall_at_50 | 0.0500 |
| validation_recall_at_500 | 0.4436 |

## Test metrics (frozen evaluation)

| metric | value |
|---|---|
| test_ap | 0.7843 |
| test_auc | 0.9402 |
| test_brier | 0.0592 |
| test_f1_at_0.5 | 0.6873 |
| test_n | 6896 |
| test_precision_at_100 | 0.9600 |
| test_precision_at_50 | 1.0000 |
| test_precision_at_500 | 0.8640 |
| test_prevalence | 0.1454 |
| test_recall_at_100 | 0.0957 |
| test_recall_at_50 | 0.0499 |
| test_recall_at_500 | 0.4307 |

## Boundary
- Track A matrices are preprocessed benchmark features; no review text, timestamps or identities exist at this level. Results are benchmark proxy-label classification, not text/timeline fraud claims.
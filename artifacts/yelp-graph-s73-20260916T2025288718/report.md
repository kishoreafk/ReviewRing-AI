# Track A run report: yelp-graph-s73-20260916T2025288718

- Model: `graph` (seed 73)
- Protocol: stratified random split, transductive; labels are Yelp filter proxy labels
- Calibration: {'coef': 0.9186446146378207, 'fitted': True, 'intercept': -1.5270053592346569, 'score_kind': 'proxy'}

## Validation metrics

| metric | value |
|---|---|
| validation_ap | 0.7754 |
| validation_auc | 0.9379 |
| validation_brier | 0.0924 |
| validation_f1_at_0.5 | 0.6539 |
| validation_n | 6892 |
| validation_precision_at_100 | 0.9900 |
| validation_precision_at_50 | 0.9800 |
| validation_precision_at_500 | 0.8640 |
| validation_prevalence | 0.1452 |
| validation_recall_at_100 | 0.0989 |
| validation_recall_at_50 | 0.0490 |
| validation_recall_at_500 | 0.4316 |

## Test metrics (frozen evaluation)

| metric | value |
|---|---|
| test_ap | 0.7850 |
| test_auc | 0.9405 |
| test_brier | 0.0596 |
| test_f1_at_0.5 | 0.6918 |
| test_n | 6896 |
| test_precision_at_100 | 0.9800 |
| test_precision_at_50 | 1.0000 |
| test_precision_at_500 | 0.8780 |
| test_prevalence | 0.1454 |
| test_recall_at_100 | 0.0977 |
| test_recall_at_50 | 0.0499 |
| test_recall_at_500 | 0.4377 |

## Boundary
- Track A matrices are preprocessed benchmark features; no review text, timestamps or identities exist at this level. Results are benchmark proxy-label classification, not text/timeline fraud claims.
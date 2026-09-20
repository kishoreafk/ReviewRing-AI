# Track A run report: yelp-numeric-s42-20260916T2032396087

- Model: `numeric` (seed 42)
- Protocol: stratified random split, transductive; labels are Yelp filter proxy labels
- Calibration: {'coef': 1.0220912103598954, 'fitted': True, 'intercept': -1.7632905906640448, 'score_kind': 'proxy'}

## Validation metrics

| metric | value |
|---|---|
| validation_ap | 0.4092 |
| validation_auc | 0.7764 |
| validation_brier | 0.1932 |
| validation_f1_at_0.5 | 0.4059 |
| validation_n | 6892 |
| validation_precision_at_100 | 0.6800 |
| validation_precision_at_50 | 0.7200 |
| validation_precision_at_500 | 0.5300 |
| validation_prevalence | 0.1452 |
| validation_recall_at_100 | 0.0679 |
| validation_recall_at_50 | 0.0360 |
| validation_recall_at_500 | 0.2647 |

## Test metrics (frozen evaluation)

| metric | value |
|---|---|
| test_ap | 0.4053 |
| test_auc | 0.7803 |
| test_brier | 0.1058 |
| test_f1_at_0.5 | 0.2409 |
| test_n | 6896 |
| test_precision_at_100 | 0.6700 |
| test_precision_at_50 | 0.7800 |
| test_precision_at_500 | 0.5060 |
| test_prevalence | 0.1454 |
| test_recall_at_100 | 0.0668 |
| test_recall_at_50 | 0.0389 |
| test_recall_at_500 | 0.2522 |

## Boundary
- Track A matrices are preprocessed benchmark features; no review text, timestamps or identities exist at this level. Results are benchmark proxy-label classification, not text/timeline fraud claims.
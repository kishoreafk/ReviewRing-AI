# Track A run report: yelp-graph-s17-20260916T2011574939

- Model: `graph` (seed 17)
- Protocol: stratified random split, transductive; labels are Yelp filter proxy labels
- Calibration: {'coef': 0.9802303912843696, 'fitted': True, 'intercept': -1.682214764918233, 'score_kind': 'proxy'}

## Validation metrics

| metric | value |
|---|---|
| validation_ap | 0.7715 |
| validation_auc | 0.9382 |
| validation_brier | 0.1001 |
| validation_f1_at_0.5 | 0.6372 |
| validation_n | 6892 |
| validation_precision_at_100 | 0.9800 |
| validation_precision_at_50 | 0.9600 |
| validation_precision_at_500 | 0.8660 |
| validation_prevalence | 0.1452 |
| validation_recall_at_100 | 0.0979 |
| validation_recall_at_50 | 0.0480 |
| validation_recall_at_500 | 0.4326 |

## Test metrics (frozen evaluation)

| metric | value |
|---|---|
| test_ap | 0.7750 |
| test_auc | 0.9367 |
| test_brier | 0.0617 |
| test_f1_at_0.5 | 0.6796 |
| test_n | 6896 |
| test_precision_at_100 | 0.9900 |
| test_precision_at_50 | 1.0000 |
| test_precision_at_500 | 0.8600 |
| test_prevalence | 0.1454 |
| test_recall_at_100 | 0.0987 |
| test_recall_at_50 | 0.0499 |
| test_recall_at_500 | 0.4287 |

## Boundary
- Track A matrices are preprocessed benchmark features; no review text, timestamps or identities exist at this level. Results are benchmark proxy-label classification, not text/timeline fraud claims.
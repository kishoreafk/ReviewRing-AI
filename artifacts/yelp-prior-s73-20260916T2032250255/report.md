# Track A run report: yelp-prior-s73-20260916T2032250255

- Model: `prior` (seed 73)
- Protocol: stratified random split, transductive; labels are Yelp filter proxy labels
- Calibration: {'fitted': False, 'score_kind': 'proxy'}

## Validation metrics

| metric | value |
|---|---|
| validation_ap | 0.1452 |
| validation_auc | 0.5000 |
| validation_brier | 0.1241 |
| validation_f1_at_0.5 | 0.0000 |
| validation_n | 6892 |
| validation_precision_at_100 | 0.0000 |
| validation_precision_at_50 | 0.0000 |
| validation_precision_at_500 | 0.0000 |
| validation_prevalence | 0.1452 |
| validation_recall_at_100 | 0.0000 |
| validation_recall_at_50 | 0.0000 |
| validation_recall_at_500 | 0.0000 |

## Test metrics (frozen evaluation)

| metric | value |
|---|---|
| test_ap | 0.1454 |
| test_auc | 0.5000 |
| test_brier | 0.1243 |
| test_f1_at_0.5 | 0.0000 |
| test_n | 6896 |
| test_precision_at_100 | 0.0000 |
| test_precision_at_50 | 0.0000 |
| test_precision_at_500 | 0.0000 |
| test_prevalence | 0.1454 |
| test_recall_at_100 | 0.0000 |
| test_recall_at_50 | 0.0000 |
| test_recall_at_500 | 0.0000 |

## Boundary
- Track A matrices are preprocessed benchmark features; no review text, timestamps or identities exist at this level. Results are benchmark proxy-label classification, not text/timeline fraud claims.
# Training review and runtime checks

The review focused on both training pipelines, fusion, masked loss, calibration,
configuration use, and checkpoint/prediction output.

Changes:
- Track B now respects configured batch size, positive-class weight cap, and
  fusion weight decay. Fusion epochs and patience inherit the training settings;
  optional `fusion_max_epochs` and `fusion_patience` override them.
- Fusion excludes labelled rows with no available expert before optimization.
  Missing scores remain NaN in predictions and are excluded from calibration,
  threshold selection, classification metrics, and missing-modality stress metrics.
  Classification summaries report `n_excluded`; gate statistics use finite rows.
- Calibration supports empty predictions and missing scores, and clears a previous
  fitted model when refitting on insufficient classes.
- Masked BCE excludes unknown labels even if the supplied mask includes them.
  Empty losses retain the correct device and gradient connection. Single-class
  positive-only data no longer receives a zero positive weight.
- Track A validates one fanout per graph layer and advances the sampling random
  sequence across batches instead of resetting it for each batch.
- Track A can save completed runs without the raw Yelp download: it records the
  source hash from the preparation manifest, explicitly labels that provenance,
  and hashes the actual processed feature file used for training.
- Missing-modality stress sampling now uses a stable modality order.

## Reproduce

Use an environment with the project dependencies and pytest installed. Run from
this repository's root (PowerShell example):

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests -q --basetemp=.pytest_cache/training-review
python scripts/dry_run_training.py
python scripts/dry_run_training.py --config configs/yelp_static.yaml --model graph --epochs 1
```

The dry-run script uses existing processed data, caps training epochs, and saves
checkpoints, histories, predictions, and summaries under `artifacts/dry_runs/`.
It exercises calibration and evaluation too. These short-run metrics are runtime
checks, not evidence of improved predictive accuracy. Original reported runs are
not overwritten. CPU execution was checked; CUDA was not validated.

Regression coverage includes complete two-epoch runs for all seven Track B model
choices with all-input-missing rows in every partition, Track A sampled graph
training, invalid fanouts, calibration edge cases, masked-loss gradients, shared
configuration defaults, and source provenance without a raw download.

Cline collaboration was unavailable because this session did not expose a native
Cline control interface. The review and checks were performed directly.

## Validation completed (2026-09-20)

- Full suite: **52 passed**, one existing SciPy deprecation warning.
- Amazon adaptive fusion: two epochs for each expert and fusion; completed
  calibration, stress evaluation, checkpoint and prediction writes.
  Run: `artifacts/dry_runs/amazon-adaptive_fusion-s42-20260920T1703218557`.
- Yelp graph: one epoch on existing processed data; completed calibration and
  all artifact writes using recorded source provenance.
  Run: `artifacts/dry_runs/yelp-graph-s42-20260920T1704127922`.
- `git diff --check` passed.
- Environment: Python 3.12.14, PyTorch 2.14.0+cpu, local `.venv`.

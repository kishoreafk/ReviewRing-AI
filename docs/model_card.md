# Model card: ReviewRing AI

## Models

### Track A: relational GraphSAGE (yelp_static)
- Two relation-aware GraphSAGE stages (64 hidden), separate convolutions per relation (`net_rur`, `net_rtr`, `net_rsr`), outputs mean-aggregated, self-feature path retained, dropout 0.2.
- Mini-batch training with per-relation neighbour sampling (fanout 10 then 5, uniform with replacement + dedup, the original GraphSAGE scheme), batch 512, AdamW lr 1e-3 / wd 1e-4, weighted BCE (`w_pos = N_neg/N_pos`, capped at 100), early stopping on validation AP, Platt calibration on a separate calibration partition, test evaluated once.
- Measured: test AP 0.782 ± 0.004 over seeds 17/42/73 (numeric baseline 0.405, prior 0.145).

### Track B: three experts + adaptive gate (raw_review_replay)
- **Language expert**: frozen MiniLM (all-MiniLM-L6-v2, 384-d) → 128 → 64 MLP; TF-IDF+SVD fallback (fit on train rows only) when the encoder is unavailable; missing text → zero vector + availability flag.
- **Timing expert**: 16 past-only features (windows 1/7/30 d, log1p + robust scaling fitted on train rows, missingness indicators) → 64-d MLP.
- **Relationship expert**: relational GraphSAGE (2 layers, 64-d) over the causal typed review graph (same_reviewer / same_target / co_target; edges point strictly backward in time with `available_at` evidence).
- **Fusion**: fixed uniform mean (B5), global learned mixture (B6), adaptive per-example gate (P1: expert 64-d states + availability + 4 past-only context counts → masked softmax). Experts trained independently and frozen; gate trained on frozen expert outputs with early stopping on validation AP.

## Measured behaviour (seed 17/42/73 mean ± std, test partition, labelled synthetic rows)

| model | AP |
|---|---|
| text | 0.760 ± 0.004 |
| time | 0.928 ± 0.001 |
| graph | 0.938 ± 0.012 |
| fixed fusion | **0.950 ± 0.007** |
| global fusion (1 seed) | 0.955 |
| adaptive gate | 0.948 ± 0.009 |
| adaptive gate w/o context (1 seed) | 0.956 |

- Principal comparison: adaptive − fixed = **−0.0022 ± 0.0024 AP** (paired by seed). The gate matches fixed fusion but does not beat it on AP in this simulator.
- Missing-modality stress (availability overridden on a random 60% of test rows): drop_time_60 AP fixed 0.902 vs adaptive 0.905 — near-identical; the single-seed global_fusion (0.932) and no-context gate (0.936) score higher under this probe, so no robustness claim is made for the gate.
- Daily replay: 24/24 planted test campaigns alerted; recall@1d 0.96, recall@7d 1.00, recall@30d 1.00; mean delay (detected) 0.29 days.
- Ring discovery (predeclared top-300 budget, maximum-weight IoU matching): campaign precision 0.93 / recall 0.54, matched-member mean IoU 0.78, no fragmentation/merging above threshold. The budget intentionally trades recall for auditability; the limitation is documented, not tuned away post hoc.
- Gate routing (test, seed 42): mean weights ≈ text 0.40 / graph 0.21 / time 0.39 with per-example variation — no expert collapse.

## Intended use / non-use

- Intended: offline research on review-coordination detection; investigation triage where every alert is examined by a human moderator with supporting review IDs.
- Non-use: autonomous banning/removal, real-time account decisions, claims about specific real accounts, platforms, or prices.

## Out-of-scope claims (explicitly NOT established)

- "Detects real fraud rings" — Track B evaluates the simulator; unmatched alerts in unlabelled background are unverified.
- Calibrated real-world probability — Platt calibration is fit on synthetic labels and estimates the synthetic task only; scores are presented as model scores.
- Causality — masking shows model-score dependence, not causal effect.
- Multilingual robustness — evaluation is English-only.

## Failure examples worth showing

- Some candidates (e.g. an earlier seed's ring-0003) report removal effect 0.000 — a documented "no sparse graph explanation exists" case (spec 14.4 requires reporting these rather than forcing a narrative); the shipped main-run candidate ring-0004 shows removal effect 0.67 with retention gap 0.03.
- Legitimate launch bursts with all-5-star ratings and template-like wording are deliberately planted as controls; some appear in the top candidate queue, demonstrating why moderator review is required.

## Training data & leakage controls

See data cards. Enforced controls: chronological splits with campaign disjointness (audit in `simulate`), past-only features/graph (tests: `test_time_isolation_*`, `test_graph_causal_direction_*`), unknown labels masked from loss/metrics (tests: `test_bce_mask_excludes_unknowns`, `test_single_class_metrics_are_none_not_invented`), scalers fitted on train rows only, test evaluated once per frozen protocol, negative-control label-shuffle test in the suite.

## Known training-protocol limitations (documented, accepted)

- Fusion heads are trained on in-sample frozen-expert outputs rather than out-of-fold predictions; the spec-permitted mitigation is the fixed-vs-gate comparison on validation, which is reported. OOF gate training is a natural next step.
- Stress-test modality drops are computed on test rows as a computational robustness probe (features unchanged); cross-variant comparisons should be read together with the global_fusion row in RESULTS.md, which also scores well under timing drops.
- The frozen test partition is evaluated once per saved run; the CLI refuses to overwrite run directories, so hyperparameter iterations create new runs rather than silently re-using test.
- Track A's mini-batch sampler symmetrises adjacency (standard GraphSAGE message-passing semantics for a transductive static snapshot); the causal directed-edge guarantee applies to the constructed graph and to Track B's full-batch message passing.

## Reproducibility

Every run directory contains `manifest.json` (resolved config, data SHA256, split hash, seeds, torch/python versions), `summary.json` (all metrics), and the model bundle. `scripts/run_all.sh` regenerates everything; `scripts/collect_results.py` rebuilds `docs/RESULTS.md` from artifacts.

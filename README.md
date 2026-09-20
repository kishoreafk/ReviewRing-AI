# ReviewRing AI

**Adaptive graph–language learning for coordinated review manipulation detection and evidence-based investigation.**

This repository implements the ReviewRing AI development specification (`docs/SPEC.md`): two tracks, one trained end-to-end from the command line — no notebooks required.

| Track | Data | Task | Result (test AP) |
|---|---|---|---|
| **A: static benchmark** | CARE-GNN YelpChi (.mat) | review-level proxy-label classification | prior 0.145 → numeric 0.405 → **relational GraphSAGE 0.781 ± 0.006** (3 seeds) |
| **B: raw-review replay** | Amazon Reviews 2023 (Subscription_Boxes) + planted campaigns | manipulated-campaign detection with three experts + adaptive gate | **fusion 0.950 ± 0.007** vs best single expert 0.938; planted campaigns 24/24 detected in daily replay |

Full measured tables: [`docs/RESULTS.md`](docs/RESULTS.md). All numbers come from saved runs under `artifacts/`.

## What is here

```
configs/            base + per-track YAML (splits, windows, caps, hyperparameters)
src/reviewring/     package: data, features, graph, models, training, simulation,
                    rings, explain, evaluation, pipelines, CLI
app/streamlit_app.py  investigation interface (queue / candidate detail / what-if / evaluation)
tests/              leakage, schema, mask, metric, parity and audit tests (pytest)
scripts/            get_data.py (Kaggle-first with validated fallbacks), run_all.sh,
                    collect_results.py
colab/              one-shot Colab setup: colab_run.sh + README_COLAB.md
docs/               SPEC (original guide), RESULTS, data cards, model card
artifacts/          saved runs: bundles, manifests, metrics, reports, figures
```

## Quick start (local or Colab)

```bash
python -m pip install -e .
python -m pip install -r requirements.txt

# 1. data (uses ~/.kaggle/kaggle.json when present; falls back with documented reasons)
python scripts/get_data.py

# 2. train + evaluate everything (Track A ~25 min CPU, Track B ~8 min CPU)
bash scripts/run_all.sh all

# 3. inspect results
python scripts/collect_results.py
```

Track-by-track, without the runner:

```bash
# Track A
python -m reviewring.cli --config configs/yelp_static.yaml prepare
python -m reviewring.cli --config configs/yelp_static.yaml split
python -m reviewring.cli --config configs/yelp_static.yaml train --model graph --seed 42
python -m reviewring.cli --config configs/yelp_static.yaml report --run <run_id>

# Track B
python -m reviewring.cli --config configs/amazon_replay.yaml prepare
python -m reviewring.cli --config configs/amazon_replay.yaml split
python -m reviewring.cli --config configs/amazon_replay.yaml simulate
python -m reviewring.cli --config configs/amazon_replay.yaml features
python -m reviewring.cli --config configs/amazon_replay.yaml graph
python -m reviewring.cli --config configs/amazon_replay.yaml train --model adaptive_fusion --seed 42
python -m reviewring.cli --config configs/amazon_replay.yaml replay  --run <run_id>
python -m reviewring.cli --config configs/amazon_replay.yaml rings   --run <run_id>
python -m reviewring.cli --config configs/amazon_replay.yaml explain --run <run_id>
python -m reviewring.cli --config configs/amazon_replay.yaml report  --run <run_id>

# investigation interface
streamlit run app/streamlit_app.py
```

## Architecture (spec section 8)

```
review events ──► validate/normalise/split ──► past-only features + causal graph
        ├──► language expert (MiniLM → MLP)              ┐
        ├──► relationship expert (relational GraphSAGE)  ├─► adaptive fusion gate ─► review scores
        └──► timing expert (past-only history → MLP)     ┘            │
                                    candidate ring discovery ◄──────────┘
                                            │
                              evidence cards + counterfactual masking
```

## Scientific honesty (read this)

- Track A labels are **platform filter proxy labels** on preprocessed features; the setting is transductive, not future-event prediction.
- Track B numbers describe the **controlled simulator** (known planted membership). Unmatched alerts in unlabelled background are *unverified*, never "false positives".
- The principal comparison (adaptive vs fixed fusion) is reported as measured: **−0.0022 ± 0.0024 AP paired by seed** — the gate matches but does not beat fixed fusion on AP (the timing expert already dominates this simulator; see the model card's stress and ablation notes). See `docs/RESULTS.md`.
- Ring discovery operates under a predeclared top-300 review budget; campaign precision 0.93 / recall 0.54 with matched-member IoU 0.78 is reported with that documented recall limitation.

## Tests

```bash
python -m pytest tests/ -q       # 40 tests: time isolation, target isolation,
                                 # campaign disjointness, mask correctness, metric
                                 # fixtures, checkpoint parity, explanation audit...
```

## Dataset access and provenance

`scripts/get_data.py` tries Kaggle first (requires your own `kaggle.json`) and **validates schema** before accepting any mirror; the only Kaggle YelpChi upload found at development time was a preprocessed `.pt` bundle that loses relation identity, so it is rejected and the canonical CARE-GNN GitHub source is used instead — loudly, never silently. Amazon background comes from the official Amazon Reviews 2023 category file. See `docs/data_cards/`.

## License

Code: MIT. Datasets keep their original terms; do not redistribute raw data from this repository.

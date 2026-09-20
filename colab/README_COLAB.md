# Running ReviewRing AI on Google Colab (no notebook required)

The project is a CLI package: clone it from GitHub into Colab and run the files directly.

## Option 1 — one-shot (recommended)

```python
# Cell 1: clone + run everything (~40 min on CPU, faster on T4)
!git clone https://github.com/<you>/ReviewRing-AI.git /content/ReviewRing-AI
%cd /content/ReviewRing-AI
!bash colab/colab_run.sh
```

To use your Kaggle token, upload `kaggle.json` to `/content/` first (Cell 0):

```python
from google.colab import files
files.upload()   # pick kaggle.json
```

`colab_run.sh` installs dependencies, downloads the datasets (Kaggle-first with validated
fallbacks — see `scripts/get_data.py`), trains both tracks, and prints the results table.

## Option 2 — step by step (paste into Colab cells)

```python
# 1. clone + install
!git clone https://github.com/<you>/ReviewRing-AI.git /content/ReviewRing-AI
%cd /content/ReviewRing-AI
!python -m pip install -q -e .
!python -m pip install -q -r requirements.txt
```

```python
# 2. data
!python scripts/get_data.py
```

```python
# 3. Track A (static benchmark, 3 seeds)
!python -m reviewring.cli --config configs/yelp_static.yaml prepare
!python -m reviewring.cli --config configs/yelp_static.yaml split
!python -m reviewring.cli --config configs/yelp_static.yaml train --model graph --seed 42
!python -m reviewring.cli --config configs/yelp_static.yaml report --run $(ls -t artifacts | grep yelp-graph | head -1)
```

```python
# 4. Track B (raw-review replay: simulate, features, graph, train, evaluate)
!python -m reviewring.cli --config configs/amazon_replay.yaml prepare
!python -m reviewring.cli --config configs/amazon_replay.yaml split
!python -m reviewring.cli --config configs/amazon_replay.yaml simulate
!python -m reviewring.cli --config configs/amazon_replay.yaml features
!python -m reviewring.cli --config configs/amazon_replay.yaml graph
!python -m reviewring.cli --config configs/amazon_replay.yaml train --model adaptive_fusion --seed 42
```

```python
# 5. replay, rings, explanations, report
import subprocess, glob
run = sorted(glob.glob('artifacts/amazon-adaptive_fusion-s42-*'))[-1].split('/')[-1]
for cmd in (["replay", "--run", run], ["rings", "--run", run],
            ["explain", "--run", run], ["report", "--run", run]):
    subprocess.run(["python", "-m", "reviewring.cli", "--config",
                    "configs/amazon_replay.yaml"] + cmd, check=True)
```

```python
# 6. results table + tests
!python scripts/collect_results.py
!python -m pytest tests/ -q
```

```python
# 7. (optional) zip artifacts for download
!cd /content && zip -q -r reviewring_results.zip ReviewRing-AI -x '*/data/raw/*' '*/.git/*'
from google.colab import files
files.download('/content/reviewring_results.zip')
```

## Optional: the investigation UI (runs in Colab too)

```python
!python -m pip install -q streamlit >/dev/null
!streamlit run app/streamlit_app.py --server.port 8711 & npx -y localtunnel --port 8711
```

## Notes

- The code is CPU-only by design (vectorised sampler, no PyG compiled extensions); GPU is unused by the trainers.
  Track A uses a vectorised relational neighbour sampler.
- Colab runtimes are ephemeral — keep the final `reviewring_results.zip` or push artifacts
  back to your repo.
- `colab_run.sh` is idempotent-safe: rerunning re-trains into NEW run directories
  (the CLI refuses to overwrite existing runs).

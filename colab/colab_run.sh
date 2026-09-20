#!/usr/bin/env bash
# ============================================================================
# ReviewRing AI - one-shot Google Colab setup + train + evaluate (no notebook)
# Run the WHOLE block in a single Colab cell, from the repository root:
#     !bash colab/colab_run.sh
# Assumes the repo was cloned into /content/ReviewRing-AI (see README_COLAB.md).
# The script resolves its own location, so it works from any working directory.
# ============================================================================
set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "== [1/6] Runtime check (CPU only by design: vectorised sampler, no PyG extensions) =="
(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no GPU detected - CPU mode is the supported path")

echo "== [2/6] Installing dependencies =="
python -m pip install -q --upgrade pip >/dev/null
# Torch ships preinstalled on Colab; install the repo package and the stack.
python -m pip install -q -e .
python -m pip install -q -r requirements.txt
python - <<'PY'
import torch, reviewring
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| reviewring", reviewring.__version__)
PY

echo "== [3/6] Downloading datasets (Kaggle-first with validated fallbacks) =="
# Option A: upload kaggle.json to /content and it will be picked up.
if [ -f /content/kaggle.json ]; then
  mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
  echo "kaggle.json detected"
else
  echo "no /content/kaggle.json - Kaggle step will be skipped and canonical sources used"
fi
python scripts/get_data.py

echo "== [4/6] Track A: YelpChi static benchmark =="
python -m reviewring.cli --config configs/yelp_static.yaml prepare
python -m reviewring.cli --config configs/yelp_static.yaml split
for seed in 17 42 73; do
  python -m reviewring.cli --config configs/yelp_static.yaml train --model graph --seed $seed
done
python -m reviewring.cli --config configs/yelp_static.yaml train --model numeric --seed 42
python -m reviewring.cli --config configs/yelp_static.yaml train --model prior --seed 42
for run in $(ls artifacts | grep yelp-graph || true); do
  # only re-report runs that still have their prediction file (fresh runs do)
  if [ -f "artifacts/$run/test_predictions.parquet" ]; then
    python -m reviewring.cli --config configs/yelp_static.yaml report --run $run
  fi
done

echo "== [5/6] Track B: Amazon replay prototype (experts + fusion + rings) =="
python -m reviewring.cli --config configs/amazon_replay.yaml prepare
python -m reviewring.cli --config configs/amazon_replay.yaml split
python -m reviewring.cli --config configs/amazon_replay.yaml simulate
python -m reviewring.cli --config configs/amazon_replay.yaml features
python -m reviewring.cli --config configs/amazon_replay.yaml graph
for seed in 17 42 73; do
  for model in text time graph; do
    python -m reviewring.cli --config configs/amazon_replay.yaml train --model $model --seed $seed
  done
done
for seed in 17 42 73; do
  python -m reviewring.cli --config configs/amazon_replay.yaml train --model fixed_fusion --seed $seed
  python -m reviewring.cli --config configs/amazon_replay.yaml train --model adaptive_fusion --seed $seed
done
python -m reviewring.cli --config configs/amazon_replay.yaml train --model global_fusion --seed 42
python -m reviewring.cli --config configs/amazon_replay.yaml train --model adaptive_fusion_noctx --seed 42

MAIN_RUN=$(ls -t artifacts | grep amazon-adaptive_fusion-s42 | head -1 || true)
if [ -z "$MAIN_RUN" ]; then echo "ERROR: no adaptive_fusion-s42 run found"; exit 1; fi
python -m reviewring.cli --config configs/amazon_replay.yaml replay  --run $MAIN_RUN
python -m reviewring.cli --config configs/amazon_replay.yaml rings   --run $MAIN_RUN
python -m reviewring.cli --config configs/amazon_replay.yaml explain --run $MAIN_RUN
python -m reviewring.cli --config configs/amazon_replay.yaml report  --run $MAIN_RUN
for run in $(ls artifacts | grep amazon- | grep -v $MAIN_RUN || true); do
  python -m reviewring.cli --config configs/amazon_replay.yaml report --run $run || true
done

echo "== [6/6] Results =="
python scripts/collect_results.py | tail -40
echo ""
echo "Artifacts saved under $REPO_DIR/artifacts/<run_id>/"
echo "Zip everything for download:"
echo "  cd /content && zip -r reviewring_results.zip ReviewRing-AI -x '*/data/raw/*' '*/.git/*'"

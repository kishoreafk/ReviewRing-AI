#!/usr/bin/env bash
# One-shot Colab setup. Options are passed to scripts/run_experiment.py.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
PY=${PYTHON:-python}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "== [1/6] Runtime: CPU trainers; text embedding may use an available GPU =="
(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU; CPU is supported")
echo "== [2/6] Install dependencies (reuse Colab's installed Torch when compatible) =="
"$PY" -m pip install -q -e . -r requirements.txt
"$PY" -c 'import torch, reviewring; print("Torch", torch.__version__, "CUDA available:", torch.cuda.is_available(), "ReviewRing", reviewring.__version__)'
echo "== [3/6] Regression tests =="
"$PY" -m pytest tests -q
echo "== [4/6] Download datasets =="
if [ -f /content/kaggle.json ]; then
  mkdir -p ~/.kaggle
  cp /content/kaggle.json ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi
"$PY" scripts/get_data.py
echo "== [5/6] Train and evaluate a fresh isolated experiment =="
"$PY" scripts/run_experiment.py "$@"
echo "== [6/6] Gate diagnosis (adaptive vs fixed; investigation only) =="
EXP_DIR="$("$PY" -c 'import json; print(json.load(open("latest_experiment.json"))["experiment_dir"])')"
"$PY" scripts/diagnose_gate.py --artifacts-dir "$EXP_DIR/artifacts"
echo "See latest_experiment.json for the output directory. Save that directory before closing Colab."

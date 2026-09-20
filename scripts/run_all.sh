#!/usr/bin/env bash
# ReviewRing AI - full experiment runner (Track A + Track B).
# Usage: bash scripts/run_all.sh [yelp|amazon|all]
# Assumes: python -m reviewring.cli prepare/split(/simulate/features/graph) already run,
# or runs them when missing. Designed to work unchanged in Google Colab.
set -e
cd "$(dirname "$0")/.."

TRACK=${1:-all}
PY=${PYTHON:-python3}

run_track_a() {
  echo "=== Track A: YelpChi static benchmark ==="
  $PY -m reviewring.cli --config configs/yelp_static.yaml prepare
  $PY -m reviewring.cli --config configs/yelp_static.yaml split
  for seed in 17 42 73; do
    $PY -m reviewring.cli --config configs/yelp_static.yaml train --model prior --seed $seed
    $PY -m reviewring.cli --config configs/yelp_static.yaml train --model numeric --seed $seed
    $PY -m reviewring.cli --config configs/yelp_static.yaml train --model graph --seed $seed
  done
  for run in $(ls artifacts | grep yelp-graph || true); do
    $PY -m reviewring.cli --config configs/yelp_static.yaml report --run $run || true
  done
}

run_track_b() {
  echo "=== Track B: Amazon replay prototype ==="
  $PY -m reviewring.cli --config configs/amazon_replay.yaml prepare
  $PY -m reviewring.cli --config configs/amazon_replay.yaml split
  $PY -m reviewring.cli --config configs/amazon_replay.yaml simulate
  $PY -m reviewring.cli --config configs/amazon_replay.yaml features
  $PY -m reviewring.cli --config configs/amazon_replay.yaml graph
  # single experts: 3 seeds
  for seed in 17 42 73; do
    for model in text time graph; do
      $PY -m reviewring.cli --config configs/amazon_replay.yaml train --model $model --seed $seed
    done
  done
  # fusion: 3 seeds for the principal comparison
  for seed in 17 42 73; do
    $PY -m reviewring.cli --config configs/amazon_replay.yaml train --model fixed_fusion --seed $seed
    $PY -m reviewring.cli --config configs/amazon_replay.yaml train --model adaptive_fusion --seed $seed
  done
  # ablations: 1 seed
  $PY -m reviewring.cli --config configs/amazon_replay.yaml train --model global_fusion --seed 42
  $PY -m reviewring.cli --config configs/amazon_replay.yaml train --model adaptive_fusion_noctx --seed 42

  MAIN_RUN=$(ls -t artifacts | grep amazon-adaptive_fusion-s42 | head -1)
  $PY -m reviewring.cli --config configs/amazon_replay.yaml replay --run $MAIN_RUN
  $PY -m reviewring.cli --config configs/amazon_replay.yaml rings --run $MAIN_RUN
  $PY -m reviewring.cli --config configs/amazon_replay.yaml explain --run $MAIN_RUN
  $PY -m reviewring.cli --config configs/amazon_replay.yaml report --run $MAIN_RUN
  for run in $(ls artifacts | grep amazon- | grep -v $MAIN_RUN || true); do
    $PY -m reviewring.cli --config configs/amazon_replay.yaml report --run $run || true
  done
  echo "main run: $MAIN_RUN"
}

case $TRACK in
  yelp) run_track_a ;;
  amazon) run_track_b ;;
  all) run_track_a; run_track_b ;;
  *) echo "unknown track $TRACK"; exit 1 ;;
esac
echo "done."

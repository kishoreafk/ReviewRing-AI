#!/usr/bin/env bash
# Full experiment runner. Usage: bash scripts/run_all.sh [yelp|amazon|all] [runner options]
set -euo pipefail
cd "$(dirname "$0")/.."
TRACK=${1:-all}
if [ "$#" -gt 0 ]; then shift; fi
exec "${PYTHON:-python3}" scripts/run_experiment.py --track "$TRACK" "$@"

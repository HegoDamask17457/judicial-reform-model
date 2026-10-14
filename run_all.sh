#!/usr/bin/env bash
# Reproduce every number and figure in the research note (~15-25 min on 32 cores).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs
regimes=("100 5" "100 8" "150 5" "150 8" "125 8" "200 8")
for i in 0 2 4; do
  for r in "${regimes[$i]}" "${regimes[$((i+1))]}"; do
    python3 code/run_regime.py $r > "results/logs/regime_${r// /_}.log" 2>&1 &
  done
  wait
done
python3 code/reform_experiment.py > results/logs/reform_experiment.log 2>&1
python3 code/make_figures.py
echo "done"

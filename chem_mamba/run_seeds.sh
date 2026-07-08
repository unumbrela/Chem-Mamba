#!/bin/bash
# Post-AuMgO queue: backfill NaCl seed-0 baselines only (analysis code landed
# mid-run). Multi-seed error bars dropped per user decision (2026-07-06);
# revisit only if reviewers ask.
set -u
cd "$(dirname "$0")/.."

# backfill seed-0 NaCl baselines with nacl_analysis + .pt predictions
for m in local-Q local+Q; do
  echo "=== backfill NaCl $m s0 ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset NaCl --model "$m" --steps 4000 --seed 0
done

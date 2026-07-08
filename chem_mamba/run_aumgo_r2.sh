#!/bin/bash
# AuMgO round 2: 4x longer training, z-ordering (PBC-safe depth profile),
# count-balanced charge loss, float64 composition baseline.
set -u
cd "$(dirname "$0")/.."
for m in ssm ssm-iso local+Q local-Q; do
  echo "=== AuMgO-r2 $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset AuMgO --model "$m" \
    --steps 12000 --batch 32 --order z --qw element --seed 0
done

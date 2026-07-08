#!/bin/bash
# AuMgO round 3: same protocol as r2, only variable changed = 30000 steps
# (r2 val-q was still falling steeply at 12000; single-variable extension).
set -u
cd "$(dirname "$0")/.."
mkdir -p chem_mamba/results/r2_aumgo
mv chem_mamba/results/AuMgO_*.json chem_mamba/results/AuMgO_*.pt \
   chem_mamba/results/r2_aumgo/ 2>/dev/null
for m in ssm ssm-iso local+Q local-Q; do
  echo "=== AuMgO-r3 $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset AuMgO --model "$m" \
    --steps 30000 --batch 32 --order z --qw element --seed 0
done

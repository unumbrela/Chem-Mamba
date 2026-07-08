#!/bin/bash
# Paper queue 3 = M2 decisive matrix (2026-07-07): AuMgO with the CORRECT
# slab-safe Ewald tail (validated: Madelung 2.6e-6, forces vs FD 3e-11).
# The question: with correct physics, does ssm get BOTH good E/F AND full
# routing (contrast -132)?  Order: the decisive pair first (ssm vs iso),
# then routing-bias, the L0 zero anchor, and the O(N^2) oracle.
set -u
cd "$(dirname "$0")/.."
for m in ssm ssm-iso ssm2 local-Q attn; do
  echo "=== AuMgO-ewald $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset AuMgO --model "$m" \
    --steps 12000 --batch 32 --order z --qw element --tail ewald --tag ewald --seed 0
done
echo "PAPER QUEUE 3 DONE"

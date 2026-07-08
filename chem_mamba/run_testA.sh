#!/bin/bash
# Test A: is the (min-image erf/r) physics tail fighting the charge labels on
# the periodic AuMgO slab?  Train with charges as a pure auxiliary head
# (--tail none).  If the Au2 separation unlocks -> tail misspecification is
# the bottleneck (motivates the M2 Ewald upgrade).  If it stays ~-36 me on
# train -> SSM routing/state capacity is the bottleneck (try d_state 32).
set -u
cd "$(dirname "$0")/.."
for m in ssm ssm-iso; do
  echo "=== AuMgO-testA $m (tail=none) ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset AuMgO --model "$m" \
    --steps 12000 --batch 32 --order z --qw element --tail none --tag notail --seed 0
done

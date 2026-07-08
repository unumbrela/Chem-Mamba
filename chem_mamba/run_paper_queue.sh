#!/bin/bash
# Paper-1 top-conf queue (2026-07-07): runs after the r3+seeds chain exits.
#   1. Test A (tail=none diagnostic) -- it was NOT actually chained after
#      r3+seeds (the earlier watcher never existed); fixed here.
#   2. New-variant matrix:
#        qeq  = differentiable charge equilibration (the 4G-HDNNP physics
#               reference inside our exact framework: same backbone/splits/
#               tail; structurally CAN concentrate charge -> complements
#               Test A in localizing the AuMgO routing bottleneck)
#        attn = distance-biased full attention, O(N^2) nonlocal oracle
#        ssm2 = summary-token two-phase scan (QEq-shaped routing bias)
set -u
cd "$(dirname "$0")/.."

bash chem_mamba/run_testA.sh >> chem_mamba/testA.log 2>&1

# AuMgO at the r2/testA budget (12000 steps) for direct comparison
for m in qeq attn ssm2; do
  echo "=== AuMgO-12k $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset AuMgO --model "$m" \
    --steps 12000 --batch 32 --order z --qw element --tag 12k --seed 0
done

# small systems at the original matrix protocols
for m in qeq attn ssm2; do
  echo "=== NaCl $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset NaCl --model "$m" \
    --steps 4000 --seed 0
done
for m in qeq attn ssm2; do
  echo "=== Carbon_chain $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset Carbon_chain --model "$m" \
    --steps 4000 --seed 0
done
for m in qeq attn; do
  echo "=== Ag_cluster $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset Ag_cluster --model "$m" \
    --steps 3000 --seed 0
done
echo "PAPER QUEUE DONE"

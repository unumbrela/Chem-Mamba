#!/bin/bash
# Paper queue 2 (2026-07-07 morning), after the decisive Test A result:
# ssm with --tail none fully unlocks AuMgO routing (contrast -132/-132).
#   1. Ag3 cutoff diagnosis: attn cracked the 63/84 meV plateau that all
#      cutoff-3.5 models share -> hypothesis: geometry blindness beyond
#      3.5 A, not backbone capacity.  cutoff 6 should close the gap.
#   2. NaCl longer runs for attn/qeq: is attn's 2x overshoot (+88 vs +40)
#      and qeq's wrong sign (-90) an underfitting artifact?
#   3. Complete the (architecture x tail) 2x2 on AuMgO: ssm2/attn + tail=none.
set -u
cd "$(dirname "$0")/.."

for m in ssm local+Q; do
  echo "=== Ag_cluster-c6 $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset Ag_cluster --model "$m" \
    --steps 3000 --cutoff 6.0 --tag c6 --seed 0
done

for m in attn qeq; do
  echo "=== NaCl-s8k $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset NaCl --model "$m" \
    --steps 8000 --tag s8k --seed 0
done

for m in ssm2 attn; do
  echo "=== AuMgO-notail $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset AuMgO --model "$m" \
    --steps 12000 --batch 32 --order z --qw element --tail none --tag notail --seed 0
done
echo "PAPER QUEUE 2 DONE"

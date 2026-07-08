#!/bin/bash
# Tier-1 training-recipe queue (2026-07-08).  Goal: close part of the E/F gap
# to the strong-backbone literature (EFA/SpookyNet) with recipe-only changes:
#   1. accuracy protocol = cutoff 6.0 on the three non-periodic systems
#      (AuMgO stays 3.5: min-image bound L/2 = 4.52 A for the 9.05 A cell;
#      NOTE the c6 runs are the ACCURACY protocol -- discriminative results
#      keep the reach-controlled cutoff-3.5 protocol, see PAPER1_DESIGN.md)
#   2. doubled steps + EMA weight averaging (--ema 0.999)
#   3. force-loss weight sweep wf in {1,3,10} on the two cheap systems
#   4. AuMgO warm-start extended 12k -> 30k (q was still falling at 12k)
# Phase B (controls at the winning wf + reach-scaling test on local models)
# is launched after this queue is judged.
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.

for wf in 1 3 10; do
  echo "=== [A] Carbon_chain ssm c6/8k/ema wf=$wf ==="
  python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm \
    --steps 8000 --cutoff 6.0 --ema 0.999 --wf "$wf" --tag "t1_wf$wf" --seed 0
done

for wf in 1 3 10; do
  echo "=== [A] NaCl ssm c6/8k/ema wf=$wf ==="
  python chem_mamba/train_4g.py --dataset NaCl --model ssm \
    --steps 8000 --cutoff 6.0 --ema 0.999 --wf "$wf" --tag "t1_wf$wf" --seed 0
done

echo "=== [B] Ag_cluster ssm c6/6k/ema (double of the 3k c6 run) ==="
python chem_mamba/train_4g.py --dataset Ag_cluster --model ssm \
  --steps 6000 --cutoff 6.0 --ema 0.999 --wf 1 --tag t1_wf1 --seed 0

echo "=== [C] AuMgO ssm warm-start 30k + ema (queue5 flags, wf=1 for comparability) ==="
python chem_mamba/train_4g.py --dataset AuMgO --model ssm \
  --steps 30000 --batch 32 --order z --qw element --tail ewald --lr 3e-4 \
  --ema 0.999 --init-from chem_mamba/results/AuMgO_ssm_s0_notail.pt \
  --tag ewald_warm30k --seed 0

echo "TIER1 QUEUE DONE"

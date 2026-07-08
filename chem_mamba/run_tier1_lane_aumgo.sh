#!/bin/bash
# Tier-1 lane 1 (critical path): AuMgO warm-start 30k + EMA.
# Split out of run_tier1.sh for parallel execution (GPU was underused: one
# python-scan run = ~20% util / 1.6 GiB of 16 GiB).
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.

python chem_mamba/train_4g.py --dataset AuMgO --model ssm \
  --steps 30000 --batch 32 --order z --qw element --tail ewald --lr 3e-4 \
  --ema 0.999 --init-from chem_mamba/results/AuMgO_ssm_s0_notail.pt \
  --tag ewald_warm30k --seed 0
echo "TIER1 LANE AUMGO DONE"

#!/bin/bash
# Tier-1 lane 3: NaCl wf sweep.
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.

for wf in 1 3 10; do
  echo "=== NaCl ssm c6/8k/ema wf=$wf ==="
  python chem_mamba/train_4g.py --dataset NaCl --model ssm \
    --steps 8000 --cutoff 6.0 --ema 0.999 --wf "$wf" --tag "t1_wf$wf" --seed 0
done
echo "TIER1 LANE NACL DONE"

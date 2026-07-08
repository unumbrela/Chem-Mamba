#!/bin/bash
# Tier-1 lane 2: remaining Carbon_chain wf sweep (wf1 already running from the
# original queue) + Ag3 doubled-budget rerun.
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.

for wf in 3 10; do
  echo "=== Carbon_chain ssm c6/8k/ema wf=$wf ==="
  python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm \
    --steps 8000 --cutoff 6.0 --ema 0.999 --wf "$wf" --tag "t1_wf$wf" --seed 0
done

echo "=== Ag_cluster ssm c6/6k/ema ==="
python chem_mamba/train_4g.py --dataset Ag_cluster --model ssm \
  --steps 6000 --cutoff 6.0 --ema 0.999 --wf 1 --tag t1_wf1 --seed 0
echo "TIER1 LANE CARBON DONE"

#!/bin/bash
# Tier-1 lane 1 RELAUNCH after the 01:20 device-level crash (WSL2 "CUDA error:
# unknown error" killed all 4 processes when total VRAM hit 15.9/16.3 GiB).
# Hardening: --eval-batch 64 caps the eval VRAM spike (the 110-atom Ewald val
# pass at bs=256 was the main reserved-memory hog); expandable_segments cuts
# allocator overshoot; train_4g now rolling-saves the best checkpoint so a
# crash no longer loses the whole 3 h run.
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python chem_mamba/train_4g.py --dataset AuMgO --model ssm \
  --steps 30000 --batch 32 --order z --qw element --tail ewald --lr 3e-4 \
  --ema 0.999 --eval-batch 64 \
  --init-from chem_mamba/results/AuMgO_ssm_s0_notail.pt \
  --tag ewald_warm30k --seed 0
echo "TIER1 LANE AUMGO2 DONE"

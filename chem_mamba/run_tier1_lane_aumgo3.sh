#!/bin/bash
# Tier-1 lane 1, third launch: NO EMA this time.
# The NaCl factorization (07-08, tier1_nacl_*.log) convicted EMA of destroying
# learned charge structure (q 19->332 me with EMA on an otherwise identical
# run); the first 9k steps of the EMA'd AuMgO warm run corroborated it
# (q stalled at ~215 vs queue5's ~100 at the same point; contrast survived).
# EMA stays in the recipe only for E/F-dominated covalent systems (carbon).
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python chem_mamba/train_4g.py --dataset AuMgO --model ssm \
  --steps 30000 --batch 32 --order z --qw element --tail ewald --lr 3e-4 \
  --eval-batch 64 \
  --init-from chem_mamba/results/AuMgO_ssm_s0_notail.pt \
  --tag ewald_warm30k --seed 0
echo "TIER1 LANE AUMGO3 DONE"

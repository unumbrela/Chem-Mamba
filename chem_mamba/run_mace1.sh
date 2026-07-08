#!/bin/bash
# MACE queue 1 (2026-07-08): first full pass of the MACE backbone swap.
# Protocols mirror the finalized per-system recipes (tier1) exactly, so every
# number is directly comparable to its schnet-backbone counterpart:
#   Ag      c6/6k/EMA/wf1  -> does the equivariant backbone break the
#                             9.3-9.8 meV/atom floor? (schnet ssm: 9.43)
#   Carbon  c6/8k/EMA/wf3  -> covalent recipe (schnet: E 1.36/F 88/q 2.67)
#   NaCl    c3.5/4k conservative baseline (schnet: E 0.889/F 40/q 19.4,
#                             dq0 +42.6 vs true +40)
#   AuMgO   Test-A notail 12k, then warm-start ewald 30k curriculum
#                             (schnet: contrast -132 notail, -131 warm30k)
# ssm-iso capacity-matched controls for Carbon/NaCl; AuMgO iso in queue 2.
# Speed check: AuMgO ewald bs32 = 1.42 steps/s, 7.5 GiB peak -> sequential
# only (WSL2 lesson: keep 3-4 GiB VRAM headroom, no parallel AuMgO).
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.
export PYTHONWARNINGS=ignore
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/8] Ag ssm mace c6/6k/ema ==="
python chem_mamba/train_4g.py --dataset Ag_cluster --model ssm --backbone mace \
  --steps 6000 --cutoff 6.0 --ema 0.999 --wf 1 --tag t1_wf1 --seed 0

echo "=== [2/8] Ag local+Q mace c6/6k/ema (backbone-floor control) ==="
python chem_mamba/train_4g.py --dataset Ag_cluster --model local+Q --backbone mace \
  --steps 6000 --cutoff 6.0 --ema 0.999 --wf 1 --tag t1_wf1 --seed 0

echo "=== [3/8] Carbon ssm mace c6/8k/ema/wf3 ==="
python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm --backbone mace \
  --steps 8000 --cutoff 6.0 --ema 0.999 --wf 3 --tag t1_wf3 --seed 0

echo "=== [4/8] Carbon ssm-iso mace (capacity control) ==="
python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm-iso --backbone mace \
  --steps 8000 --cutoff 6.0 --ema 0.999 --wf 3 --tag t1_wf3 --seed 0

echo "=== [5/8] NaCl ssm mace c3.5/4k (conservative baseline) ==="
python chem_mamba/train_4g.py --dataset NaCl --model ssm --backbone mace --seed 0

echo "=== [6/8] NaCl ssm-iso mace (capacity control) ==="
python chem_mamba/train_4g.py --dataset NaCl --model ssm-iso --backbone mace --seed 0

echo "=== [7/8] AuMgO ssm mace Test-A (tail=none, routing capability) ==="
python chem_mamba/train_4g.py --dataset AuMgO --model ssm --backbone mace \
  --steps 12000 --batch 32 --order z --qw element --tail none \
  --eval-batch 64 --tag notail --seed 0

echo "=== [8/8] AuMgO ssm mace warm-start ewald 30k (curriculum) ==="
python chem_mamba/train_4g.py --dataset AuMgO --model ssm --backbone mace \
  --steps 30000 --batch 32 --order z --qw element --tail ewald --lr 3e-4 \
  --eval-batch 64 \
  --init-from chem_mamba/results/AuMgO_ssm_s0_mace_notail.pt \
  --tag ewald_warm30k --seed 0

echo "MACE QUEUE 1 DONE"

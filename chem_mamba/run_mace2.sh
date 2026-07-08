#!/bin/bash
# MACE queue 2 (2026-07-08): feature-scale fix (LayerNorm at the backbone
# boundary) after queue 1 diagnosis:
#   - raw MACE scalars are ~8x smaller (std 0.11 vs 0.84) with ~20x less
#     per-atom spread (0.02 vs 0.38) than schnet features
#   - AuMgO notail routing CRAWLED: contrast +10 -> -4 over 12k steps,
#     never unlocked (schnet: -132 = truth at same budget)
#   - NaCl absolute charges offset ~-120 me (undertrained symptom)
# Single changed variable (LN); lr/steps/recipes stay identical to queue 1
# so the fix is cleanly attributable.  _v2 tags mark the LN protocol.
# Queue 1 runs 8 (warm-start from dead-routing ckpt) was killed as moot.
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.
export PYTHONWARNINGS=ignore
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/6] AuMgO ssm mace notail 12k v2 (DECISIVE: does LN unlock routing?) ==="
python chem_mamba/train_4g.py --dataset AuMgO --model ssm --backbone mace \
  --steps 12000 --batch 32 --order z --qw element --tail none \
  --eval-batch 64 --tag notail_v2 --seed 0

echo "=== [2/6] AuMgO ssm-iso mace notail 12k v2 (capacity control) ==="
python chem_mamba/train_4g.py --dataset AuMgO --model ssm-iso --backbone mace \
  --steps 12000 --batch 32 --order z --qw element --tail none \
  --eval-batch 64 --tag notail_v2 --seed 0

echo "=== [3/6] NaCl ssm mace 4k v2 (LN effect at queue-1 budget) ==="
python chem_mamba/train_4g.py --dataset NaCl --model ssm --backbone mace \
  --tag v2 --seed 0

echo "=== [4/6] NaCl ssm mace 8k v2 (budget knob on top of LN) ==="
python chem_mamba/train_4g.py --dataset NaCl --model ssm --backbone mace \
  --steps 8000 --tag v2_8k --seed 0

echo "=== [5/6] Carbon ssm mace v2 (does the good result hold under LN?) ==="
python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm --backbone mace \
  --steps 8000 --cutoff 6.0 --ema 0.999 --wf 3 --tag t1_wf3_v2 --seed 0

echo "=== [6/6] Ag ssm mace v2 ==="
python chem_mamba/train_4g.py --dataset Ag_cluster --model ssm --backbone mace \
  --steps 6000 --cutoff 6.0 --ema 0.999 --wf 1 --tag t1_wf1_v2 --seed 0

echo "MACE QUEUE 2 DONE"

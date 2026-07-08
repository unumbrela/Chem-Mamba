#!/bin/bash
# Backfill the two small v2 runs that were killed to protect VRAM headroom
# while the decisive AuMgO notail continue-18k was running.  Waits for the
# 18k run to write its result JSON, then runs sequentially (no AuMgO overlap).
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.
export PYTHONWARNINGS=ignore
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[backfill] waiting for AuMgO notail_v2c18k to finish..."
until [ -f chem_mamba/results/AuMgO_ssm_s0_mace_notail_v2c18k.json ]; do sleep 60; done
sleep 30   # let the process release GPU memory

echo "=== [1/2] NaCl ssm mace 8k v2 (budget knob, rerun) ==="
python chem_mamba/train_4g.py --dataset NaCl --model ssm --backbone mace \
  --steps 8000 --tag v2_8k --seed 0

echo "=== [2/2] Carbon ssm mace v2 (LN check, rerun) ==="
python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm --backbone mace \
  --steps 8000 --cutoff 6.0 --ema 0.999 --wf 3 --tag t1_wf3_v2 --seed 0

echo "MACE BACKFILL DONE"

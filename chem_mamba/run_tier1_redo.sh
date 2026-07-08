#!/bin/bash
# Tier-1 redo lane: re-run the three victims of the 01:20 device crash
# (carbon wf1 was at step ~6000, carbon wf3 ~2000, NaCl wf1 ~2000; results
# are only written at the end, so all three were lost).  Waits for the
# currently running carbon wf10 to finish so total concurrency stays at
# AuMgO + 2 small runs (~8 GiB, >= 8 GiB headroom).
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[redo] waiting for carbon wf10 to finish..."
until [ -f chem_mamba/results/Carbon_chain_ssm_s0_t1_wf10.json ]; do sleep 60; done

for spec in "Carbon_chain 1" "Carbon_chain 3" "NaCl 1"; do
  set -- $spec
  echo "=== [redo] $1 ssm c6/8k/ema wf=$2 ==="
  python chem_mamba/train_4g.py --dataset "$1" --model ssm \
    --steps 8000 --cutoff 6.0 --ema 0.999 --wf "$2" --eval-batch 128 \
    --tag "t1_wf$2" --seed 0
done
echo "TIER1 REDO LANE DONE"

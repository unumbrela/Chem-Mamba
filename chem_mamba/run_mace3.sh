#!/bin/bash
# MACE queue 3: curriculum stage 2 on the MACE backbone + small backfills.
# The notail continue-18k run phase-transitioned at step ~5.5k (contrast
# -9 -> -141, true -132), confirming the budget diagnosis.  As soon as it
# finishes, warm-start the Ewald tail from its checkpoint (exact tier1
# protocol: lr 3e-4, 30k) -- the MACE analog of the schnet full-victory run
# (schnet: contrast -131, E 0.388).  Small reruns follow sequentially;
# no AuMgO-class run ever overlaps another job (VRAM discipline).
set -u
cd /home/zihao/code/Chem-Mamba
export PYTHONPATH=.
export PYTHONWARNINGS=ignore
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[mace3] waiting for AuMgO notail_v2c18k to finish..."
until [ -f chem_mamba/results/AuMgO_ssm_s0_mace_notail_v2c18k.json ]; do sleep 60; done
sleep 30

echo "=== [1/3] AuMgO ssm mace warm-start ewald 30k (curriculum stage 2) ==="
python chem_mamba/train_4g.py --dataset AuMgO --model ssm --backbone mace \
  --steps 30000 --batch 32 --order z --qw element --tail ewald --lr 3e-4 \
  --eval-batch 64 \
  --init-from chem_mamba/results/AuMgO_ssm_s0_mace_notail_v2c18k.pt \
  --tag ewald_warm30k --seed 0

echo "=== [2/3] NaCl ssm mace 8k v2 (budget knob, rerun) ==="
python chem_mamba/train_4g.py --dataset NaCl --model ssm --backbone mace \
  --steps 8000 --tag v2_8k --seed 0

echo "=== [3/3] Carbon ssm mace v2 (LN check, rerun) ==="
python chem_mamba/train_4g.py --dataset Carbon_chain --model ssm --backbone mace \
  --steps 8000 --cutoff 6.0 --ema 0.999 --wf 3 --tag t1_wf3_v2 --seed 0

echo "MACE QUEUE 3 DONE"

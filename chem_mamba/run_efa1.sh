#!/bin/bash
# EFA queue 1: launch the official-config EFA AuMgO training as soon as the
# mace3 queue (PID passed as $1) releases the GPU.
#
# Budget note: official config = 1M steps, checkpoint every 5000 steps
# (orbax).  Early stopping at any checkpoint IS the reduced-budget tier, so
# we launch the full config and decide the stop point from measured steps/s.
# Smoke at MEM_FRACTION=.20 OOM'd at model init (needs >3.3 GiB); with the
# GPU free this config is expected to fit.  PREALLOCATE=false so any small
# co-running eval cannot be starved.
set -u
QUEUE_PID=${1:?usage: run_efa1.sh <pid-to-wait-for>}
EFA=/home/zihao/code/Chem-Mamba/external/euclidean_fast_attention
LOG=/home/zihao/code/Chem-Mamba/efa_aumgo_train.log

echo "[efa1] waiting for PID $QUEUE_PID (mace3 queue) to exit..."
while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 300; done
sleep 120   # let the last job's teardown finish

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "[efa1] GPU free: ${free_mib} MiB; launching official EFA AuMgO training"

rm -rf "$EFA/runs/4ghdnnp/aumgo/efa_official"
cd "$EFA"
nohup env WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false \
  conda run --no-capture-output -n llm python \
  "$EFA/euclidean_fast_attention/main.py" \
  --config "$EFA/euclidean_fast_attention/configs/config.py" \
  --config.wandb.group 4ghdnnp \
  --config.wandb.name efa_aumgo_official \
  --optimizer_config "$EFA/euclidean_fast_attention/configs/optimizer/default.py" \
  --model_config "$EFA/euclidean_fast_attention/configs/model/4ghdnnp_base_model_pbc.py" \
  --trainer_config "$EFA/euclidean_fast_attention/configs/trainer/4ghdnnp_AuMgO.py" \
  --trainer_config.datafile /home/zihao/code/Chem-Mamba/data/efa_datasets/AuMgO_preprocessed.npz \
  --workdir "$EFA/runs/4ghdnnp/aumgo/efa_official" \
  > "$LOG" 2>&1 &
echo "[efa1] launched, pid $!, log $LOG"

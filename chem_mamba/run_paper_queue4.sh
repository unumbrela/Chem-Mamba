#!/bin/bash
# Paper queue 4 (2026-07-07): resolve the queue3 cliffhanger --- WHY does the
# SSM's spatial routing collapse the moment charge is coupled into Coulomb
# energy (Test A: contrast -132; erf/r: -15; Ewald: +2), while attention is
# unaffected?  Two hypotheses, two knobs:
#
#   H1 "hard gradient competition": energy loss backprops into the charge head
#      and overwrites the (fragile, sequential) routing solution that SGD found.
#      Knob = --detach-elec: charges still feed E_elec in the forward pass, but
#      energy/force grads cannot reach the charge head (Test A's clean gradient).
#      Prediction if H1: ssm-detach routing snaps back to ~-132; iso stays ~0.
#
#   H2 "just undertraining": ssm-ewald val q was still dropping at 12k (no
#      plateau).  Knob = 30000 steps (aligns with the erf/r r3 budget).
#      Prediction if H2: ssm-ewald-30k routing recovers on its own.
#
# The four runs cross both knobs.  Ordered fastest-decisive first so the novel
# detach result lands in ~1.5h; the expensive 30k control runs last.
# True values: sep -282 me, contrast -132 me.  Clean judge = CONTRAST.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
COMMON="--dataset AuMgO --batch 32 --order z --qw element --tail ewald --seed 0"

echo "=== [1/4] ssm-iso  detach-elec 12k  (control for the novel run) ==="
python chem_mamba/train_4g.py --model ssm-iso $COMMON --steps 12000 \
  --detach-elec --tag ewald_detach

echo "=== [2/4] ssm      detach-elec 12k  (H1 test: does routing snap back?) ==="
python chem_mamba/train_4g.py --model ssm     $COMMON --steps 12000 \
  --detach-elec --tag ewald_detach

echo "=== [3/4] ssm-iso  plain ewald 30k  (undertraining control, cheap) ==="
python chem_mamba/train_4g.py --model ssm-iso $COMMON --steps 30000 \
  --tag ewald30k

echo "=== [4/4] ssm      plain ewald 30k  (H2 test: does patience alone fix it?) ==="
python chem_mamba/train_4g.py --model ssm     $COMMON --steps 30000 \
  --tag ewald30k

echo "PAPER QUEUE 4 DONE"

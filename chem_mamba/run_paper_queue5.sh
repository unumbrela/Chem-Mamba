#!/bin/bash
# Paper queue 5 (2026-07-07): the CORRECTED coupling-suppression probe.
# queue4's --detach-elec DIVERGED (full stop-grad makes E_elec a param-gradient
# dead-end: q detached, pos is input, sigma constant -> zero grad -> the Ewald
# term is an uncontrolled additive nuisance -> blow-up).  Cleaner two-stage
# design instead: WARM-START from the tail=none Test A checkpoint (routing
# already = -132 = truth), then turn the physical Ewald tail ON and keep
# training with a gentle LR.  Live per-500-step `contrast` trace shows whether
# routing HOLDS or DECAYS toward 0 under energy coupling.
#
#   routing holds (~-132)  => coupling is compatible; warm-start is the recipe
#                             => "full victory (with the right training curriculum)"
#   routing decays to ~0   => coupling actively destroys the SSM's routing
#                             solution even when handed it for free => the
#                             SSM-vs-attention inductive-bias story is the spine
#
# True values: sep -282 me, contrast -132 me.  Clean judge = CONTRAST.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# hold until the queue4 ssm-30k run writes its result (frees the ~16 GB GPU)
echo "[queue5] waiting for queue4 ssm-ewald-30k to finish (GPU is near-full)..."
until [ -f chem_mamba/results/AuMgO_ssm_s0_ewald30k.json ]; do sleep 60; done
sleep 45   # let the process fully release GPU memory

COMMON="--dataset AuMgO --batch 32 --order z --qw element --tail ewald --seed 0 --lr 3e-4 --steps 12000"

echo "=== [1/2] ssm  warm-start from Test A routing (-132), Ewald ON ==="
python chem_mamba/train_4g.py --model ssm $COMMON \
  --init-from chem_mamba/results/AuMgO_ssm_s0_notail.pt --tag ewald_warm

echo "=== [2/2] ssm-iso  warm-start (capacity-matched control) ==="
python chem_mamba/train_4g.py --model ssm-iso $COMMON \
  --init-from chem_mamba/results/AuMgO_ssm-iso_s0_notail.pt --tag ewald_warm

echo "PAPER QUEUE 5 DONE"

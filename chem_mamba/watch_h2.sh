#!/bin/bash
# Pure watcher: waits for the #4 ssm-ewald-30k run to write its result, then
# exits so the harness notifies. Launches NOTHING (user asked to stop after #4).
cd "$(dirname "$0")/.."
until [ -f chem_mamba/results/AuMgO_ssm_s0_ewald30k.json ]; do sleep 20; done
sleep 25   # allow .pt save + process exit
echo "H2_CONTROL_DONE"

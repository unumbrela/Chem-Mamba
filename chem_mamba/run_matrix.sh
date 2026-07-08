#!/bin/bash
# Run the 4-model matrix on the remaining 4G-HDNNP benchmarks.
# NaCl / Ag_cluster: small, fast.  AuMgO: 110 atoms, periodic, smaller batch.
set -u
cd "$(dirname "$0")/.."

for m in local-Q local+Q ssm ssm-iso; do
  echo "=== NaCl $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset NaCl --model "$m" --steps 4000
done

for m in local-Q local+Q ssm ssm-iso; do
  echo "=== Ag_cluster $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset Ag_cluster --model "$m" --steps 3000
done

for m in local-Q local+Q ssm ssm-iso; do
  echo "=== AuMgO $m ==="
  PYTHONPATH=. python chem_mamba/train_4g.py --dataset AuMgO --model "$m" \
    --steps 3000 --batch 32
done

#!/usr/bin/env bash
# Train every network used in the write-up (several hours on a 4-core laptop).
# Run from the repository root:  bash scripts/train_all.sh
set -e
python -m brain_sandbox train --config configs/main_modular.yaml     --parallel 4   # 5 seeds, the main model
python -m brain_sandbox train --config configs/control_nopitch.yaml  --parallel 3   # never practises the pitch task
python -m brain_sandbox train --config configs/pressure_modular.yaml --parallel 3   # more noise + stronger penalties
python -m brain_sandbox train --config configs/flat.yaml             --parallel 3   # one unstructured block

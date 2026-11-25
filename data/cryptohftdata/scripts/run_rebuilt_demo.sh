#!/usr/bin/env bash
# Optional rebuilt mode on two subsets of the months without anchors. Writes only under dataset names rebuilt_demo*.
set -euo pipefail
cd "$(dirname "$0")/.."
C=configs/rebuilt_demo.yaml
for args in "" "--set dataset.name=rebuilt_demo_crash --set dataset.start=2025-10-09 --set dataset.end=2025-10-11"; do
  .venv/bin/chdds features --config $C $args --workers 3
  .venv/bin/chdds book --config $C $args --workers 2
  .venv/bin/chdds assemble --config $C $args
done
.venv/bin/python scripts/verify_dataset.py rebuilt_demo --days 2025-12-15
.venv/bin/python scripts/verify_dataset.py rebuilt_demo_crash --days 2025-10-10

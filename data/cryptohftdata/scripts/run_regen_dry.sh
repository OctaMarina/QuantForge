#!/usr/bin/env bash
# End-to-end dry run on the windows that exercise each correction. Writes only under dataset names regen_dry_*.
set -euo pipefail
cd "$(dirname "$0")/.."
C=configs/btcusdt_v2.yaml
run() {  # name start end day_to_verify
  A="--set dataset.name=regen_dry_$1 --set dataset.start=$2 --set dataset.end=$3"
  .venv/bin/python -m chdds.cli features --config $C $A --workers 3
  .venv/bin/chdds book --config $C $A --workers 2
  .venv/bin/chdds assemble --config $C $A
  .venv/bin/python scripts/verify_dataset.py regen_dry_$1 --days $4
}
run cross 2025-08-28 2025-08-30 2025-08-29       # crossed stream
run liq 2026-04-23 2026-04-25 2026-04-24         # missing liquidation files, mark/ticker outage
run storm 2026-05-24 2026-05-25 2026-05-25       # reconnect storm
run marks 2026-09-16 2026-09-17 2026-09-17       # hourly checkpoints, mark-only gaps
run oi 2026-10-14 2026-10-17 2026-10-16          # open interest incident

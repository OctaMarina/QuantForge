#!/usr/bin/env bash
# Second part of the full build after the 2026-11-04 kernel bounds fix: clean L2 replay from the history start,
# recompute the adaptive-CUSUM feature parts (engine fix), then assemble. Logs in derived/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
C=configs/btcusdt_initial.yaml
run() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) START $name" | tee -a derived/logs/full_summary.log
  /usr/bin/time -v "$@" > "derived/logs/full_${name}.log" 2>&1
  echo "$(date -u +%FT%TZ) END $name wall=$(grep 'Elapsed (wall' derived/logs/full_${name}.log | awk '{print $8}') max_rss_kb=$(grep 'Maximum resident' derived/logs/full_${name}.log | awk '{print $6}')" | tee -a derived/logs/full_summary.log
}
rm -rf derived/book_union derived/book_grid derived/book_state derived/book_features
run book .venv/bin/python -m chdds.cli book --config "$C" --workers 4
rm -rf derived/features/btcusdt_initial/cusum_adapt_c1-749bed5a6f
run features_cusum_adapt .venv/bin/python -m chdds.cli features --config configs/_recompute_cusum_adapt.yaml --workers 4
run assemble .venv/bin/python -m chdds.cli assemble --config "$C"
du -sh derived/* | tee -a derived/logs/full_summary.log

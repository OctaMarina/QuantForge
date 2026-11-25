#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) START $name" | tee -a derived/logs/full_summary.log
  /usr/bin/time -v "$@" > "derived/logs/full_${name}.log" 2>&1
  echo "$(date -u +%FT%TZ) END $name wall=$(grep 'Elapsed (wall' derived/logs/full_${name}.log | awk '{print $8}') max_rss_kb=$(grep 'Maximum resident' derived/logs/full_${name}.log | awk '{print $6}')" | tee -a derived/logs/full_summary.log
}
run finish_book .venv/bin/python scripts/finish_book.py configs/btcusdt_initial.yaml
rm -rf derived/features/btcusdt_initial/cusum_adapt_c1-749bed5a6f
run features_cusum_adapt .venv/bin/python -m chdds.cli features --config configs/_recompute_cusum_adapt.yaml --workers 4
run assemble .venv/bin/python -m chdds.cli assemble --config configs/btcusdt_initial.yaml
du -sh derived/* | tee -a derived/logs/full_summary.log

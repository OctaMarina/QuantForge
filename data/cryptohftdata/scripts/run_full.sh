#!/usr/bin/env bash
# Full-history build of the initial documented configuration, stage by stage, with time/RSS measurement.
# Resumable: every stage skips/continues completed work. Logs: derived/logs/full_<stage>.log
set -euo pipefail
cd "$(dirname "$0")/.."
C=${1:-configs/btcusdt_initial.yaml}
W_FEAT=${W_FEAT:-4}
W_BOOK=${W_BOOK:-4}
mkdir -p derived/logs
run() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) START $name" | tee -a derived/logs/full_summary.log
  /usr/bin/time -v "$@" > "derived/logs/full_${name}.log" 2>&1
  local rss wall
  rss=$(grep "Maximum resident" "derived/logs/full_${name}.log" | awk '{print $6}')
  wall=$(grep "Elapsed (wall" "derived/logs/full_${name}.log" | awk '{print $8}')
  echo "$(date -u +%FT%TZ) END $name wall=$wall max_rss_kb=$rss" | tee -a derived/logs/full_summary.log
}
run bars     .venv/bin/python -m chdds.cli bars --config "$C" --workers 3
run features .venv/bin/python -m chdds.cli features --config "$C" --workers "$W_FEAT"
run book     .venv/bin/python -m chdds.cli book --config "$C" --workers "$W_BOOK"
run assemble .venv/bin/python -m chdds.cli assemble --config "$C"
du -sh derived/* | tee -a derived/logs/full_summary.log

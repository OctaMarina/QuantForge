#!/usr/bin/env bash
# Regeneration of the dataset under a new name; the previous dataset and the raw data are not touched. Re-running continues after a stop.
# Run under: systemd-run --user --unit=regen --property=MemoryMax=20G --working-directory=$PWD bash scripts/run_regen.sh
# Stops at a checkpoint if free disk falls below CHDDS_MIN_FREE_GB or derived/regen_check/STOP exists (exit 75). It removes only temporary
# files of this run: the book union once split, and per-day book files once their month is written.
set -uo pipefail
cd "$(dirname "$0")/.."
export CHDDS_MIN_FREE_GB=${CHDDS_MIN_FREE_GB:-10}
NAME=${REGEN_NAME:-btcusdt_v2}
C=configs/btcusdt_v2.yaml
PY=.venv/bin/python
SET="--set dataset.name=$NAME ${REGEN_SET:-}"
CHK=derived/regen_check/$NAME
mkdir -p "$CHK"

step() { "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "step failed with code $rc: $*"; exit $rc; fi; }

monitor() {
  local i=0 du=""
  while true; do
    if [ $((i % 10)) -eq 0 ]; then
      du=$(du -sm derived/features/$NAME derived/book_union/$NAME derived/book_features/$NAME derived/book_grid/$NAME derived/book_state/$NAME derived/datasets/$NAME 2>/dev/null | awk '{s+=$1} END {print s+0}')
    fi
    echo "$(date -u +%FT%TZ),$(df -BM --output=avail derived | tail -1 | tr -dc 0-9),$du" >> "$CHK/disk_monitor.csv"
    i=$((i + 1))
    sleep 60
  done
}

if [ -z "${REGEN_SKIP_PREFLIGHT:-}" ]; then step $PY scripts/regen_preflight.py; fi
[ -f "$CHK/disk_monitor.csv" ] || echo "utc,free_mib,run_mib" > "$CHK/disk_monitor.csv"
monitor & MON=$!
trap 'kill $MON 2>/dev/null' EXIT

if [ ! -f "$CHK/oi.done" ]; then
  step $PY -c "from chdds.canonical import slow; slow.build_oi()"
  touch "$CHK/oi.done"
fi
if [ ! -f "$CHK/features.done" ]; then
  step $PY -m chdds.cli features --config $C $SET --workers 8
  touch "$CHK/features.done"
fi
if [ ! -f "$CHK/book.done" ]; then
  END=$($PY -c "import sys; from chdds.config import load_config as L; print(L('$C', dict(x.split('=',1) for x in '$SET'.replace('--set ','').split())).end)")
  LAST=$(ls derived/book_state/$NAME 2>/dev/null | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' | sort | tail -1)
  if [ -z "$LAST" ]; then
    step $PY -m chdds.cli book --config $C $SET --workers 4
  elif [[ "$LAST" < "$END" ]]; then
    step $PY -m chdds.cli book --config $C $SET --workers 4 --resume-from "$LAST" --set features.book_start="$(date -I -d "$LAST + 1 day")"
  fi
  rm -rf "derived/book_union/$NAME"
  touch "$CHK/book.done"
fi
if [ ! -f "$CHK/assemble.done" ]; then
  step $PY -m chdds.cli assemble --config $C $SET --drop-book-intermediates
  touch "$CHK/assemble.done"
fi
if [ -n "${REGEN_VERIFY_DAYS-2025-06-29,2025-08-29,2025-10-10,2025-12-15,2026-04-24,2026-05-25,2026-09-17,2026-10-02,2026-10-16}" ]; then
  step $PY scripts/verify_dataset.py $NAME --days "${REGEN_VERIFY_DAYS-2025-06-29,2025-08-29,2025-10-10,2025-12-15,2026-04-24,2026-05-25,2026-09-17,2026-10-02,2026-10-16}"
fi
echo "regeneration finished: derived/datasets/$NAME"

#!/usr/bin/env bash
# Re-run of the L2 replay after making integration independent of the query set (2026-11-04), then assembly,
# demo rebuild and independent verification.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) START $name" | tee -a derived/logs/full_summary.log
  /usr/bin/time -v "$@" > "derived/logs/full_${name}.log" 2>&1
  echo "$(date -u +%FT%TZ) END $name wall=$(grep 'Elapsed (wall' derived/logs/full_${name}.log | awk '{print $8}') max_rss_kb=$(grep 'Maximum resident' derived/logs/full_${name}.log | awk '{print $6}')" | tee -a derived/logs/full_summary.log
}
rm -rf derived/book_union/btcusdt_initial derived/book_grid/btcusdt_initial derived/book_state/btcusdt_initial derived/book_features/btcusdt_initial
run book2 .venv/bin/python -m chdds.cli book --config configs/btcusdt_initial.yaml --workers 4
run assemble2 .venv/bin/python -m chdds.cli assemble --config configs/btcusdt_initial.yaml
rm -rf derived/features/demo_3days derived/book_union/demo_3days derived/book_grid/demo_3days derived/book_state/demo_3days derived/book_features/demo_3days derived/datasets/demo_3days
run demo .venv/bin/python -m chdds.cli build --config configs/demo_3days.yaml --workers 2
run verify_full .venv/bin/python scripts/verify_dataset.py btcusdt_initial --days 2025-06-29,2025-12-15,2026-04-24,2026-10-02,2026-10-16 --sample 200
run verify_demo .venv/bin/python scripts/verify_dataset.py demo_3days --days 2025-06-29,2025-06-30 --sample 300
run validity .venv/bin/python -c "from chdds.audit.report import write_validity_map; write_validity_map()"
du -sh derived/* | tee -a derived/logs/full_summary.log

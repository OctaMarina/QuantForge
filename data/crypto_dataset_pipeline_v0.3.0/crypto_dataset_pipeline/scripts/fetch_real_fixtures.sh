#!/usr/bin/env bash
# Fetch small REAL market-data samples used by the integration tests and `sample-real`.
# Source: nautilus_trader repository test_data (LGPL-3.0), pinned commit below.  These are
# third-party copies of Binance / Tardis data; they could not be re-verified against the Binance
# CHECKSUM files from this environment (data.binance.vision blocked), see docs/STATUS.md.
set -euo pipefail
COMMIT=16de62c98b59c322129f9a85c98b8c9c6bb52b04
DEST="${1:-$(dirname "$0")/../tests/fixtures/real}"
mkdir -p "$DEST"
BASE="https://raw.githubusercontent.com/nautechsystems/nautilus_trader/${COMMIT}/test_data"
for f in binance/ethusdt-trades.csv binance/btcusdt-trades.parquet binance/btcusdt-quotes.parquet \
         binance/btcusdt-depth-snap.csv binance/btcusdt-depth-update.csv \
         tardis/deribit_incremental_book_L2_BTC-PERPETUAL.csv tardis/binance-futures_book_snapshot_25_BTCUSDT.csv; do
  out="$DEST/$(echo "$f" | tr '/' '_')"
  if [ ! -s "$out" ]; then
    curl -fsSL "$BASE/$f" -o "$out" || { echo "failed: $f"; exit 1; }
  fi
done
sha256sum "$DEST"/* > "$DEST/SHA256SUMS"
echo "fixtures in $DEST"

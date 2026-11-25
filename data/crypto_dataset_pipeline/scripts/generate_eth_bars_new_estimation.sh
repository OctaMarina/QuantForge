#!/usr/bin/env bash
set -euo pipefail

# Directorul proiectului
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=".venv/bin/python"
CONFIG="configs/btc_demo.yaml"

# Perioada comună pentru toate dataseturile
START="2021-09-01T00:00:00Z"
END="2026-09-01T00:00:00Z"

COMMON_ARGS=(
    --config "$CONFIG"
    --set "instrument.base=ETH"
    --set "instrument.quote=USDT"
    --set "instrument.symbol=ETHUSDT"
    --set "instrument.market_type=spot"
    --set "period.start=$START"
    --set "period.end=$END"
    --set "period.bar_origin=$START"
)

generate() {
    local type="$1"
    local threshold="$2"
    local target="$3"

    echo "========================================"
    echo "Generating ETHUSDT $type bars"
    echo "Threshold: $threshold"
    echo "Target: ~$target bars/day"
    echo "========================================"

    "$PYTHON" crypto_dataset.py build \
        "${COMMON_ARGS[@]}" \
        --set "bars.types=[$type]" \
        --set "bars.$type.threshold=$threshold" \
        --fetch

    echo "Completed: $type / $target bars per day"
    echo
}

# VOLUME BARS (ETH)
generate volume 19600 25
generate volume 9800 50
generate volume 4900 100

# NOTIONAL / DOLLAR BARS (USDT)
generate notional 48300000 25
generate notional 24150000 50
generate notional 12075000 100

echo "========================================"
echo "All 6 datasets generated successfully!"
echo "========================================"
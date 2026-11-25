#!/usr/bin/env bash
# Generează BNBUSDT volume + notional bars (~25 / 50 / 100 bare pe zi) pe Binance spot,
# cu premium Coinbase pentru BTC, ETH și BNB. Toată logica e în build_asset.sh (vezi antetul lui
# pentru variabilele opționale: PLAN_OVERRIDE, PRICE_DECIMALS, QTY_DECIMALS, PREMIUM_ASSETS, ...).
set -euo pipefail
BUILD_CALLER="$0" exec "$(dirname "${BASH_SOURCE[0]}")/build_asset.sh" BNB "$@"

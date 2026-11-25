#!/usr/bin/env bash
# Generează <BASE>USDT volume + notional bars (~25 / 50 / 100 bare pe zi) pe Binance spot.
#
#   ./scripts/build_asset.sh SOL            (sau prin wrapper: ./scripts/build_sol.sh)
#
# Pragurile se calculează automat din lumânările zilnice Binance pentru aceeași perioadă,
# deci se potrivesc cu sursa folosită de crypto_dataset.py (arhiva de trades Binance).
#
# Premium Coinbase (premium_raw_coinbase_<asset>_*, premium_fx_coinbase_<asset>_*): se calculează
# pentru PREMIUM_ASSETS (context, implicit "BTC ETH") + activul construit (BASE). Pentru SOL ies
# coloanele _btc_, _eth_ și _sol_. Doar activul propriu:  PREMIUM_ASSETS="" ./scripts/build_sol.sh
# Produsul USD de pe Coinbase e "<ASSET>-USD"; dacă diferă, ex. USD_PRODUCTS="{POL: POL-USD}".
# Unde activul nu era listat pe Coinbase (ex. XRP-USD înainte de 2023-07), coloanele lui sunt null.
#
# Fiecare dataset ajunge în data/datasets/<SYMBOL>_spot_<bară>_<prag>_<țintă>pd_<start>-<end>_<hash8>/,
# ex. SOLUSDT_spot_volume_168000_25pd_20210901-20260901_1a2b3c4d.
#
# Loguri: logs/<SYMBOL>_<timestamp>/ (sau LOG_DIR=...) cu câte un fișier per dataset
# (<SYMBOL>_<bară>_<țintă>pd.log) și summary.tsv (status, dataset, director, durată).
# Un dataset care eșuează e trecut FAILED în summary, iar scriptul continuă cu următorul (exit 1 la final).
#
# Variabile opționale: QUOTE, START, END, TARGETS="25 50 100" (filtrează și PLAN_OVERRIDE), PLAN_OVERRIDE,
# PRICE_DECIMALS, QTY_DECIMALS, PREMIUM_ASSETS, USD_PRODUCTS, CONFIG, LOG_DIR,
# EXTRA_SETS="cheie=valoare ..." (override-uri suplimentare, ex. EXTRA_SETS="storage.root=/tmp/x").
set -euo pipefail

if [[ $# -lt 1 || -z "$1" ]]; then
    echo "Utilizare: $0 <BASE>   ex. $0 SOL"
    exit 1
fi

# Directorul proiectului
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=".venv/bin/python"
CONFIG="${CONFIG:-configs/btc_demo.yaml}"

BASE="${1^^}"
QUOTE="${QUOTE:-USDT}"
SYMBOL="${BASE}${QUOTE}"          # NU se mai scrie de mână: derivat din BASE
CALLER="${BUILD_CALLER:-$0 $BASE}"

# Perioada comună pentru toate dataseturile
START="${START:-2021-09-01T00:00:00Z}"
END="${END:-2026-09-01T00:00:00Z}"
read -r -a TARGETS <<< "${TARGETS:-25 50 100}"

# Activele de context pentru premium (activul construit se adaugă oricum de pipeline)
PREMIUM_ASSETS="${PREMIUM_ASSETS-BTC ETH}"
PREMIUM_LIST="[$(echo $PREMIUM_ASSETS | tr '[:lower:]' '[:upper:]' | tr ' ' ',')]"
USD_PRODUCTS="${USD_PRODUCTS:-}"
[[ -z "$USD_PRODUCTS" ]] && USD_PRODUCTS="{}"
EXTRA_ARGS=()
for kv in ${EXTRA_SETS:-}; do EXTRA_ARGS+=(--set "$kv"); done

# Opțional: suprascrie pragurile manual, ex.  PLAN_OVERRIDE=$'volume 100 97.2\nnotional 100 5280000' ./scripts/build_sol.sh
PLAN="${PLAN_OVERRIDE:-}"
if [[ -z "$PLAN" ]]; then
    echo "Calibrez pragurile pentru $SYMBOL din lumânările zilnice Binance..."
    PLAN="$("$PYTHON" scripts/calibrate_thresholds.py --symbol "$SYMBOL" \
            --start "${START:0:10}" --end "${END:0:10}" --targets "${TARGETS[@]}" --plan)"
fi
# Zecimale preț / cantitate: pipeline-ul le are predefinite doar pentru BTCUSDT și ETHUSDT.
# Se citesc direct din arhiva de trades Binance (inclusiv anul dinaintea lui START, descărcat ca istoric
# de builder, când tickSize-ul era mai fin). Suprascrie cu PRICE_DECIMALS=.. QTY_DECIMALS=.. dacă e nevoie.
if [[ -z "${PRICE_DECIMALS:-}" || -z "${QTY_DECIMALS:-}" ]]; then
    echo "Citesc zecimalele din arhiva de trades $SYMBOL (câteva minute)..."
    DEC="$("$PYTHON" scripts/find_price_decimals.py --symbol "$SYMBOL" \
           --start "${START:0:10}" --end "${END:0:10}")"   # set -e: se oprește dacă eșuează
    read -r AUTO_PD AUTO_QD <<< "$DEC"
    PRICE_DECIMALS="${PRICE_DECIMALS:-${AUTO_PD:-}}"
    QTY_DECIMALS="${QTY_DECIMALS:-${AUTO_QD:-}}"
fi
if [[ ! "${PRICE_DECIMALS:-}" =~ ^[0-9]+$ || ! "${QTY_DECIMALS:-}" =~ ^[0-9]+$ ]]; then
    echo "EROARE: zecimale invalide pentru $SYMBOL (price='${PRICE_DECIMALS:-}', qty='${QTY_DECIMALS:-}')."
    echo "Rulează scripts/find_price_decimals.py --symbol $SYMBOL sau dă-le manual:"
    echo "  PRICE_DECIMALS=.. QTY_DECIMALS=8 $CALLER"
    exit 1
fi
echo "Zecimale $SYMBOL: price_decimals=$PRICE_DECIMALS qty_decimals=$QTY_DECIMALS"

# păstrează doar țintele din TARGETS (și pentru PLAN_OVERRIDE)
PLAN="$(echo "$PLAN" | awk -v t=" ${TARGETS[*]} " 'NF == 3 && index(t, " " $2 " ")')"
if [[ -z "$PLAN" ]]; then echo "EROARE: planul de praguri e gol pentru $SYMBOL (TARGETS='${TARGETS[*]}')"; exit 1; fi

echo
echo "Plan de build pentru $SYMBOL:"
echo "$PLAN" | awk '{ printf "  %-8s ~%3s bare/zi  prag %s\n", $1, $2, $3 }'
echo "Premium Coinbase: context $PREMIUM_LIST + $BASE"
echo

LOG_DIR="${LOG_DIR:-logs/${SYMBOL}_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"
[[ -f "$SUMMARY" ]] || printf 'status\tsymbol\tbar\ttarget_pd\tthreshold\tout_dir\tminutes\tlog\n' > "$SUMMARY"
echo "Loguri: $LOG_DIR"
echo

BUILT=()
FAILED=()

generate() {
    local type="$1"
    local target="$2"
    local threshold="$3"
    local log="$LOG_DIR/${SYMBOL}_${type}_${target}pd.log"
    local t0=$SECONDS

    echo "========================================"
    echo "Generating $SYMBOL $type bars"
    echo "Threshold: $threshold"
    echo "Target: ~$target bars/day"
    echo "Log: $log"
    echo "========================================"

    local rc=0 marker
    marker="$(mktemp)"                      # manifest.json mai nou decât acest fișier = build terminat acum
    "$PYTHON" crypto_dataset.py build \
        --config "$CONFIG" \
        --set "instrument.base=$BASE" \
        --set "instrument.quote=$QUOTE" \
        --set "instrument.market_type=spot" \
        --set "instrument.price_decimals=$PRICE_DECIMALS" \
        --set "instrument.qty_decimals=$QTY_DECIMALS" \
        --set "period.start=$START" \
        --set "period.end=$END" \
        --set "period.bar_origin=$START" \
        --set "bars.types=[$type]" \
        --set "bars.$type.threshold=$threshold" \
        --set "features.premium.context_assets=$PREMIUM_LIST" \
        --set "features.premium.include_instrument_base=true" \
        --set "features.premium.usd_products=$USD_PRODUCTS" \
        --set "storage.dataset_tag=${target}pd" \
        "${EXTRA_ARGS[@]}" \
        --fetch > "$log" 2>&1 || rc=$?

    local out_dir status="OK"
    out_dir="$(sed -n 's/^  "out_dir": "\(.*\)"$/\1/p' "$log" | tail -1)"
    if [[ $rc -ne 0 && -n "$out_dir" && "$out_dir/manifest.json" -nt "$marker" ]]; then
        # manifestul e ultimul fișier scris de build: un exit != 0 după el vine de la oprirea interpretorului
        echo "ATENȚIE: build-ul a ieșit cu $rc după ce a scris manifest.json — datasetul e complet"
        rc=0
    fi
    rm -f "$marker"
    if [[ $rc -ne 0 || -z "$out_dir" || ! -f "$out_dir/$type.parquet" || ! -f "$out_dir/manifest.json" ]]; then
        status="FAILED"
        FAILED+=("$SYMBOL $type ${target}pd (exit $rc, vezi $log)")
        echo "EROARE: build-ul $SYMBOL $type ~$target/zi a eșuat (exit $rc). Ultimele linii din $log:"
        tail -20 "$log"
    else
        echo "Completed: $SYMBOL $type / $target bars per day -> $out_dir"
        BUILT+=("$out_dir/$type.parquet")
        if ! "$PYTHON" scripts/check_bars.py "$out_dir/$type.parquet" --expect "$SYMBOL" --target "$target" 2>&1 \
                | tee -a "$log"; then
            status="CHECK"
            echo "ATENȚIE: verificarea a semnalat probleme pentru $out_dir"
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$status" "$SYMBOL" "$type" "$target" "$threshold" \
        "${out_dir:-}" "$(( (SECONDS - t0) / 60 ))" "$log" >> "$SUMMARY"
    echo
}

while read -r type target threshold; do
    [[ -z "${type:-}" ]] && continue
    generate "$type" "$target" "$threshold"
done <<< "$PLAN"

echo "========================================"
echo "$SYMBOL: ${#BUILT[@]} dataseturi generate:"
for f in "${BUILT[@]}"; do echo "  $f"; done
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "$SYMBOL: ${#FAILED[@]} eșuate:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
fi
echo "Rezumat: $SUMMARY"
echo "========================================"
[[ ${#FAILED[@]} -eq 0 ]]

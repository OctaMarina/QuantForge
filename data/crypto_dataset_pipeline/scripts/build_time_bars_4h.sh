#!/usr/bin/env bash
# Generează TIME bars de 4 ore (Binance spot, <BASE>USDT) pentru mai multe active, aceeași perioadă ca
# dataseturile volume/notional (2021-09-01 -> 2026-09-01), cu premium Coinbase (context BTC + ETH + activul).
#
#   ./scripts/build_time_bars_4h.sh                 # lista implicită (19 active, fără BTC — deja construit)
#   ASSETS="ETH SOL" ./scripts/build_time_bars_4h.sh
#   JOBS=1 ./scripts/build_time_bars_4h.sh          # secvențial
#
# Zecimale: pipeline-ul le are predefinite doar pentru BTCUSDT/ETHUSDT (2/8), iar 8 e prea mult la time bars:
# scara notional = 10^(price_dec + qty_dec), iar suma pe bară e int64 (max ~9,2e18). La BTC, o bară de 4h
# trece de ~9e8 USDT (ScaleOverflow). De aceea se folosesc zecimalele OBSERVATE în arhiva Binance
# (scripts/find_price_decimals.py), nu "max(observat, 10 - preț)" ca la volume/notional. O valoare cu mai multe
# zecimale decât cele declarate oprește build-ul cu PrecisionError (nu se rotunjește niciodată).
# Suprascriere per activ:  DECIMALS_OVERRIDE="ADA=4:1 DOGE=7:0"   (preț:cantitate)
#
# Spațiu pe disc: zip-urile de tranzacții descărcate de acest script (active fără cache la pornire) se șterg după
# un build reușit (PURGE_RAW=auto, implicit). Cache-ul preexistent NU se atinge. PURGE_RAW=none păstrează tot,
# PURGE_RAW=all șterge și cache-ul preexistent al activelor construite (nu recomandat).
# Înainte de fiecare activ se cere MIN_FREE_GB liberi (implicit 20), altfel activul e sărit (SKIPPED).
#
# Loguri: logs/time4h_<timestamp>/ (sau LOG_DIR=...): <SYMBOL>.log, <SYMBOL>.decimals.log și summary.tsv.
# Un activ care eșuează e trecut FAILED, iar restul continuă (exit 1 la final).
#
# Variabile opționale: ASSETS, JOBS, START, END, INTERVAL, CONFIG, LOG_DIR, PREMIUM_ASSETS, PURGE_RAW,
# MIN_FREE_GB, DECIMALS_OVERRIDE, EXTRA_SETS="cheie=valoare ...".
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SELF="scripts/build_time_bars_4h.sh"
PYTHON=".venv/bin/python"
CONFIG="${CONFIG:-configs/btc_demo.yaml}"
START="${START:-2021-09-01T00:00:00Z}"
END="${END:-2026-09-01T00:00:00Z}"
INTERVAL="${INTERVAL:-4h}"
QUOTE="USDT"
PREMIUM_ASSETS="${PREMIUM_ASSETS-BTC ETH}"
PREMIUM_LIST="[$(echo $PREMIUM_ASSETS | tr '[:lower:]' '[:upper:]' | tr ' ' ',')]"
PURGE_RAW="${PURGE_RAW:-auto}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
DECIMALS_OVERRIDE="${DECIMALS_OVERRIDE:-}"
TRADES_ROOT="data/raw/binance/data/spot/daily/trades"
DEFAULT_ASSETS="ETH BNB SOL XRP ADA DOGE LINK LTC BCH DOT AVAX UNI AAVE ATOM NEAR FIL ETC XLM TRX"

EXTRA_ARGS=()
for kv in ${EXTRA_SETS:-}; do EXTRA_ARGS+=(--set "$kv"); done

worker() {
    local BASE="${1^^}"
    local SYMBOL="${BASE}${QUOTE}"
    local log="$LOG_DIR/${SYMBOL}.log"
    local declog="$LOG_DIR/${SYMBOL}.decimals.log"
    local t0=$SECONDS status="OK" out_dir="" pd="" qd="" rc=0

    finish() {
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$status" "$SYMBOL" "${pd:+$pd/}${qd}" "$INTERVAL" \
            "${out_dir:-}" "$(( (SECONDS - t0) / 60 ))" "$log" >> "$LOG_DIR/summary.tsv"
    }

    local free_gb
    free_gb="$(df --output=avail -BG . | tail -1 | tr -dc '0-9')"
    if (( free_gb < MIN_FREE_GB )); then
        status="SKIPPED"; echo "SKIPPED: doar ${free_gb} GB liberi (< MIN_FREE_GB=${MIN_FREE_GB})" > "$log"
        echo "[$SYMBOL] SKIPPED: ${free_gb} GB liberi"; finish; return 0
    fi

    # 1) zecimale
    local ov
    ov="$(tr ' ' '\n' <<< "$DECIMALS_OVERRIDE" | sed -n "s/^${BASE}=\([0-9]*\):\([0-9]*\)$/\1 \2/p" | head -1)"
    if [[ -n "$ov" ]]; then
        read -r pd qd <<< "$ov"
        echo "override: price_decimals=$pd qty_decimals=$qd" > "$declog"
    else
        if ! "$PYTHON" scripts/find_price_decimals.py --symbol "$SYMBOL" \
                --start "${START:0:10}" --end "${END:0:10}" > /dev/null 2> "$declog"; then
            status="FAILED"; echo "[$SYMBOL] FAILED la citirea zecimalelor (vezi $declog)"; tail -3 "$declog"
            finish; return 0
        fi
        # "=> observat: price_decimals P, qty_decimals Q; în build: ..." -> valorile observate
        pd="$(sed -n 's/^=> observat: price_decimals \([0-9]*\), qty_decimals \([0-9]*\);.*/\1/p' "$declog" | tail -1)"
        qd="$(sed -n 's/^=> observat: price_decimals \([0-9]*\), qty_decimals \([0-9]*\);.*/\2/p' "$declog" | tail -1)"
    fi
    if [[ ! "$pd" =~ ^[0-9]+$ || ! "$qd" =~ ^[0-9]+$ ]] || (( pd + qd > 12 )); then
        status="FAILED"; echo "[$SYMBOL] FAILED: zecimale invalide (price='$pd', qty='$qd'), vezi $declog"
        finish; return 0
    fi
    echo "[$SYMBOL] zecimale: price=$pd qty=$qd (scară notional 10^$((pd + qd)))"

    # 2) build
    local marker; marker="$(mktemp)"
    "$PYTHON" crypto_dataset.py build \
        --config "$CONFIG" \
        --set "instrument.base=$BASE" \
        --set "instrument.quote=$QUOTE" \
        --set "instrument.market_type=spot" \
        --set "instrument.price_decimals=$pd" \
        --set "instrument.qty_decimals=$qd" \
        --set "period.start=$START" \
        --set "period.end=$END" \
        --set "period.bar_origin=$START" \
        --set "bars.types=[time]" \
        --set "bars.time.interval=$INTERVAL" \
        --set "features.premium.context_assets=$PREMIUM_LIST" \
        --set "features.premium.include_instrument_base=true" \
        --set "storage.dataset_tag=$INTERVAL" \
        "${EXTRA_ARGS[@]}" \
        --fetch > "$log" 2>&1 || rc=$?

    out_dir="$(sed -n 's/^  "out_dir": "\(.*\)"$/\1/p' "$log" | tail -1)"
    if [[ $rc -ne 0 && -n "$out_dir" && "$out_dir/manifest.json" -nt "$marker" ]]; then
        echo "[$SYMBOL] ATENȚIE: exit $rc după scrierea manifest.json — datasetul e complet"; rc=0
    fi
    rm -f "$marker"
    if [[ $rc -ne 0 || -z "$out_dir" || ! -f "$out_dir/time.parquet" || ! -f "$out_dir/manifest.json" ]]; then
        status="FAILED"
        echo "[$SYMBOL] FAILED (exit $rc). Ultimele linii din $log:"; tail -5 "$log"
        finish; return 0
    fi

    # 3) verificare
    if ! "$PYTHON" scripts/check_bars.py "$out_dir/time.parquet" --expect "$SYMBOL" >> "$log" 2>&1; then
        status="CHECK"; echo "[$SYMBOL] ATENȚIE: check_bars a semnalat probleme (vezi $log)"
    fi

    # 4) curățare zip-uri (doar ce a descărcat acest script, în mod implicit)
    local purge=0
    if [[ "$PURGE_RAW" == "all" ]]; then purge=1
    elif [[ "$PURGE_RAW" == "auto" ]] && ! grep -qx "$SYMBOL" "$LOG_DIR/preexisting.txt"; then purge=1; fi
    if (( purge )); then
        if [[ -d "$TRADES_ROOT/$SYMBOL" ]]; then
            local gb; gb="$(du -sBG "$TRADES_ROOT/$SYMBOL" | cut -f1)"
            rm -rf -- "${TRADES_ROOT:?}/$SYMBOL"
            echo "[$SYMBOL] șters cache-ul de tranzacții ($gb)" >> "$log"
        fi
    fi
    echo "[$SYMBOL] $status -> $out_dir ($(( (SECONDS - t0) / 60 )) min)"
    finish
}

if [[ "${1:-}" == "--worker" ]]; then
    worker "$2"
    exit 0
fi

# ---- orchestrare ----
ASSETS="${ASSETS:-$DEFAULT_ASSETS}"
JOBS="${JOBS:-3}"
LOG_DIR="${LOG_DIR:-logs/time4h_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
export LOG_DIR
SUMMARY="$LOG_DIR/summary.tsv"
[[ -f "$SUMMARY" ]] || printf 'status\tsymbol\tprice/qty_dec\tinterval\tout_dir\tminutes\tlog\n' > "$SUMMARY"
# active ale căror tranzacții erau deja în cache la pornire: acestea nu se șterg niciodată (PURGE_RAW=auto)
: > "$LOG_DIR/preexisting.txt"
for a in $ASSETS; do
    s="${a^^}${QUOTE}"
    [[ -d "$TRADES_ROOT/$s" ]] && echo "$s" >> "$LOG_DIR/preexisting.txt"
done
export START END INTERVAL CONFIG QUOTE PREMIUM_LIST PURGE_RAW MIN_FREE_GB DECIMALS_OVERRIDE PYTHON TRADES_ROOT
export EXTRA_SETS="${EXTRA_SETS:-}"

echo "Time bars $INTERVAL | $START -> $END | premium $PREMIUM_LIST + activul | JOBS=$JOBS | PURGE_RAW=$PURGE_RAW"
echo "Active: $ASSETS"
echo "Cache preexistent (nu se șterge): $(tr '\n' ' ' < "$LOG_DIR/preexisting.txt")"
echo "Loguri: $LOG_DIR"
echo

# shellcheck disable=SC2086
printf '%s\n' $ASSETS | xargs -P "$JOBS" -I{} "$SELF" --worker {}

echo
echo "========================================"
column -t -s $'\t' "$SUMMARY"
echo "========================================"
! grep -q -E '^(FAILED|SKIPPED|CHECK)' "$SUMMARY"

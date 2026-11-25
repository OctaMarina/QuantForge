#!/usr/bin/env bash
# Generează câte 6 dataseturi pe activ (volume + notional × ~25 / 50 / 100 bare pe zi), Binance spot,
# 2021-09-01 -> 2026-09-01, pentru fiecare activ din ASSETS (implicit: BTC SOL BNB XRP).
#
#   ./scripts/build_multiasset__bars.sh
#   ASSETS="SOL" ./scripts/build_multiasset__bars.sh
#   nohup ./scripts/build_multiasset__bars.sh > /dev/null 2>&1 &      # logurile ajung oricum în logs/
#
# Fiecare activ rulează prin build_asset.sh (aceeași logică ca build_sol.sh etc.), deci:
#   * premium Coinbase pentru BTC, ETH + activul propriu (premium_*_coinbase_<asset>_*);
#   * instrument.base = activul (symbol derivat), deci DVOL-vs-RV, unitățile și premium-ul sunt pe activul corect;
#   * nume dataset: data/datasets/<SYMBOL>_spot_<bară>_<prag>_<țintă>pd_20210901-20260901_<hash8>/;
#   * check_bars.py după fiecare build (bare/zi, simbol, coloane premium).
#
# Loguri: logs/multiasset_<timestamp>/
#   run.log                       tot ce apare în terminal
#   <SYMBOL>/<SYMBOL>_<bară>_<țintă>pd.log   logul complet al fiecărui build (JSON-ul cu out_dir inclus)
#   summary.tsv                   un rând per dataset: OK / CHECK / FAILED, director, durată
#
# Praguri: calibrate_thresholds.py pe lumânările zilnice Binance SPOT 2021-09-01 -> 2026-09-01 (aceeași bursă
# și piață ca arhiva de trades). Vechile praguri din acest script erau calibrate pe OKX (SOL, BTC) și ar fi dat
# de ~4x (SOL) / ~7x (BTC) mai multe bare decât ținta.
#
# Zecimale: find_price_decimals.py pe arhiva de trades. qty_decimals e ales astfel încât scara notionalului
# să fie 10^10 (ca la BTCUSDT): cu qty_decimals=8 (vechea regulă) SOL avea 10^12 și sumele int64 pe bară
# depășeau limita (notional negativ / greșit în datasetul SOL vechi). Builder-ul refuză acum explicit o
# astfel de depășire (ScaleOverflow) și o cantitate/preț cu mai multe zecimale (PrecisionError).
set -uo pipefail

# Directorul proiectului
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ASSETS="${ASSETS:-BTC SOL BNB XRP}"  # BTC și SOL primele: trades deja în cache
export START="${START:-2021-09-01T00:00:00Z}"
export END="${END:-2026-09-01T00:00:00Z}"     # planurile fixate mai jos sunt calibrate pe această perioadă
export PREMIUM_ASSETS="${PREMIUM_ASSETS-BTC ETH}"

RUN_DIR="${RUN_DIR:-logs/multiasset_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

# Planuri fixate (tip, bare/zi, prag) — rezultatul calibrate_thresholds.py --plan pentru perioada de mai sus.
plan_for() {
    case "$1" in
        BTC) echo $'volume 25 2650\nnotional 25 95500000\nvolume 50 1320\nnotional 50 47800000\nvolume 100 662\nnotional 100 23900000' ;;
        SOL) echo $'volume 25 168000\nnotional 25 17500000\nvolume 50 84200\nnotional 50 8750000\nvolume 100 42100\nnotional 100 4380000' ;;
        BNB) echo $'volume 25 19900\nnotional 25 9160000\nvolume 50 9970\nnotional 50 4580000\nvolume 100 4980\nnotional 100 2290000' ;;
        XRP) echo $'volume 25 12500000\nnotional 25 12100000\nvolume 50 6270000\nnotional 50 6050000\nvolume 100 3130000\nnotional 100 3030000' ;;
        *)   echo "" ;;                    # alt activ: build_asset.sh calibrează automat
    esac
}
# Zecimale fixate (price qty) — find_price_decimals.py; BTC/ETH sunt predefinite în pipeline.
decimals_for() {
    case "$1" in
        BTC) echo "2 8" ;;
        ETH) echo "2 8" ;;
        SOL) echo "4 6" ;;
        BNB) echo "4 6" ;;
        XRP) echo "5 5" ;;
        *)   echo "" ;;                    # alt activ: build_asset.sh le citește din arhivă
    esac
}

echo "========================================"
echo "Multi-asset build: $ASSETS"
echo "Perioada: $START -> $END"
echo "Premium Coinbase: [$PREMIUM_ASSETS] + activul"
echo "Loguri: $RUN_DIR"
echo "========================================"

SUMMARY="$RUN_DIR/summary.tsv"
printf 'status\tsymbol\tbar\ttarget_pd\tthreshold\tout_dir\tminutes\tlog\n' > "$SUMMARY"
FAILED=()

for asset in $ASSETS; do
    asset="${asset^^}"
    read -r pd qd <<< "$(decimals_for "$asset")"
    echo
    echo "################ ${asset}USDT  (start $(date -u +%H:%M:%SZ)) ################"
    rc=0
    PLAN_OVERRIDE="$(plan_for "$asset")" \
    PRICE_DECIMALS="${pd:-}" QTY_DECIMALS="${qd:-}" \
    LOG_DIR="$RUN_DIR/${asset}USDT" BUILD_CALLER="ASSETS=$asset $0" \
        ./scripts/build_asset.sh "$asset" || rc=$?
    if [[ -f "$RUN_DIR/${asset}USDT/summary.tsv" ]]; then
        tail -n +2 "$RUN_DIR/${asset}USDT/summary.tsv" >> "$SUMMARY"
    fi
    if [[ $rc -ne 0 ]]; then
        FAILED+=("${asset}USDT (exit $rc)")
        echo "EROARE: ${asset}USDT s-a terminat cu exit $rc — continui cu următorul activ."
    fi
done

echo
echo "========================================"
echo "Rezumat ($SUMMARY):"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Active cu probleme: ${FAILED[*]}"
    echo "========================================"
    exit 1
fi
echo "Toate activele au fost generate."
echo "========================================"

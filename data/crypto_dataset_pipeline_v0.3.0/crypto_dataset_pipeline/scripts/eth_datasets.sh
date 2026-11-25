#!/usr/bin/env bash
# ETHUSDT spot datasets: volume / notional bars (your thresholds) + information-driven bars (runs and
# imbalance, calibrated to the same 25 / 50 / 100 bars per day).  See docs/RUN_ETH.md.
#
#   bash scripts/eth_datasets.sh                 # all steps, in order
#   bash scripts/eth_datasets.sh download        # only one step (download | classic | calibrate | information | report)
#   bash scripts/eth_datasets.sh classic report  # several steps
#
# Every step can be re-run: downloads are cached, builds continue from their saved state, finished days
# are skipped.  A build that cannot complete exits with code 3 and the script stops (set -e) with the
# reason printed by the build (INCOMPLETE / HALTED lines).
#
# Settings can be changed with environment variables, e.g.
#   TARGETS="25 50" PILOT_MONTHS="2023-05" bash scripts/eth_datasets.sh calibrate information
#   EXTRA_SETS="storage.root=/mnt/big/cryptods" bash scripts/eth_datasets.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/eth_spot.yaml}"
START="${START:-2021-09-01T00:00:00Z}"
END="${END:-2026-09-01T00:00:00Z}"
# one month before START: bar-window features and the information-bar estimates are warmed up before the export
ORIGIN="${ORIGIN:-2021-08-01T00:00:00Z}"
VOLUME_THRESHOLDS="${VOLUME_THRESHOLDS:-19600 9800 4900}"            # ETH per bar      (~25 / 50 / 100 bars/day)
NOTIONAL_THRESHOLDS="${NOTIONAL_THRESHOLDS:-48300000 24150000 12075000}"  # USDT per bar
TARGETS="${TARGETS:-25 50 100}"                                       # information bars: mean bars per day
PILOT_MONTHS="${PILOT_MONTHS:-2022-06,2024-03}"
SIGN="${SIGN:-aggressor}"                                             # aggressor (observed) | tick_rule (AFML)
CALIB_DIR="${CALIB_DIR:-calibration}"
EXTRA_SETS="${EXTRA_SETS:-}"                                          # space separated key=value overrides

COMMON=(--config "$CONFIG" --set "period.start=$START" --set "period.end=$END" --set "period.bar_origin=$ORIGIN")
for kv in $EXTRA_SETS; do COMMON+=(--set "$kv"); done

banner() { echo; echo "========== $* =========="; }

step_download() {
    banner "download (resumable; checksums verified)"
    "$PYTHON" crypto_dataset.py download "${COMMON[@]}"
}

step_classic() {
    for thr in $VOLUME_THRESHOLDS; do
        banner "volume bars, threshold $thr ETH"
        "$PYTHON" crypto_dataset.py build "${COMMON[@]}" --set "bars.types=[volume]" --set "bars.volume.threshold=$thr"
    done
    for thr in $NOTIONAL_THRESHOLDS; do
        banner "notional bars, threshold $thr USDT"
        "$PYTHON" crypto_dataset.py build "${COMMON[@]}" --set "bars.types=[notional]" --set "bars.notional.threshold=$thr"
    done
}

step_calibrate() {
    banner "calibrate information bars for targets: $TARGETS bars/day (pilots: $PILOT_MONTHS)"
    "$PYTHON" scripts/calibrate_information_bars.py "${COMMON[@]}" --targets "$(echo $TARGETS | tr ' ' ',')" \
        --pilot-months "$PILOT_MONTHS" --sign "$SIGN" --out "$CALIB_DIR"
}

step_information() {
    for n in $TARGETS; do
        f="$CALIB_DIR/target_${n}.args"
        if [ ! -f "$f" ]; then
            echo "missing $f: run the calibrate step first" >&2
            exit 2
        fi
        IB_ARGS=()
        while IFS= read -r line || [ -n "$line" ]; do
            [ -n "$line" ] && IB_ARGS+=("$line")
        done < "$f"
        banner "information bars (runs + imbalance), target ~$n bars/day"
        "$PYTHON" crypto_dataset.py build "${COMMON[@]}" "${IB_ARGS[@]}"
    done
}

step_report() {
    banner "summary of all datasets"
    "$PYTHON" scripts/report_datasets.py "${COMMON[@]}"
}

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ] || [ "${STEPS[0]}" = "all" ]; then
    STEPS=(download classic calibrate information report)
fi
for s in "${STEPS[@]}"; do
    case "$s" in
        download|classic|calibrate|information|report) "step_$s" ;;
        *) echo "unknown step '$s' (download | classic | calibrate | information | report | all)" >&2; exit 2 ;;
    esac
done
banner "done: ${STEPS[*]}"

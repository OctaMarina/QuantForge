#!/usr/bin/env python
"""
Verifică un dataset de bare după build: simbol, prag, bare/zi (total și pe an).

    python scripts/check_bars.py data/datasets/BTCUSDT_spot_volume_.../volume.parquet --expect BTCUSDT --target 100

Verifică și coloanele de premium Coinbase: activul construit (instrument.base din manifest.json) trebuie
să aibă premium_raw_coinbase_<asset>_1m / premium_fx_coinbase_<asset>_1m.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect", help="simbolul așteptat, ex. BTCUSDT")
    ap.add_argument("--target", type=float, help="bare/zi țintă")
    a = ap.parse_args()

    d = pd.read_parquet(a.path, columns=["series_id", "bar_type", "threshold_frozen", "close", "low", "high", "volume",
                                         "notional_usdt", "bar_end_us"])
    t = pd.to_datetime(d["bar_end_us"], unit="us", utc=True)
    days = (t.max() - t.min()).total_seconds() / 86400
    series = d["series_id"].iloc[0]
    bpd = len(d) / days
    print(f"series   : {series}")
    print(f"bar_type : {d['bar_type'].iloc[0]} | prag {d['threshold_frozen'].iloc[0]:,} | {len(d):,} bare | {t.min().date()} -> {t.max().date()}")
    print(f"preț     : {d['close'].iloc[0]:,.4f} -> {d['close'].iloc[-1]:,.4f}")
    print(f"bare/zi  : {bpd:.1f}")
    per_year = t.groupby(t.dt.year).agg(lambda x: len(x) / max((x.max() - x.min()).total_seconds() / 86400, 1))
    print(f"pe an    : {per_year.round(0).astype(int).to_dict()}")
    ok = True
    if a.expect and a.expect not in series:
        print(f"EROARE: seria nu e {a.expect}! (simbolul din config nu corespunde activului)"); ok = False
    if a.target and not (0.7 * a.target <= bpd <= 1.3 * a.target):
        print(f"ATENȚIE: {bpd:.1f} bare/zi față de ținta {a.target:g} (±30%)"); ok = False
    # notional / volume trebuie să fie în [low, high] (o depășire int64 dă notional negativ / aberant)
    v = d["volume"] > 0
    vw = d.loc[v, "notional_usdt"] / d.loc[v, "volume"]
    bad = int(((vw < d.loc[v, "low"] * (1 - 1e-6)) | (vw > d.loc[v, "high"] * (1 + 1e-6))).sum())
    neg = int((d["notional_usdt"] < 0).sum())
    print(f"notional : {bad:,} bare cu notional/volum în afara [low, high], {neg:,} negative")
    if bad or neg:
        print("EROARE: notional incoerent (depășire int64?) — refă cu qty_decimals mai mic"); ok = False
    ok &= check_premium(Path(a.path))
    print("OK" if ok else "VERIFICĂ")
    # os._exit: pyarrow abortează uneori la oprirea interpretorului (după ce totul a fost scris) -> exit 134 fals
    sys.stdout.flush(), sys.stderr.flush()
    os._exit(0 if ok else 1)


def check_premium(path: Path) -> bool:
    names = pq.read_schema(path).names
    prem = [n for n in names if n.startswith("premium_raw_coinbase_") and n.endswith("_1m")]
    assets = [n[len("premium_raw_coinbase_"):-len("_1m")] for n in prem]
    man = path.parent / "manifest.json"
    base = json.loads(man.read_text())["instrument"]["base"].lower() if man.exists() else None
    if prem:
        share = pd.read_parquet(path, columns=prem).notna().mean()
        print("premium  : " + ", ".join(f"{a} {share[c]:.1%} valid" for a, c in zip(assets, prem)))
    if base and base not in assets:
        print(f"EROARE: lipsesc coloanele premium_*_coinbase_{base}_* pentru activul construit")
        return False
    return True


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Calibrează pragurile pentru volume / notional bars din lumânările zilnice Binance SPOT
(aceeași bursă și același tip de piață ca arhiva de trades folosită de crypto_dataset.py).

    prag volume   = volumul mediu zilnic (monedă de bază)  / bare dorite pe zi
    prag notional = quote volume mediu zilnic (USDT)        / bare dorite pe zi

Exemple:
    python scripts/calibrate_thresholds.py --symbol BTCUSDT
    python scripts/calibrate_thresholds.py --symbol BTCUSDT --plan            # linii "tip ținta prag" pentru scripturile bash
    python scripts/calibrate_thresholds.py --symbol ETHUSDT                   # verificare: ~4.900 ETH / ~12,1 mil. USDT la 100 bare/zi
"""
import argparse
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone

URL = "https://api.binance.com/api/v3/klines?symbol={s}&interval=1d&startTime={a}&endTime={b}&limit=1000"
INFO_URL = "https://api.binance.com/api/v3/exchangeInfo?symbol={s}"


def n_decimals(step: str) -> int:
    """'0.01000000' -> 2, '1.00000000' -> 0"""
    frac = step.split(".")[1].rstrip("0") if "." in step else ""
    return len(frac)


def historical_price_decimals(klines) -> int:
    """Numărul maxim de zecimale semnificative din prețurile OHLC ale perioadei.
    Binance a schimbat tickSize-ul în timp, iar arhiva de trades păstrează prețurile cu pasul de atunci."""
    return max(n_decimals(x[i]) for x in klines for i in (1, 2, 3, 4))


def symbol_decimals(symbol: str, klines):
    """price_decimals = max(zecimale tickSize curent, zecimale observate în prețurile istorice ale perioadei).
    qty_decimals = 8 (precizia arhivei de trades Binance; aceeași valoare ca specificația ETHUSDT din pipeline)."""
    with urllib.request.urlopen(INFO_URL.format(s=symbol), timeout=30) as r:
        info = json.loads(r.read())["symbols"][0]
    f = {x["filterType"]: x for x in info["filters"]}
    tick = f["PRICE_FILTER"]["tickSize"]
    hist = historical_price_decimals(klines) if klines else 0
    return max(n_decimals(tick), hist), 8, tick, hist


def ms(d: str) -> int:
    return int(datetime.strptime(d[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def daily_klines(symbol: str, start: str, end: str):
    a, b, rows = ms(start), ms(end) - 1, []
    while a < b:
        with urllib.request.urlopen(URL.format(s=symbol, a=a, b=b), timeout=30) as r:
            k = json.loads(r.read())
        if not k:
            break
        rows += k
        a = k[-1][6] + 1
        time.sleep(0.2)
    return rows   # [open_time, o, h, l, c, base_volume, close_time, quote_volume, ...]


def round_sig(x: float, sig: int = 3) -> float:
    if x <= 0:
        return x
    p = sig - int(math.floor(math.log10(x))) - 1
    v = round(x, p)
    return int(v) if v >= 1 and float(v).is_integer() else v


def summarize(k, start, end):
    expected_days = (ms(end) - ms(start)) // 86_400_000
    days = len(k)
    vol = sum(float(x[5]) for x in k) / days
    quote = sum(float(x[7]) for x in k) / days
    yearly = {}
    for x in k:
        y = datetime.fromtimestamp(x[0] / 1000, timezone.utc).year
        yearly.setdefault(y, []).append((float(x[5]), float(x[7])))
    return dict(days=days, expected_days=expected_days, vol_per_day=vol, quote_per_day=quote, yearly=yearly)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", default="2021-09-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--targets", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--types", nargs="+", default=["volume", "notional"])
    ap.add_argument("--plan", action="store_true", help="emite doar liniile 'tip ținta prag' (pentru bash)")
    ap.add_argument("--decimals", action="store_true", help="emite 'price_decimals qty_decimals' și iese")
    a = ap.parse_args()

    if a.decimals:
        k = daily_klines(a.symbol, a.start, a.end)
        pd_, qd, tick, hist = symbol_decimals(a.symbol, k)
        print(f"{a.symbol}: tickSize curent {tick} ({n_decimals(tick)} zecimale), prețuri istorice {a.start}->{a.end} "
              f"cu max {hist} zecimale -> price_decimals {pd_} | qty_decimals {qd}", file=sys.stderr)
        print(f"{pd_} {qd}")
        return

    k = daily_klines(a.symbol, a.start, a.end)
    if not k:
        sys.exit(f"Binance nu a întors lumânări pentru {a.symbol} în {a.start} -> {a.end}")
    s = summarize(k, a.start, a.end)
    log = sys.stderr if a.plan else sys.stdout
    base = a.symbol.replace("USDT", "")
    print(f"{a.symbol} Binance spot {a.start} -> {a.end}: {s['days']} zile cu date (din {s['expected_days']})", file=log)
    if s["days"] < 0.95 * s["expected_days"]:
        print(f"ATENȚIE: lipsesc {s['expected_days'] - s['days']} zile (listare târzie / delistare?)", file=log)
    print(f"  medie/zi: {s['vol_per_day']:,.2f} {base} | {s['quote_per_day']:,.0f} USDT", file=log)

    plan = []
    for t in a.targets:
        if "volume" in a.types:
            plan.append(("volume", t, round_sig(s["vol_per_day"] / t)))
        if "notional" in a.types:
            plan.append(("notional", t, round_sig(s["quote_per_day"] / t)))
    for typ, t, thr in plan:
        y = {yr: round(sum(v[0 if typ == "volume" else 1] for v in vals) / len(vals) / thr)
             for yr, vals in sorted(s["yearly"].items())}
        print(f"  {typ:8s} ~{t:>3}/zi  prag {thr:>14,}   bare/zi pe an: {y}", file=log)
        if a.plan:
            print(f"{typ} {t} {thr}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Află câte zecimale au prețurile și cantitățile direct în arhiva de trades Binance spot
(aceeași sursă ca builder-ul), pe câteva zile din perioadă.

    python scripts/find_price_decimals.py --symbol SOLUSDT
    python scripts/find_price_decimals.py --symbol SOLUSDT --days 2021-09-01 2023-01-01

Ultima linie e "price_decimals qty_decimals", ca la calibrate_thresholds.py --decimals.

qty_decimals = max(zecimalele observate, min(8, 10 - price_decimals)): scara notionalului rămâne cel mult
10^10 (ca la BTCUSDT 2+8), deci sumele int64 pe bară încap până la ~9e8 USDT. Cu 4+8 (=10^12), o bară de
17,5 mil. USDT depășește int64. O cantitate cu mai multe zecimale decât qty_decimals oprește build-ul
(PrecisionError), nu e rotunjită.

O zi care nu poate fi citită din motive de rețea (după reîncercări) oprește scriptul cu cod 1, ca să nu
iasă un maxim calculat pe zile lipsă. Zilele inexistente în arhivă (404, înainte de listare) se sar.
"""
import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter

URL = "https://data.binance.vision/data/spot/daily/trades/{s}/{s}-trades-{d}.zip"


def default_days(start: str = "2021-09-01", end: str = "2026-09-01"):
    """Câte o zi pe lună în anul dinaintea lui START (builder-ul descarcă istoric pentru features,
    iar atunci tickSize-ul era adesea mai fin) + câte o zi la ~10 luni în perioada propriu-zisă."""
    import datetime as dt
    s, e = dt.date.fromisoformat(start[:10]), dt.date.fromisoformat(end[:10])
    days = []
    for k in range(12, 0, -1):
        y, m = s.year, s.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        days.append(dt.date(y, m, 1))
    d = s
    while d < e:
        days.append(d)
        y, m = d.year, d.month + 10
        while m > 12:
            y, m = y + 1, m - 12
        d = dt.date(y, m, 1)
    return [x.isoformat() for x in days]


def n_dec(x: str) -> int:
    x = x.strip()
    if "e" in x.lower():                      # notație științifică -> forțează verificare manuală
        return 99
    return len(x.split(".")[1].rstrip("0")) if "." in x else 0


def fetch(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise                                  # 404 etc.: nu se reîncearcă
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def scan_day(symbol, day):
    z = zipfile.ZipFile(io.BytesIO(fetch(URL.format(s=symbol, d=day))))
    pc, qc, ex = Counter(), Counter(), {}
    with z.open(z.namelist()[0]) as f:
        for row in csv.reader(io.TextIOWrapper(f)):
            if not row or not row[0].isdigit():   # sare peste header, dacă există
                continue
            p, q = row[1], row[2]
            dp = n_dec(p)
            pc[dp] += 1
            qc[n_dec(q)] += 1
            ex.setdefault(dp, p)
    return pc, qc, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", default="2021-09-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--days", nargs="+", default=None)
    a = ap.parse_args()
    a.days = a.days or default_days(a.start, a.end)
    max_p = max_q = 0
    net_errors = []
    for d in a.days:
        try:
            pc, qc, ex = scan_day(a.symbol, d)
        except urllib.error.HTTPError as e:       # zi înainte de listare / lipsă în arhivă
            print(f"{d}: lipsă în arhivă ({e.code})", file=sys.stderr)
            continue
        except Exception as e:
            print(f"{d}: nu am putut citi ({e})", file=sys.stderr)
            net_errors.append(d)
            continue
        mp, mq = max(pc), max(qc)
        max_p, max_q = max(max_p, mp), max(max_q, mq)
        dist = ", ".join(f"{k}z: {v:,}" for k, v in sorted(pc.items()))
        print(f"{d}: preț max {mp} zecimale ({dist}); exemplu cu {mp} zecimale: {ex[mp]} | cantitate max {mq} zecimale",
              file=sys.stderr)
    if net_errors:
        sys.exit(f"EROARE: {len(net_errors)} zile necitite din cauza rețelei ({', '.join(net_errors)}); rulează din nou")
    if max_p == 0 and max_q == 0:
        sys.exit("nicio zi citită din arhivă")
    qd = max(max_q, min(8, 10 - max_p))
    if max_p + qd > 12:
        sys.exit(f"EROARE: price_decimals {max_p} + qty_decimals {qd} > 12 (limita pipeline-ului)")
    print(f"=> observat: price_decimals {max_p}, qty_decimals {max_q}; în build: {max_p} {qd} "
          f"(scară notional 10^{max_p + qd})", file=sys.stderr)
    print(f"{max_p} {qd}")


if __name__ == "__main__":
    main()

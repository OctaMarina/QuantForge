"""Fill the REST kline caches for the time-bar frequencies of the bar-type study (30m, 1h, 1d; 4h exists).

    python download_time_klines.py [--end 2026-11-19] [--intervals 1d 1h 30m]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents] if (p / "quant_forge").exists())
sys.path.insert(0, str(ROOT))

from quant_forge.data import ALL_ASSETS, load_klines  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--end", default="2026-11-19")
ap.add_argument("--intervals", nargs="*", default=["1d", "1h", "30m"])
args = ap.parse_args()

t0 = time.time()
for iv in args.intervals:
    for a in ALL_ASSETS:
        t = time.time()
        for attempt in range(3):
            try:
                k = load_klines(f"{a}USDT", iv, end=args.end, source="rest")
                print(f"{iv:4s} {a:6s} {len(k):>8,} rows  {time.time() - t:5.0f} s", flush=True)
                break
            except Exception as exc:
                print(f"{iv:4s} {a:6s} attempt {attempt + 1} failed: {exc!r}", flush=True)
                time.sleep(10)
print(f"done in {(time.time() - t0) / 60:.1f} min")

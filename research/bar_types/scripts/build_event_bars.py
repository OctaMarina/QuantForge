"""Build the volume / dollar bars of the bar-type study from the Binance aggTrades archive (40 assets, 8 series:
volume and dollar bars at 48, 24, 6 and 1 bars per day, 30-day trailing thresholds, 4-hour outage gap).

Resumable and incremental: the cache (~/.cache/quantforge/binance_event_bars/<SYMBOL>/) is committed after every
archive file, so re-running continues where it stopped.  Progress: <cache>/build.log.

    python build_event_bars.py [--end 2026-11-19] [--workers 6] [--assets BTC ETH ...]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents] if (p / "quant_forge").exists())
sys.path.insert(0, str(ROOT))

from quant_forge.data import ALL_ASSETS  # noqa: E402
from quant_forge.data.event_bars import DEFAULT_EVENT_SPECS, build_event_bars_many  # noqa: E402

# largest archives first (sizes of the monthly aggTrades files), so the long jobs start immediately
BY_SIZE = ["BTC", "ETH", "BNB", "XRP", "DOGE", "SOL", "ADA", "LTC", "LINK", "DOT", "TRX", "ZEC", "AVAX", "AAVE", "NEAR",
           "VET", "SAND", "UNI", "FIL", "ATOM", "XLM", "BCH", "MANA", "HBAR", "ETC", "CHZ", "ICP", "CRV", "ALGO", "THETA",
           "AXS", "GRT", "DASH", "XTZ", "ENJ", "ZIL", "IOTA", "QTUM", "COMP", "SNX"]

ap = argparse.ArgumentParser()
ap.add_argument("--end", default="2026-11-19")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--assets", nargs="*", default=None)
args = ap.parse_args()
assets = args.assets or [a for a in BY_SIZE if a in ALL_ASSETS]
assert set(assets) <= set(ALL_ASSETS)

t0 = time.time()
res = build_event_bars_many([f"{a}USDT" for a in assets], DEFAULT_EVENT_SPECS, end=args.end, workers=args.workers)
failed = {k: v for k, v in res.items() if isinstance(v, str)}
print(f"done in {(time.time() - t0) / 60:.1f} min; {len(res) - len(failed)} ok, failed: {failed}")
for sym, st in res.items():
    if not isinstance(st, str):
        print(sym, st["first_day"], st["next_day"], f"{st['trades']:,} trades", f"{st['bytes'] / 1e9:.1f} GB",
              "missing files:", st["missing"])

"""Example: load an assembled dataset, keep only rows whose required families are valid, list feature metadata.

  .venv/bin/python examples/load_dataset.py btcusdt_initial volume_50btc
"""
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chdds import paths as P          # noqa: E402
from chdds.registry import match      # noqa: E402

name, bar_name = sys.argv[1], sys.argv[2]
d = P.DERIVED / "datasets" / name
man = json.loads((d / "manifest.json").read_text())
spec_id = next(k for k, v in man["bar_sets"].items() if v["name"] == bar_name)
df = pl.scan_parquet(str(d / spec_id / "*.parquet"))
# validity masks are explicit columns: never fill nulls with zeros
ok = (pl.col("complete") == 1) & (pl.col("feed_complete") == 1) & (pl.col("tw_state_15m") == 0) & \
     (pl.col("mark_state") == 0) & (pl.col("oi_state") == 0)
cols = ["decision_time_ns", "bar_return_log", "bar_duration_s", "tfi_ratio_2m", "rv_trade_15m_grid1s",
        "basis_mark_index_bps", "funding_rate_asof", "oi_log_change_known_1h", "liq_notional_long_15m_usdt"]
sub = df.filter(ok).select(cols).collect()
print(sub.describe())
print("book anchored or certified:", df.select(((pl.col("book_state") == 0) | (pl.col("book_state") == 9)).mean()).collect().item())
print("default model columns:", len(man["bar_sets"][spec_id]["model_feature_columns"]))
for c in cols[1:4]:
    f = match(c)[0]
    print(f"{c}: {f.formula} [{f.unit}] clock={f.clock} validity={f.validity}")

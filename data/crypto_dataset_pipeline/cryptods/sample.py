"""Build a small dataset from REAL fixture trades (no network needed after fetch_real_fixtures.sh)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from . import timeutil as tu
from .sources.binance import normalize_trades_table

HERE = Path(__file__).resolve().parent.parent


def load_nautilus_trades_csv(path: Path, spec) -> tuple[pd.DataFrame, dict]:
    """nautilus test_data CSV: timestamp (ISO), trade_id, price, quantity, buyer_maker."""
    raw = pd.read_csv(path, dtype={"price": str, "quantity": str, "buyer_maker": str})
    ts = pd.to_datetime(raw["timestamp"], utc=True, format="ISO8601").dt.as_unit("us").astype("int64")
    t = pa.table({"id": pa.array(raw["trade_id"].astype(np.int64)), "price": pa.array(raw["price"]),
                  "qty": pa.array(raw["quantity"]), "time": pa.array((ts // 1000).astype(np.int64)),  # ms
                  "is_buyer_maker": pa.array(raw["buyer_maker"])})
    day = pd.Timestamp(ts.iloc[0], unit="us", tz="UTC").date()
    df, q = normalize_trades_table(t, spec, "spot", day, str(path), ts_unit="ms")
    return df, q


def build_real_sample(cfg, fixtures: str | None):
    from .pipeline import Pipeline
    fx = Path(fixtures) if fixtures else HERE / "tests" / "fixtures" / "real"
    f = fx / "binance_ethusdt-trades.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing - run scripts/fetch_real_fixtures.sh first")
    p = Pipeline(cfg, offline=True)
    df, q = load_nautilus_trades_csv(f, p.spec)
    res = p.build(allow_fetch=False, live_trades=df)
    res["input_quality"] = {k: v for k, v in q.items() if k != "internal_gap_ranges"}
    res["input_file"] = str(f)
    return res

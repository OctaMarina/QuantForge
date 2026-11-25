import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cryptods.instruments import InstrumentSpec  # noqa: E402

REAL = ROOT / "tests" / "fixtures" / "real"


@pytest.fixture
def spec():
    return InstrumentSpec("binance", "spot", "BTCUSDT", "BTC", "USDT", price_decimals=2, qty_decimals=8)


def mk_trades(spec, ids, ts, px, qty, side=None):
    """Build normalised trades from decimal strings / numbers (exact via Decimal)."""
    ps, qs = spec.price_scale, spec.qty_scale
    p = np.array([int(Decimal(str(x)) * ps) for x in px], dtype=np.int64)
    q = np.array([int(Decimal(str(x)) * qs) for x in qty], dtype=np.int64)
    if side is None:
        side = np.ones(len(ids), dtype=np.int8)
    return pd.DataFrame({"trade_id": np.asarray(ids, np.int64), "ts_us": np.asarray(ts, np.int64),
                         "price_i": p, "qty_i": q, "notional_i": p * q, "aggressor": np.asarray(side, np.int8)})


def random_trades(spec, n=5000, seed=0, t0=1_700_000_000_000_000, mean_dt=50_000, start_id=1):
    rng = np.random.default_rng(seed)
    dt = rng.exponential(mean_dt, n).astype(np.int64)
    dt[rng.random(n) < 0.2] = 0          # identical timestamps happen
    ts = t0 + np.cumsum(dt)
    steps = rng.choice([-1, 0, 0, 1], n)
    price = 30000_00 + np.cumsum(steps)  # cents
    qty = (rng.lognormal(-3, 1.5, n) * 1e8).astype(np.int64) + 1
    side = rng.choice([-1, 1], n).astype(np.int8)
    ids = np.arange(start_id, start_id + n, dtype=np.int64)
    return pd.DataFrame({"trade_id": ids, "ts_us": ts, "price_i": price.astype(np.int64), "qty_i": qty,
                         "notional_i": price.astype(np.int64) * qty, "aggressor": side})


@pytest.fixture
def real_dir():
    if not (REAL / "binance_ethusdt-trades.csv").exists():
        pytest.skip("real fixtures missing: run scripts/fetch_real_fixtures.sh")
    return REAL

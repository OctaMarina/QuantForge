import os
from datetime import date

import numpy as np
import pytest

from chdds import paths as P

REAL_DAY = date(2025, 6, 28)


def have_raw():
    return P.raw_path(REAL_DAY, 5, "orderbook").exists()


def have_canonical():
    return (P.DERIVED / "canonical" / "trades" / "2025-06-28.parquet").exists()


requires_raw = pytest.mark.skipif(not have_raw(), reason="raw CryptoHFTData files not available")
requires_canonical = pytest.mark.skipif(not have_canonical(), reason="canonical layer not built")


@pytest.fixture(scope="session")
def book_hours():
    """Decoded real book hours 2025-06-28 05..07 (REST snapshots every 5 min -> exact verification)."""
    from chdds.book.driver import load_book_hour
    return {h: load_book_hour("2025-06-28", h) for h in (5, 6, 7)}


def synthetic_trades(n=20000, seed=0, p0=100000.0, gaps_at=()):
    rng = np.random.default_rng(seed)
    steps = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]) * 10   # price_e2 moves in 0.1 USDT ticks
    px = (int(p0 * 100) + np.cumsum(steps)).astype(np.int64)
    qx = rng.integers(1, 3000, size=n).astype(np.int64)
    recv = (1_751_000_000_000_000_000 + np.cumsum(rng.integers(1, 50_000_000, size=n))).astype(np.int64)
    tt = recv // 1_000_000 - 100
    sign = rng.choice([-1, 1], size=n).astype(np.int64)
    seq = np.arange(n, dtype=np.int64)
    tid = seq + 1000
    seg = np.zeros(n, np.int64)
    for g in gaps_at:
        seg[g:] += 1
    return dict(seq=seq, tid=tid, recv=recv, tt=tt, px=px, qx=qx, sign=sign, seg=seg)

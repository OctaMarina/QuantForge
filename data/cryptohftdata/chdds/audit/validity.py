"""Validity map per family and interval (EXHAUSTIVE: evaluated on a 1 s availability grid over the whole history;
book on the 100 ms replay grid).

Families and the condition used (same thresholds as the initial feature configuration):
  trades      : no trade-continuity gap (missing ids / missing file) covering the second; after history start
  mark        : last mark row with market age <= 3 s (fast) / <= 10 s (slow context)
  ticker      : last ticker row with market age <= 10 s
  oi          : reference point age <= 15 min (by availability)
  liquidations: liquidation file present for the hour and no shared market-stream reception gap > 5 s
  book_strict : replay state 0 (anchored, chained, not crossed, fresh <= 1 s, top inside certified domain)
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pyarrow as pa
import polars as pl
import pyarrow.parquet as pq

from .. import paths as P

CAN = P.DERIVED / "canonical"
NS = 1_000_000_000


def _asof_age(times_avail, times_mkt, grid):
    j = np.searchsorted(times_avail, grid, side="right") - 1
    age = np.full(grid.size, np.inf)
    ok = j >= 0
    age[ok] = (grid[ok] - times_mkt[j[ok]]) / NS
    return age


def hourly_validity(log=print):
    gaps = pq.read_table(CAN / "trades_gaps.parquet").to_pydict() if (CAN / "trades_gaps.parquet").exists() else {}
    g0 = np.array([x if x is not None else 0 for x in gaps.get("t0", [])], np.int64)
    g1 = np.array([x if x is not None else 0 for x in gaps.get("t1", [])], np.int64)
    oi = pq.read_table(CAN / "open_interest.parquet")
    oi = oi.filter(pa.array(oi["status"].to_numpy() == 0))
    oi_recv = oi["recv_ns"].to_numpy()
    oi_ts = oi["ts_ms"].to_numpy() * P.MS_NS
    best = np.maximum.accumulate(oi_ts)
    liq_hours = set(pq.read_table(CAN / "liquidation_file_hours.parquet")["hour_index"].to_numpy().tolist())
    first_trade = None
    rows = []
    d = P.HISTORY_START
    prev_mark = None
    while d <= P.HISTORY_END:
        day0 = P.hour_start_ns(d, 0)
        grid = day0 + np.arange(1, 86401, dtype=np.int64) * NS
        tp = CAN / "trades" / f"{d.isoformat()}.parquet"
        if first_trade is None and tp.exists():
            first_trade = int(pq.read_table(tp, columns=["recv_ns"])["recv_ns"][0].as_py())
        tr_ok = np.ones(grid.size, bool) if first_trade is not None else np.zeros(grid.size, bool)
        if first_trade is not None:
            tr_ok &= grid >= first_trade
        if not tp.exists():
            tr_ok[:] = False
        for a, b in zip(g0, g1):
            if b > day0 and a < day0 + 86400 * NS:
                tr_ok &= ~((grid >= a) & (grid <= (b if b > 0 else day0 + 86400 * NS)))
        # missing trade hours
        for h in range(24):
            if not P.raw_path(d, h, "trades").exists():
                tr_ok[h * 3600:(h + 1) * 3600] = False
        mk = []
        for dd in (d - timedelta(days=1), d):
            p = CAN / "mark" / f"{dd.isoformat()}.parquet"
            if p.exists():
                mk.append(pq.read_table(p, columns=["recv_ns", "event_ms"]))
        if mk:
            mr = np.concatenate([m["recv_ns"].to_numpy() for m in mk])
            me = np.concatenate([m["event_ms"].to_numpy() for m in mk]) * P.MS_NS
            o = np.argsort(mr, kind="stable")
            mr, me = mr[o], me[o]
            mage = _asof_age(mr, me, grid)
            # shared market-stream reception gaps (> 5 s since last mark reception)
            ragej = np.searchsorted(mr, grid, side="right") - 1
            rage = np.where(ragej >= 0, (grid - mr[np.maximum(ragej, 0)]) / NS, np.inf)
        else:
            mage = np.full(grid.size, np.inf)
            rage = np.full(grid.size, np.inf)
        tk = []
        for dd in (d - timedelta(days=1), d):
            p = CAN / "ticker" / f"{dd.isoformat()}.parquet"
            if p.exists():
                tk.append(pq.read_table(p, columns=["recv_ns", "event_ms"]))
        if tk:
            tr_ = np.concatenate([m["recv_ns"].to_numpy() for m in tk])
            te = np.concatenate([m["event_ms"].to_numpy() for m in tk]) * P.MS_NS
            o = np.argsort(tr_, kind="stable")
            tage = _asof_age(tr_[o], te[o], grid)
        else:
            tage = np.full(grid.size, np.inf)
        j = np.searchsorted(oi_recv, grid, side="right") - 1
        oage = np.where(j >= 0, (grid - best[np.maximum(j, 0)]) / NS, np.inf)
        hidx = (grid - 1 - P.hour_start_ns(P.HISTORY_START, 0)) // P.HOUR_NS
        liq_file = np.array([h in liq_hours for h in hidx])
        fam = dict(trades=tr_ok, mark_fast=mage <= 3, mark_slow=mage <= 10, ticker=tage <= 10, oi=oage <= 900,
                   liquidations=liq_file & (rage <= 5))
        for h in range(24):
            sl = slice(h * 3600, (h + 1) * 3600)
            rows.append(dict(day=d.isoformat(), hour=h, **{k: float(v[sl].mean()) for k, v in fam.items()}))
        if d.day == 1:
            log(f"  validity {d}")
        d += timedelta(days=1)
    df = pl.DataFrame(rows)
    df.write_parquet(P.DERIVED / "audit" / "validity_hourly.parquet")
    return df


def book_hourly(grid_dir):
    rows = []
    for f in sorted(grid_dir.glob("*.parquet")):
        g = pq.read_table(f, columns=["t_ns", "state"])
        t = g["t_ns"].to_numpy()
        s = g["state"].to_numpy()
        h = ((t - 1) // P.HOUR_NS) * P.HOUR_NS
        df = pl.DataFrame({"h": h, "s": s}).group_by("h").agg(
            (pl.col("s") == 0).mean().alias("book_strict"), (pl.col("s") == 1).mean().alias("book_unanchored"),
            (pl.col("s") == 2).mean().alias("book_stale"), (pl.col("s") == 7).mean().alias("book_top_uncertified"))
        rows.append(df)
    if not rows:
        return None
    out = pl.concat(rows).with_columns(
        pl.from_epoch(pl.col("h"), time_unit="ns").dt.strftime("%Y-%m-%d").alias("day"),
        pl.from_epoch(pl.col("h"), time_unit="ns").dt.hour().cast(pl.Int64).alias("hour")).drop("h")
    return out


def invalid_intervals(df: pl.DataFrame, col: str, thr: float = 0.5, min_hours: int = 2):
    """Runs of consecutive hours with valid fraction < thr."""
    x = df.sort(["day", "hour"])
    bad = (x[col] < thr).to_list()
    days, hours = x["day"].to_list(), x["hour"].to_list()
    out = []
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if j - i >= min_hours:
                out.append(dict(family=col, start=f"{days[i]} {hours[i]:02d}:00",
                                end_excl=f"{days[j]} {hours[j]:02d}:00" if j < len(bad) else "end", hours=j - i))
            i = j
        else:
            i += 1
    return out

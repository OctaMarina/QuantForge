"""Adapters turning raw book data into the engine's message stream.

Message dict: {"type": snapshot|update|disconnect, "ts_us", "recv_us", "bids": [(p_i, q_i)],
"asks": [...], "U", "u", "pu", "depth_limit"} with exact scaled integers.

Time stamp used for ordering checkpoints: exchange event time in market-time replay, receipt time in
observed-availability replay.  A REST snapshot has no exchange time on Binance spot; it is placed at
the time of the last message before it and only becomes *valid* once the first bridging diff is
applied (L2Book.awaiting_first), so it can never make the book look valid earlier than the exchange
time of that diff.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from .. import timeutil as tu
from ..storage import iter_jsonl_gz


def _lv(levels, ps: int, qs: int):
    out = []
    for p, q in levels:
        pi = Decimal(p) * ps
        qi = Decimal(q) * qs
        if pi != pi.to_integral_value() or qi != qi.to_integral_value():
            raise ValueError(f"book level {p}/{q} not representable at instrument scale")
        out.append((int(pi), int(qi)))
    return out


def binance_live_messages(depth_files, snapshot_files, conn_files, spec, mode: str = "market_time", futures=False):
    """Merge live-collected Binance diff-depth messages, REST snapshots and connection events
    (all logged with received_at_us) into an ordered message stream."""
    ps, qs = spec.price_scale, spec.qty_scale
    items = []
    for rec in iter_jsonl_gz(depth_files):
        p = rec["payload"]
        p = p.get("data", p)
        if p.get("e") != "depthUpdate":
            continue
        ex_ts = int(p.get("E")) * 1000 if p.get("E") is not None else rec["received_at_us"]
        items.append((rec["received_at_us"], 1, {"type": "update", "ex_ts": ex_ts, "recv_us": rec["received_at_us"],
                                                  "bids": _lv(p["b"], ps, qs), "asks": _lv(p["a"], ps, qs),
                                                  "U": int(p["U"]), "u": int(p["u"]),
                                                  "pu": int(p["pu"]) if "pu" in p else None}))
    for rec in iter_jsonl_gz(snapshot_files):
        p = rec["payload"]
        items.append((rec["received_at_us"], 0, {"type": "snapshot", "ex_ts": None, "recv_us": rec["received_at_us"],
                                                  "bids": _lv(p["bids"], ps, qs), "asks": _lv(p["asks"], ps, qs),
                                                  "u": int(p["lastUpdateId"]), "depth_limit": rec.get("limit")}))
    for rec in iter_jsonl_gz(conn_files):
        ev = rec["payload"].get("event")
        if ev in ("close", "error", "disconnect"):
            items.append((rec["received_at_us"], 2, {"type": "disconnect", "ex_ts": None, "recv_us": rec["received_at_us"],
                                                      "bids": [], "asks": []}))
    items.sort(key=lambda x: (x[0], x[1]))
    last_ts = None
    for _, _, m in items:
        if mode == "observed_availability":
            ts = m["recv_us"]
        else:
            ts = m["ex_ts"] if m["ex_ts"] is not None else (last_ts if last_ts is not None else m["recv_us"])
        if last_ts is not None and ts < last_ts:
            ts = last_ts  # receipt order is authoritative for application; keep checkpoint time monotonic
        m["ts_us"] = ts
        last_ts = ts
        yield m


def tardis_incremental_messages(df: pd.DataFrame, spec, mode: str = "market_time"):
    """Group Tardis incremental_book_L2 rows into messages.  Rows before the first snapshot are
    skipped; a snapshot after updates resets the book.  The snapshot's depth limit is unknown, so the
    deepest snapshot level bounds the known-complete region (conservative)."""
    ps, qs = spec.price_scale, spec.qty_scale
    tcol = "received_at_us" if mode == "observed_availability" else "ts_us"
    df = df.reset_index(drop=True)
    key = df["ts_us"].astype(str) + "|" + df["received_at_us"].astype(str) + "|" + df["is_snapshot"].astype(str)
    grp_id = (key != key.shift()).cumsum()
    seen_snapshot = False
    prev_snap = False
    for _, g in df.groupby(grp_id, sort=False):
        snap = bool(g["is_snapshot"].iloc[0])
        if not seen_snapshot and not snap:
            continue
        seen_snapshot = True
        bids = _lv(zip(g.loc[g.side == 1, "price"], g.loc[g.side == 1, "qty"]), ps, qs)
        asks = _lv(zip(g.loc[g.side == -1, "price"], g.loc[g.side == -1, "qty"]), ps, qs)
        ts = int(g[tcol].iloc[0])
        if snap:
            if prev_snap:
                # continuation of the same snapshot (split across timestamps) is not expected; treat as new
                pass
            yield {"type": "snapshot", "ts_us": ts, "recv_us": int(g["received_at_us"].iloc[0]), "bids": bids, "asks": asks,
                   "u": None, "depth_limit": max(len(bids), 1) if bids else None}
        else:
            yield {"type": "update", "ts_us": ts, "recv_us": int(g["received_at_us"].iloc[0]), "bids": bids, "asks": asks}
        prev_snap = snap


def nautilus_binance_futures_messages(snap_csv: str, update_csv: str, spec):
    """Real Binance USD-M BTCUSDT fixture from nautilus_trader test_data (snapshot + diff rows)."""
    ps, qs = spec.price_scale, spec.qty_scale
    s = pd.read_csv(snap_csv, dtype={"price": str, "qty": str})
    u = pd.read_csv(update_csv, dtype={"price": str, "qty": str})
    yield {"type": "snapshot", "ts_us": int(s["timestamp"].iloc[0]) * 1000, "recv_us": None,
           "bids": _lv(zip(s.loc[s.side == "b", "price"], s.loc[s.side == "b", "qty"]), ps, qs),
           "asks": _lv(zip(s.loc[s.side == "a", "price"], s.loc[s.side == "a", "qty"]), ps, qs),
           "u": int(s["last_update_id"].iloc[0]), "depth_limit": int((s.side == "b").sum())}
    for (U, uu, pu), g in u.groupby(["first_update_id", "last_update_id", "pu"], sort=False):
        yield {"type": "update", "ts_us": int(g["timestamp"].iloc[0]) * 1000, "recv_us": None,
               "bids": _lv(zip(g.loc[g.side == "b", "price"], g.loc[g.side == "b", "qty"]), ps, qs),
               "asks": _lv(zip(g.loc[g.side == "a", "price"], g.loc[g.side == "a", "qty"]), ps, qs),
               "U": int(U), "u": int(uu), "pu": int(pu)}

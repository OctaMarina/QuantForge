"""Turn live logs into the same normalised tables the historical path produces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from .. import timeutil as tu
from ..sources.binance import normalize_trades_table
from ..sources.deribit import normalize_ticker
from ..sources.publications import parse_fng, parse_farside, farside_daily, vintages_from_snapshots
from ..storage import iter_jsonl_gz


def files(root: Path, stream: str):
    return sorted((Path(root) / "live" / stream).glob("*/*.jsonl.gz"))


def uptime(root: Path, conn_name: str):
    """[(open_us, close_us)] for a connection name from the _conn log."""
    ivs, cur = [], None
    for rec in iter_jsonl_gz(files(root, "_conn")):
        p = rec["payload"]
        if p.get("stream") != conn_name:
            continue
        if p["event"] == "open":
            cur = rec["received_at_us"]
        elif p["event"] in ("close", "error") and cur is not None:
            ivs.append((cur, rec["received_at_us"]))
            cur = None
    if cur is not None:
        ivs.append((cur, tu.now_us()))
    return ivs


def trades(root: Path, spec, futures: bool) -> pd.DataFrame:
    tag = "binance_um" if futures else "binance_spot"
    rows = [r for r in iter_jsonl_gz(files(root, f"{tag}_trades"))]
    if not rows:
        return pd.DataFrame()
    p = [r["payload"] for r in rows]
    if futures:   # aggTrade
        t = pa.table({"id": pa.array([x["a"] for x in p], pa.int64()), "price": pa.array([x["p"] for x in p]),
                      "qty": pa.array([x["q"] for x in p]), "first_trade_id": pa.array([x["f"] for x in p], pa.int64()),
                      "last_trade_id": pa.array([x["l"] for x in p], pa.int64()),
                      "time": pa.array([x["T"] for x in p], pa.int64()), "is_buyer_maker": pa.array([x["m"] for x in p])})
        unit = "ms"
    else:
        t = pa.table({"id": pa.array([x["t"] for x in p], pa.int64()), "price": pa.array([x["p"] for x in p]),
                      "qty": pa.array([x["q"] for x in p]), "time": pa.array([x["T"] for x in p], pa.int64()),
                      "is_buyer_maker": pa.array([x["m"] for x in p])})
        unit = "us"  # collector subscribes with timeUnit=MICROSECOND
    day = tu.us_to_datetime_index([tu.normalize_exchange_ts([t.column("time")[0].as_py()], unit)[0]])[0].date()
    df, _ = normalize_trades_table(t, spec, spec.archive_market(), day, f"live:{tag}", aggregated=futures, ts_unit=unit)
    rec = pd.DataFrame({"trade_id": [x["a"] if futures else x["t"] for x in p],
                        "received_at_us": [r["received_at_us"] for r in rows]}).drop_duplicates("trade_id")
    return df.merge(rec, on="trade_id", how="left")


def liquidations(root: Path) -> dict:
    out = {}
    ev = []
    for r in iter_jsonl_gz(files(root, "binance_um_forceOrder")) or []:
        o = r["payload"].get("o", {})
        if not o:
            continue
        ev.append({"ts_us": int(o["T"]) * 1000, "received_at_us": r["received_at_us"],
                   "liquidated_position": "long" if o["S"] == "SELL" else "short",
                   "qty": float(o["z"]), "price": float(o["ap"]), "status": o.get("X")})
    out["binance"] = {"events": pd.DataFrame(ev, columns=["ts_us", "received_at_us", "liquidated_position", "qty", "price", "status"]),
                      "uptime": uptime(root, "binance_um")}
    ev = []
    for r in iter_jsonl_gz(files(root, "bybit_allLiquidation")):
        for d in r["payload"].get("data", []) or []:
            ev.append({"ts_us": int(d["T"]) * 1000, "received_at_us": r["received_at_us"],
                       "liquidated_position": "long" if d["S"] == "Buy" else "short",
                       "qty": float(d["v"]), "price": float(d["p"])})
    out["bybit"] = {"events": pd.DataFrame(ev, columns=["ts_us", "received_at_us", "liquidated_position", "qty", "price"]),
                    "uptime": uptime(root, "bybit_linear")}
    return out


def dvol(root: Path, mode: str) -> dict:
    out = {}
    rows = []
    for r in iter_jsonl_gz(files(root, "deribit_dvol")):
        d = r["payload"]["params"]["data"]
        rows.append({"currency": d["index_name"].split("_")[0].upper(), "ts_us": int(d["timestamp"]) * 1000,
                     "received_at_us": r["received_at_us"], "close": float(d["volatility"])})
    df = pd.DataFrame(rows)
    for cur, g in (df.groupby("currency") if len(df) else []):
        g = g.sort_values("ts_us").copy()
        g["period_start_us"] = g["ts_us"]
        g["period_end_us"] = g["received_at_us"] if mode == "observed_availability" else g["ts_us"]
        out[cur] = g
    return out


def options(root: Path) -> pd.DataFrame:
    rows = []
    for r in iter_jsonl_gz(files(root, "deribit_option_ticker")):
        d = r["payload"]["params"]["data"]
        x = normalize_ticker(d, r["received_at_us"])
        if x:
            rows.append(x)
    return pd.DataFrame(rows)


def quotes(root: Path, spec) -> pd.DataFrame:
    rows = []
    for r in iter_jsonl_gz(files(root, "binance_spot_bookTicker")):
        p = r["payload"]
        rows.append({"ts_us": r["received_at_us"], "received_at_us": r["received_at_us"], "update_id": p.get("u"),
                     "bid": float(p["b"]), "bid_qty": float(p["B"]), "ask": float(p["a"]), "ask_qty": float(p["A"])})
    return pd.DataFrame(rows)


def publications(root: Path):
    fng_frames, far = [], {"BTC": [], "ETH": []}
    for r in iter_jsonl_gz(files(root, "poll_fear_greed")):
        b = r["payload"]
        if b.get("status") == 200:
            try:
                fng_frames.append((parse_fng(b["body"].encode()), r["received_at_us"]))
            except Exception:
                pass
    for asset in ("BTC", "ETH"):
        for r in iter_jsonl_gz(files(root, f"poll_farside_{asset.lower()}")):
            b = r["payload"]
            if b.get("status") == 200:
                try:
                    far[asset].append((farside_daily(parse_farside(b["body"].encode(), asset)), r["received_at_us"]))
                except Exception:
                    pass
    fng = vintages_from_snapshots(fng_frames, ["value"], ["economic_date"]) if fng_frames else None
    if fng is not None and len(fng):
        fng["evidence"] = "live_poll_received_at"
    etf = {}
    for asset, frames in far.items():
        if frames:
            v = vintages_from_snapshots(frames, ["total_musd", "sum_funds_musd", "n_funds_reported"], ["economic_date"])
            v["evidence"] = "live_poll_received_at"
            etf[asset] = v
    return fng, etf

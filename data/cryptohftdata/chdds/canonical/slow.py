"""Canonical slow streams: mark/index/funding, ticker, open interest (revision history), liquidations.

Identity rules (report §3.3):
  mark/ticker  : (event_time, full payload); identical copies -> first authentic receipt kept;
                 the same event_time with a different payload is a distinct observation (counted).
  OI           : (timestamp) with versions = distinct (oi, oi_value); each version keeps its first receipt and a status
                 (classify_oi_versions). A version is rejected only if it is demonstrably impossible; a point with one
                 surviving version takes it, available only from its reception; with several it stays in quarantine.
  liquidations : conservative fingerprint of all exchange fields; exact copies (except received_time) dropped;
                 rows sharing (trade_time, side, price, quantity) with different fill state are kept and flagged
                 `maybe_same_order` (no order_id -> identity cannot be proven).
Market-stream liveness (ticker/mark/liquidations share one WebSocket per the vendor wire format): the socket is silent
only if neither mark nor ticker is received. This and missing files are evidence, not proof, of liquidation-feed state.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import paths as P
from ..audit.run import staged_path

OUT = P.DERIVED / "canonical"


def _dedup_payload(recv, evt, payload_cols):
    synth = recv == evt * P.MS_NS
    keys = [recv, synth] + payload_cols[::-1] + [evt]
    o = np.lexsort(keys)
    same = np.ones(o.size - 1, bool) if o.size > 1 else np.zeros(0, bool)
    for c in [evt] + payload_cols:
        same &= c[o][1:] == c[o][:-1]
    keep = np.ones(o.size, bool)
    keep[1:] = ~same
    first = np.nonzero(keep)[0]
    ncop = np.diff(np.append(first, o.size)).astype(np.int8)
    sel = o[keep]
    ao = np.argsort(recv[sel], kind="stable")
    return sel[ao], ncop[ao]


def build_mark(start=P.HISTORY_START, end=P.HISTORY_END, log=print):
    out = OUT / "mark"
    out.mkdir(parents=True, exist_ok=True)
    stats = dict(rows=0, kept=0, conflicting_evt=0)
    d = start
    while d <= end:
        tabs = []
        for h in range(24):
            sp = staged_path("mark_price", d, h)
            if sp.exists():
                tabs.append(pq.read_table(sp))
        if tabs:
            t = pa.concat_tables(tabs)
            c = {k: t[k].to_numpy() for k in t.column_names}
            pay = [c["mark_e8"], c["index_e8"], c["settle_e8"], c["funding_e8"], c["next_funding_ms"]]
            sel, ncop = _dedup_payload(c["recv_ns"], c["event_ms"], pay)
            stats["rows"] += t.num_rows
            stats["kept"] += sel.size
            stats["conflicting_evt"] += int(sel.size - np.unique(c["event_ms"][sel]).size)
            tab = pa.table({"recv_ns": c["recv_ns"][sel], "event_ms": c["event_ms"][sel],
                            "mark_e8": c["mark_e8"][sel], "index_e8": c["index_e8"][sel],
                            "settle_e8": c["settle_e8"][sel], "settle_null": c["settle_null"][sel],
                            "funding_e8": c["funding_e8"][sel], "funding_null": c["funding_null"][sel],
                            "next_funding_ms": c["next_funding_ms"][sel], "n_copies": ncop})
            pq.write_table(tab, out / f"{d.isoformat()}.parquet", compression="zstd")
        d += timedelta(days=1)
    (OUT / "mark_build.json").write_text(json.dumps(stats))
    log(f"mark canonical: {stats}")
    return stats


def build_ticker(start=P.HISTORY_START, end=P.HISTORY_END, log=print):
    out = OUT / "ticker"
    out.mkdir(parents=True, exist_ok=True)
    stats = dict(rows=0, kept=0, conflicting_evt=0)
    d = start
    while d <= end:
        tabs = [pq.read_table(staged_path("ticker", d, h)) for h in range(24) if staged_path("ticker", d, h).exists()]
        if tabs:
            t = pa.concat_tables(tabs)
            c = {k: t[k].to_numpy() for k in t.column_names}
            pay = [c[k] for k in ("last_price", "price_change", "price_change_percent", "weighted_average_price",
                                  "open_price", "high_price", "low_price", "base_asset_volume", "quote_asset_volume",
                                  "last_quantity", "stat_open_ms", "stat_close_ms", "first_trade_id", "last_trade_id",
                                  "total_trades")]
            sel, ncop = _dedup_payload(c["recv_ns"], c["event_ms"], pay)
            stats["rows"] += t.num_rows
            stats["kept"] += sel.size
            stats["conflicting_evt"] += int(sel.size - np.unique(c["event_ms"][sel]).size)
            cols = {k: c[k][sel] for k in c if k not in ("row",)}
            cols["n_copies"] = ncop
            pq.write_table(pa.table(cols), out / f"{d.isoformat()}.parquet", compression="zstd")
        d += timedelta(days=1)
    (OUT / "ticker_build.json").write_text(json.dumps(stats))
    log(f"ticker canonical: {stats}")
    return stats


# OI version status. Rejections can act retroactively: they only withhold a value. Nothing is ever available before its receipt.
OI_OK, OI_FOREIGN, OI_FROZEN, OI_PRICE, OI_UNREACHABLE, OI_CONFLICT = 0, 1, 2, 3, 4, 5
OI_STATUS_NAMES = ["accepted", "foreign_source_zero_value", "frozen_value", "implied_price_inconsistent",
                   "unreachable_under_volume_bound", "unresolved_conflict"]
OI_FROZEN_MIN_TS = 3            # identical OI on >= 3 points never happens on single-version points (max 2 in the whole history)
OI_MAX_PRICE_DEV_BPS = 100.0    # value/OI vs last trade at ts; 78 bps is the maximum on single-version points
OI_VOLUME_MARGIN = 1.5          # |dOI| <= margin * volume; 1.02 is the maximum on 141,770 adjacent single-version pairs


def classify_oi_versions(ts, recv, oi, ov, volume_between, price_at):
    """Status of every version (ts ms, recv ns, integer oi and value). Only points with several versions can be rejected by
    the frozen, price and volume rules, so unaffected observations are never touched. A step between two 5-minute points cannot
    exceed the volume traded around them, because a trade changes OI by at most its size."""
    st = np.full(ts.size, OI_OK, np.int8)
    st[(ov == 0) & (oi > 0)] = OI_FOREIGN
    u, inv, cnt = np.unique(ts, return_inverse=True, return_counts=True)
    multi = cnt[inv] > 1
    if not multi.any():
        return st
    # frozen: same OI on several points; a row received strictly before any repetition existed stays a candidate
    uo, oinv = np.unique(oi, return_inverse=True)
    for g in range(uo.size):
        rows = np.nonzero(oinv == g)[0]
        if np.unique(ts[rows]).size >= OI_FROZEN_MIN_TS:
            first = recv[rows].min()
            early = rows[recv[rows] == first]
            for r in rows:
                if multi[r] and st[r] == OI_OK and not (early.size == 1 and r == early[0]):
                    st[r] = OI_FROZEN
    for r in np.nonzero(multi & (st == OI_OK))[0]:
        ref = price_at(int(ts[r]))
        if ref > 0 and abs(1e4 * (ov[r] / oi[r] / ref - 1)) > OI_MAX_PRICE_DEV_BPS:
            st[r] = OI_PRICE
    # reachability over runs of consecutive contested points, between neighbours with a single accepted value
    cand = {int(t): np.nonzero((ts == t) & (st == OI_OK))[0] for t in u}
    contested = sorted(int(t) for t, c in zip(u, cnt) if c > 1)

    def linked(ra, rb):
        a, b = int(ts[ra]), int(ts[rb])
        return abs(float(oi[rb]) - float(oi[ra])) <= OI_VOLUME_MARGIN * volume_between(min(a, b) - 300000, max(a, b) + 300000)

    i = 0
    while i < len(contested):
        j = i
        while j + 1 < len(contested) and contested[j + 1] == contested[j] + 300000:
            j += 1
        run = contested[i:j + 1]
        left = cand.get(run[0] - 300000, np.zeros(0, np.int64))
        right = cand.get(run[-1] + 300000, np.zeros(0, np.int64))
        reach = {}
        prev = left
        for t in run:
            ok = [r for r in cand[t] if prev.size == 0 or any(linked(p_, r) for p_ in prev)]
            reach[t] = set(ok)
            prev = np.array(ok, np.int64) if ok else cand[t]
        prev = right
        for t in reversed(run):
            ok = [r for r in cand[t] if prev.size == 0 or any(linked(r, p_) for p_ in prev)]
            reach[t] &= set(ok)
            prev = np.array(sorted(reach[t]), np.int64) if reach[t] else cand[t]
        for t in run:
            for r in cand[t]:
                if r not in reach[t]:
                    st[r] = OI_UNREACHABLE
        i = j + 1
    for t in contested:
        alive = np.nonzero((ts == t) & (st == OI_OK))[0]
        if alive.size > 1:
            st[alive] = OI_CONFLICT
    return st


def _oi_trade_context(ts_ms):
    """Volume and last price around the contested points, from the canonical trades. A trade gap removes the volume bound."""
    from ..features.trades import load_gaps
    d0 = date(1970, 1, 1) + timedelta(days=int(ts_ms.min() // 86400000) - 1)
    d1 = date(1970, 1, 1) + timedelta(days=int(ts_ms.max() // 86400000) + 1)
    tabs = []
    d = d0
    while d <= d1:
        f = OUT / "trades" / f"{d.isoformat()}.parquet"
        if f.exists():
            tabs.append(pq.read_table(f, columns=["trade_time_ms", "price_e2", "qty_e3"]))
        d += timedelta(days=1)
    t = pa.concat_tables(tabs)
    tt = t["trade_time_ms"].to_numpy()
    o = np.argsort(tt, kind="stable")
    tt = tt[o]
    px = t["price_e2"].to_numpy()[o] / 100.0
    cv = np.concatenate([[0], np.cumsum(t["qty_e3"].to_numpy()[o])])
    g0, g1 = load_gaps()

    def volume_between(a, b):
        if g0.size and np.any((g0 < b * P.MS_NS) & (g1 > a * P.MS_NS)):
            return np.inf
        return cv[np.searchsorted(tt, b, side="right")] - cv[np.searchsorted(tt, a, side="right")]

    def price_at(a):
        j = np.searchsorted(tt, a, side="right") - 1
        return float(px[j]) if j >= 0 else 0.0

    return volume_between, price_at


def build_oi(start=P.HISTORY_START, end=P.HISTORY_END, log=print):
    rows = []
    for day, h in P.iter_hours(start, end):
        sp = staged_path("open_interest", day, h)
        if sp.exists():
            t = pq.read_table(sp)
            t = t.append_column("src_hour", pa.array(np.full(t.num_rows, P.hour_index(P.hour_start_ns(day, h)),
                                                              np.int32)))
            rows.append(t)
    t = pa.concat_tables(rows)
    # QUARANTINE (audit finding 2026-10-15): points whose timestamp is not on the 5-minute grid do not belong to
    # the documented source (openInterestHist period=5m); they are excluded and kept for review.
    ts_all = t["ts_ms"].to_numpy()
    offgrid = ts_all % 300000 != 0
    if offgrid.any():
        pq.write_table(t.filter(pa.array(offgrid)), OUT / "open_interest_quarantine_offgrid.parquet")
        t = t.filter(pa.array(~offgrid))
    recv = t["recv_ns"].to_numpy()
    ts = t["ts_ms"].to_numpy()
    oi = t["oi_e8"].to_numpy()
    ov = t["oi_value_e8"].to_numpy()
    src = t["src_hour"].to_numpy()
    o = np.lexsort((recv, ov, oi, ts))
    same = np.zeros(o.size, bool)
    same[1:] = (ts[o][1:] == ts[o][:-1]) & (oi[o][1:] == oi[o][:-1]) & (ov[o][1:] == ov[o][:-1])
    first = o[~same]
    grp_start = np.nonzero(~same)[0]
    ncop = np.diff(np.append(grp_start, o.size))
    # versions per ts ordered by first receipt
    v_ts, v_recv = ts[first], recv[first]
    vo = np.lexsort((v_recv, v_ts))
    first = first[vo]
    ncop = ncop[vo]
    v_ts = ts[first]
    ver = np.zeros(first.size, np.int32)
    for i in range(1, first.size):
        ver[i] = ver[i - 1] + 1 if v_ts[i] == v_ts[i - 1] else 0
    nver = np.zeros(first.size, np.int32)
    _, inv_, cnt_ = np.unique(v_ts, return_inverse=True, return_counts=True)
    nver = cnt_[inv_].astype(np.int32)
    fo, fv, fr = oi[first], ov[first], recv[first]
    if (nver > 1).any():
        vol_e3, price_at = _oi_trade_context(v_ts[nver > 1])
        # OI is stored at 1e-8 BTC and traded quantity at 1e-3 BTC
        status = classify_oi_versions(v_ts, fr, fo, fv, lambda a, b: vol_e3(a, b) * 100000.0, lambda a: price_at(a))
    else:
        status = np.where((fv == 0) & (fo > 0), OI_FOREIGN, OI_OK).astype(np.int8)
    tab = pa.table({"ts_ms": v_ts, "recv_ns": recv[first], "oi_e8": oi[first], "oi_value_e8": ov[first],
                    "version": ver, "n_versions": nver, "n_copies": ncop.astype(np.int32), "src_hour": src[first],
                    "status": status})
    tab = tab.sort_by("recv_ns")
    pq.write_table(tab, OUT / "open_interest.parquet")
    stats = dict(rows_on_grid=int(t.num_rows), quarantined_offgrid=int(offgrid.sum()),
                 versions=int(first.size), unique_ts=int(np.unique(v_ts).size),
                 ts_with_conflicting_versions=int(np.unique(v_ts[ver > 0]).size),
                 revised_ts=int(np.count_nonzero(ver > 0)), copies_dropped=int(t.num_rows - first.size),
                 ts_not_5m=int(np.count_nonzero(v_ts % 300000)),
                 status={n: int(np.count_nonzero(status == k)) for k, n in enumerate(OI_STATUS_NAMES)},
                 ts_contested=int(np.unique(v_ts[nver > 1]).size),
                 ts_resolved=int(np.unique(v_ts[(nver > 1) & (status == OI_OK)]).size),
                 ts_quarantined=int(np.unique(v_ts[status == OI_CONFLICT]).size))
    (OUT / "oi_build.json").write_text(json.dumps(stats))
    log(f"OI canonical: {stats}")
    return stats


def build_liquidations(start=P.HISTORY_START, end=P.HISTORY_END, log=print):
    rows = []
    hours_present = []
    for day, h in P.iter_hours(start, end):
        sp = staged_path("liquidations", day, h)
        hi = P.hour_index(P.hour_start_ns(day, h))
        if sp.exists():
            t = pq.read_table(sp)
            rows.append(t.append_column("src_hour", pa.array(np.full(t.num_rows, hi, np.int32))))
            hours_present.append(hi)
    t = pa.concat_tables(rows)
    c = {k: t[k].to_numpy() for k in t.column_names}
    pay = [c[k] for k in ("trade_time_ms", "side", "status", "order_type", "tif", "qty_e3", "price_e2",
                          "avg_price_e8", "last_filled_e3", "filled_e3")]
    sel, ncop = _dedup_payload(c["recv_ns"], c["event_ms"], pay)
    k2 = np.stack([c["trade_time_ms"][sel], c["side"][sel].astype(np.int64), c["qty_e3"][sel], c["price_e2"][sel]], 1)
    _, inv, cnt = np.unique(k2, axis=0, return_inverse=True, return_counts=True)
    maybe = cnt[inv.ravel()] > 1
    cols = {k: c[k][sel] for k in c if k != "row"}
    cols["n_copies"] = ncop
    cols["maybe_same_order"] = maybe
    cols["notional_usdt"] = c["avg_price_e8"][sel] / 1e8 * c["filled_e3"][sel] / 1e3
    pq.write_table(pa.table(cols), OUT / "liquidations.parquet")
    pq.write_table(pa.table({"hour_index": np.array(hours_present, np.int32)}), OUT / "liquidation_file_hours.parquet")
    stats = dict(rows=int(t.num_rows), kept=int(sel.size), exact_copies_dropped=int(t.num_rows - sel.size),
                 maybe_same_order_rows=int(maybe.sum()), hours_with_file=len(hours_present),
                 non_filled=int(np.count_nonzero(c["status"][sel] != 0)))
    (OUT / "liq_build.json").write_text(json.dumps(stats))
    log(f"liquidations canonical: {stats}")
    return stats


def market_stream_gaps(threshold_s: float = 5.0):
    """Reception gaps of the shared market WebSocket as [t0, t1] intervals: no mark row and no ticker row. A gap in mark
    alone is a mark gap (documented by the vendor, ticker keeps arriving on the same connection), not a socket gap."""
    files = sorted((OUT / "mark").glob("*.parquet")) + sorted((OUT / "ticker").glob("*.parquet"))
    recv = np.concatenate([pq.read_table(f, columns=["recv_ns"])["recv_ns"].to_numpy() for f in files])
    recv.sort()
    d = np.diff(recv)
    idx = np.nonzero(d > threshold_s * P.NS)[0]
    return recv[idx], recv[idx + 1]

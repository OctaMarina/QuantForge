"""Exhaustive per-file audit: every row of every file is read. Returns (stats, events, staged).

stats  : flat dict of scalars (JSON-serialisable) for files_<stream>.parquet
events : list of dicts (gaps, chain breaks, snapshots, duplicates) for events_<stream>.parquet
staged : pyarrow.Table with exact scaled-integer columns (or None for orderbook)
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..decimal import parse_decimal, ST_OK, ST_INEXACT, ST_MALFORMED, ST_NULL, ST_OVERFLOW
from ..paths import HOUR_NS, MS_NS, NS
from ..rawio import col_i64
from . import kernels as K

# Scales of the exact integer representation (number of decimals kept).
SCALES = {
    "price": 2, "qty": 3,
    "mark": 8, "funding": 8,
    "oi": 8, "oi_value": 8,
    "tk_price": 2, "tk_pct": 4, "tk_base": 3, "tk_quote": 4,
    "lq_price": 2, "lq_avg": 8, "lq_qty": 3,
}

GAP_EVENT_NS = {"trades": 5 * NS, "orderbook": 5 * NS, "mark_price": 3 * NS, "ticker": 10 * NS}


def _q(x, qs=(0.0, 0.01, 0.5, 0.99, 0.999, 1.0)):
    if x.size == 0:
        return [None] * len(qs)
    return [float(v) for v in np.quantile(x, qs)]


def _vc(arr) -> str:
    vc = pc.value_counts(arr).to_pylist()
    return json.dumps({str(d["values"]): int(d["counts"]) for d in vc}, sort_keys=True)


def _status_counts(prefix, st, ndec, stats):
    stats[f"{prefix}_inexact"] = int(np.count_nonzero(st == ST_INEXACT))
    stats[f"{prefix}_malformed"] = int(np.count_nonzero(st == ST_MALFORMED))
    stats[f"{prefix}_overflow"] = int(np.count_nonzero(st == ST_OVERFLOW))
    stats[f"{prefix}_null"] = int(np.count_nonzero(st == ST_NULL))
    stats[f"{prefix}_maxdec"] = int(ndec.max()) if ndec.size else None


def _common(table, info, h0, stats):
    stats["n_rows"] = table.num_rows
    stats["nulls"] = json.dumps({c: table.column(c).null_count for c in table.column_names})
    if "symbol" in table.column_names:
        stats["symbols"] = _vc(table.column("symbol"))


def _gaps(t_sorted, h0, threshold, stream, kind, events, stats, prefix="recv"):
    """Gaps between consecutive (sorted) availability times, plus head/tail gaps to hour edges."""
    if t_sorted.size == 0:
        stats[f"{prefix}_max_gap_s"] = None
        return
    d = np.diff(t_sorted)
    stats[f"{prefix}_head_gap_s"] = float((t_sorted[0] - h0) / NS)
    stats[f"{prefix}_tail_gap_s"] = float((h0 + HOUR_NS - t_sorted[-1]) / NS)
    stats[f"{prefix}_max_gap_s"] = float(d.max() / NS) if d.size else 0.0
    for thr in (1, 3, 5, 10, 30, 60, 300):
        stats[f"{prefix}_n_gap_gt_{thr}s"] = int(np.count_nonzero(d > thr * NS))
    stats[f"{prefix}_gap_time_gt_5s_s"] = float(d[d > 5 * NS].sum() / NS) if d.size else 0.0
    idx = np.nonzero(d > threshold)[0]
    for i in idx[:5000]:
        events.append(dict(kind=kind, t0=int(t_sorted[i]), t1=int(t_sorted[i + 1]),
                           dur_s=float(d[i] / NS)))


# --------------------------------------------------------------------------------------- trades
def audit_trades(table, info, h0):
    stats, events = {}, []
    _common(table, info, h0, stats)
    n = table.num_rows
    recv, _ = col_i64(table, "received_time")
    evt, _ = col_i64(table, "event_time")
    tt, _ = col_i64(table, "trade_time")
    tid, _ = col_i64(table, "trade_id")
    price, pst, pdec = parse_decimal(table.column("price"), SCALES["price"])
    qty, qst, qdec = parse_decimal(table.column("quantity"), SCALES["qty"])
    _status_counts("price", pst, pdec, stats)
    _status_counts("qty", qst, qdec, stats)
    ot = table.column("order_type")
    stats["order_types"] = _vc(ot)
    is_na = pc.equal(ot, "NA").to_numpy(zero_copy_only=False)
    is_mkt = pc.equal(ot, "MARKET").to_numpy(zero_copy_only=False)
    ot_code = np.where(is_mkt, 0, np.where(is_na, 1, 2)).astype(np.int8)
    ibm = table.column("is_buyer_maker").to_numpy(zero_copy_only=False)
    pos = (price > 0) & (qty > 0)
    stats["n_na"] = int(is_na.sum())
    stats["n_na_nonzero"] = int(np.count_nonzero(is_na & ((price != 0) | (qty != 0))))
    stats["n_market"] = int(is_mkt.sum())
    stats["n_other_type"] = int(n - is_na.sum() - is_mkt.sum())
    stats["n_nonna_nonpos"] = int(np.count_nonzero(~is_na & ~pos))
    stats["n_economic"] = int(np.count_nonzero(~is_na & pos))
    stats["n_offtick"] = int(np.count_nonzero(pos & (price % 10 != 0)))
    stats["n_qty_offstep"] = 0  # qty scale == step (0.001) by construction; inexact flagged above
    stats["n_buyer_maker"] = int(ibm.sum())
    econ = ~is_na & pos
    if econ.any():
        stats["price_min"] = float(price[econ].min() / 100)
        stats["price_max"] = float(price[econ].max() / 100)
        stats["qty_max"] = float(qty[econ].max() / 1000)
        stats["notional_usdt"] = float((price[econ].astype(np.float64) * qty[econ]).sum() / 1e5)
        stats["volume_btc"] = float(qty[econ].sum() / 1000)
    # identity / duplicates
    uniq, first_idx, counts = np.unique(tid, return_index=True, return_counts=True)
    stats["id_min"] = int(uniq[0]) if n else None
    stats["id_max"] = int(uniq[-1]) if n else None
    stats["n_unique_id"] = int(uniq.size)
    stats["n_dup_id_rows"] = int(n - uniq.size)
    n_conf = n_dup_exact = n_dup_norecv = 0
    if n - uniq.size:
        o2 = np.lexsort((recv, tid))
        same = tid[o2][1:] == tid[o2][:-1]
        a_, b_ = o2[:-1][same], o2[1:][same]
        payload_eq = ((price[a_] == price[b_]) & (qty[a_] == qty[b_]) & (tt[a_] == tt[b_]) & (evt[a_] == evt[b_])
                      & (ibm[a_] == ibm[b_]) & (ot_code[a_] == ot_code[b_]))
        conf = ~payload_eq
        n_conf = int(conf.sum())
        n_dup_exact = int(np.count_nonzero(payload_eq & (recv[a_] == recv[b_])))
        n_dup_norecv = int(np.count_nonzero(payload_eq & (recv[a_] != recv[b_])))
        for i, j in zip(a_[conf][:500], b_[conf][:500]):
            events.append(dict(kind="trade_id_conflict", t0=int(recv[i]), t1=int(recv[j]), id0=int(tid[i])))
        stats["dup_copy_delay_ms_max"] = float(((recv[b_] - recv[a_]) / 1e6).max()) if a_.size else None
    stats["n_dup_conflict"] = n_conf
    stats["n_dup_exact"] = n_dup_exact
    stats["n_dup_identical_payload_diff_recv"] = n_dup_norecv
    # id gaps (missing ids inside the file)
    gi = K.id_gaps(uniq)
    missing = (uniq[gi + 1] - uniq[gi] - 1) if gi.size else np.zeros(0, np.int64)
    stats["n_id_gaps"] = int(gi.size)
    stats["n_missing_ids"] = int(missing.sum())
    stats["max_id_gap"] = int(missing.max()) if gi.size else 0
    recv_by_id = recv[first_idx]
    tt_by_id = tt[first_idx]
    for j, i in enumerate(gi[:5000]):
        events.append(dict(kind="trade_id_gap", id0=int(uniq[i]), id1=int(uniq[i + 1]), missing=int(missing[j]),
                           t0=int(recv_by_id[i]), t1=int(recv_by_id[i + 1]),
                           m0=int(tt_by_id[i]), m1=int(tt_by_id[i + 1]),
                           dur_s=float((tt_by_id[i + 1] - tt_by_id[i]) / 1000)))
    # ordering in file order
    stats["n_id_regress_fileorder"] = int(np.count_nonzero(np.diff(tid) <= 0))
    dr = np.diff(recv)
    stats["n_recv_regress_fileorder"] = int(np.count_nonzero(dr < 0))
    stats["max_recv_regress_s"] = float(-dr.min() / NS) if dr.size and dr.min() < 0 else 0.0
    # availability order vs id order: inversions when sorted by (recv, id)
    o = np.lexsort((tid, recv))
    stats["n_id_inversion_in_recv_order"] = int(np.count_nonzero(np.diff(tid[o]) < 0))
    stats["n_tt_regress_in_id_order"] = int(np.count_nonzero(np.diff(tt_by_id) < 0))
    stats["n_evt_ne_tt"] = int(np.count_nonzero(evt != tt))
    delay = (recv - tt * MS_NS) / 1e6  # ms
    q = _q(delay)
    for name, v in zip(("p0", "p01", "p50", "p99", "p999", "max"), q):
        stats[f"delay_ms_{name}"] = v
    stats["n_neg_delay"] = int(np.count_nonzero(delay < 0))
    stats["n_synthetic_recv"] = int(np.count_nonzero(recv == evt * MS_NS))
    stats["n_recv_ms_aligned"] = int(np.count_nonzero(recv % MS_NS == 0))
    stats["n_recv_outside_hour"] = int(np.count_nonzero((recv < h0) | (recv >= h0 + HOUR_NS)))
    stats["n_tt_outside_hour"] = int(np.count_nonzero((tt * MS_NS < h0) | (tt * MS_NS >= h0 + HOUR_NS)))
    stats["recv_first"] = int(recv.min()) if n else None
    stats["recv_last"] = int(recv.max()) if n else None
    stats["tt_first"] = int(tt.min()) if n else None
    stats["tt_last"] = int(tt.max()) if n else None
    # gaps between economic trades on the availability clock
    rs = np.sort(recv)
    _gaps(rs, h0, GAP_EVENT_NS["trades"], "trades", "recv_gap", events, stats)
    flags = (pst.astype(np.uint8) | (qst.astype(np.uint8) << 3))
    staged = pa.table({
        "recv_ns": recv, "event_ms": evt, "trade_time_ms": tt, "trade_id": tid,
        "price_e2": price, "qty_e3": qty, "is_buyer_maker": ibm, "order_type": ot_code,
        "parse_flags": flags, "row": np.arange(n, dtype=np.int32),
    })
    return stats, events, staged


# ------------------------------------------------------------------------------------ orderbook
def _codes(arr, mapping, other):
    out = np.full(len(arr), other, np.int8)
    for val, code in mapping.items():
        out[pc.equal(arr, val).to_numpy(zero_copy_only=False)] = code
    return out


def audit_orderbook(table, info, h0):
    from ..book.messages import build_messages
    stats, events = {}, []
    _common(table, info, h0, stats)
    n = table.num_rows
    stats["has_order_count_col"] = "order_count" in table.column_names
    if "order_count" in table.column_names:
        stats["n_order_count_nonnull"] = int(n - table.column("order_count").null_count)
    stats["event_types"] = _vc(table.column("event_type"))
    stats["sides"] = _vc(table.column("side"))
    bm = build_messages(table)
    price, qty = bm.price, bm.qty
    _status_counts("price", bm.price_status, np.zeros(1, np.int8), stats)
    _status_counts("qty", bm.qty_status, np.zeros(1, np.int8), stats)
    stats.pop("price_maxdec"); stats.pop("qty_maxdec")
    stats["n_zero_qty_rows"] = int(np.count_nonzero(qty == 0))
    stats["n_neg_qty_rows"] = int(np.count_nonzero(qty < 0))
    stats["n_offtick"] = int(np.count_nonzero((price > 0) & (price % 10 != 0)))
    stats["n_nonpos_price_rows"] = int(np.count_nonzero((price <= 0) & (bm.side < 2)))
    m = bm.m_recv.size
    m_recv, m_evt, m_txn, m_txn_null = bm.m_recv, bm.m_evt, bm.m_txn, bm.m_txn_null
    m_et, m_U, m_u, m_pu, m_L = bm.m_et, bm.m_U, bm.m_u, bm.m_pu, bm.m_L
    is_up = m_et == K.ET_UPDATE
    is_sn = m_et == K.ET_SNAPSHOT
    stats["n_reception_groups"] = bm.n_groups
    stats["n_msgs"] = int(m)
    stats["n_msgs_update"] = int(is_up.sum())
    stats["n_msgs_snapshot"] = int(is_sn.sum())
    stats["n_msgs_other"] = int(m - is_up.sum() - is_sn.sum())
    stats["n_msgs_inconsistent"] = int(bm.m_incons.sum())
    stats["n_msgs_duplevel"] = int(bm.m_duplev.sum())
    stats["n_msgs_noop"] = int(np.count_nonzero(bm.m_nnoop > 0))
    stats["copies_max"] = int(bm.m_ncopies.max()) if m else 0
    stats["copies_mean"] = float(bm.m_ncopies.mean()) if m else 0.0
    stats["n_msgs_multi_copy"] = int(np.count_nonzero(bm.m_ncopies > 1))
    stats["n_msgs_copy_conflict"] = int(np.count_nonzero(~bm.m_ident))
    stats["n_msgs_synthetic_only"] = int(np.count_nonzero(bm.m_synth_only))
    can_rows = (bm.gstarts[bm.m_group + 1] - bm.gstarts[bm.m_group]).sum()
    stats["n_rows_in_canonical"] = int(can_rows)
    stats["n_rows_dropped_copies"] = int(n - can_rows)
    for k in np.nonzero(~bm.m_ident)[0][:500]:
        events.append(dict(kind="book_copy_conflict", t1=int(m_recv[k]), id2=int(m_u[k]), etype=int(m_et[k]),
                           ncopies=int(bm.m_ncopies[k])))
    lv = (bm.m_nbid + bm.m_nask)[is_up]
    stats["levels_per_update_p50"] = float(np.median(lv)) if lv.size else None
    stats["levels_per_update_max"] = int(lv.max()) if lv.size else None
    code, ref, last_u, mode = K.chain_classify(m_et, m_U, m_u, m_pu, m_L, -1, 0)
    for nm, c in (("ok", K.CH_OK), ("connect", K.CH_CONNECT), ("covered", K.CH_COVERED), ("dup", K.CH_DUP),
                  ("stale", K.CH_STALE), ("break", K.CH_BREAK), ("noconnect", K.CH_NOCONNECT)):
        stats[f"chain_{nm}"] = int(np.count_nonzero(code == c))
    names = {K.CH_BREAK: "book_chain_break", K.CH_STALE: "book_chain_stale",
             K.CH_NOCONNECT: "book_snapshot_noconnect", K.CH_DUP: "book_chain_dup", K.CH_COVERED: "book_covered"}
    for k in np.nonzero(np.isin(code, list(names)))[0][:5000]:
        events.append(dict(kind=names[int(code[k])], t0=int(m_recv[k - 1]) if k else None, t1=int(m_recv[k]),
                           m1=int(m_evt[k]), id0=int(ref[k]), id1=int(m_pu[k]), id2=int(m_u[k]), id3=int(m_U[k]),
                           msg_index=int(k)))
    for k in np.nonzero(is_sn)[0]:
        synthetic = bool(bm.m_synth_only[k])
        checkpoint = bool(m_recv[k] == h0 and m_txn_null[k])
        nxt = k + 1
        while nxt < m and not is_up[nxt]:
            nxt += 1
        connects = None
        if nxt < m:
            connects = bool((m_U[nxt] <= m_L[k] <= m_u[nxt]) or m_pu[nxt] == m_L[k])
        events.append(dict(kind="book_snapshot", t1=int(m_recv[k]), m1=int(m_evt[k]), id0=int(ref[k]),
                           id2=int(m_L[k]), nbid=int(bm.m_nbid[k]), nask=int(bm.m_nask[k]), nzero=int(bm.m_nzero[k]),
                           best_bid=int(bm.m_bbid[k]), best_ask=int(bm.m_bask[k]), min_bid=int(bm.m_mbid[k]),
                           max_ask=int(bm.m_mask[k]),
                           crossed=bool(bm.m_bbid[k] >= 0 and bm.m_bask[k] >= 0 and bm.m_bbid[k] >= bm.m_bask[k]),
                           synthetic=synthetic, checkpoint=checkpoint, txn_null=bool(m_txn_null[k]),
                           ncopies=int(bm.m_ncopies[k]), connects_next=connects, msg_index=int(k)))
    stats["n_snap_synthetic_only"] = int(np.count_nonzero(is_sn & bm.m_synth_only))
    stats["n_snap_checkpoint"] = int(np.count_nonzero(is_sn & (m_recv == h0) & m_txn_null))
    stats["n_snap_multi_copy"] = int(np.count_nonzero(is_sn & (bm.m_ncopies > 1)))
    stats["n_update_synthetic_only"] = int(np.count_nonzero(is_up & bm.m_synth_only))
    stats["n_update_txn_null"] = int(np.count_nonzero(is_up & m_txn_null))
    up_idx = np.nonzero(is_up)[0]
    stats["first_msg_type"] = int(m_et[0]) if m else None
    stats["first_msg_recv"] = int(m_recv[0]) if m else None
    stats["first_msg_u"] = int(m_u[0]) if m else None
    stats["first_msg_L"] = int(m_L[0]) if m else None
    if up_idx.size:
        f, l = up_idx[0], up_idx[-1]
        stats.update(first_up_U=int(m_U[f]), first_up_u=int(m_u[f]), first_up_pu=int(m_pu[f]),
                     first_up_recv=int(m_recv[f]), last_up_u=int(m_u[l]), last_up_recv=int(m_recv[l]),
                     last_up_evt=int(m_evt[l]))
    stats["last_msg_type"] = int(m_et[-1]) if m else None
    stats["last_msg_u"] = int(m_u[-1]) if m else None
    stats["final_chain_last_u"] = int(last_u)
    stats["final_chain_mode"] = int(mode)
    du = np.diff(m_u[is_up])
    stats["n_u_regress_in_avail_order"] = int(np.count_nonzero(du <= 0))
    de = np.diff(m_evt[is_up])
    stats["n_evt_regress_in_avail_order"] = int(np.count_nonzero(de < 0))
    q = _q(de.astype(np.float64)) if de.size else [None] * 6
    for name, v in zip(("p0", "p01", "p50", "p99", "p999", "max"), q):
        stats[f"msg_interval_ms_{name}"] = v
    delay = (m_recv[is_up] - m_evt[is_up] * MS_NS) / 1e6
    q = _q(delay)
    for name, v in zip(("p0", "p01", "p50", "p99", "p999", "max"), q):
        stats[f"delay_ms_{name}"] = v
    stats["n_neg_delay"] = int(np.count_nonzero(delay < 0))
    tdiff = (m_evt - m_txn)[is_up & ~m_txn_null]
    stats["evt_minus_txn_ms_p50"] = float(np.median(tdiff)) if tdiff.size else None
    stats["evt_minus_txn_ms_max"] = float(tdiff.max()) if tdiff.size else None
    stats["n_recv_outside_hour"] = int(np.count_nonzero((m_recv < h0) | (m_recv >= h0 + HOUR_NS)))
    stats["recv_first"] = int(m_recv.min()) if m else None
    stats["recv_last"] = int(m_recv.max()) if m else None
    stats["evt_first"] = int(m_evt.min()) if m else None
    stats["evt_last"] = int(m_evt.max()) if m else None
    if up_idx.size:
        _gaps(m_recv[is_up], h0, GAP_EVENT_NS["orderbook"], "orderbook", "recv_gap", events, stats)
        stats["price_min"] = float(price[price > 0].min() / 100) if np.any(price > 0) else None
        stats["price_max"] = float(price.max() / 100)
    return stats, events, None


# --------------------------------------------------------------------------------------- ticker
def audit_ticker(table, info, h0):
    stats, events = {}, []
    _common(table, info, h0, stats)
    n = table.num_rows
    recv, _ = col_i64(table, "received_time")
    evt, _ = col_i64(table, "event_time")
    cols = {}
    for c, sc in (("price_change", "tk_price"), ("price_change_percent", "tk_pct"),
                  ("weighted_average_price", "tk_price"), ("last_price", "tk_price"),
                  ("last_quantity", "tk_base"), ("open_price", "tk_price"), ("high_price", "tk_price"),
                  ("low_price", "tk_price"), ("base_asset_volume", "tk_base"), ("quote_asset_volume", "tk_quote")):
        v, st, nd = parse_decimal(table.column(c), SCALES[sc])
        _status_counts(c, st, nd, stats)
        cols[c] = v
        cols[c + "_st"] = st
    so, _ = col_i64(table, "statistics_open_time", -1)
    scl, _ = col_i64(table, "statistics_close_time", -1)
    fid, _ = col_i64(table, "first_trade_id", -1)
    lid, _ = col_i64(table, "last_trade_id", -1)
    tot, _ = col_i64(table, "total_trades", -1)
    win = (scl - so) / 1000.0
    for name, v in zip(("p0", "p50", "max"), _q(win, (0, 0.5, 1))):
        stats[f"window_s_{name}"] = v
    ok = cols["price_change_st"] == 0
    stats["n_pc_mismatch"] = int(np.count_nonzero(ok & (cols["price_change"] != cols["last_price"] - cols["open_price"])))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_calc = 100.0 * (cols["last_price"] / cols["open_price"] - 1.0)
    pct = cols["price_change_percent"] / 1e4
    stats["max_abs_pct_diff"] = float(np.nanmax(np.abs(pct - pct_calc))) if n else None
    stats["n_pct_mismatch_gt_0p001"] = int(np.count_nonzero(np.abs(pct - pct_calc) > 0.001))
    stats["n_vwap_outside_hl"] = int(np.count_nonzero((cols["weighted_average_price"] < cols["low_price"]) |
                                                        (cols["weighted_average_price"] > cols["high_price"])))
    stats["n_last_outside_hl"] = int(np.count_nonzero((cols["last_price"] < cols["low_price"]) |
                                                        (cols["last_price"] > cols["high_price"])))
    stats["n_last_id_regress"] = int(np.count_nonzero(np.diff(lid) < 0))
    d = tot - (lid - fid + 1)
    stats["trades_count_minus_idspan_min"] = int(d.min()) if n else None
    stats["trades_count_minus_idspan_max"] = int(d.max()) if n else None
    de = np.diff(evt)
    for name, v in zip(("p0", "p50", "p99", "max"), _q(de.astype(np.float64), (0, 0.5, 0.99, 1))):
        stats[f"interval_ms_{name}"] = v
    delay = (recv - evt * MS_NS) / 1e6
    for name, v in zip(("p0", "p01", "p50", "p99", "p999", "max"), _q(delay)):
        stats[f"delay_ms_{name}"] = v
    stats["n_neg_delay"] = int(np.count_nonzero(delay < 0))
    stats["n_recv_regress"] = int(np.count_nonzero(np.diff(recv) < 0))
    stats["n_recv_outside_hour"] = int(np.count_nonzero((recv < h0) | (recv >= h0 + HOUR_NS)))
    stats["n_dup_evt"] = int(n - np.unique(evt).size)
    pay = np.stack([evt, cols["last_price"], cols["base_asset_volume"], cols["quote_asset_volume"], lid, tot, so, scl], axis=1)
    stats["n_distinct_payload"] = int(np.unique(pay, axis=0).shape[0]) if n else 0
    stats["n_evt_with_conflicting_payload"] = int(stats["n_distinct_payload"] - np.unique(evt).size)
    stats["recv_first"] = int(recv.min()) if n else None
    stats["recv_last"] = int(recv.max()) if n else None
    _gaps(np.sort(recv), h0, GAP_EVENT_NS["ticker"], "ticker", "recv_gap", events, stats)
    staged = pa.table({"recv_ns": recv, "event_ms": evt,
                       **{k: v for k, v in cols.items() if not k.endswith("_st")},
                       "stat_open_ms": so, "stat_close_ms": scl, "first_trade_id": fid, "last_trade_id": lid,
                       "total_trades": tot,
                       "parse_flags": np.bitwise_or.reduce(np.stack([cols[k] for k in cols if k.endswith("_st")]), axis=0).astype(np.uint8),
                       "row": np.arange(n, dtype=np.int32)})
    return stats, events, staged


# ---------------------------------------------------------------------------------- mark_price
def audit_mark(table, info, h0):
    stats, events = {}, []
    _common(table, info, h0, stats)
    n = table.num_rows
    recv, _ = col_i64(table, "received_time")
    evt, _ = col_i64(table, "event_time")
    nft, nft_null = col_i64(table, "next_funding_time", -1)
    vals = {}
    flags = np.zeros(n, np.uint8)
    for c, sc in (("mark_price", "mark"), ("index_price", "mark"), ("estimated_settle_price", "mark"),
                  ("funding_rate", "funding")):
        v, st, nd = parse_decimal(table.column(c), SCALES[sc])
        _status_counts(c, st, nd, stats)
        vals[c] = v
        vals[c + "_null"] = st == ST_NULL
        flags |= (st != ST_OK) & (st != ST_NULL)
    stats["nft_values"] = json.dumps(sorted(set(int(x) for x in np.unique(nft) if x > 0)))
    de = np.diff(evt)
    for name, v in zip(("p0", "p50", "p99", "max"), _q(de.astype(np.float64), (0, 0.5, 0.99, 1))):
        stats[f"interval_ms_{name}"] = v
    stats["n_dup_evt"] = int(n - np.unique(evt).size)
    pay = np.stack([evt, vals["mark_price"], vals["index_price"], vals["estimated_settle_price"], vals["funding_rate"], nft], axis=1)
    stats["n_distinct_payload"] = int(np.unique(pay, axis=0).shape[0]) if n else 0
    stats["n_evt_with_conflicting_payload"] = int(stats["n_distinct_payload"] - np.unique(evt).size)
    stats["n_evt_regress"] = int(np.count_nonzero(de < 0))
    delay = (recv - evt * MS_NS) / 1e6
    for name, v in zip(("p0", "p01", "p50", "p99", "p999", "max"), _q(delay)):
        stats[f"delay_ms_{name}"] = v
    stats["n_delay_gt_3s"] = int(np.count_nonzero(delay > 3000))
    stats["n_delay_gt_10s"] = int(np.count_nonzero(delay > 10000))
    stats["n_neg_delay"] = int(np.count_nonzero(delay < 0))
    stats["n_recv_regress"] = int(np.count_nonzero(np.diff(recv) < 0))
    stats["n_recv_outside_hour"] = int(np.count_nonzero((recv < h0) | (recv >= h0 + HOUR_NS)))
    ok = (vals["mark_price"] > 0) & (vals["index_price"] > 0)
    if ok.any():
        b = 1e4 * (vals["mark_price"][ok] / vals["index_price"][ok] - 1)
        stats["basis_mark_index_bps_min"] = float(b.min())
        stats["basis_mark_index_bps_max"] = float(b.max())
    fr = vals["funding_rate"][~vals["funding_rate_null"]] / 1e8
    stats["funding_min"] = float(fr.min()) if fr.size else None
    stats["funding_max"] = float(fr.max()) if fr.size else None
    stats["n_nonpos_mark"] = int(np.count_nonzero(vals["mark_price"] <= 0))
    stats["n_nonpos_index"] = int(np.count_nonzero(vals["index_price"] <= 0))
    stats["recv_first"] = int(recv.min()) if n else None
    stats["recv_last"] = int(recv.max()) if n else None
    stats["evt_first"] = int(evt.min()) if n else None
    stats["evt_last"] = int(evt.max()) if n else None
    _gaps(np.sort(recv), h0, GAP_EVENT_NS["mark_price"], "mark_price", "recv_gap", events, stats)
    # market-clock gaps (event_time), independent of receive delays
    es = np.sort(evt) * MS_NS
    st2 = {}
    _gaps(es, h0, GAP_EVENT_NS["mark_price"], "mark_price", "evt_gap", events, st2, prefix="evt")
    stats.update(st2)
    staged = pa.table({"recv_ns": recv, "event_ms": evt, "mark_e8": vals["mark_price"], "index_e8": vals["index_price"],
                       "settle_e8": vals["estimated_settle_price"], "settle_null": vals["estimated_settle_price_null"],
                       "funding_e8": vals["funding_rate"], "funding_null": vals["funding_rate_null"],
                       "next_funding_ms": nft, "parse_flags": flags, "row": np.arange(n, dtype=np.int32)})
    return stats, events, staged


# -------------------------------------------------------------------------------- open_interest
def audit_oi(table, info, h0):
    stats, events = {}, []
    _common(table, info, h0, stats)
    n = table.num_rows
    recv, _ = col_i64(table, "received_time")
    ts, _ = col_i64(table, "timestamp")
    oi, s1, d1 = parse_decimal(table.column("sum_open_interest"), SCALES["oi"])
    oiv, s2, d2 = parse_decimal(table.column("sum_open_interest_value"), SCALES["oi_value"])
    _status_counts("oi", s1, d1, stats)
    _status_counts("oi_value", s2, d2, stats)
    stats["repairs"] = info.kv_metadata.get("chd.repairs", "")
    stats["n_unique_ts"] = int(np.unique(ts).size)
    stats["n_ts_not_5m"] = int(np.count_nonzero(ts % 300_000 != 0))
    stats["n_batches"] = int(np.unique(recv).size)
    stats["ts_first"] = int(ts.min()) if n else None
    stats["ts_last"] = int(ts.max()) if n else None
    stats["recv_first"] = int(recv.min()) if n else None
    stats["recv_last"] = int(recv.max()) if n else None
    stats["n_ts_before_hour"] = int(np.count_nonzero(ts * MS_NS < h0))
    stats["n_recv_outside_hour"] = int(np.count_nonzero((recv < h0) | (recv >= h0 + HOUR_NS)))
    delay = (recv - ts * MS_NS) / NS
    for name, v in zip(("p0", "p50", "max"), _q(delay, (0, 0.5, 1))):
        stats[f"delay_s_{name}"] = v
    ub = np.unique(recv)
    if ub.size > 1:
        db = np.diff(ub) / NS
        stats["batch_interval_s_p50"] = float(np.median(db))
        stats["batch_interval_s_max"] = float(db.max())
    stats["n_nonpos_oi"] = int(np.count_nonzero(oi <= 0))
    staged = pa.table({"recv_ns": recv, "ts_ms": ts, "oi_e8": oi, "oi_value_e8": oiv,
                       "parse_flags": (s1 | (s2 << 3)).astype(np.uint8), "row": np.arange(n, dtype=np.int32)})
    return stats, events, staged


# -------------------------------------------------------------------------------- liquidations
def audit_liq(table, info, h0):
    stats, events = {}, []
    _common(table, info, h0, stats)
    n = table.num_rows
    recv, _ = col_i64(table, "received_time")
    evt, _ = col_i64(table, "event_time")
    tt, _ = col_i64(table, "trade_time")
    for c in ("side", "order_type", "time_in_force", "order_status"):
        stats[c + "_values"] = _vc(table.column(c))
    side = _codes(table.column("side"), {"BUY": 1, "SELL": -1}, 0)
    status = _codes(table.column("order_status"), {"FILLED": 0, "PARTIALLY_FILLED": 1, "NEW": 2, "EXPIRED": 3,
                                                   "CANCELED": 4}, 9)
    otype = _codes(table.column("order_type"), {"LIMIT": 0, "MARKET": 1}, 9)
    tif = _codes(table.column("time_in_force"), {"IOC": 0, "GTC": 1, "FOK": 2, "GTX": 3}, 9)
    q, s1, d1 = parse_decimal(table.column("quantity"), SCALES["lq_qty"])
    p, s2, d2 = parse_decimal(table.column("price"), SCALES["lq_price"])
    ap, s3, d3 = parse_decimal(table.column("average_price"), SCALES["lq_avg"])
    lf, s4, d4 = parse_decimal(table.column("last_filled_quantity"), SCALES["lq_qty"])
    fq, s5, d5 = parse_decimal(table.column("filled_quantity"), SCALES["lq_qty"])
    for nm, st, nd in (("qty", s1, d1), ("price", s2, d2), ("avg_price", s3, d3), ("last_filled", s4, d4),
                       ("filled", s5, d5)):
        _status_counts(nm, st, nd, stats)
    stats["n_filled_eq_qty"] = int(np.count_nonzero(fq == q))
    stats["n_filled_lt_qty"] = int(np.count_nonzero(fq < q))
    stats["n_lastfilled_gt_filled"] = int(np.count_nonzero(lf > fq))
    stats["n_lastfilled_lt_filled"] = int(np.count_nonzero(lf < fq))
    stats["n_avg_nonpos"] = int(np.count_nonzero(ap <= 0))
    stats["n_side_unknown"] = int(np.count_nonzero(side == 0))
    if n:
        notional = ap.astype(np.float64) / 1e8 * fq / 1e3
        stats["notional_usdt"] = float(notional.sum())
        stats["notional_buy_usdt"] = float(notional[side == 1].sum())
        stats["notional_sell_usdt"] = float(notional[side == -1].sum())
    delay = (recv - evt * MS_NS) / 1e6
    for name, v in zip(("p0", "p50", "max"), _q(delay, (0, 0.5, 1))):
        stats[f"delay_ms_{name}"] = v
    stats["evt_minus_tt_ms_max"] = float((evt - tt).max()) if n else None
    stats["n_neg_delay"] = int(np.count_nonzero(delay < 0))
    stats["n_recv_outside_hour"] = int(np.count_nonzero((recv < h0) | (recv >= h0 + HOUR_NS)))
    # duplicates: exact (all fields) and identical except received_time; same order-ish key with different state
    payload = np.stack([evt, tt, side.astype(np.int64), q, p, ap, lf, fq, status.astype(np.int64)], axis=1)
    full = np.concatenate([recv[:, None], payload], axis=1)
    stats["n_dup_exact"] = int(n - np.unique(full, axis=0).shape[0]) if n else 0
    stats["n_dup_except_recv"] = int(n - np.unique(payload, axis=0).shape[0]) if n else 0
    k2 = np.stack([tt, side.astype(np.int64), q, p], axis=1)
    stats["n_same_order_key_rows"] = int(n - np.unique(k2, axis=0).shape[0]) if n else 0
    stats["recv_first"] = int(recv.min()) if n else None
    stats["recv_last"] = int(recv.max()) if n else None
    staged = pa.table({"recv_ns": recv, "event_ms": evt, "trade_time_ms": tt, "side": side, "status": status,
                       "order_type": otype, "tif": tif, "qty_e3": q, "price_e2": p, "avg_price_e8": ap,
                       "last_filled_e3": lf, "filled_e3": fq,
                       "parse_flags": (s1 | s2 | s3 | s4 | s5).astype(np.uint8), "row": np.arange(n, dtype=np.int32)})
    return stats, events, staged


AUDITORS = {"trades": audit_trades, "orderbook": audit_orderbook, "ticker": audit_ticker,
            "mark_price": audit_mark, "open_interest": audit_oi, "liquidations": audit_liq}

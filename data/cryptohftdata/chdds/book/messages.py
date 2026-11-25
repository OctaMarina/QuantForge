"""Reassemble raw L2 rows of one hourly file into complete, de-duplicated messages.

A raw row is one price level. A *reception group* is the set of rows with identical
(event_type, final_update_id, received_time); a *message* is the set of reception groups
with identical (event_type, final_update_id) — i.e. copies of the same exchange message
received several times (early 2025 files carry ~3 interleaved copies; REST snapshots can
have a second copy with synthetic received_time = event_time*1e6). A snapshot written in two halves with different
received_time (different levels, same event_time and last_update_id) is merged first: see _merge_split_snapshots.

Canonical copy = earliest non-synthetic reception (report §3.3: keep the first authentic
receipt; a synthetic receive time does not prove zero-latency availability).
Messages are returned in availability order (received_time of the canonical copy, then
first file row), which is the order of the causal replay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow.compute as pc
from numba import njit

from ..audit import kernels as K
from ..decimal import parse_decimal
from ..rawio import col_i64


@njit(cache=True)
def _gather(order, gstarts, m_group, row_start):
    out = np.empty(row_start[-1], np.int64)
    for t in range(m_group.shape[0]):
        g = m_group[t]
        s = gstarts[g]
        e = gstarts[g + 1]
        o = row_start[t]
        for j in range(e - s):
            out[o + j] = order[s + j]
    return out


def _codes(arr, mapping, other):
    out = np.full(len(arr), other, np.int8)
    for val, code in mapping.items():
        out[pc.equal(arr, val).to_numpy(zero_copy_only=False)] = code
    return out


@dataclass
class BookMessages:
    # row level
    side: np.ndarray
    price: np.ndarray       # price * 100 (int64)
    qty: np.ndarray         # quantity * 1000 (int64)
    price_status: np.ndarray
    qty_status: np.ndarray
    order: np.ndarray       # permutation of rows grouped by (etype,u,recv)
    gstarts: np.ndarray     # group boundaries into `order`
    # message level (canonical copies, availability order)
    m_group: np.ndarray
    m_recv: np.ndarray
    m_evt: np.ndarray
    m_txn: np.ndarray
    m_txn_null: np.ndarray
    m_et: np.ndarray
    m_U: np.ndarray
    m_u: np.ndarray
    m_pu: np.ndarray
    m_L: np.ndarray
    m_nbid: np.ndarray
    m_nask: np.ndarray
    m_nnoop: np.ndarray
    m_nzero: np.ndarray
    m_ncopies: np.ndarray
    m_ident: np.ndarray
    m_synth_only: np.ndarray
    m_incons: np.ndarray
    m_duplev: np.ndarray
    m_bbid: np.ndarray
    m_bask: np.ndarray
    m_mbid: np.ndarray
    m_mask: np.ndarray
    m_minrow: np.ndarray
    m_hash: np.ndarray
    n_groups: int
    n_rows: int

    def rows_of(self, t: int) -> np.ndarray:
        g = self.m_group[t]
        return self.order[self.gstarts[g]:self.gstarts[g + 1]]


def _merge_split_snapshots(recv, evt, L, etc, sdc, price):
    """The vendor can write one snapshot in two halves with different received_time; it must be identified by
    (event_time, last_update_id). Parts that are identical copies stay separate (copy logic picks the earliest real
    one). Parts that hold DIFFERENT levels are one message, available only when its last part arrived: their received
    time becomes the latest real one. A synthetic copy (received_time = event_time*1e6) proves no availability."""
    s = np.nonzero(etc == K.ET_SNAPSHOT)[0]
    if s.size == 0:
        return recv
    out = recv
    real = s[recv[s] != evt[s] * 1_000_000]
    if real.size == 0:
        return recv
    o = np.lexsort((recv[real], L[real], evt[real]))
    r = real[o]
    brk = np.nonzero((evt[r][1:] != evt[r][:-1]) | (L[r][1:] != L[r][:-1]))[0] + 1
    for g in np.split(r, brk):
        parts = np.unique(recv[g])
        if parts.size < 2:
            continue
        levels = price[g] * 8 + sdc[g]
        largest = max(int(np.unique(levels[recv[g] == x]).size) for x in parts)
        if np.unique(levels).size > largest:
            if out is recv:
                out = recv.copy()
            out[g] = parts[-1]
    return out


def build_messages(table) -> BookMessages:
    n = table.num_rows
    recv, _ = col_i64(table, "received_time")
    evt, _ = col_i64(table, "event_time")
    txn, txn_null = col_i64(table, "transaction_time", -1)
    if txn_null is None:
        txn_null = np.zeros(n, bool)
    U, _ = col_i64(table, "first_update_id", -1)
    u, _ = col_i64(table, "final_update_id", -1)
    pu, _ = col_i64(table, "prev_final_update_id", -1)
    L, _ = col_i64(table, "last_update_id", -1)
    etc = _codes(table.column("event_type"), {"update": K.ET_UPDATE, "snapshot": K.ET_SNAPSHOT}, K.ET_OTHER)
    sdc = _codes(table.column("side"), {"bid": K.SD_BID, "ask": K.SD_ASK, "noop": K.SD_NOOP}, K.SD_OTHER)
    price, pst, _ = parse_decimal(table.column("price"), 2)
    qty, qst, _ = parse_decimal(table.column("quantity"), 3)
    recv = _merge_split_snapshots(recv, evt, L, etc, sdc, price)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, recv, u, etc))
    (gstarts, g_nbid, g_nask, g_nnoop, g_nzero, g_hash, g_incons, g_duplev, g_minrow,
     g_bbid, g_bask, g_mbid, g_mask, msg_group, msg_ncopies, msg_ident, msg_synth) = K.group_copies(
        order, recv, etc, U, u, pu, L, evt, txn, sdc, price, qty)
    first_row = order[gstarts[msg_group]]
    m_recv = recv[first_row]
    m_minrow = g_minrow[msg_group]
    av = np.lexsort((m_minrow, m_recv))
    msg_group = msg_group[av]
    first_row = first_row[av]
    g = msg_group
    return BookMessages(
        side=sdc, price=price, qty=qty, price_status=pst, qty_status=qst, order=order, gstarts=gstarts,
        m_group=g, m_recv=recv[first_row], m_evt=evt[first_row], m_txn=txn[first_row],
        m_txn_null=txn_null[first_row], m_et=etc[first_row], m_U=U[first_row], m_u=u[first_row],
        m_pu=pu[first_row], m_L=L[first_row], m_nbid=g_nbid[g], m_nask=g_nask[g], m_nnoop=g_nnoop[g],
        m_nzero=g_nzero[g], m_ncopies=msg_ncopies[av], m_ident=msg_ident[av], m_synth_only=msg_synth[av],
        m_incons=g_incons[g], m_duplev=g_duplev[g], m_bbid=g_bbid[g], m_bask=g_bask[g], m_mbid=g_mbid[g],
        m_mask=g_mask[g], m_minrow=g_minrow[g], m_hash=g_hash[g], n_groups=int(gstarts.size - 1), n_rows=n)


def gather_rows(bm: BookMessages):
    """Flatten canonical-copy rows in message (availability) order: returns (row_start, side, idx, qty, offtick)."""
    sizes = bm.gstarts[bm.m_group + 1] - bm.gstarts[bm.m_group]
    row_start = np.zeros(sizes.size + 1, np.int64)
    np.cumsum(sizes, out=row_start[1:])
    idx_rows = _gather(bm.order, bm.gstarts, bm.m_group, row_start)
    price = bm.price[idx_rows]
    return (row_start, bm.side[idx_rows].astype(np.int8), (price // 10).astype(np.int64),
            bm.qty[idx_rows].astype(np.int64), (price % 10 != 0))

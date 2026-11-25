"""Slow, obviously-correct reference L2 replay (pure Python dicts) for differential testing.

Consumes the same application sequence as the numba kernel (chain.resolve output) and exposes the
book state; used by tests and diagnostics, never by the production pipeline.
"""
from __future__ import annotations

from .chain import Q_ANCHOR, Q_INVALIDATE, Q_UPDATE, Q_VERIFY


class RefBook:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.anchored = False
        self.cert_lo = None   # bids certified for idx >= cert_lo
        self.cert_hi = None   # asks certified for idx <= cert_hi
        self.log = []

    def load_snapshot(self, rows):
        self.bids = {}
        self.asks = {}
        for side, idx, qty, off in rows:
            if off:
                continue
            if side == 0 and qty > 0:
                self.bids[idx] = qty
            elif side == 1 and qty > 0:
                self.asks[idx] = qty
        self.anchored = True
        self.cert_lo = min(i for s, i, q, o in rows if s == 0 and not o)
        self.cert_hi = max(i for s, i, q, o in rows if s == 1 and not o)

    def apply(self, rows):
        for side, idx, qty, off in rows:
            if off:
                continue
            book = self.bids if side == 0 else self.asks if side == 1 else None
            if book is None:
                continue
            if qty == 0:
                book.pop(idx, None)
            else:
                book[idx] = qty

    def best(self):
        bb = max(self.bids) if self.bids else None
        ba = min(self.asks) if self.asks else None
        return bb, ba

    def top(self, k):
        b = sorted(self.bids.items(), reverse=True)[:k]
        a = sorted(self.asks.items())[:k]
        return b, a


def rows_of(batch, m):
    rs = batch["row_start"]
    return [(int(batch["side"][r]), int(batch["idx"][r]), int(batch["qty"][r]), bool(batch["off"][r]))
            for r in range(rs[m], rs[m + 1])]


def compare_snapshot(book: RefBook, snap_rows, exclude_rows=()):
    """Mismatches where both the snapshot and the replayed state are certified, excluding levels of the
    overlay message. On success the snapshot is adopted outside our certified domain (domain extension)."""
    ex = {(s, i) for s, i, q, o in exclude_rows}
    sb = {i: q for s, i, q, o in snap_rows if s == 0 and not o and q > 0}
    sa = {i: q for s, i, q, o in snap_rows if s == 1 and not o and q > 0}
    lo_b = min(i for s, i, q, o in snap_rows if s == 0 and not o)
    hi_a = max(i for s, i, q, o in snap_rows if s == 1 and not o)
    cb = max(lo_b, book.cert_lo)
    ca = min(hi_a, book.cert_hi)
    mis = []
    for i in {i for i in sb if i >= cb} | {i for i in book.bids if i >= cb}:
        if (0, i) in ex:
            continue
        if book.bids.get(i, 0) != sb.get(i, 0):
            mis.append((0, i, book.bids.get(i, 0), sb.get(i, 0)))
    for i in {i for i in sa if i <= ca} | {i for i in book.asks if i <= ca}:
        if (1, i) in ex:
            continue
        if book.asks.get(i, 0) != sa.get(i, 0):
            mis.append((1, i, book.asks.get(i, 0), sa.get(i, 0)))
    if not mis:
        for i in [i for i in book.bids if lo_b <= i < book.cert_lo and (0, i) not in ex]:
            del book.bids[i]
        for i in [i for i in book.asks if book.cert_hi < i <= hi_a and (1, i) not in ex]:
            del book.asks[i]
        for i, q in sb.items():
            if i < book.cert_lo and (0, i) not in ex:
                book.bids[i] = q
        for i, q in sa.items():
            if i > book.cert_hi and (1, i) not in ex:
                book.asks[i] = q
        book.cert_lo = min(book.cert_lo, lo_b)
        book.cert_hi = max(book.cert_hi, hi_a)
    return mis


def run_reference(batches_and_seqs, on_verify=None):
    """batches_and_seqs: iterable of (batch, seq_msg, seq_kind, seq_aux, seq_time)."""
    book = RefBook()
    results = []
    for batch, seq_msg, seq_kind, seq_aux, seq_time in batches_and_seqs:
        for m, k, aux, t in zip(seq_msg, seq_kind, seq_aux, seq_time):
            if k == Q_INVALIDATE:
                book.anchored = False
            elif k == Q_ANCHOR:
                book.load_snapshot(rows_of(batch, m))
            elif k == Q_VERIFY:
                ex = rows_of(batch, aux) if aux >= 0 else []
                mis = compare_snapshot(book, rows_of(batch, m), ex) if book.anchored else None
                results.append((int(t), mis))
                if on_verify:
                    on_verify(int(t), mis, book)
                if mis:
                    book.load_snapshot(rows_of(batch, m))
                    if aux >= 0:
                        book.apply(ex)
            elif k == Q_UPDATE:
                if book.anchored:
                    book.apply(rows_of(batch, m))
    return book, results

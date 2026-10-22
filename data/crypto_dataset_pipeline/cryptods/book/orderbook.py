"""Exact L2 book with venue synchronisation rules and band maintenance.

Verified rules (2026-10-20):
* Binance spot diff depth (web-socket-streams.md, "How to manage a local order book correctly"):
  buffer events, take REST snapshot (lastUpdateId), drop events with u <= lastUpdateId, the first
  applied event must satisfy U <= lastUpdateId+1 <= u; afterwards an event with U > book_id+1 means
  missed events -> discard the book and resync; quantities are absolute; 0 removes the level; levels
  beyond the snapshot depth (max 5000/side) are unknown until they change.
* Binance USD-M futures: same, but continuity is checked with pu == previous u; the first event must
  satisfy U <= lastUpdateId <= u.
* Bybit v5 orderbook: snapshot resets the book, deltas set absolute sizes, size 0 deletes, a new
  snapshot (incl. u == 1 after service restart) overwrites; RPI orders are excluded from the feed.
* Tardis incremental_book_L2: absolute sizes; snapshot rows reset the book; no sequence numbers.

The book stores exact integer prices/quantities.  Bands are maintained incrementally and exactly:
for a band of k bps around the mid m = (a+b)/2, a bid level p is inside iff
2e4 * p >= (a+b) * (1e4 - k), an ask level iff 2e4 * p <= (a+b) * (1e4 + k).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sortedcontainers import SortedDict


class BookGap(Exception):
    pass


@dataclass
class BandState:
    k: int
    thr: int | None = None        # bid: lowest included price; ask: highest included price
    depth: int = 0                # exact notional (instrument notional scale)
    added: int = 0                # same-price increases inside the band (since last reset)
    withdrawn: int = 0            # same-price decreases inside the band (since last reset)


@dataclass
class L2Book:
    venue_rule: str               # binance_spot | binance_futures | bybit | tardis | snapshot_only
    notional_fn: object           # (price_i, qty_i) -> notional int
    bands_bps: tuple = (10, 25, 50)
    bids: SortedDict = field(default_factory=SortedDict)
    asks: SortedDict = field(default_factory=SortedDict)
    synced: bool = False
    update_id: int | None = None
    last_ts_us: int | None = None
    complete_bid_floor: int | None = None   # bid prices >= floor are known completely (None = all)
    complete_ask_cap: int | None = None     # ask prices <= cap are known completely
    awaiting_first: bool = False
    resyncs: int = 0
    gaps: list = field(default_factory=list)
    pending: deque = field(default_factory=lambda: deque(maxlen=50_000))

    def __post_init__(self):
        self.bid_bands = {k: BandState(k) for k in self.bands_bps}
        self.ask_bands = {k: BandState(k) for k in self.bands_bps}

    # ------------------------------------------------------------- basic state
    def best_bid(self):
        return self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self):
        return self.asks.peekitem(0)[0] if self.asks else None

    def crossed(self) -> bool:
        b, a = self.best_bid(), self.best_ask()
        return b is not None and a is not None and b >= a

    def valid(self) -> bool:
        """Synced, bridged by the first diff (Binance procedure), both sides present, not crossed."""
        return (self.synced and not self.awaiting_first and bool(self.bids) and bool(self.asks)
                and not self.crossed())

    def invalidate(self, reason: str, ts_us: int | None):
        self.synced = False
        self.pending.clear()
        self.gaps.append((ts_us, reason))
        self.bids.clear()
        self.asks.clear()
        for bs in list(self.bid_bands.values()) + list(self.ask_bands.values()):
            bs.thr, bs.depth = None, 0

    # ------------------------------------------------------------- snapshot / updates
    def apply_snapshot(self, bids, asks, update_id, ts_us, depth_limit: int | None = None):
        """``depth_limit``: number of levels per side requested; if a side is full, prices beyond its
        deepest level are unknown (completeness bound)."""
        self.bids.clear()
        self.asks.clear()
        for p, q in bids:
            if q > 0:
                self.bids[p] = q
        for p, q in asks:
            if q > 0:
                self.asks[p] = q
        self.update_id = update_id
        self.last_ts_us = ts_us
        self.synced = True
        self.awaiting_first = self.venue_rule in ("binance_spot", "binance_futures")
        self.complete_bid_floor = None
        self.complete_ask_cap = None
        if depth_limit is not None:
            if len(self.bids) >= depth_limit:
                self.complete_bid_floor = self.bids.peekitem(0)[0]
            if len(self.asks) >= depth_limit:
                self.complete_ask_cap = self.asks.peekitem(-1)[0]
        self._rebuild_bands()
        # Binance procedure: diffs buffered while unsynced are applied now (u <= lastUpdateId dropped)
        buffered = list(self.pending)
        self.pending.clear()
        for (b_, a_, ts_, U_, u_, pu_) in buffered:
            if u_ is not None and update_id is not None and u_ <= update_id:
                continue
            self.apply_update(b_, a_, ts_, U_, u_, pu_)

    def apply_update(self, bids, asks, ts_us, U=None, u=None, pu=None):
        """Apply a diff message.  Returns 'applied' | 'ignored' | 'buffered'.  Raises BookGap on missed
        events.  While unsynced, Binance diffs are buffered for the next snapshot (official procedure)."""
        if not self.synced:
            if self.venue_rule in ("binance_spot", "binance_futures"):
                self.pending.append((bids, asks, ts_us, U, u, pu))
                return "buffered"
            return "ignored"
        rule = self.venue_rule
        if rule == "binance_spot":
            if u <= self.update_id:
                return "ignored"
            if self.awaiting_first:
                if not (U <= self.update_id + 1 <= u):
                    raise BookGap(f"first event [{U},{u}] does not cover lastUpdateId+1={self.update_id + 1}")
                self.awaiting_first = False
            elif U > self.update_id + 1:
                raise BookGap(f"missed events: U={U} > book_id+1={self.update_id + 1}")
        elif rule == "binance_futures":
            if u < self.update_id:
                return "ignored"
            if self.awaiting_first:
                if not (U <= self.update_id <= u):
                    raise BookGap(f"first event [{U},{u}] does not contain lastUpdateId={self.update_id}")
                self.awaiting_first = False
            elif pu != self.update_id:
                raise BookGap(f"pu={pu} != previous u={self.update_id}")
        self._apply_levels(bids, asks)
        if u is not None:
            self.update_id = u
        self.last_ts_us = ts_us if self.last_ts_us is None else max(self.last_ts_us, ts_us)
        return "applied"

    def _apply_levels(self, bids, asks):
        bb = self.best_bid()
        ba = self.best_ask()
        # (1) same-price changes, attributed with the pre-message band thresholds
        for side_book, bands, levels, is_bid in ((self.bids, self.bid_bands, bids, True),
                                                 (self.asks, self.ask_bands, asks, False)):
            for p, q in levels:
                old = side_book.get(p, 0)
                if q > 0:
                    side_book[p] = q
                elif old:
                    del side_book[p]
                if old == q:
                    continue
                dn = self.notional_fn(p, q) - self.notional_fn(p, old)
                for bs in bands.values():
                    if bs.thr is None:
                        continue
                    inside = p >= bs.thr if is_bid else p <= bs.thr
                    if inside:
                        bs.depth += dn
                        if dn > 0:
                            bs.added += dn
                        else:
                            bs.withdrawn += -dn
        # (2) mid move -> move the band boundaries exactly
        nb, na = self.best_bid(), self.best_ask()
        if nb is None or na is None:
            for bs in list(self.bid_bands.values()) + list(self.ask_bands.values()):
                bs.thr, bs.depth = None, 0
            return
        if (nb, na) != (bb, ba) or any(bs.thr is None for bs in self.bid_bands.values()):
            self._move_bands(nb, na)

    def _thr(self, b, a, k, is_bid):
        s = a + b
        if is_bid:   # smallest integer p with 2e4 p >= s (1e4 - k)
            num = s * (10_000 - k)
            return -((-num) // 20_000)
        num = s * (10_000 + k)  # largest integer p with 2e4 p <= s (1e4 + k)
        return num // 20_000

    def _move_bands(self, b, a):
        for k, bs in self.bid_bands.items():
            new = self._thr(b, a, k, True)
            if bs.thr is None:
                bs.thr = new
                bs.depth = sum(self.notional_fn(p, q) for p, q in self.bids.items()[self.bids.bisect_left(new):])
                continue
            if new < bs.thr:
                for p in self.bids.irange(new, bs.thr - 1):
                    bs.depth += self.notional_fn(p, self.bids[p])
            elif new > bs.thr:
                for p in self.bids.irange(bs.thr, new - 1):
                    bs.depth -= self.notional_fn(p, self.bids[p])
            bs.thr = new
        for k, bs in self.ask_bands.items():
            new = self._thr(b, a, k, False)
            if bs.thr is None:
                bs.thr = new
                bs.depth = sum(self.notional_fn(p, q) for p, q in self.asks.items()[: self.asks.bisect_right(new)])
                continue
            if new > bs.thr:
                for p in self.asks.irange(bs.thr + 1, new):
                    bs.depth += self.notional_fn(p, self.asks[p])
            elif new < bs.thr:
                for p in self.asks.irange(new + 1, bs.thr):
                    bs.depth -= self.notional_fn(p, self.asks[p])
            bs.thr = new

    def _rebuild_bands(self):
        for bs in list(self.bid_bands.values()) + list(self.ask_bands.values()):
            bs.thr, bs.depth = None, 0
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            self._move_bands(b, a)

    # ------------------------------------------------------------- queries
    def band_covered(self, k: int, is_bid: bool) -> bool:
        bs = (self.bid_bands if is_bid else self.ask_bands)[k]
        if bs.thr is None:
            return False
        if is_bid:
            return self.complete_bid_floor is None or bs.thr >= self.complete_bid_floor
        return self.complete_ask_cap is None or bs.thr <= self.complete_ask_cap

    def coverage_bps(self, is_bid: bool):
        """Distance from mid (bps) up to which the side is known completely (inf = whole book)."""
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        m2 = a + b
        if is_bid:
            if self.complete_bid_floor is None:
                return float("inf")
            return 1e4 * (m2 - 2 * self.complete_bid_floor) / m2
        if self.complete_ask_cap is None:
            return float("inf")
        return 1e4 * (2 * self.complete_ask_cap - m2) / m2

    def walk(self, qty_i: int, is_buy: bool):
        """Static market order walk consuming levels in price order; partial use of the last level.
        Returns (sum of price_i*qty_i, filled qty_i, levels used) or None if depth (within the known-
        complete region) is insufficient."""
        side = self.asks if is_buy else self.bids
        keys = side.keys() if is_buy else reversed(side.keys())
        remaining = qty_i
        cost = 0
        used = 0
        for p in keys:
            if is_buy and self.complete_ask_cap is not None and p > self.complete_ask_cap:
                return None
            if (not is_buy) and self.complete_bid_floor is not None and p < self.complete_bid_floor:
                return None
            q = side[p]
            take = q if q < remaining else remaining
            cost += p * take
            remaining -= take
            used += 1
            if remaining == 0:
                return cost, qty_i, used
        return None

    def levels_within(self, lo_price: int, hi_price: int, is_bid: bool):
        side = self.bids if is_bid else self.asks
        return [(p, side[p]) for p in side.irange(lo_price, hi_price)]

    def reset_band_flows(self):
        for bs in list(self.bid_bands.values()) + list(self.ask_bands.values()):
            bs.added = 0
            bs.withdrawn = 0

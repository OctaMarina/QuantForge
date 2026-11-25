"""L2 liquidity features (task section 7, 9.4, 9.5): spread, depth bands, imbalance, static
market-order cost, concentrations ("walls") and their observed persistence, net withdrawal/refill,
and explicit L2 x aggressive-flow interactions.

Everything describes DISPLAYED liquidity reconstructed from the public feed: no hidden orders, no
order identities, no participant intent.  State at a query time q = book after all messages with
exchange time < q (other stream than trades -> equal timestamps are treated as not yet known).
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .. import timeutil as tu
from ..book.orderbook import L2Book, BookGap
from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .context import Ctx
from .stats import safe_div

KIND_ORDER = {"end": 0, "start": 1, "decision": 2, "sample": 3}


class BookEngine:
    def __init__(self, spec, cfg_book: dict, venue_rule: str, bar_start, bar_end, decision,
                 wall_bin_size_i: int | None):
        self.spec = spec
        self.cb = cfg_book
        self.bands = [int(k) for k in cfg_book["bands_bps"]]
        lin = spec.contract_type == "linear"
        self.linear = lin
        self.book = L2Book(venue_rule=venue_rule, bands_bps=tuple(self.bands),
                           notional_fn=(lambda p, q: p * q) if lin else (lambda p, q: q * int(spec.contract_size) * spec.price_scale))
        self.S = np.asarray(bar_start, dtype=np.int64)
        self.E = np.asarray(bar_end, dtype=np.int64)
        self.D = np.asarray(decision, dtype=np.int64)
        n = len(self.S)
        self.n = n
        self.max_age = tu.parse_duration(cfg_book["max_book_age"])
        w = cfg_book["wall"]
        self.wall_bin = wall_bin_size_i
        self.wall_maxd = float(w["max_distance_bps"])
        self.wall_mode = w["threshold_mode"]
        self.wall_q = float(w["quantile"])
        self.wall_abs = w.get("absolute_notional")
        self.wall_mult = float(w.get("median_multiple", 5.0))
        self.hist_window = tu.parse_duration(w["history_window"])
        self.sample_every = tu.parse_duration(w["sample_every"])
        self.min_hist = int(w["min_history_samples"])
        self.match_tol = int(w["match_tolerance_bins"])
        self.hist = {True: deque(), False: deque()}
        self.tracked = {True: [], False: []}
        # outputs
        f = lambda: np.full(n, np.nan)
        self.o = {"valid_dec": np.zeros(n, bool), "valid_start": np.zeros(n, bool), "valid_end": np.zeros(n, bool),
                  "age_dec": np.full(n, -1, np.int64), "bid": f(), "ask": f(), "mid_dec": f(), "mid_start": f(), "mid_end": f()}
        for k in self.bands:
            for s in ("bid", "ask"):
                for t in ("dec", "start", "end"):
                    self.o[f"d_{s}_{k}_{t}"] = f()
                self.o[f"twa_{s}_{k}"] = f()
                self.o[f"twcov_{s}_{k}"] = np.zeros(n)
                self.o[f"add_{s}_{k}"] = f()
                self.o[f"wd_{s}_{k}"] = f()
        self.o["twa_spread_bps"] = f()
        self.o["tw_valid_frac"] = np.zeros(n)
        self.o["cov_bid_bps"] = f()
        self.o["cov_ask_bps"] = f()
        self.sizes_base = [str(x) for x in cfg_book["sizes_base"]]
        self.sizes_quote = [str(x) for x in cfg_book["sizes_quote"]]
        for sz in self.sizes_base + [f"q{x}" for x in self.sizes_quote]:
            self.o[f"slip_buy_{sz}"] = f()
            self.o[f"slip_sell_{sz}"] = f()
            self.o[f"slipr_buy_{sz}"] = np.full(n, CODE["missing_source_data"], np.int8)
            self.o[f"slipr_sell_{sz}"] = np.full(n, CODE["missing_source_data"], np.int8)
        for s in ("bid", "ask"):
            self.o[f"wall_dist_{s}"] = f()
            self.o[f"wall_size_{s}"] = f()
            self.o[f"wall_age_{s}"] = f()
            self.o[f"wallr_{s}"] = np.full(n, CODE["missing_source_data"], np.int8)
        # integration state
        self.cur_bar = -1
        self.t_last = None
        self.acc = None

    # ---------------------------------------------------------------- helpers
    def _mid(self):
        b, a = self.book.best_bid(), self.book.best_ask()
        return (a + b) / 2 / self.spec.price_scale

    def _state_ok(self, q):
        bk = self.book
        return bk.valid() and bk.last_ts_us is not None and (q - bk.last_ts_us) <= self.max_age

    def _depth(self, k, is_bid):
        bk = self.book
        if not bk.band_covered(k, is_bid):
            return np.nan
        bs = (bk.bid_bands if is_bid else bk.ask_bands)[k]
        return bs.depth / self.spec.notional_scale

    def _integrate_to(self, t):
        """Integrate the piecewise-constant book state over [t_last, t).  Only the part of the interval
        during which the book is synced, uncrossed and not older than max_book_age counts."""
        if self.cur_bar < 0 or self.acc is None or self.t_last is None:
            self.t_last = t
            return
        bk = self.book
        if t > self.t_last and bk.valid() and bk.last_ts_us is not None:
            hi = min(t, bk.last_ts_us + self.max_age + 1)
            dt = hi - self.t_last
            if dt > 0:
                self.acc["valid"] += dt
                b, a = bk.best_bid(), bk.best_ask()
                self.acc["spread"] += dt * (1e4 * (a - b) / ((a + b) / 2))
                for k in self.bands:
                    for s, isb in (("bid", True), ("ask", False)):
                        if bk.band_covered(k, isb):
                            bs = (bk.bid_bands if isb else bk.ask_bands)[k]
                            self.acc[f"{s}{k}"] += dt * bs.depth
                            self.acc[f"c{s}{k}"] += dt
        self.t_last = t

    def _begin_bar(self, i, t):
        self.cur_bar = i
        self.t_last = t
        self.acc = {"valid": 0, "spread": 0.0}
        for k in self.bands:
            for s in ("bid", "ask"):
                self.acc[f"{s}{k}"] = 0
                self.acc[f"c{s}{k}"] = 0
        self.book.reset_band_flows()
        self.flows_valid = self.book.valid()

    def _end_bar(self, i, t):
        if self.cur_bar != i or self.acc is None:
            return
        dur = t - self.S[i]
        if dur > 0:
            self.o["tw_valid_frac"][i] = self.acc["valid"] / dur
            if self.acc["valid"] == dur:
                self.o["twa_spread_bps"][i] = self.acc["spread"] / dur
            for k in self.bands:
                for s in ("bid", "ask"):
                    cv = self.acc[f"c{s}{k}"]
                    self.o[f"twcov_{s}_{k}"][i] = cv / dur
                    if cv == dur:  # full-bar variant only when the band was covered for the whole bar
                        self.o[f"twa_{s}_{k}"][i] = self.acc[f"{s}{k}"] / dur / self.spec.notional_scale
        if self.flows_valid and self.book.valid() and self.acc["valid"] == dur:
            for k in self.bands:
                for s, isb in (("bid", True), ("ask", False)):
                    bs = (self.book.bid_bands if isb else self.book.ask_bands)[k]
                    self.o[f"add_{s}_{k}"][i] = bs.added / self.spec.notional_scale
                    self.o[f"wd_{s}_{k}"][i] = bs.withdrawn / self.spec.notional_scale
        self.cur_bar = -1
        self.acc = None

    def _snapshot_point(self, i, t, kind):
        ok = self._state_ok(t)
        if kind == "start":
            self.o["valid_start"][i] = ok
            if ok:
                self.o["mid_start"][i] = self._mid()
                for k in self.bands:
                    self.o[f"d_bid_{k}_start"][i] = self._depth(k, True)
                    self.o[f"d_ask_{k}_start"][i] = self._depth(k, False)
        elif kind == "end":
            self.o["valid_end"][i] = ok
            if ok:
                self.o["mid_end"][i] = self._mid()
                for k in self.bands:
                    self.o[f"d_bid_{k}_end"][i] = self._depth(k, True)
                    self.o[f"d_ask_{k}_end"][i] = self._depth(k, False)
        elif kind == "decision":
            bk = self.book
            if bk.last_ts_us is not None:
                self.o["age_dec"][i] = t - bk.last_ts_us
            self.o["valid_dec"][i] = ok
            if not ok:
                return
            ps = self.spec.price_scale
            b, a = bk.best_bid(), bk.best_ask()
            self.o["bid"][i] = b / ps
            self.o["ask"][i] = a / ps
            self.o["mid_dec"][i] = (a + b) / 2 / ps
            self.o["cov_bid_bps"][i] = bk.coverage_bps(True)
            self.o["cov_ask_bps"][i] = bk.coverage_bps(False)
            for k in self.bands:
                self.o[f"d_bid_{k}_dec"][i] = self._depth(k, True)
                self.o[f"d_ask_{k}_dec"][i] = self._depth(k, False)
            if self.linear:
                self._costs(i, b, a)
            self._walls(i, t, record=True)

    def _costs(self, i, b, a):
        from decimal import Decimal
        qs = self.spec.qty_scale
        m2 = a + b
        items = [(sz, int(Decimal(sz) * qs)) for sz in self.sizes_base]
        for sz in self.sizes_quote:
            # reference notional N -> quantity q = N / mid (rounded down to the qty scale); the actual
            # amount spent is VWAP*q, which differs from N by the slippage
            qref = int(Decimal(sz) * qs * 2 * self.spec.price_scale / m2)
            items.append((f"q{sz}", qref))
        for key, qty_i in items:
            if qty_i <= 0:
                continue
            for side, is_buy in (("buy", True), ("sell", False)):
                w = self.book.walk(qty_i, is_buy)
                if w is None:
                    self.o[f"slipr_{side}_{key}"][i] = CODE["insufficient_book_depth"]
                    continue
                cost, filled, _ = w
                # VWAP / m = 2*cost / (filled * (a+b))   (exact rational -> float at the end)
                ratio = (2 * cost) / (filled * m2)
                v = 1e4 * (ratio - 1) if is_buy else 1e4 * (1 - ratio)
                self.o[f"slip_{side}_{key}"][i] = v
                self.o[f"slipr_{side}_{key}"][i] = VALID

    # ---------------------------------------------------------------- walls
    def _bins(self, is_bid):
        bk = self.book
        b, a = bk.best_bid(), bk.best_ask()
        m2 = a + b
        if is_bid:
            lo = -((-(m2 * (10_000 - int(self.wall_maxd)))) // 20_000)
            floor = bk.complete_bid_floor
            partial = floor is not None and floor > lo
            lev = bk.levels_within(max(lo, floor) if floor is not None else lo, b, True)
        else:
            hi = (m2 * (10_000 + int(self.wall_maxd))) // 20_000
            cap = bk.complete_ask_cap
            partial = cap is not None and cap < hi
            lev = bk.levels_within(a, min(hi, cap) if cap is not None else hi, False)
        acc: dict = {}
        pw: dict = {}   # sum of price * notional per bin (for the notional-weighted price of a zone)
        for p, q in lev:
            bi = p // self.wall_bin
            nv = self.book.notional_fn(p, q)
            acc[bi] = acc.get(bi, 0) + nv
            pw[bi] = pw.get(bi, 0) + p * nv
        return acc, pw, partial

    def _walls(self, i, t, record: bool):
        if self.wall_bin is None:
            for s in ("bid", "ask"):
                if record:
                    self.o[f"wallr_{s}"][i] = CODE["not_applicable"]
            return
        mid2 = self.book.best_ask() + self.book.best_bid()
        for is_bid, s in ((True, "bid"), (False, "ask")):
            acc, near, partial = self._bins(is_bid)  # near = price*notional sums per bin
            vals = np.array(list(acc.values()), dtype=np.float64) / self.spec.notional_scale
            # threshold from strictly earlier samples only
            hist = self.hist[is_bid]
            while hist and hist[0][0] < t - self.hist_window:
                hist.popleft()
            if self.wall_mode == "absolute":
                thr = float(self.wall_abs) if self.wall_abs is not None else None
                enough = thr is not None
            elif self.wall_mode == "snapshot_median_multiple":
                # relative to the CURRENT observable book on this side (causal; no absorption of a
                # persistent concentration into its own baseline)
                enough = len(vals) >= 3
                thr = float(np.median(vals)) * self.wall_mult if enough else None
            else:
                enough = len(hist) >= self.min_hist
                thr = float(np.quantile(np.concatenate([h[1] for h in hist]), self.wall_q)) if enough and any(len(h[1]) for h in hist) else None
                enough = enough and thr is not None
            zones = []
            if enough:
                elig = sorted(bi for bi, v in acc.items() if v / self.spec.notional_scale > thr)  # strict: ties are not walls
                cur = None
                for bi in elig:
                    if cur and bi == cur[1] + 1:
                        cur[1] = bi
                        cur[2] += acc[bi]
                    else:
                        if cur:
                            zones.append(cur)
                        cur = [bi, bi, acc[bi]]
                if cur:
                    zones.append(cur)
            # persistence: match with tracked zones (overlap within tolerance), age = observed so far
            newtrack = []
            for z in zones:
                first = t
                for (lo, hi, fs) in self.tracked[is_bid]:
                    if z[0] <= hi + self.match_tol and z[1] >= lo - self.match_tol:
                        first = min(first, fs)
                newtrack.append((z[0], z[1], first))
            self.tracked[is_bid] = newtrack
            if record:
                if not enough:
                    self.o[f"wallr_{s}"][i] = CODE["insufficient_history"]
                elif not zones:
                    self.o[f"wallr_{s}"][i] = CODE["insufficient_book_depth"] if partial else CODE["no_observation"]
                else:
                    zbest = max(zones, key=lambda z: z[1]) if is_bid else min(zones, key=lambda z: z[0])
                    bins = range(zbest[0], zbest[1] + 1)
                    px = sum(near[b] for b in bins if b in near) / zbest[2]   # notional-weighted zone price
                    dist = 1e4 * ((mid2 - 2 * px) / mid2 if is_bid else (2 * px - mid2) / mid2)
                    self.o[f"wall_dist_{s}"][i] = dist
                    self.o[f"wall_size_{s}"][i] = zbest[2] / self.spec.notional_scale
                    first = [fs for (lo, hi, fs) in newtrack if lo == zbest[0] and hi == zbest[1]][0]
                    self.o[f"wall_age_{s}"][i] = (t - first) / tu.SEC
                    self.o[f"wallr_{s}"][i] = VALID
            hist.append((t, vals))

    # ---------------------------------------------------------------- main loop
    def run(self, messages):
        """``messages``: iterable of dicts sorted by ts_us (see book adapters).  Checkpoints (bar start,
        bar end, decision) and wall-history samples at time q are evaluated on the book state built
        from messages with ts < q; a message at exactly q is applied afterwards (tie rule)."""
        cps = []
        for i in range(self.n):
            cps.append((int(self.S[i]), KIND_ORDER["start"], i, "start"))
            cps.append((int(self.E[i]), KIND_ORDER["end"], i, "end"))
            cps.append((int(self.D[i]), KIND_ORDER["decision"], i, "decision"))
        heapq.heapify(cps)
        last_cp = max((c[0] for c in cps), default=-1)
        state = {"next_sample": None}

        def process_until(T):
            while True:
                ns = state["next_sample"]
                tc = cps[0][0] if cps else None
                cand = [x for x in (tc, ns) if x is not None and x <= T and x <= last_cp]
                if not cand:
                    return
                t = min(cand)
                if tc is not None and tc == t:
                    _, _, i, kind = heapq.heappop(cps)
                    self._integrate_to(t)
                    if kind == "start":
                        self._snapshot_point(i, t, "start")
                        self._begin_bar(i, t)
                    elif kind == "end":
                        self._snapshot_point(i, t, "end")
                        self._end_bar(i, t)
                    else:
                        self._snapshot_point(i, t, "decision")
                else:
                    self._integrate_to(t)
                    if self.book.valid() and self.wall_bin is not None:
                        self._walls(-1, t, record=False)
                    state["next_sample"] = ns + self.sample_every

        for msg in messages:
            ts = int(msg["ts_us"])
            process_until(ts)
            if state["next_sample"] is None:
                state["next_sample"] = (ts // self.sample_every + 1) * self.sample_every
            self._integrate_to(ts)
            typ = msg["type"]
            try:
                if typ == "snapshot":
                    self.book.apply_snapshot(msg["bids"], msg["asks"], msg.get("u"), ts, msg.get("depth_limit"))
                    self.tracked = {True: [], False: []}
                    if self.cur_bar >= 0:
                        self.flows_valid = False
                elif typ == "update":
                    self.book.apply_update(msg["bids"], msg["asks"], ts, msg.get("U"), msg.get("u"), msg.get("pu"))
                elif typ in ("disconnect", "gap"):
                    self.book.invalidate(typ, ts)
                    self.tracked = {True: [], False: []}
                    self.flows_valid = False
            except BookGap as e:
                self.book.invalidate(f"gap: {e}", ts)
                self.tracked = {True: [], False: []}
                self.flows_valid = False
        # after the last message the book is unknown (feed ended): evaluate remaining checkpoints
        # as invalid rather than extrapolating the last state indefinitely
        if self.book.last_ts_us is not None:
            process_until(self.book.last_ts_us + self.max_age)
        self.book.invalidate("end_of_feed", None)
        process_until(2 ** 62)
        return self.o


def compute(ctx: Ctx):
    cb = ctx.cfg["features"]["book"]
    n = ctx.n
    src = ctx.tables.get("book_messages")
    b = ctx.bars
    reason_missing = ctx.tables.get("book_missing_reason", "not_available_in_period")
    if src is not None:
        venue_rule = ctx.tables.get("book_venue_rule", "binance_spot")
        wall_bin = ctx.tables.get("wall_bin_size_i")
        eng = BookEngine(ctx.spec, cb, venue_rule, b["bar_start_us"].to_numpy(), b["bar_end_us"].to_numpy(),
                         ctx.decision_us, wall_bin)
        o = eng.run(src())
    else:
        o = None
    bands = [int(k) for k in cb["bands_bps"]]
    quote = ctx.spec.quote
    fam = "l2_book"
    common = dict(provider=f"{ctx.spec.exchange} L2", inputs="L2 snapshots + diffs", clock="decision time",
                  availability="decision time (book messages with ts < decision)",
                  staleness=f"book age <= {cb['max_book_age']}")

    def col_from(vals, okmask, default_reason, extra_reason=None):
        c = Col(n, reason_missing if o is None else default_reason)
        if o is not None:
            v = np.asarray(vals, dtype=np.float64)
            m = np.asarray(okmask, bool) & np.isfinite(v)
            c.set_valid(m, v)
            if extra_reason is not None:
                c.reason[~m & extra_reason[0]] = CODE[extra_reason[1]]
        return c

    vd = o["valid_dec"] if o is not None else np.zeros(n, bool)
    # ------------- spread (from the reconstructed book)
    if o is not None:
        sp = o["ask"] - o["bid"]
        mid = o["mid_dec"]
        sp_bps = 1e4 * (o["ask"] - o["bid"]) / mid
    else:
        sp = sp_bps = mid = np.full(n, np.nan)
    ctx.add(FeatureSpec(name="spread", family=fam, formula="a - b (best ask - best bid) at decision", unit=quote,
                        nature="observed", null_reasons=["stale_input", "feed_gap", reason_missing], **common),
            col_from(sp, vd, "stale_input"))
    ctx.add(FeatureSpec(name="spread_bps", family=fam, formula="1e4 (a - b) / m, m = (a+b)/2", unit="bps",
                        nature="observed", null_reasons=["stale_input", "feed_gap", reason_missing], **common),
            col_from(sp_bps, vd, "stale_input"))
    ctx.add(FeatureSpec(name="mid", family=fam, formula="(a+b)/2 at decision", unit=quote, nature="observed",
                        null_reasons=["stale_input", reason_missing], **common), col_from(mid, vd, "stale_input"))
    age = Col(n, reason_missing if o is None else "no_observation")
    if o is not None:
        age.set_valid(o["age_dec"] >= 0, o["age_dec"] / tu.SEC)
    ctx.add(FeatureSpec(name="book_age_s", family=fam, formula="decision - exchange time of last applied book message",
                        unit="seconds", nature="observed", null_reasons=[reason_missing], **common), age)
    tws = col_from(o["twa_spread_bps"] if o is not None else np.full(n, np.nan), np.ones(n, bool), "incomplete_coverage")
    ctx.add(FeatureSpec(name="twa_spread_bps", family=fam, formula="(1/|B|) int_B spread_bps(t) dt; requires valid book over the whole bar",
                        unit="bps", nature="observed", clock="bar interval", inputs=common["inputs"],
                        provider=common["provider"], availability="bar close",
                        null_reasons=["incomplete_coverage", reason_missing]), tws)
    vf = Col(n, reason_missing if o is None else "no_observation")
    if o is not None:
        vf.set_valid(np.ones(n, bool), o["tw_valid_frac"])
    ctx.add(FeatureSpec(name="book_valid_time_frac", family=fam, formula="valid synced book time / bar duration",
                        unit="fraction", nature="observed", clock="bar interval", inputs=common["inputs"],
                        provider=common["provider"], availability="bar close", null_reasons=[reason_missing]), vf)

    # ------------- depth bands & imbalance
    for k in bands:
        for s in ("bid", "ask"):
            ctx.add(FeatureSpec(name=f"depth_{s}_{k}bps", family=fam,
                                formula=(f"sum p_i q_i over {s} levels with p {'>=' if s == 'bid' else '<='} m(1{'-' if s == 'bid' else '+'}{k}/1e4)"),
                                unit=f"{quote} notional", nature="observed",
                                null_reasons=["insufficient_book_depth", "stale_input", reason_missing],
                                notes="Null when the band extends beyond the known-complete part of the book.", **common),
                    col_from(o[f"d_{s}_{k}_dec"] if o is not None else np.full(n, np.nan), vd, "insufficient_book_depth",
                             (vd, "insufficient_book_depth")))
        if o is not None:
            imb, rr = safe_div(o[f"d_bid_{k}_dec"] - o[f"d_ask_{k}_dec"], o[f"d_bid_{k}_dec"] + o[f"d_ask_{k}_dec"])
        else:
            imb = np.full(n, np.nan)
        ctx.add(FeatureSpec(name=f"depth_imbalance_{k}bps", family=fam, formula="(D_bid - D_ask) / (D_bid + D_ask)",
                            unit="ratio", nature="observed",
                            null_reasons=["insufficient_book_depth", "zero_denominator", reason_missing], **common),
                col_from(imb, vd, "insufficient_book_depth", (vd, "insufficient_book_depth")))
        for s in ("bid", "ask"):
            ctx.add(FeatureSpec(name=f"twa_depth_{s}_{k}bps", family=fam,
                                formula=f"(1/|B|) int_B D_{s},{k}(t) dt; only when the band is covered for the whole bar",
                                unit=f"{quote} notional", nature="observed", clock="bar interval",
                                inputs=common["inputs"], provider=common["provider"], availability="bar close",
                                null_reasons=["incomplete_coverage", reason_missing]),
                    col_from(o[f"twa_{s}_{k}"] if o is not None else np.full(n, np.nan), np.ones(n, bool), "incomplete_coverage"))
            ctx.add(FeatureSpec(name=f"liq_added_{s}_{k}bps", family="l2_withdrawal_refill",
                                formula="sum of same-price size increases inside the band during the bar (net observed, incl. new orders)",
                                unit=f"{quote} notional", nature="observed", clock="bar interval", inputs=common["inputs"],
                                provider=common["provider"], availability="bar close",
                                null_reasons=["incomplete_coverage", reason_missing],
                                notes="Aggregated L2 cannot separate simultaneous cancels and new orders: this is net observed refill."),
                    col_from(o[f"add_{s}_{k}"] if o is not None else np.full(n, np.nan), np.ones(n, bool), "incomplete_coverage"))
            ctx.add(FeatureSpec(name=f"liq_withdrawn_{s}_{k}bps", family="l2_withdrawal_refill",
                                formula="sum of same-price size decreases inside the band during the bar (cancels AND executions)",
                                unit=f"{quote} notional", nature="observed", clock="bar interval", inputs=common["inputs"],
                                provider=common["provider"], availability="bar close",
                                null_reasons=["incomplete_coverage", reason_missing],
                                notes="Not a cancellation rate: executions also reduce displayed size."),
                    col_from(o[f"wd_{s}_{k}"] if o is not None else np.full(n, np.nan), np.ones(n, bool), "incomplete_coverage"))
            if o is not None:
                shift = o[f"d_{s}_{k}_end"] - o[f"d_{s}_{k}_start"] - (o[f"add_{s}_{k}"] - o[f"wd_{s}_{k}"])
            else:
                shift = np.full(n, np.nan)
            ctx.add(FeatureSpec(name=f"depth_change_band_shift_{s}_{k}bps", family="l2_withdrawal_refill",
                                formula="D(end) - D(start) - (added - withdrawn): mechanical change from the moving mid",
                                unit=f"{quote} notional", nature="observed", clock="bar interval", inputs=common["inputs"],
                                provider=common["provider"], availability="bar close",
                                null_reasons=["incomplete_coverage", reason_missing]),
                    col_from(shift, np.ones(n, bool), "incomplete_coverage"))

    # ------------- static execution cost from the displayed book
    fee = cb.get("taker_fee_bps")
    for sz in [str(x) for x in cb["sizes_base"]] + [f"q{x}" for x in cb["sizes_quote"]]:
        lab = f"{sz}{ctx.spec.base}" if not sz.startswith("q") else f"{sz[1:]}{quote}"
        for side in ("buy", "sell"):
            c = Col(n, reason_missing)
            if o is not None:
                if ctx.spec.contract_type != "linear":
                    c.reason[:] = CODE["unsupported_instrument"]
                else:
                    c.values[:] = o[f"slip_{side}_{sz}"]
                    c.reason[:] = o[f"slipr_{side}_{sz}"]
                    c.reason[~vd] = CODE["stale_input"]
            frm = ("1e4 (VWAP_ask(q)/m - 1)" if side == "buy" else "1e4 (1 - VWAP_bid(q)/m)") + \
                  ("" if not sz.startswith("q") else f"; q = {sz[1:]} {quote} / m (reference notional; actual spend = VWAP*q)")
            ctx.add(FeatureSpec(name=f"cost_{side}_{lab}_bps", family="execution_cost", formula=frm + "; includes spread crossing",
                                unit="bps vs mid", nature="estimated",
                                null_reasons=["insufficient_book_depth", "stale_input", reason_missing],
                                notes="Static walk of the displayed book at decision; excludes latency, queue position, hidden "
                                      "liquidity and impact beyond observed depth. No half-spread is added on top.", **common), c)
        rt = Col(n, reason_missing)
        if o is not None and ctx.spec.contract_type == "linear":
            vb, vs = o[f"slip_buy_{sz}"], o[f"slip_sell_{sz}"]
            okrt = (o[f"slipr_buy_{sz}"] == VALID) & (o[f"slipr_sell_{sz}"] == VALID) & vd
            rt.set_valid(okrt, vb + vs)
            rt.reason[~okrt & vd] = CODE["insufficient_book_depth"]
        ctx.add(FeatureSpec(name=f"hypo_roundtrip_cost_{lab}_current_book_bps", family="execution_cost",
                            formula="cost_buy + cost_sell on the SAME current book (hypothetical; the future exit cost is unknown)",
                            unit="bps", nature="estimated", null_reasons=["insufficient_book_depth", reason_missing], **common), rt)
        if fee is not None:
            for side in ("buy", "sell"):
                base_c = ctx.out[f"cost_{side}_{lab}_bps"]
                c = Col.from_arrays(base_c.values + float(fee), base_c.reason.copy())
                ctx.add(FeatureSpec(name=f"cost_{side}_{lab}_incl_fee_bps", family="execution_cost",
                                    formula=f"cost_{side} + taker fee {fee} bps (ASSUMED via config)", unit="bps",
                                    nature="estimated", null_reasons=["insufficient_book_depth", reason_missing], **common), c)

    # ------------- walls & persistence
    for s in ("bid", "ask"):
        for key, unit, frm in (("dist", "bps", "distance mid -> notional-weighted price of the nearest eligible concentration"),
                               ("size", f"{quote} notional", "notional of that concentration (contiguous eligible bins)"),
                               ("age", "seconds", "observed age so far of the same concentration (overlap matching)")):
            c = Col(n, reason_missing)
            if o is not None:
                c.values[:] = o[f"wall_{key}_{s}"]
                c.reason[:] = o[f"wallr_{s}"]
                c.reason[~vd] = CODE["stale_input"]
            w = cb["wall"]
            ctx.add(FeatureSpec(name=f"wall_{key}_{s}", family="l2_concentration", formula=frm,
                                unit=unit, nature="constructed",
                                windows=f"history {w['history_window']}, samples every {w['sample_every']}",
                                null_reasons=["insufficient_history", "no_observation", "insufficient_book_depth", reason_missing],
                                notes=(f"Fixed absolute price bins of {w.get('bin_size')} quote units; threshold mode "
                                       f"{w['threshold_mode']} (snapshot_median_multiple: bin > {w.get('median_multiple', 5.0)} x median "
                                       f"bin of the current side; history_quantile: bin > {w['quantile']}-quantile of strictly "
                                       f"earlier samples within {w['history_window']}); within {w['max_distance_bps']} bps; "
                                       "contiguous bins merged; age resolution = sample period; price levels only - no order identity."),
                                **common), c)

    # ------------- L2 x aggressive flow interactions (bar interval)
    for k in bands:
        for side, num_col, dep in (("buy", "buy_notional", "ask"), ("sell", "sell_notional", "bid")):
            c = Col(n, reason_missing)
            if o is not None:
                flow = b[num_col].to_numpy(np.float64)
                okf = (b["unknown_side_notional_i"].to_numpy() == 0) & (b["complete"].to_numpy() == 1)
                den = o[f"d_{dep}_{k}_start"]
                v, rr = safe_div(flow, den)
                good = okf & o["valid_start"] & np.isfinite(den)
                c.values[good] = v[good]
                c.reason[good] = rr[good]
                c.reason[~good] = CODE["incomplete_coverage"]
            ctx.add(FeatureSpec(name=f"{side}_pressure_{k}bps", family="l2_flow_interaction",
                                formula=f"aggressive {side} notional in bar / {dep} depth within {k} bps at bar start",
                                unit="ratio", nature="constructed", clock="bar interval", inputs="trades + L2",
                                provider=common["provider"], availability="bar close",
                                null_reasons=["incomplete_coverage", "zero_denominator", reason_missing]), c)
        mv = Col(n, reason_missing)
        if o is not None:
            okm = o["valid_start"] & o["valid_end"]
            mv.set_valid(okm, 1e4 * (o["mid_end"] / o["mid_start"] - 1))
            mv.reason[~okm] = CODE["incomplete_coverage"]
        if k == bands[0]:
            ctx.add(FeatureSpec(name="mid_move_bps", family="l2_flow_interaction", formula="1e4 (m_end / m_start - 1) over the bar",
                                unit="bps", nature="observed", clock="bar interval", inputs="L2", provider=common["provider"],
                                availability="bar close", null_reasons=["incomplete_coverage", reason_missing]), mv)
        for side in ("buy", "sell"):
            c = Col(n, reason_missing)
            if o is not None:
                pr = ctx.out[f"{side}_pressure_{k}bps"]
                mm = ctx.out["mid_move_bps"]
                v, rr = safe_div(mm.values, pr.values)
                good = (pr.reason == VALID) & (mm.reason == VALID)
                c.values[good] = v[good]
                c.reason[good] = rr[good]
                c.reason[~good] = CODE["incomplete_coverage"]
            ctx.add(FeatureSpec(name=f"impact_per_{side}_pressure_{k}bps", family="l2_flow_interaction",
                                formula=f"mid_move_bps / {side}_pressure_{k}bps (low |value| with high pressure = absorbed flow)",
                                unit="bps per unit of depth consumed", nature="constructed", clock="bar interval",
                                inputs="trades + L2", provider=common["provider"], availability="bar close",
                                null_reasons=["incomplete_coverage", "zero_denominator", reason_missing],
                                notes="Constructed 'absorption' proxy defined only by this formula; uses no later price "
                                      "reversal and claims no participant intent."), c)

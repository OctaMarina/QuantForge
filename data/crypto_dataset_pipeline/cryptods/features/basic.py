"""Mandatory base features: returns, realized volatility, durations and intensities (task section 9)."""
from __future__ import annotations

import numpy as np

from .. import timeutil as tu
from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .context import Ctx
from .stats import window_sums, safe_div


def previous_tick_close(ctx: Ctx):
    """Close series usable on a bar clock.

    * bars with trades: the last executed price (exact close);
    * complete time bars without trades: last trade price of an earlier bar in the same segment
      (the feed is complete, so the last observed price still stands); age is tracked;
    * incomplete / gap-invalidated bars: null and the series is broken there.
    Returns (price, valid, age_us, chain_id) where chain_id increments at every break.
    """
    b = ctx.bars
    n = len(b)
    close = b["close"].to_numpy(np.float64)
    status = b["status"].to_numpy()
    complete = b["complete"].to_numpy() == 1
    seg = b["segment_id"].to_numpy()
    last_ts = b["last_trade_ts_us"].to_numpy()
    end = b["bar_end_us"].to_numpy()
    px = np.full(n, np.nan)
    age = np.full(n, -1, dtype=np.int64)
    chain = np.zeros(n, dtype=np.int64)
    cur_px, cur_ts, cid, prev_seg = np.nan, -1, 0, None
    for i in range(n):
        if prev_seg is not None and seg[i] != prev_seg:
            cid += 1
            cur_px, cur_ts = np.nan, -1
        prev_seg = seg[i]
        if not complete[i]:
            cid += 1
            cur_px, cur_ts = np.nan, -1
            chain[i] = cid
            cid += 1
            continue
        if np.isfinite(close[i]):
            cur_px, cur_ts = close[i], last_ts[i]
        chain[i] = cid
        if np.isfinite(cur_px):
            px[i] = cur_px
            age[i] = end[i] - cur_ts
    return px, np.isfinite(px), age, chain


def _px_at_grid(ctx: Ctx, g: np.ndarray):
    """Previous-tick price at grid time g: last trade strictly before g.  Returns price, ts_eff."""
    p1 = ctx.px1s
    sec = p1["sec_start_us"].to_numpy(np.int64)
    idx = np.searchsorted(sec, np.asarray(g, dtype=np.int64) - tu.SEC, side="right") - 1
    price = np.full(len(g), np.nan)
    ts_eff = np.full(len(g), -1, dtype=np.int64)
    ok = idx >= 0
    price[ok] = p1["price"].to_numpy(np.float64)[idx[ok]]
    ts_eff[ok] = p1["last_ts_us"].to_numpy(np.int64)[idx[ok]]
    return price, ts_eff


def compute(ctx: Ctx):
    cfg = ctx.cfg["features"]
    b = ctx.bars
    n = ctx.n
    sym = ctx.spec.symbol
    quote = ctx.spec.quote
    pt, ptv, pt_age, chain = previous_tick_close(ctx)

    # ---------------- bar-clock returns
    for k in cfg["returns"]["bar_horizons"]:
        col = Col(n, "insufficient_history")
        if n > k:
            a, c = pt[k:], pt[:-k]
            same = chain[k:] == chain[:-k]
            ok = same & ptv[k:] & ptv[:-k] & (a > 0) & (c > 0)
            vals = np.full(n - k, np.nan)
            vals[ok] = np.log(a[ok] / c[ok])
            col.values[k:] = vals
            col.reason[k:] = np.where(ok, VALID, np.where(~same, CODE["feed_gap"], CODE["missing_source_data"]))
        ctx.add(FeatureSpec(
            name=f"ret_bars_{k}", family="returns", formula=f"log(P_t / P_(t-{k} bars)), P = previous-tick close on the bar clock",
            unit="log return", clock=f"{ctx.bar_type} bars", inputs="trades", provider=ctx.spec.exchange,
            availability="bar close", windows=f"{k} bars", nature="observed",
            null_reasons=["insufficient_history", "feed_gap"],
            notes="Bar-count horizon; on volume/notional bars this is NOT a calendar horizon."), col)

    # ---------------- calendar returns
    tol = tu.parse_duration(cfg["returns"]["ref_tolerance"])
    ref = ctx.ref_us
    if ctx.px1s is not None and len(ctx.px1s):
        for h in cfg["returns"]["calendar_horizons"]:
            hu = tu.parse_duration(h)
            target = ref - hu
            g = (target // tu.SEC) * tu.SEC
            p0, ts0 = _px_at_grid(ctx, g)
            age = target - ts0
            col = Col(n, "insufficient_history")
            agecol = Col(n, "insufficient_history")
            okp = np.isfinite(p0) & (ts0 >= 0)
            if ctx.data_start_us is not None:
                okp &= target >= ctx.data_start_us
            stale = okp & (age > tol)
            # Only the as-of lookup itself must be gap-free: a gap inside (ts_ref, t-h] means the true
            # last trade before t-h may be missing.  (Gaps between t-h and t do not alter P_t or P_(t-h).)
            gap = okp & ctx.gap_overlaps(ts0, target)
            ok = okp & ~stale & ~gap & ptv & (pt > 0) & (p0 > 0)
            col.values[ok] = np.log(pt[ok] / p0[ok])
            col.reason[ok] = VALID
            col.reason[stale] = CODE["stale_input"]
            col.reason[gap] = CODE["feed_gap"]
            col.reason[okp & ~ptv] = CODE["feed_gap"]
            agecol.values[okp] = age[okp] / tu.SEC
            agecol.reason[okp] = VALID
            lab = tu.duration_label(hu)
            ctx.add(FeatureSpec(
                name=f"ret_{lab}", family="returns",
                formula=f"log(P_t / P(t-{lab})); P_t = bar close (previous-tick for empty time bars); "
                        "P(x) = last trade strictly before the latest 1-second grid point <= x",
                unit="log return", clock="calendar time", inputs="trades", provider=ctx.spec.exchange,
                availability="bar close", windows=lab, staleness=f"reference age <= {cfg['returns']['ref_tolerance']}",
                null_reasons=["insufficient_history", "stale_input", "feed_gap"]), col)
            ctx.add(FeatureSpec(
                name=f"ret_{lab}_ref_age_s", family="returns", formula="(t-h) - timestamp of the reference trade",
                unit="seconds", clock="calendar time", inputs="trades", provider=ctx.spec.exchange,
                availability="bar close", nature="observed", null_reasons=["insufficient_history"]), agecol)

        # ---------------- realized volatility on an explicit calendar grid
        rvc = cfg["realized_vol"]
        step = tu.parse_duration(rvc["grid"])
        stale_g = tu.parse_duration(rvc["max_grid_staleness"])
        t_year = tu.parse_duration(rvc["year"])
        g0 = ((int(ctx.px1s["sec_start_us"].iloc[0]) // step) + 1) * step
        g1 = (int(ref.max()) // step) * step
        grid = np.arange(g0, g1 + step, step, dtype=np.int64)
        gp, gts = _px_at_grid(ctx, grid)
        gvalid = np.isfinite(gp) & (grid - gts <= stale_g) & ~ctx.gap_overlaps(gts, grid)
        r = np.full(grid.size, np.nan)
        rv_ok = np.zeros(grid.size, dtype=bool)
        rv_ok[1:] = gvalid[1:] & gvalid[:-1]
        r[1:] = np.where(rv_ok[1:], np.log(np.where(rv_ok[1:], gp[1:], 1.0) / np.where(rv_ok[1:], gp[:-1], 1.0)), np.nan)
        r2 = np.where(rv_ok, r * r, 0.0)
        G = (ref // step) * step
        gi = ((G - g0) // step).astype(np.int64)
        for w in rvc["windows"]:
            wu = tu.parse_duration(w)
            m = wu // step
            sums, okw = window_sums(r2, rv_ok, int(m))
            lab = tu.duration_label(wu)
            cv, cs, ca = Col(n, "insufficient_history"), Col(n, "insufficient_history"), Col(n, "insufficient_history")
            inr = (gi >= 0) & (gi < grid.size)
            idx = np.where(inr, gi, 0)
            good = inr & okw[idx]
            rv = np.where(good, sums[idx], np.nan)
            cv.set_valid(good, rv)
            cs.set_valid(good, np.sqrt(np.where(good, rv, 0)))
            ca.set_valid(good, np.sqrt(np.where(good, rv, 0) * t_year / wu))
            # distinguish "window touches a gap / stale grid point" from "not enough history"
            had_hist = inr & (gi >= m)
            for c in (cv, cs, ca):
                c.reason[had_hist & ~good] = CODE["stale_input"]
            base = dict(family="realized_vol", clock=f"calendar grid {rvc['grid']}", inputs="trades",
                        provider=ctx.spec.exchange, availability="bar close (grid point <= bar end)",
                        windows=lab, staleness=f"grid price age <= {rvc['max_grid_staleness']}", nature="estimated",
                        null_reasons=["insufficient_history", "stale_input"],
                        notes="Previous-tick sampling on the grid (P(g) = last trade strictly before g). "
                              "Estimates quadratic variation of log price incl. microstructure noise at this "
                              "frequency; cannot observe jumps' timing within a grid step.")
            ctx.add(FeatureSpec(name=f"rv_{lab}", formula=f"sum_(g in (G-{lab}, G]) r_g^2", unit="log-return^2", **base), cv)
            ctx.add(FeatureSpec(name=f"rvol_{lab}", formula=f"sqrt(rv_{lab})", unit="log-return", **base), cs)
            ctx.add(FeatureSpec(name=f"rvol_ann_{lab}", formula=f"sqrt(rv_{lab} * T_year / T_W), T_year={rvc['year']}",
                                unit="annualized log-return (fraction)", **base), ca)
    else:
        # keep the schema stable: same columns, null with an explicit reason
        ctx.quality["returns"] = "no 1-second trade price series available; calendar features null"
        for h in cfg["returns"]["calendar_horizons"]:
            lab = tu.duration_label(tu.parse_duration(h))
            for nm in (f"ret_{lab}", f"ret_{lab}_ref_age_s"):
                ctx.add(FeatureSpec(name=nm, family="returns", formula="calendar return (see docs)", unit="log return",
                                    clock="calendar time", inputs="trades", provider=ctx.spec.exchange,
                                    availability="bar close", null_reasons=["missing_source_data"]), Col(n, "missing_source_data"))
        for w in cfg["realized_vol"]["windows"]:
            lab = tu.duration_label(tu.parse_duration(w))
            for nm in (f"rv_{lab}", f"rvol_{lab}", f"rvol_ann_{lab}"):
                ctx.add(FeatureSpec(name=nm, family="realized_vol", formula="calendar-grid realised variance (see docs)",
                                    unit="log-return^2", clock=f"calendar grid {cfg['realized_vol']['grid']}", inputs="trades",
                                    provider=ctx.spec.exchange, availability="bar close", nature="estimated",
                                    null_reasons=["missing_source_data"]), Col(n, "missing_source_data"))

    # ---------------- bar-clock realized variance (no annualisation)
    lr = np.full(n, np.nan)
    okr = np.zeros(n, dtype=bool)
    if n > 1:
        okr[1:] = (chain[1:] == chain[:-1]) & ptv[1:] & ptv[:-1]
        lr[1:] = np.where(okr[1:], np.log(np.where(okr[1:], pt[1:], 1) / np.where(okr[1:], pt[:-1], 1)), np.nan)
    for N in cfg["realized_vol"]["bar_windows"]:
        sums, okw = window_sums(np.where(okr, lr * lr, 0.0), okr, int(N))
        col = Col(n, "insufficient_history")
        col.set_valid(okw, sums)
        ctx.add(FeatureSpec(
            name=f"rv_bars_{N}", family="realized_vol", formula=f"sum of squared log close-to-close returns over last {N} bars",
            unit="log-return^2", clock=f"{ctx.bar_type} bars", inputs="trades", provider=ctx.spec.exchange,
            availability="bar close", windows=f"{N} bars", nature="estimated",
            null_reasons=["insufficient_history"], notes="Not annualised: a bar count has no fixed calendar length."), col)

    # ---------------- durations and intensities
    start = b["bar_start_us"].to_numpy(np.int64)
    end = b["bar_end_us"].to_numpy(np.int64)
    ft = b["first_trade_ts_us"].to_numpy(np.int64)
    lt = b["last_trade_ts_us"].to_numpy(np.int64)
    nmsg = b["n_messages"].to_numpy()
    nex = b["n_trades"].to_numpy()
    complete = b["complete"].to_numpy() == 1
    dur = (end - start) / tu.SEC
    dcol = Col(n, "feed_gap")
    dcol.set_valid(complete, dur)
    ctx.add(FeatureSpec(
        name="bar_duration_s", family="duration",
        formula="time bars: interval length; event bars: close(prev bar or segment origin) -> close(this bar)",
        unit="seconds", clock="calendar time", inputs="trades", provider=ctx.spec.exchange, availability="bar close",
        null_reasons=["feed_gap"]), dcol)
    span = Col(n, "not_applicable")
    has = complete & (nmsg > 0)
    span.set_valid(has, (lt - ft) / tu.SEC)
    ctx.add(FeatureSpec(name="trade_span_s", family="duration", formula="last trade ts - first trade ts in bar",
                        unit="seconds", clock="calendar time", inputs="trades", provider=ctx.spec.exchange,
                        availability="bar close", null_reasons=["not_applicable", "feed_gap"]), span)
    vol = b["volume"].to_numpy(np.float64)
    notl = b["notional"].to_numpy(np.float64)
    avg = Col(n, "not_applicable")
    v, rr = safe_div(vol, nex)
    avg.set_valid(has & (rr == VALID), v)
    ctx.add(FeatureSpec(name="avg_trade_size", family="intensity", formula="volume / n_trades (executions)",
                        unit=ctx.spec.base, clock="bar", inputs="trades", provider=ctx.spec.exchange,
                        availability="bar close", null_reasons=["not_applicable"]), avg)
    for name, num, unit in (("trade_intensity_per_s", nex.astype(float), "executions/s"),
                            ("volume_per_s", vol, f"{ctx.spec.base}/s"),
                            ("notional_per_s", notl, f"{quote}/s")):
        col = Col(n, "feed_gap")
        val, rr = safe_div(num, dur)
        col.values[complete] = val[complete]
        col.reason[complete] = rr[complete]
        ctx.add(FeatureSpec(name=name, family="intensity", formula=f"{name.split('_per')[0]} / bar_duration_s",
                            unit=unit, clock="bar", inputs="trades", provider=ctx.spec.exchange,
                            availability="bar close", null_reasons=["feed_gap", "zero_denominator"],
                            notes="Denominator = bar_duration_s; null (zero_denominator) for an instantaneous "
                                  "event bar - no epsilon is added."), col)
    if (b["n_messages"] != b["n_trades"]).any():
        ctx.quality["intensity"] = "aggregated trade messages: n_messages != n_trades (executions from id range)"
    _ = sym

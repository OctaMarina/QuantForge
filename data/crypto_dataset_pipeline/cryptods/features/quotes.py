"""Top-of-book spread from a best bid/ask stream (task 9.4), independent of the L2 reconstruction.

Sources: live Binance ``bookTicker`` logs (spot bookTicker carries no exchange timestamp -> receipt
time is used in both replay modes), Binance USD-M ``bookTicker`` daily archives, Tardis ``quotes``.
Columns are prefixed ``quote_`` and carry the source in the registry, so they are never confused with
the spread derived from the reconstructed L2 book.
"""
from __future__ import annotations

import numpy as np

from .. import timeutil as tu
from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .context import Ctx


def compute(ctx: Ctx):
    q = ctx.tables.get("quotes")
    n = ctx.n
    cb = ctx.cfg["features"]["book"]
    max_age = tu.parse_duration(cb["max_quote_age"])
    src = ctx.tables.get("quotes_source", "best bid/ask stream")
    kw = dict(family="quotes_spread", provider=src, inputs="best bid/ask updates", clock="decision time",
              availability="decision time (quotes with ts < decision)", nature="observed",
              staleness=f"quote age <= {cb['max_quote_age']}",
              null_reasons=["not_available_in_period", "stale_input", "invalid_quote"])
    names = ["quote_spread", "quote_spread_bps", "quote_age_s", "quote_twa_spread_bps", "quote_valid_time_frac"]
    if q is None or len(q) == 0:
        for nm in names:
            ctx.add(FeatureSpec(name=nm, formula="see quotes.py", unit="varies", **kw), Col(n, "not_available_in_period"))
        return
    q = q.sort_values("ts_us", kind="stable")
    ts = q["ts_us"].to_numpy(np.int64)
    b = q["bid"].to_numpy(np.float64)
    a = q["ask"].to_numpy(np.float64)
    good = (b > 0) & (a > b) & np.isfinite(a) & np.isfinite(b)    # crossed/locked/empty -> invalid
    sp_bps = np.where(good, 1e4 * (a - b) / ((a + b) / 2), np.nan)
    tie = ctx.cfg["timing"]["cross_stream_tie"]
    d = ctx.decision_us
    idx = np.searchsorted(ts, d, side="left" if tie == "exclude_equal" else "right") - 1
    has = idx >= 0
    j = np.where(has, idx, 0)
    age = d - ts[j]
    fresh = has & (age <= max_age)
    ok = fresh & good[j]
    c1, c2, c3 = Col(n, "no_observation"), Col(n, "no_observation"), Col(n, "no_observation")
    c1.set_valid(ok, (a - b)[j])
    c2.set_valid(ok, sp_bps[j])
    for c in (c1, c2):
        c.reason[has & ~fresh] = CODE["stale_input"]
        c.reason[fresh & ~good[j]] = CODE["invalid_quote"]
    c3.set_valid(has, age / tu.SEC)
    ctx.add(FeatureSpec(name="quote_spread", formula="a - b from the latest valid quote", unit=ctx.spec.quote, **kw), c1)
    ctx.add(FeatureSpec(name="quote_spread_bps", formula="1e4 (a-b)/m, m=(a+b)/2", unit="bps", **kw), c2)
    ctx.add(FeatureSpec(name="quote_age_s", formula="decision - quote timestamp", unit="seconds", **kw), c3)
    # time-weighted spread over the bar [S, E): requires a valid, fresh quote over the whole bar
    S = ctx.bars["bar_start_us"].to_numpy(np.int64)
    E = ctx.bars["bar_end_us"].to_numpy(np.int64)
    tw, vf = Col(n, "incomplete_coverage"), Col(n, "no_observation")
    nq = len(ts)
    for i in range(n):
        dur = E[i] - S[i]
        if dur <= 0:
            tw.reason[i] = CODE["zero_denominator"]
            continue
        k = max(int(np.searchsorted(ts, S[i], side="right")) - 1, 0)   # quote in effect at S (if any)
        acc, vt = 0.0, 0
        while k < nq and ts[k] < E[i]:
            lo = max(S[i], ts[k])
            hi = min(E[i], ts[k + 1] if k + 1 < nq else E[i], ts[k] + max_age)
            if hi > lo and good[k]:
                acc += sp_bps[k] * (hi - lo)
                vt += hi - lo
            k += 1
        vf.values[i] = vt / dur
        vf.reason[i] = VALID
        if vt == dur:
            tw.values[i] = acc / dur
            tw.reason[i] = VALID
    ctx.add(FeatureSpec(name="quote_twa_spread_bps", formula="(1/|B|) int_B spread_bps(t) dt; valid quote over the whole bar",
                        unit="bps", **{**kw, "clock": "bar interval", "availability": "bar close"}), tw)
    ctx.add(FeatureSpec(name="quote_valid_time_frac", formula="time with a valid fresh quote / bar duration",
                        unit="fraction", **{**kw, "clock": "bar interval", "availability": "bar close"}), vf)

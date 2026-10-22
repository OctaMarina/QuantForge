"""Intraday liquidation features from real venue events (task 6.7).

Per-source conventions (verified 2026-10-20):
* Binance USD-M ``<symbol>@forceOrder``: "only the latest one liquidation order within 1000ms will be
  pushed as the snapshot" -> SAMPLED feed: columns are named ``binance_reported_sampled_liq_*``.
  ``S`` is the side of the liquidation ORDER (SELL closes a long -> long liquidated); ``z`` = filled
  accumulated quantity (base units for USD-M), ``ap`` = average execution price -> executed notional
  = z * ap.
* Bybit v5 ``allLiquidation.{symbol}``: pushes all liquidations (500 ms batches); ``S`` is the POSITION
  side ("Buy" -> a long position was liquidated); ``v`` executed size; ``p`` = BANKRUPTCY price -> only
  a notional *valued at bankruptcy price* (estimated), never executed notional.
* Tardis liquidations: side buy -> short liquidated, sell -> long liquidated.
Windows overlapping a collector disconnection are null (feed_gap): no message means unknown, not 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import timeutil as tu
from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .context import Ctx


def _covered(uptime: list, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    ok = np.zeros(lo.shape, dtype=bool)
    for a, b in uptime:
        ok |= (lo >= a) & (hi <= b)
    return ok


def compute(ctx: Ctx):
    lc = ctx.cfg["features"]["liquidations"]
    srcs = ctx.tables.get("liquidations") or {}
    mode = ctx.cfg["timing"]["replay_mode"]
    n = ctx.n
    d = ctx.decision_us
    for src, prefix, exec_px in (("binance", "binance_reported_sampled_liq", True), ("bybit", "bybit_liq", False)):
        info = srcs.get(src)
        kw = dict(family="liquidations", provider=src, inputs="liquidation events", clock="calendar windows",
                  availability="event time (market-time) or receipt time (observed mode)", nature="observed",
                  null_reasons=["feed_gap", "not_available_in_period", "insufficient_history"],
                  notes=("Binance stream is sampled (latest liquidation per symbol per 1000 ms)." if src == "binance"
                         else "Bybit price is the bankruptcy price."))
        for w in lc["windows"]:
            wu = tu.parse_duration(w)
            lab = tu.duration_label(wu)
            cols = {f"{prefix}_long_qty_{lab}": None, f"{prefix}_short_qty_{lab}": None,
                    f"{prefix}_count_{lab}": None, f"{prefix}_imbalance_{lab}": None,
                    f"{prefix}_intensity_per_min_{lab}": None,
                    f"{prefix}_{'notional_exec' if exec_px else 'notional_at_bankruptcy_px'}_{lab}": None,
                    f"{prefix}_surprise_pctile_{lab}": None}
            if info is None or len(info.get("events", [])) == 0 and not info.get("uptime"):
                for nm in cols:
                    ctx.add(FeatureSpec(name=nm, formula="see liquidations.py", unit="varies", **kw),
                            Col(n, "not_available_in_period"))
                continue
            ev = info["events"].sort_values("ts_us")
            tcol = "received_at_us" if mode == "observed_availability" and "received_at_us" in ev else "ts_us"
            t = ev[tcol].to_numpy(np.int64)
            o = np.argsort(t, kind="stable")
            t = t[o]
            longm = (ev["liquidated_position"].to_numpy() == "long")[o]
            qty = ev["qty"].to_numpy(np.float64)[o]
            px = ev["price"].to_numpy(np.float64)[o]
            cl = np.concatenate([[0.0], np.cumsum(np.where(longm, qty, 0.0))])
            cs = np.concatenate([[0.0], np.cumsum(np.where(~longm, qty, 0.0))])
            cn = np.concatenate([[0.0], np.cumsum(qty * px)])
            # window (d - w, d): events with t > d-w and t < d (strict: same-time events not yet known)
            i1 = np.searchsorted(t, d, side="left")
            i0 = np.searchsorted(t, d - wu, side="right")
            lq, sq = cl[i1] - cl[i0], cs[i1] - cs[i0]
            cnt = (i1 - i0).astype(float)
            notl = cn[i1] - cn[i0]
            cov = _covered(info.get("uptime", []), d - wu, d)
            vals = {f"{prefix}_long_qty_{lab}": lq, f"{prefix}_short_qty_{lab}": sq, f"{prefix}_count_{lab}": cnt,
                    f"{prefix}_intensity_per_min_{lab}": cnt / (wu / tu.MIN),
                    f"{prefix}_{'notional_exec' if exec_px else 'notional_at_bankruptcy_px'}_{lab}": notl}
            for nm, v in vals.items():
                c = Col(n, "feed_gap")
                c.set_valid(cov, v)
                frm = {"long_qty": "sum of qty of liquidated LONG positions in (t-w, t)",
                       "short_qty": "sum of qty of liquidated SHORT positions in (t-w, t)",
                       "count": "number of liquidation messages in (t-w, t)",
                       "intensity_per_min": "count / window minutes"}
                key = next((k for k in frm if f"_{k}_" in nm), None)
                f = frm.get(key, "sum qty * price" + (" (average execution price)" if exec_px else " (BANKRUPTCY price: estimated valuation, not executed)"))
                ctx.add(FeatureSpec(name=nm, formula=f, unit="base units" if "qty" in nm else ("count" if "count" in nm else
                        ("1/min" if "intensity" in nm else ctx.spec.quote)), windows=lab, **kw), c)
            imb = Col(n, "feed_gap")
            tot = lq + sq
            okb = cov & (tot > 0)
            imb.set_valid(okb, (lq - sq) / np.where(okb, tot, 1))
            imb.reason[cov & (tot == 0)] = CODE["zero_denominator"]
            ctx.add(FeatureSpec(name=f"{prefix}_imbalance_{lab}", formula="(long - short) / (long + short), qty", unit="ratio",
                                windows=lab, **kw), imb)
            # surprise: percentile of the current window total vs totals on a regular grid of
            # strictly earlier windows (lookback), only fully covered windows
            lb = tu.parse_duration(lc["surprise_lookback"])
            grid_end = np.arange((int(d.min()) - lb) // wu * wu, int(d.max()) + wu, wu, dtype=np.int64)
            g1 = np.searchsorted(t, grid_end, side="left")
            g0 = np.searchsorted(t, grid_end - wu, side="right")
            gtot = (cl[g1] - cl[g0]) + (cs[g1] - cs[g0])
            gcov = _covered(info.get("uptime", []), grid_end - wu, grid_end)
            sp = Col(n, "insufficient_history")
            for i in range(n):
                if not cov[i]:
                    sp.reason[i] = CODE["feed_gap"]
                    continue
                m = gcov & (grid_end <= d[i] - wu) & (grid_end > d[i] - lb)
                if m.sum() < int(lc["surprise_min_obs"]):
                    continue
                sp.values[i] = float(np.mean(gtot[m] <= tot[i]))
                sp.reason[i] = VALID
            ctx.add(FeatureSpec(name=f"{prefix}_surprise_pctile_{lab}",
                                formula=f"share of earlier non-overlapping {lab} windows (last {lc['surprise_lookback']}, fully covered) with total <= current",
                                unit="fraction", windows=f"{lab}, lookback {lc['surprise_lookback']}", **kw), sp)

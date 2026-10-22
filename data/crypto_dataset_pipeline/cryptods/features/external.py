"""External publications and implied volatility: spot ETF flows (BTC, ETH), Fear & Greed, DVOL
(task 6.3, 6.4, 8).

Strict mode: a published value is eligible only from a *verifiable* availability time (own live poll
``received_at`` or archive capture time).  Values without such evidence stay null with reason
``unknown_publication_time``.  Research mode adds separate ``__research_assumed_lag`` columns that use
an assumed publication lag; they are never point-in-time certified and are excluded from the strict
dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import timeutil as tu
from ..asof import asof_values, point_in_time_latest
from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .context import Ctx
from .stats import causal_zscore


def _pit_series_with_z(vint: pd.DataFrame, value: str, z_n: int, z_min: int) -> pd.DataFrame:
    """For each availability instant: the latest economic date known, its value (latest vintage),
    and the z-score of that value vs the previous z_n economic observations as known at that instant."""
    if vint is None or len(vint) == 0:
        return pd.DataFrame(columns=["available_at_us", "economic_date", value, "z"])
    v = vint.dropna(subset=["available_at_us"]).sort_values("available_at_us", kind="stable")
    known: dict = {}
    rows = []
    for avail, grp in v.groupby("available_at_us", sort=True):
        for r in grp.itertuples(index=False):
            known[getattr(r, "economic_date")] = getattr(r, value)
        dates = sorted(d for d, x in known.items() if x is not None and np.isfinite(x))
        if not dates:
            continue
        last = dates[-1]
        prev = known[dates[-2]] if len(dates) >= 2 else np.nan
        hist = np.array([known[d] for d in dates[:-1]][-z_n:], dtype=float)
        z = np.nan
        if hist.size >= z_min:
            sd = hist.std(ddof=1)
            if sd > 0:
                z = (known[last] - hist.mean()) / sd
        rows.append({"available_at_us": int(avail), "economic_date": last, value: known[last], "z": z,
                     "prev": prev, "change_prev": known[last] - prev, "n_hist": int(hist.size)})
    return pd.DataFrame(rows)


def _add_pub_family(ctx: Ctx, prefix: str, vint: pd.DataFrame, value: str, unit: str, family: str,
                    provider: str, max_age: str, z_n: int | None, z_min: int | None, research_lag: str | None,
                    formula: str, notes: str, with_change: bool = False, research_table=None):
    n = ctx.n
    tie = ctx.cfg["timing"]["cross_stream_tie"]
    strict = ctx.cfg["timing"]["strictness"] == "strict"
    max_age_us = tu.parse_duration(max_age)
    has_evidence = vint is not None and len(vint) and vint["available_at_us"].notna().any()
    kw = dict(family=family, provider=provider, inputs="published daily values (vintages)", clock="publication events",
              availability="first verifiable availability (live poll received_at or archive capture time)",
              null_reasons=["unknown_publication_time", "stale_input", "insufficient_history", "missing_source_data"])

    def emit(name, col, f, u, nature="observed", strict_ok=True, extra=None):
        ctx.add(FeatureSpec(name=name, formula=f, unit=u, nature=nature, strict_eligible=strict_ok,
                            staleness=f"age <= {max_age}", notes=notes, **{**kw, **(extra or {})}), col)

    if not has_evidence:
        reason = "unknown_publication_time" if vint is not None and len(vint) else "missing_source_data"
        emit(f"{prefix}", Col(n, reason), formula, unit)
        emit(f"{prefix}_age_s", Col(n, reason), "decision - availability", "seconds")
        emit(f"{prefix}_economic_age_d", Col(n, reason), "decision date - economic date", "days")
        if z_n:
            emit(f"{prefix}_z{z_n}", Col(n, reason), f"causal z vs previous {z_n} observations", "z-score")
        if with_change:
            emit(f"{prefix}_change_prev", Col(n, reason), "latest value - previous economic observation (both known)", unit)
    else:
        pit = _pit_series_with_z(vint, value, z_n or 1, z_min or 10 ** 9)
        col, age, idx = asof_values(pit, value, ctx.decision_us, max_age_us, tie=tie)
        emit(prefix, col, formula, unit)
        ac = Col(n, "no_observation")
        ac.set_valid(age >= 0, age / tu.SEC)
        emit(f"{prefix}_age_s", ac, "decision - availability time of the value", "seconds")
        ec = Col(n, "no_observation")
        if len(pit):
            ed = pd.to_datetime(pit["economic_date"]).to_numpy("datetime64[us]").astype(np.int64)
            j = np.where(idx >= 0, idx, 0)
            ec.set_valid(idx >= 0, (ctx.decision_us - ed[j]) / tu.DAY)
        emit(f"{prefix}_economic_age_d", ec, "decision time - economic date (00:00 UTC)", "days")
        if z_n:
            zc, _, _ = asof_values(pit, "z", ctx.decision_us, max_age_us, tie=tie)
            zc.reason[(col.reason == VALID) & (zc.reason != VALID)] = CODE["insufficient_history"]
            emit(f"{prefix}_z{z_n}", zc, f"(x - mean(prev {z_n} valid obs)) / sd(prev, ddof=1), current excluded, as known at availability",
                 "z-score", extra={"windows": f"{z_n} previous observations (min {z_min}); missing sessions skipped, never filled"})
        if with_change:
            cc, _, _ = asof_values(pit, "change_prev", ctx.decision_us, max_age_us, tie=tie)
            emit(f"{prefix}_change_prev", cc, "latest value - value of the previous economic observation (both as known)", unit)
    rsrc = research_table if research_table is not None and len(research_table) else vint
    if not strict and research_lag is not None and rsrc is not None and len(rsrc):
        # research-only assumed-lag variant: economic date 00:00 UTC + lag
        lag = tu.parse_duration(research_lag)
        r = rsrc.copy()
        r["available_at_us"] = pd.to_datetime(r["economic_date"]).to_numpy("datetime64[us]").astype(np.int64) + lag
        r = r.sort_values("available_at_us").drop_duplicates("economic_date", keep="first")
        pit = _pit_series_with_z(r, value, z_n or 1, z_min or 10 ** 9)
        col, _, _ = asof_values(pit, value, ctx.decision_us, max_age_us, tie=tie)
        emit(f"{prefix}__research_assumed_lag", col, formula + f"; ASSUMED availability = economic date + {research_lag}",
             unit, nature="observed", strict_ok=False,
             extra={"availability": f"ASSUMED economic date 00:00 UTC + {research_lag} (NOT point-in-time certified)"})


def compute(ctx: Ctx):
    T = ctx.tables
    cfg = ctx.cfg["features"]
    lags = ctx.cfg["timing"]["research_assumed_lags"]
    ec = cfg["etf"]
    rr = T.get("research_raw") or {}
    for asset in ("BTC", "ETH"):
        v = (T.get("etf") or {}).get(asset)
        if v is not None and len(v):
            v = v.copy()
            v["total_usd"] = v["total_musd"] * 1e6
            v["sum_funds_usd"] = v["sum_funds_musd"] * 1e6
        rv = (rr.get("etf") or {}).get(asset)
        if rv is not None and len(rv):
            rv = rv.assign(total_usd=rv["total_musd"] * 1e6, sum_funds_usd=rv["sum_funds_musd"] * 1e6)
        _add_pub_family(ctx, f"etf_{asset.lower()}_net_flow_total_usd", v, "total_usd", "USD", "etf_flows",
                        f"Farside ({asset} spot ETF table, US$m converted x1e6)", ec["max_age"], int(ec["z_window_sessions"]),
                        int(ec["z_min_obs"]), lags.get("etf_flows"),
                        "Farside 'Total' column for the latest US trading session known (not summed with the fund rows)",
                        "Weekends/holidays have no row (never zero); pre-launch funds blank -> not zero; 'Total' as published.",
                        research_table=rv)
        _add_pub_family(ctx, f"etf_{asset.lower()}_sum_reported_funds_usd", v, "sum_funds_usd", "USD", "etf_flows",
                        "Farside", ec["max_age"], None, None, None,
                        "sum of fund cells reported for the session (differs from Total when cells are blank)",
                        "Compare with n_funds_reported to judge completeness.")
        _add_pub_family(ctx, f"etf_{asset.lower()}_n_funds_reported", v, "n_funds_reported", "count", "etf_flows",
                        "Farside", ec["max_age"], None, None, None,
                        "number of fund cells with a numeric value for the session", "")
    fg = T.get("fng")
    _add_pub_family(ctx, "fear_greed_value", fg, "value", "index 0-100", "fear_greed", "alternative.me",
                    cfg["fear_greed"]["max_age"], None, None, lags.get("fear_greed"),
                    "Alternative.me Fear & Greed value for the latest economic day known",
                    "Bitcoin-centric market context, not coin-specific sentiment; API timestamp is the economic day, "
                    "not a publication time; price/volume components overlap with market data.", with_change=True,
                    research_table=rr.get("fng"))

    # ---------------- DVOL
    dc = cfg["dvol"]
    tie = ctx.cfg["timing"]["cross_stream_tie"]
    for cur in dc["currencies"]:
        d = (T.get("dvol") or {}).get(cur)
        name = f"dvol_{cur.lower()}"
        kw = dict(family="implied_vol", provider=f"Deribit DVOL {cur}", inputs="DVOL candles (close)",
                  clock=f"DVOL candles res {dc['resolution']}", availability="candle start + resolution (conservative)",
                  unit="annualised implied vol, percentage points (per Deribit methodology article; API docs give no unit)",
                  null_reasons=["missing_source_data", "stale_input", "not_available_in_period"],
                  notes=("Market context for other assets; it is not the implied volatility of "
                         f"{ctx.spec.base}." if cur != ctx.spec.base else ""))
        missing = d is None or len(d) == 0
        miss_reason = T.get("dvol_missing_reason", "missing_source_data")
        if missing:
            col = Col(ctx.n, miss_reason)
        else:
            dd = d.copy()
            dd["available_at_us"] = dd["period_end_us"]
            col, age, idx = asof_values(dd, "close", ctx.decision_us, tu.parse_duration(dc["max_age"]), tie=tie)
        ctx.add(FeatureSpec(name=name, formula="DVOL close of the last closed candle", **kw), col)
        for h in ("1h", "1d"):
            hu = tu.parse_duration(h)
            ch = Col(ctx.n, miss_reason if missing else "insufficient_history")
            if not missing:
                col0, _, _ = asof_values(dd, "close", ctx.decision_us - hu, tu.parse_duration(dc["max_age"]), tie=tie)
                ok = (col.reason == VALID) & (col0.reason == VALID)
                ch.set_valid(ok, col.values - col0.values)
            ctx.add(FeatureSpec(name=f"{name}_change_{h}", formula=f"DVOL(t) - DVOL(t-{h}), both as-of known values",
                                **{**kw, "unit": "percentage points"}), ch)
        if cur == ctx.spec.base:
            g = Col(ctx.n, miss_reason if missing else "missing_source_data")
            rv = ctx.out.get("rvol_ann_30d")
            if rv is not None and not missing:
                ok = (col.reason == VALID) & (rv.reason == VALID)
                g.set_valid(ok, col.values - 100.0 * rv.values)
            elif rv is None:
                g.reason[:] = CODE["not_applicable"]   # 30d realised-vol window not configured
            ctx.add(FeatureSpec(name=f"{name}_minus_trailing_rvol30d_pp", family="implied_vol",
                                formula="DVOL (forward 30d IV, pp) - 100 * trailing 30d realised vol (annualised, 1-min grid)",
                                unit="percentage points", provider="Deribit + trades", inputs="DVOL, trades", clock="decision",
                                availability="decision", nature="constructed",
                                null_reasons=["missing_source_data", "stale_input", "not_applicable"],
                                notes="Descriptive IV - trailing RV gap; horizons differ (forward vs trailing); not a "
                                      "rigorously estimated variance risk premium."), g)

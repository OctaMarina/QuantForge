"""Cross-market features: USD-vs-USDT premium (raw and FX-adjusted), 30-day volatility-scaled trend,
perpetual-spot basis (last and mark/index), exchange premium index, funding (task 6.1, 6.2, 6.6).

Price-time convention: 1-minute candles describe [start, start+1min) and become available at the
period end (market-time replay).  Pairs are matched on identical, closed minutes; a missing candle
on either side makes that minute invalid (never forward-filled into a comparison).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import timeutil as tu
from ..asof import asof_values
from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .context import Ctx
from .stats import causal_zscore


def _minute_join(a: pd.DataFrame, b: pd.DataFrame, cols=("close", "close")) -> pd.DataFrame:
    m = a[["period_start_us", "period_end_us", cols[0]]].rename(columns={cols[0]: "a"}).merge(
        b[["period_start_us", cols[1]]].rename(columns={cols[1]: "b"}), on="period_start_us", how="inner")
    return m.sort_values("period_start_us").reset_index(drop=True)


def _asof_add(ctx: Ctx, name: str, obs: pd.DataFrame, value: str, max_age: str, spec_kw: dict,
              age_name: str | None = None):
    max_age_us = tu.parse_duration(max_age)
    tie = ctx.cfg["timing"]["cross_stream_tie"]
    col, age, _ = asof_values(obs, value, ctx.decision_us, max_age_us, tie=tie, age_from="available_at_us")
    ctx.add(FeatureSpec(name=name, staleness=f"age <= {max_age}", **spec_kw), col)
    if age_name:
        ac = Col(ctx.n, "no_observation")
        ac.set_valid(age >= 0, age / tu.SEC)
        ctx.add(FeatureSpec(name=age_name, **{**spec_kw, "formula": f"decision - availability of {name}",
                                              "unit": "seconds", "nature": "observed"}), ac)


def daily_mean_with_coverage(minutes: pd.DataFrame, value: str, min_cov: float) -> pd.DataFrame:
    d = minutes.assign(day=(minutes["period_start_us"] // tu.DAY) * tu.DAY)
    g = d.groupby("day")[value].agg(["mean", "count"]).reset_index()
    g["coverage"] = g["count"] / 1440.0
    g["valid"] = g["coverage"] >= min_cov
    g["period_end_us"] = g["day"] + tu.DAY
    return g


def compute(ctx: Ctx):
    T = ctx.tables
    cfg = ctx.cfg["features"]
    tie = ctx.cfg["timing"]["cross_stream_tie"]
    n = ctx.n
    pc = cfg["premium"]
    venue = pc["usd_venue"]
    lat_d = tu.parse_duration(pc["daily_finalization_delay"])
    assets = list(dict.fromkeys(list(pc["context_assets"]) + [ctx.spec.base]))
    for asset in assets:
        usd = (T.get("usd_1m") or {}).get(asset)
        usdt = (T.get("usdt_1m") or {}).get(asset)
        fx = T.get("fx_1m")
        lab = f"{venue}_{asset.lower()}"
        base_kw = dict(family="premium", clock="1-minute candles", provider=f"{venue} {asset}-USD vs binance {asset}USDT",
                       inputs="closed 1m candles (close vs close, identical minute)",
                       availability="minute end (market-time replay)", nature="observed",
                       null_reasons=["missing_source_data", "stale_input", "source_unreachable"])
        reason = T.get("premium_missing_reason", "missing_source_data")
        if usd is None or usdt is None or len(usd) == 0 or len(usdt) == 0:
            for nm in (f"premium_raw_{lab}_1m", f"premium_fx_{lab}_1m", f"premium_raw_{lab}_1d",
                       f"premium_raw_{lab}_1d_z{pc['z_window_days']}", f"premium_raw_{lab}_1d_coverage"):
                ctx.add(FeatureSpec(name=nm, formula="see FORMULAS.md (premium)", unit="log ratio", **base_kw), Col(n, reason))
            continue
        m = _minute_join(usd, usdt)
        okm = (m["a"] > 0) & (m["b"] > 0)
        m = m[okm].copy()
        m["premium_raw"] = np.log(m["a"] / m["b"])
        m["available_at_us"] = m["period_end_us"]
        _asof_add(ctx, f"premium_raw_{lab}_1m", m, "premium_raw", "2min",
                  dict(formula=f"log(P_USD/P_USDT), {venue} {asset}-USD close vs Binance {asset}USDT close, same closed minute",
                       unit="log ratio", **base_kw))
        if fx is not None and len(fx):
            mf = m.merge(fx[["period_start_us", "close"]].rename(columns={"close": "fx"}), on="period_start_us", how="inner")
            mf = mf[mf["fx"] > 0].copy()
            mf["premium_fx"] = np.log(mf["a"] / (mf["b"] * mf["fx"]))
            _asof_add(ctx, f"premium_fx_{lab}_1m", mf, "premium_fx", "2min",
                      dict(formula=f"log(P_USD / (P_USDT * FX)), FX = {venue} USDT-USD close (USD per USDT), same minute",
                           unit="log ratio", **base_kw))
        else:
            ctx.add(FeatureSpec(name=f"premium_fx_{lab}_1m", formula="log(P_USD/(P_USDT*FX))", unit="log ratio", **base_kw),
                    Col(n, "missing_source_data"))
        daily = daily_mean_with_coverage(m, "premium_raw", float(pc["daily_min_coverage"]))
        daily["available_at_us"] = daily["period_end_us"] + lat_d
        dv = daily[daily["valid"]].reset_index(drop=True)
        dkw = dict(base_kw, clock="UTC day", availability="day end + finalization delay")
        _asof_add(ctx, f"premium_raw_{lab}_1d", dv, "mean", "2d",
                  dict(dkw, formula=f"mean of valid synchronised minute premia of the UTC day (coverage >= {pc['daily_min_coverage']})",
                       unit="log ratio"))
        z = causal_zscore(daily["mean"].to_numpy(), daily["valid"].to_numpy(), int(pc["z_window_days"]), int(pc["z_min_obs"]))
        daily["z"] = z.values
        dz = daily[np.isfinite(daily["z"])].reset_index(drop=True)
        _asof_add(ctx, f"premium_raw_{lab}_1d_z{pc['z_window_days']}", dz, "z", "2d",
                  dict(dkw, formula=f"(x_d - mean(prev {pc['z_window_days']} valid days)) / sd(prev, ddof=1); current day excluded",
                       unit="z-score", windows=f"{pc['z_window_days']} prior valid days (min {pc['z_min_obs']})"))
        _asof_add(ctx, f"premium_raw_{lab}_1d_coverage", daily, "coverage", "2d",
                  dict(dkw, formula="valid synchronised minutes / 1440", unit="fraction"))

    # ---------------- 30-day volatility-scaled trend on UTC daily closes
    tc = cfg["trend"]
    dc = T.get("daily_close")
    kw = dict(family="trend", clock="UTC day", provider=f"{ctx.spec.exchange} {ctx.spec.symbol} 1d", inputs="daily close",
              availability="day end", nature="observed", null_reasons=["insufficient_history", "zero_denominator", "missing_source_data"])
    N = int(tc["days"])
    name = f"trend_{N}d_volscaled"
    if dc is None or len(dc) == 0:
        ctx.add(FeatureSpec(name=name, formula="log(C_d/C_(d-30)) / (s_30 sqrt(30))", unit="ratio", **kw), Col(n, "missing_source_data"))
    else:
        d = dc.sort_values("period_start_us").reset_index(drop=True)
        day = (d["period_start_us"] // tu.DAY).to_numpy()
        c = d["close"].to_numpy(np.float64)
        vals = np.full(len(d), np.nan)
        for i in range(N, len(d)):
            if day[i] - day[i - N] != N:      # require N+1 consecutive daily closes
                continue
            seg = c[i - N:i + 1]
            if not np.all(seg > 0):
                continue
            r = np.diff(np.log(seg))
            s = r.std(ddof=int(tc["ddof"]))
            if s > 0:
                vals[i] = np.log(c[i] / c[i - N]) / (s * np.sqrt(N))
        d["trend"] = vals
        d["available_at_us"] = d["period_end_us"]
        dd = d[np.isfinite(d["trend"])]
        _asof_add(ctx, name, dd, "trend", "2d",
                  dict(kw, formula=f"log(C_d/C_(d-{N})) / (s_{N},d sqrt({N})), s = sd(ddof={tc['ddof']}) of the {N} daily log returns ending at d",
                       unit="ratio", windows=f"{N} days"))

    # ---------------- basis & premium index & funding
    bkw = dict(family="basis", clock="1-minute candles", inputs="closed 1m candles", availability="minute end",
               nature="observed", null_reasons=["missing_source_data", "stale_input"])
    spot, perp = T.get("kl_spot_1m"), T.get("kl_perp_1m")
    max_age = cfg["basis"]["max_age"]
    if spot is not None and perp is not None and len(spot) and len(perp):
        m = _minute_join(perp, spot)
        m = m[(m.a > 0) & (m.b > 0)].copy()
        m["v"] = np.log(m["a"] / m["b"])
        m["available_at_us"] = m["period_end_us"]
        _asof_add(ctx, "basis_last_perp_spot_1m", m, "v", max_age,
                  dict(bkw, provider=f"binance {ctx.spec.symbol} USD-M perp vs spot", unit="log ratio",
                       formula="log(P_perp/P_spot), last-trade closes of the same closed minute (both quoted in USDT)"))
    else:
        ctx.add(FeatureSpec(name="basis_last_perp_spot_1m", formula="log(P_perp/P_spot)", unit="log ratio",
                            provider="binance", **bkw), Col(n, "missing_source_data"))
    mk, ix = T.get("mark_1m"), T.get("index_1m")
    if mk is not None and ix is not None and len(mk) and len(ix):
        m = _minute_join(mk, ix)
        m = m[(m.a > 0) & (m.b > 0)].copy()
        m["v"] = np.log(m["a"] / m["b"])
        m["available_at_us"] = m["period_end_us"]
        _asof_add(ctx, "basis_mark_index_1m", m, "v", max_age,
                  dict(bkw, provider="binance USD-M markPriceKlines / indexPriceKlines", unit="log ratio",
                       formula="log(mark close / index close), same closed minute"))
    else:
        ctx.add(FeatureSpec(name="basis_mark_index_1m", formula="log(mark/index)", unit="log ratio", provider="binance", **bkw),
                Col(n, "missing_source_data"))
    pi = T.get("premium_index_1m")
    if pi is not None and len(pi):
        p2 = pi.copy()
        p2["available_at_us"] = p2["period_end_us"]
        _asof_add(ctx, "premium_index_binance_1m", p2, "close", max_age,
                  dict(bkw, provider="binance USD-M premiumIndexKlines", unit="fraction (as published)",
                       formula="exchange-published Premium Index, close of the last closed minute (taken directly, not perp/spot-1)"))
    else:
        ctx.add(FeatureSpec(name="premium_index_binance_1m", formula="exchange premium index", unit="fraction",
                            provider="binance", **bkw), Col(n, "missing_source_data"))
    fr = T.get("funding")
    fkw = dict(family="funding", clock="funding events", provider="binance USD-M fundingRate history",
               inputs="realised funding events", availability="funding time", nature="observed",
               null_reasons=["missing_source_data", "stale_input"])
    if fr is not None and len(fr):
        f = fr.sort_values("funding_time_us").reset_index(drop=True).copy()
        f["available_at_us"] = f["funding_time_us"]
        f["interval_h"] = f["funding_time_us"].diff() / tu.HOUR
        _asof_add(ctx, "funding_rate_last", f, "funding_rate", cfg["funding"]["max_age"],
                  dict(fkw, formula="most recent realised funding rate with funding_time < decision", unit="fraction per interval"),
                  age_name="funding_rate_last_age_s")
        _asof_add(ctx, "funding_interval_h_observed", f, "interval_h", cfg["funding"]["max_age"],
                  dict(fkw, formula="hours between the last two realised funding times (not assumed to be 8h)", unit="hours"))
    else:
        for nm in ("funding_rate_last", "funding_rate_last_age_s", "funding_interval_h_observed"):
            ctx.add(FeatureSpec(name=nm, formula="realised funding", unit="fraction", **fkw), Col(n, "missing_source_data"))

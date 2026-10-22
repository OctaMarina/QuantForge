"""25-delta risk reversal skew at a constant 30-day tenor from contemporaneous option quotes
(task 6.5).

skew_25d,T = IV_put(delta=-0.25, T) - IV_call(delta=+0.25, T)

Method (explicit surface estimation from contemporaneous valid quotes only):
1. Snapshot at time s: for each instrument the latest quote with ts < s and age <= max_quote_age.
2. Filters: state open, bid > 0, ask >= bid, bid_iv > 0, ask_iv >= bid_iv, finite delta, expiry > s.
3. IV source: mid of bid_iv/ask_iv (``bid_ask_mid``, default) or the venue mark IV (``mark``).
4. Per expiry: linear interpolation of IV in the exchange-provided delta between the two quotes
   bracketing -0.25 (puts) and +0.25 (calls).  No extrapolation: an expiry without bracketing quotes
   is unusable.
5. Constant 30-day tenor: linear interpolation of total variance w = IV^2 * tau (tau in years,
   ACT/365 from s to expiry) between the two usable expiries bracketing 30 days, separately for the
   put and the call legs; IV_30 = sqrt(w_30 / tau_30).  No extrapolation in tenor.
Delta convention = the venue's (Deribit: Black-Scholes delta per expiry).  IV unit = venue's
(Deribit: percentage points, not documented by the API).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import timeutil as tu
from ..asof import asof_values
from ..nulls import Col
from ..registry import FeatureSpec
from .context import Ctx

YEAR = 365 * tu.DAY


def _interp_delta(d: np.ndarray, iv: np.ndarray, target: float):
    o = np.argsort(d)
    d, iv = d[o], iv[o]
    lo = np.flatnonzero(d <= target)
    hi = np.flatnonzero(d >= target)
    if lo.size == 0 or hi.size == 0:
        return np.nan
    i, j = lo[-1], hi[0]
    if d[j] == d[i]:
        return float(iv[i])
    return float(iv[i] + (target - d[i]) * (iv[j] - iv[i]) / (d[j] - d[i]))


def skew_at(quotes: pd.DataFrame, s_us: int, cfg: dict) -> dict:
    """Compute the 30d 25-delta skew from the latest valid quotes before ``s_us``."""
    maxage = tu.parse_duration(cfg["max_quote_age"])
    q = quotes[(quotes["ts_us"] < s_us) & (quotes["ts_us"] >= s_us - maxage)]
    if len(q) == 0:
        return {"ok": False, "reason": "no_observation"}
    q = q.sort_values("ts_us").groupby("instrument", as_index=False).last()
    q = q[(q["expiry_us"] > s_us) & (q["state"].fillna("open") == "open")]
    if cfg["iv_source"] == "bid_ask_mid":
        ok = (q["bid_price"] > 0) & (q["ask_price"] >= q["bid_price"]) & (q["bid_iv"] > 0) & (q["ask_iv"] >= q["bid_iv"])
        q = q[ok].assign(iv=lambda x: (x["bid_iv"] + x["ask_iv"]) / 2)
    else:
        q = q[q["mark_iv"] > 0].assign(iv=lambda x: x["mark_iv"])
    q = q[np.isfinite(q["delta"])]
    target_d = float(cfg["delta"])
    per_exp = []
    for exp, g in q.groupby("expiry_us"):
        p = g[g["type"] == "put"]
        c = g[g["type"] == "call"]
        ivp = _interp_delta(p["delta"].to_numpy(float), p["iv"].to_numpy(float), -target_d) if len(p) >= 2 else np.nan
        ivc = _interp_delta(c["delta"].to_numpy(float), c["iv"].to_numpy(float), target_d) if len(c) >= 2 else np.nan
        if np.isfinite(ivp) and np.isfinite(ivc):
            per_exp.append((exp, (exp - s_us) / YEAR, ivp, ivc))
    tau_t = float(cfg["target_days"]) * tu.DAY / YEAR
    lo = [x for x in per_exp if x[1] <= tau_t]
    hi = [x for x in per_exp if x[1] >= tau_t]
    if not lo or not hi:
        return {"ok": False, "reason": "insufficient_book_depth", "n_expiries": len(per_exp)}
    e1 = max(lo, key=lambda x: x[1])
    e2 = min(hi, key=lambda x: x[1])
    out = {}
    for leg, k in (("put", 2), ("call", 3)):
        if e1[1] == e2[1]:
            out[leg] = e1[k]
            continue
        w1, w2 = e1[k] ** 2 * e1[1], e2[k] ** 2 * e2[1]
        w = w1 + (w2 - w1) * (tau_t - e1[1]) / (e2[1] - e1[1])
        out[leg] = float(np.sqrt(w / tau_t)) if w > 0 else np.nan
    if not (np.isfinite(out["put"]) and np.isfinite(out["call"])):
        return {"ok": False, "reason": "invalid_quote"}
    return {"ok": True, "skew": out["put"] - out["call"], "iv_put": out["put"], "iv_call": out["call"],
            "exp1_us": e1[0], "exp2_us": e2[0], "n_expiries": len(per_exp)}


def skew_series(quotes: pd.DataFrame, start_us: int, end_us: int, cfg: dict, step_us: int = tu.MIN) -> pd.DataFrame:
    rows = []
    if quotes is None or len(quotes) == 0:
        return pd.DataFrame(columns=["available_at_us", "skew", "iv_put", "iv_call"])
    qs = quotes.sort_values("ts_us")
    t0 = max(start_us, int(qs["ts_us"].iloc[0]))
    t1 = min(end_us, int(qs["ts_us"].iloc[-1]) + tu.parse_duration(cfg["max_quote_age"]))
    s = (t0 // step_us + 1) * step_us
    while s <= t1:
        r = skew_at(qs, s, cfg)
        if r.get("ok"):
            rows.append({"available_at_us": s, **{k: r[k] for k in ("skew", "iv_put", "iv_call", "exp1_us", "exp2_us")}})
        s += step_us
    return pd.DataFrame(rows)


def compute(ctx: Ctx):
    sc = ctx.cfg["features"]["skew"]
    cur = sc["currency"]
    q = ctx.tables.get("options")
    tie = ctx.cfg["timing"]["cross_stream_tie"]
    kw = dict(family="options_skew", provider=f"Deribit {cur} options (live tickers or Tardis options_chain)",
              inputs="contemporaneous option quotes", clock="1-minute snapshots",
              availability="snapshot time (quotes strictly earlier)", nature="estimated",
              unit="IV points (venue unit; Deribit: percentage points)",
              staleness=f"quotes <= {sc['max_quote_age']}, snapshot <= {sc['max_snapshot_age']}",
              null_reasons=["not_available_in_period", "insufficient_book_depth", "no_observation", "stale_input"])
    names = (f"skew25d_{sc['target_days']}d_{cur.lower()}", f"iv_put25d_{sc['target_days']}d_{cur.lower()}",
             f"iv_call25d_{sc['target_days']}d_{cur.lower()}")
    if q is None or len(q) == 0:
        for nm in names:
            ctx.add(FeatureSpec(name=nm, formula="see options.py docstring", **kw),
                    Col(ctx.n, ctx.tables.get("options_missing_reason", "not_available_in_period")))
        return
    lo = int(ctx.decision_us.min()) - tu.HOUR
    hi = int(ctx.decision_us.max())
    ss = skew_series(q, lo, hi, sc)
    for nm, v in zip(names, ("skew", "iv_put", "iv_call")):
        col, _, _ = asof_values(ss, v, ctx.decision_us, tu.parse_duration(sc["max_snapshot_age"]), tie=tie)
        frm = {"skew": "IV_put(-25d, 30d) - IV_call(+25d, 30d)", "iv_put": "IV at put delta -0.25, 30d (total-variance interp.)",
               "iv_call": "IV at call delta +0.25, 30d (total-variance interp.)"}[v]
        ctx.add(FeatureSpec(name=nm, formula=frm, **kw), col)

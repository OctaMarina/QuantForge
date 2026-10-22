"""Order-flow features: observed-aggressor flow, tick-rule flow, Kyle's and Hasbrouck's lambdas.

Two distinct sign sources are never mixed in one column:

* ``aggr``: sign from the venue's aggressor flag (Binance isBuyerMaker: true -> seller aggressed -> -1;
  false -> buyer aggressed -> +1).
* ``tick``: AFML tick rule b_i = sign(p_i - p_(i-1)), carrying b_(i-1) when unchanged.  State is
  carried across files/bars; the first trade of a segment has no previous price -> sign unknown (never
  back-filled from a later price change).

Lambdas (classical OLS standard errors, no HAC; t = coef / sqrt(var(coef))):

* Kyle, trade level, within the bar: dp_i = lambda * (b_i q_i) + e_i, no intercept.
* Hasbrouck, trade level, within the bar: r_i = lambda * b_i sqrt(p_i q_i) + e_i, r_i = log(p_i/p_(i-1)).
* Bar-level rolling variants over the last W bars: dP_B = lambda * sum_i b_i q_i and
  r_B = lambda * sum_i b_i sqrt(p_i q_i) (the sum of per-trade root-notionals, NOT sqrt of the sum).
"""
from __future__ import annotations

import numpy as np

from ..nulls import Col, CODE, VALID
from ..registry import FeatureSpec
from .basic import previous_tick_close
from .context import Ctx
from .stats import imbalance, ols_no_intercept_from_sums, window_sums

AFML19 = ("AFML (2018) ch. 19 'Microstructural Features' (Kyle / Hasbrouck lambdas, tick rule). "
          "Primary text not accessible in this build (O'Reilly preview only, checked 2026-10-20); "
          "regression form = task specification section 5.6-5.8.")


def compute(ctx: Ctx):
    b = ctx.bars
    n = ctx.n
    base, quote = ctx.spec.base, ctx.spec.quote
    complete = b["complete"].to_numpy() == 1
    has = complete & (b["n_messages"].to_numpy() > 0)
    mk = lambda **kw: FeatureSpec(provider=ctx.spec.exchange, availability="bar close", inputs="trades", **kw)

    # ---------------- observed aggressor volumes & imbalance
    for unit_kind, suf, u in (("volume", "volume", base), ("notional", "notional", quote)):
        buy = b[f"buy_{suf}"].to_numpy(np.float64)
        sell = b[f"sell_{suf}"].to_numpy(np.float64)
        unk = b[f"unknown_side_{suf}"].to_numpy(np.float64)
        tot = b[suf].to_numpy(np.float64)
        cov_val = np.where(tot > 0, (buy + sell) / np.where(tot > 0, tot, 1), np.nan)
        for nm, val, frm in ((f"aggr_buy_{suf}", buy, "sum of qty with buyer as aggressor"),
                             (f"aggr_sell_{suf}", sell, "sum of qty with seller as aggressor"),
                             (f"aggr_signed_{suf}", buy - sell, "buy - sell (observed aggressor)")):
            c = Col(n, "feed_gap")
            okc = complete & (unk == 0)
            c.set_valid(okc, val)
            c.reason[complete & (unk > 0)] = CODE["incomplete_coverage"]
            ctx.add(mk(name=nm, family="flow_aggressor", formula=frm, unit=u, clock="bar",
                       null_reasons=["feed_gap", "incomplete_coverage"],
                       notes="Binance isBuyerMaker mapping: true -> -1 (seller aggressor), false -> +1."), c)
        cc = Col(n, "not_applicable")
        cc.set_valid(complete & (tot > 0), cov_val)
        ctx.add(mk(name=f"aggr_classified_frac_{suf}", family="flow_aggressor",
                   formula="(buy + sell) / total", unit="fraction", clock="bar",
                   null_reasons=["not_applicable"]), cc)
        imb, rr = imbalance(buy, sell)
        c = Col(n, "feed_gap")
        okc = complete & (unk == 0)
        c.values[okc] = imb[okc]
        c.reason[okc] = rr[okc]
        c.reason[complete & (unk > 0)] = CODE["incomplete_coverage"]
        ctx.add(mk(name=f"aggr_imbalance_{suf}", family="flow_aggressor",
                   formula="(V_buy - V_sell) / (V_buy + V_sell)", unit="ratio in [-1,1]", clock="bar",
                   null_reasons=["feed_gap", "incomplete_coverage", "zero_denominator"]), c)

        # tick-rule variants (distinct names)
        tb = b[f"tick_buy_{suf}"].to_numpy(np.float64)
        ts_ = b[f"tick_sell_{suf}"].to_numpy(np.float64)
        tu_ = b[f"tick_unknown_{suf}"].to_numpy(np.float64)
        imb, rr = imbalance(tb, ts_)
        c = Col(n, "feed_gap")
        okc = complete & (tu_ == 0)
        c.values[okc] = imb[okc]
        c.reason[okc] = rr[okc]
        c.reason[complete & (tu_ > 0)] = CODE["incomplete_coverage"]
        ctx.add(mk(name=f"tick_imbalance_{suf}", family="flow_tick_rule",
                   formula="(V_tick+ - V_tick-) / (V_tick+ + V_tick-), tick rule b_i", unit="ratio in [-1,1]",
                   clock="bar", reference=AFML19,
                   null_reasons=["feed_gap", "incomplete_coverage", "zero_denominator"],
                   notes="Trades before the first price change of a segment have unknown tick sign -> "
                         "bar imbalance null (incomplete_coverage)."), c)
        c2 = Col(n, "feed_gap")
        c2.set_valid(complete, tb - ts_)
        c2.reason[complete & (tu_ > 0)] = CODE["incomplete_coverage"]
        c2.values[c2.reason != VALID] = np.nan
        ctx.add(mk(name=f"tick_signed_{suf}", family="flow_tick_rule", formula="V_tick+ - V_tick-",
                   unit=u, clock="bar", reference=AFML19, null_reasons=["feed_gap", "incomplete_coverage"]), c2)

    # ---------------- trade-level lambdas within the bar
    min_n = int(ctx.cfg["features"]["flow"]["min_trades_regression"])
    for model, ylab, xlab, unit in (("kyle", "dp_i (quote)", "b_i q_i (base)", f"{quote} per {base}"),
                                    ("hasb", "log(p_i/p_(i-1))", "b_i sqrt(p_i q_i)", f"log-return per sqrt({quote})")):
        for sgn in ("tick", "aggr"):
            nn = b[f"{model}_{sgn}_n"].to_numpy()
            lam, t, rb, rt = ols_no_intercept_from_sums(nn, b[f"{model}_{sgn}_sxx"], b[f"{model}_{sgn}_sxy"],
                                                         b[f"{model}_{sgn}_syy"], min_n)
            full = "kyle" if model == "kyle" else "hasbrouck"
            for nm, val, rr in ((f"{full}_lambda_trades_{sgn}", lam, rb), (f"{full}_tstat_trades_{sgn}", t, rt)):
                c = Col(n, "feed_gap")
                c.values[complete] = val[complete]
                c.reason[complete] = rr[complete]
                ctx.add(mk(name=nm, family="lambda", clock="trades within the bar",
                           formula=f"OLS {ylab} = lambda * {xlab} + e (no intercept), trades of the bar; "
                                   f"sign={'tick rule' if sgn == 'tick' else 'observed aggressor'}"
                                   + ("; t = lambda / sqrt(s^2/Sxx), s^2 = RSS/(n-1)" if 'tstat' in nm else ""),
                           unit=unit if "lambda" in nm else "t-statistic", nature="estimated",
                           windows=f"bar (>= {min_n} trades)", reference=AFML19,
                           variant=("literal tick-rule" if sgn == "tick" else "observed-aggressor variant"),
                           null_reasons=["insufficient_history", "invalid_regression", "feed_gap"],
                           notes="First trade after a segment start has no previous price and is excluded; "
                                 "dp of the first trade of a bar uses the previous trade of the earlier bar."), c)
            cn = Col(n, "feed_gap")
            cn.set_valid(complete, nn.astype(float))
            ctx.add(mk(name=f"{'kyle' if model == 'kyle' else 'hasbrouck'}_n_trades_{sgn}", family="lambda",
                       formula="observations used", unit="count", clock="bar", null_reasons=["feed_gap"]), cn)

    # ---------------- bar-level rolling lambdas
    pt, ptv, _, chain = previous_tick_close(ctx)
    dP = np.full(n, np.nan)
    rB = np.full(n, np.nan)
    okd = np.zeros(n, dtype=bool)
    if n > 1:
        okd[1:] = (chain[1:] == chain[:-1]) & ptv[1:] & ptv[:-1] & complete[1:]
        dP[1:] = pt[1:] - pt[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rB[1:] = np.log(pt[1:] / pt[:-1])
    for W in ctx.cfg["features"]["flow"]["bar_windows"]:
        for sgn in ("aggr", "tick"):
            if sgn == "aggr":
                xk = b["buy_volume"].to_numpy(np.float64) - b["sell_volume"].to_numpy(np.float64)
                xh = b["hasb_aggr_xb"].to_numpy(np.float64)
                okx = b["unknown_side_volume_i"].to_numpy() == 0
            else:
                xk = b["tick_buy_volume"].to_numpy(np.float64) - b["tick_sell_volume"].to_numpy(np.float64)
                xh = b["hasb_tick_xb"].to_numpy(np.float64)
                okx = b["tick_unknown_volume_i"].to_numpy() == 0
            for model, x, y in (("kyle", xk, dP), ("hasbrouck", xh, rB)):
                ok = okd & okx & np.isfinite(x) & np.isfinite(y)
                xx = np.where(ok, x * x, 0.0)
                xy = np.where(ok, x * y, 0.0)
                yy = np.where(ok, y * y, 0.0)
                sxx, okw = window_sums(xx, ok, int(W))
                sxy, _ = window_sums(xy, ok, int(W))
                syy, _ = window_sums(yy, ok, int(W))
                lam, t, rb, rt = ols_no_intercept_from_sums(np.where(okw, W, 0), sxx, sxy, syy, 3)
                for nm, val, rr in ((f"{model}_lambda_bars{W}_{sgn}", lam, rb), (f"{model}_tstat_bars{W}_{sgn}", t, rt)):
                    c = Col(n, "insufficient_history")
                    c.values[:] = val
                    c.reason[:] = rr
                    c.reason[~okw & (np.arange(n) >= W)] = CODE["feed_gap"]
                    c.values[c.reason != VALID] = np.nan
                    if model == "kyle":
                        frm = (f"rolling OLS over last {W} bars: (P_B - P_(B-1)) = lambda * sum_(i in B) b_i q_i, "
                               "no intercept; regressor from trade-level signed volume (not sign(close-open)*volume)")
                        unit = f"{quote} per {base}"
                    else:
                        frm = (f"rolling OLS over last {W} bars: log(P_B/P_(B-1)) = lambda * sum_(i in B) b_i sqrt(p_i q_i), "
                               "no intercept (sum of per-trade root notionals, not root of summed notional)")
                        unit = f"log-return per sqrt({quote})"
                    ctx.add(mk(name=nm, family="lambda", formula=frm if "lambda" in nm else frm + "; t = coef/sqrt(var)",
                               unit=unit if "lambda" in nm else "t-statistic", clock=f"{ctx.bar_type} bars",
                               windows=f"{W} bars", nature="estimated", reference=AFML19,
                               variant=("observed-aggressor" if sgn == "aggr" else "tick rule"),
                               null_reasons=["insufficient_history", "feed_gap", "invalid_regression"],
                               notes="Construction choice for the bar-aggregated form (task 5.6/5.7); the AFML "
                                     "bar-level definition could not be checked against the primary text."), c)

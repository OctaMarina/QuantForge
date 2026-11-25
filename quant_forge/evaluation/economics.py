"""Economic evaluation of sized trade lists: monthly PnL, Sharpe, exposure, drawdown, bootstrap and permutation tests.

Conventions
-----------
* A trade's PnL is ``notional * size * trade_return`` booked in its **exit month**; every calendar month between
  the first and the last exit counts (empty months are zeros).  Sharpe is annualised from monthly PnL
  (``mean / sd * sqrt(12)``, ``ddof = 1``).
* Differences between rules use a **paired monthly block bootstrap** (the same resampled months for all
  rules) and, for model-driven rules, a **permutation test**: sizes are shuffled among the trades of the same
  group (strategy x window by default), which keeps the size distribution but destroys the information; the
  p-value is the share of shuffles whose Sharpe gain over ``U`` is at least the observed one.
* The capital budget ``K`` is the 95th percentile (time-weighted) of the gross open exposure of the uniform rule
  ``U`` (at least one notional); "return on K" is annualised PnL / K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..validation.calendar_kfold import to_int64_ns

__all__ = ["monthly_pnl", "annualized_sharpe", "exposure_path", "max_drawdown", "SizingEvaluation",
           "evaluate_sizing", "paired_sharpe_table", "verdict"]


def monthly_pnl(exit_time: pd.Series, returns, sizes, notional: float = 10_000.0) -> tuple[np.ndarray, np.ndarray]:
    """``(PnL per calendar month, month index of every trade)``; month 0 = month of the first exit."""
    t = exit_time
    mi = ((t.dt.year - t.dt.year.min()) * 12 + (t.dt.month - t.loc[t.idxmin()].month)).to_numpy()
    mi = mi - mi.min()
    nm = mi.max() + 1
    return np.bincount(mi, weights=notional * np.asarray(sizes) * np.asarray(returns), minlength=nm), mi


def annualized_sharpe(x):
    """Annualised Sharpe of monthly PnL along the last axis (NaN when the sd is 0); vectorised over rows."""
    x = np.asarray(x, dtype=float)
    sd = x.std(-1, ddof=1)
    out = np.where(sd > 0, x.mean(-1) / np.where(sd > 0, sd, 1) * np.sqrt(12), np.nan)
    return float(out) if np.ndim(out) == 0 else out


def exposure_path(entry_time, exit_time, sizes, notional: float = 10_000.0) -> dict:
    """Time-weighted mean / p95 / max of gross open notional and mean / max number of open positions."""
    w = np.asarray(sizes, dtype=float)
    t = np.concatenate([to_int64_ns(entry_time), to_int64_ns(exit_time)])
    v = np.concatenate([w * notional, -w * notional])
    c = np.concatenate([(w > 0).astype(float), -(w > 0).astype(float)])
    o = np.argsort(t, kind="stable")
    t, v, c = t[o], np.cumsum(v[o]), np.cumsum(c[o])
    dt = np.diff(t).astype(float)
    if dt.sum() <= 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "mean_pos": 0.0, "max_pos": 0.0}
    ww = dt / dt.sum()
    order = np.argsort(v[:-1])
    cw = np.cumsum(ww[order])
    p95 = v[:-1][order][np.searchsorted(cw, 0.95)]
    return {"mean": float((v[:-1] * ww).sum()), "p95": float(p95), "max": float(v.max()),
            "mean_pos": float((c[:-1] * ww).sum()), "max_pos": float(c.max())}


def max_drawdown(monthly) -> float:
    """Largest peak-to-trough fall of cumulative monthly PnL (a negative number)."""
    eq = np.cumsum(monthly)
    return float((eq - np.maximum.accumulate(eq)).min())


@dataclass
class SizingEvaluation:
    """Output of :func:`evaluate_sizing`: the summary ``table`` (one row per rule), ``monthly`` PnL per rule,
    ``boot`` = bootstrap Sharpe samples per rule (paired), ``budget`` K and number of ``months``."""

    table: pd.DataFrame
    monthly: dict
    boot: dict
    budget: float
    months: int

    def __iter__(self):            # allows ``tab, M, B = evaluate_sizing(...)``
        return iter((self.table, self.monthly, self.boot))


def evaluate_sizing(df: pd.DataFrame, sizes: Mapping[str, np.ndarray] | pd.DataFrame, *, rules: Sequence[str] | None = None,
                    perm_rules: Sequence[str] = ("F_tau", "F_be", "R", "P", "Pd", "DxF_tau", "DxF_be", "DxR", "DxP"),
                    baseline: str = "U", benchmark: str = "D", notional: float = 10_000.0, n_boot: int = 2000,
                    n_perm: int = 500, seed: int = 0, perm_group_cols: Sequence[str] = ("strategy", "window"),
                    return_col: str = "trade_return", entry_col: str = "entry_time",
                    exit_col: str = "exit_time") -> SizingEvaluation:
    """Sharpe, paired bootstrap CIs against ``baseline`` and ``benchmark``, permutation p-values, PnL, return on
    budget, drawdown, exposure, share traded and concurrency for every rule.

    ``df`` holds the trades (returns, entry / exit times, permutation-group columns); ``sizes`` maps rule ->
    sizes aligned with ``df`` rows.  Random draws: the bootstrap months first, then the permutations of the
    rules in ``rules`` order (reproducible with ``seed``).
    """
    df = df.reset_index(drop=True)
    rules = list(rules if rules is not None else (sizes.columns if isinstance(sizes, pd.DataFrame) else sizes))
    W = {r: np.asarray(sizes[r]) for r in rules}
    M, mi = {}, None
    for r in rules:
        M[r], mi = monthly_pnl(df[exit_col], df[return_col], W[r], notional)
    nm = len(M[baseline])
    years = nm / 12
    entry, exit_ = df[entry_col], df[exit_col]
    K = max(exposure_path(entry, exit_, W[baseline], notional)["p95"], notional)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, nm, size=(n_boot, nm))
    B = {r: annualized_sharpe(M[r][ix]) for r in rules}
    key = df[perm_group_cols[0]].astype(str)
    for c in perm_group_cols[1:]:
        key = key + df[c].astype(str)
    key = key.to_numpy()
    groups = [np.flatnonzero(key == g) for g in np.unique(key)]
    r_ = df[return_col].to_numpy() * notional
    s_u = annualized_sharpe(M[baseline])
    has_bench = benchmark in M

    def perm_p(w, observed):
        res = np.empty(n_perm)
        for i in range(n_perm):
            ww = w.copy()
            for g in groups:
                ww[g] = w[rng.permutation(g)]
            res[i] = annualized_sharpe(np.bincount(mi, weights=ww * r_, minlength=nm)) - s_u
        return float((res >= observed).mean())

    rows = []
    for r in rules:
        ex = exposure_path(entry, exit_, W[r], notional)
        s_r = annualized_sharpe(M[r])
        d_u = B[r] - B[baseline]
        row = {"rule": r, "Sharpe": s_r, f"Δ vs {baseline}": s_r - s_u,
               f"CI {baseline} lo": np.quantile(d_u, 0.025), f"CI {baseline} hi": np.quantile(d_u, 0.975)}
        if has_bench:
            d_d = B[r] - B[benchmark]
            row.update({f"Δ vs {benchmark}": s_r - annualized_sharpe(M[benchmark]),
                        f"CI {benchmark} lo": np.quantile(d_d, 0.025), f"CI {benchmark} hi": np.quantile(d_d, 0.975)})
        row.update({"p perm": perm_p(W[r], s_r - s_u) if r in perm_rules else np.nan,
                    "PnL (kUSD)": M[r].sum() / 1e3, "return on K (ann.)": M[r].sum() / K / years,
                    "bps / notional": M[r].sum() / (notional * W[r].sum()) * 1e4 if W[r].sum() else np.nan,
                    "max DD / K": max_drawdown(M[r]) / K, "exposure / K": ex["mean"] / K,
                    "traded": float((W[r] > 0).mean()), "mean open pos.": ex["mean_pos"],
                    "max open pos.": ex["max_pos"]})
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("rule")
    tab.attrs["K"], tab.attrs["months"] = K, nm
    return SizingEvaluation(tab, M, B, K, nm)


def paired_sharpe_table(evaluation: SizingEvaluation, pairs: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """``a − b`` Sharpe differences with paired bootstrap 95 % CI and ``P(Δ > 0)`` for each ``(a, b)``."""
    M, B = evaluation.monthly, evaluation.boot
    rows = []
    for a, b in pairs:
        d = B[a] - B[b]
        rows.append({"comparison": f"{a} − {b}", "Δ Sharpe": annualized_sharpe(M[a]) - annualized_sharpe(M[b]),
                     "CI low": np.quantile(d, 0.025), "CI high": np.quantile(d, 0.975), "P(Δ > 0)": float((d > 0).mean())})
    return pd.DataFrame(rows).set_index("comparison")


def verdict(row) -> str:
    """"yes" / "no (worse)" / "not distinguishable from 0" from a row with ``CI low`` and ``CI high``."""
    return "yes" if row["CI low"] > 0 else ("no (worse)" if row["CI high"] < 0 else "not distinguishable from 0")

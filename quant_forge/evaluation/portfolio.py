"""Calendar-based portfolio evaluation of sized trade lists: daily mark-to-market PnL, a common capital budget and
paired comparisons across rules *and* across trade lists (e.g. strategies run on different bar series).

Why daily mark-to-market.  :func:`quant_forge.evaluation.monthly_pnl` books a trade's whole PnL in its exit
month, which is fine for one trade list but not when holding periods differ by orders of magnitude (a strategy
on 30-minute bars holds for hours, on daily bars for months).  Here every trade is valued at each UTC day end
between its entry and its exit with the asset's daily close (:func:`daily_close_panel`), the entry fee is booked
on the entry day and the exit fee on the exit day, so the daily PnL of a trade sums exactly to
``notional * size * trade_return``.

* :func:`expand_trade_days` - the per-day PnL of every trade per unit of entry notional (computed once per trade
  list; any sizing rule is a multiplication);
* :func:`daily_pnl` - daily PnL of one or several sizing vectors, optionally split by a group label;
* :func:`portfolio_table` - net PnL, return on the budget, Sharpe (daily, x sqrt(365)), max drawdown, turnover,
  exposure and trade counts;
* :func:`bootstrap_sharpe` - Sharpe of many daily PnL series under the **same** monthly block-bootstrap draws,
  so that differences between any two series (two rules, two bar types) are paired;
* :func:`paired_differences` - ``a - b`` Sharpe differences with percentile intervals from those draws.

All series must share the calendar (the same days), which is what makes configurations comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .economics import exposure_path

__all__ = ["daily_close_panel", "TradeDays", "expand_trade_days", "daily_pnl", "portfolio_table", "max_drawdown_daily",
           "bootstrap_sharpe", "paired_differences", "month_labels"]

_DAY_NS = 86_400 * 10 ** 9


def daily_close_panel(daily_bars: Mapping[str, pd.DataFrame], calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Close at the end of each UTC day per asset (rows = day starts, tz UTC; columns = assets).

    ``daily_bars`` are 1-day time bars (e.g. ``load_bars(assets, BarConfig("time", 1), ...)``): the close of the bar
    that starts on a day is that day's close.  Missing days are forward-filled; days before an asset's first bar
    are back-filled (only used if a trade starts before the asset's first daily bar).
    """
    cols = {}
    for a, b in daily_bars.items():
        s = pd.Series(b["close"].to_numpy(float), index=pd.DatetimeIndex(b["bar_start"]).tz_convert("UTC").normalize())
        cols[a] = s[~s.index.duplicated(keep="last")]
    panel = pd.DataFrame(cols).sort_index()
    if calendar is not None:
        panel = panel.reindex(panel.index.union(calendar)).ffill().reindex(calendar)
    return panel.ffill().bfill()


@dataclass
class TradeDays:
    """Per-day mark-to-market of a trade list.

    ``trade`` (positional row of the trade list), ``day`` (index into ``calendar``) and ``unit`` (PnL on that day
    per unit of entry notional, fees included) are aligned arrays with one entry per (trade, day held);
    ``calendar`` are the day starts (UTC); ``n_trades`` the length of the trade list.
    """

    trade: np.ndarray
    day: np.ndarray
    unit: np.ndarray
    calendar: pd.DatetimeIndex
    n_trades: int

    def unit_return(self) -> np.ndarray:
        """Sum of ``unit`` per trade (equals the net trade return, a consistency check)."""
        return np.bincount(self.trade, weights=self.unit, minlength=self.n_trades)


def _day_index(times, calendar: pd.DatetimeIndex) -> np.ndarray:
    t = pd.DatetimeIndex(times).tz_convert("UTC").as_unit("ns").asi8
    c0 = calendar.as_unit("ns").asi8[0]
    return (t - c0) // _DAY_NS


def expand_trade_days(trades: pd.DataFrame, closes: pd.DataFrame, *, fee_bps: float, calendar: pd.DatetimeIndex | None = None,
                      asset_col: str = "asset", side_col: str = "side", entry_col: str = "entry_time",
                      exit_col: str = "exit_time", entry_price_col: str = "entry_price",
                      gross_col: str = "trade_return_gross") -> TradeDays:
    """Daily path of every trade between entry and exit (see the module docstring).

    The exit price is recovered from the gross return (``exit = entry * (1 + side * gross)``); ``closes`` is a
    :func:`daily_close_panel` covering the calendar (default: its own index).  Trades must lie inside the calendar.
    """
    calendar = closes.index if calendar is None else calendar
    C = closes.reindex(calendar).reindex(columns=sorted(set(trades[asset_col]) | set(closes.columns))).to_numpy(float)
    col_of = {a: j for j, a in enumerate(sorted(set(trades[asset_col]) | set(closes.columns)))}
    n = len(trades)
    de = _day_index(trades[entry_col], calendar)
    dx = _day_index(trades[exit_col], calendar)
    if n and (de.min() < 0 or dx.max() >= len(calendar)):
        raise ValueError("trades fall outside the calendar")
    L = (dx - de + 1).astype(np.int64)
    tr = np.repeat(np.arange(n), L)
    off = np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L)
    day = de[tr] + off
    k = np.array([col_of[a] for a in trades[asset_col]], dtype=np.int64)[tr]
    pe = trades[entry_price_col].to_numpy(float)
    s = trades[side_col].to_numpy(float)
    px = pe * (1 + s * trades[gross_col].to_numpy(float))
    first, last = day == de[tr], day == dx[tr]
    c_end = C[day, k]
    c_prev = C[np.maximum(day - 1, 0), k]
    p_end = np.where(last, px[tr], c_end)
    p_start = np.where(first, pe[tr], c_prev)
    f = fee_bps / 1e4
    unit = s[tr] * (p_end - p_start) / pe[tr] - f * first - f * (px[tr] / pe[tr]) * last
    return TradeDays(tr, day.astype(np.int64), unit, calendar, n)


def daily_pnl(td: TradeDays, sizes, *, notional: float = 10_000.0, group: np.ndarray | None = None,
              n_groups: int | None = None) -> np.ndarray:
    """Daily PnL of sizing vector(s) ``sizes`` (shape ``(n_trades,)`` or ``(n_rules, n_trades)``).

    With ``group`` (one integer label per trade) the result has an extra leading axis of length ``n_groups``.
    Shapes: ``(n_days,)``, ``(n_rules, n_days)``, ``(n_groups, n_days)`` or ``(n_groups, n_rules, n_days)``.
    """
    one = np.ndim(sizes) == 1
    W = np.atleast_2d(np.asarray(sizes, dtype=float))
    nd = len(td.calendar)
    if group is None:
        out = np.vstack([np.bincount(td.day, weights=notional * w[td.trade] * td.unit, minlength=nd) for w in W])
        return out[0] if one else out
    g = np.asarray(group)[td.trade]
    G = int(n_groups if n_groups is not None else (g.max() + 1 if len(g) else 0))
    out = np.zeros((G, len(W), nd))
    for gi in range(G):
        m = g == gi
        for ri, w in enumerate(W):
            out[gi, ri] = np.bincount(td.day[m], weights=notional * w[td.trade[m]] * td.unit[m], minlength=nd)
    return out[:, 0] if one else out


def max_drawdown_daily(daily) -> float:
    """Largest peak-to-trough fall of cumulative daily PnL (negative number)."""
    eq = np.cumsum(np.asarray(daily, dtype=float))
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    return float((eq - peak).min()) if len(eq) else 0.0


def _sharpe(x, days_per_year: float = 365.0):
    x = np.asarray(x, dtype=float)
    sd = x.std(-1, ddof=1)
    return np.where(sd > 0, x.mean(-1) / np.where(sd > 0, sd, 1) * np.sqrt(days_per_year), np.nan)


def portfolio_table(daily: Mapping[str, np.ndarray], *, budget: float, trades: Mapping[str, pd.DataFrame] | None = None,
                    sizes: Mapping[str, np.ndarray] | None = None, notional: float = 10_000.0,
                    days_per_year: float = 365.0, entry_col: str = "entry_time", exit_col: str = "exit_time") -> pd.DataFrame:
    """One row per series: net PnL, annualised return on ``budget``, Sharpe of daily PnL, max drawdown / budget,
    and (with ``trades`` and ``sizes``) trades taken, turnover (entry + exit notional per year / budget), mean gross
    exposure / budget, mean and max open positions and mean size of the trades taken."""
    rows = []
    for name, x in daily.items():
        x = np.asarray(x, dtype=float)
        years = len(x) / days_per_year
        row = {"series": name, "net PnL": x.sum(), "return on K (ann.)": x.sum() / budget / years,
               "Sharpe": float(_sharpe(x, days_per_year)), "max DD / K": max_drawdown_daily(x) / budget}
        if trades is not None and sizes is not None:
            w = np.asarray(sizes[name], dtype=float)
            tdf = trades[name]
            taken = w > 0
            row.update({"trades": int(taken.sum()),
                        "turnover (x K / yr)": 2 * notional * w.sum() / budget / years,
                        "mean size": float(w[taken].mean()) if taken.any() else np.nan})
            if taken.any():
                ex = exposure_path(tdf[entry_col][taken], tdf[exit_col][taken], w[taken], notional)
                row.update({"exposure / K": ex["mean"] / budget, "mean open pos.": ex["mean_pos"],
                            "max open pos.": ex["max_pos"]})
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def month_labels(calendar: pd.DatetimeIndex) -> np.ndarray:
    """Integer calendar-month label of every day (0 = first month)."""
    c = calendar.tz_convert("UTC") if calendar.tz is not None else calendar
    m = (c.year - c.year[0]) * 12 + (c.month - c.month[0])
    return np.asarray(m - m.min(), dtype=np.int64)


def bootstrap_sharpe(daily: np.ndarray, months: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
                     days_per_year: float = 365.0, chunk: int = 64) -> np.ndarray:
    """Bootstrap Sharpe ratios (shape ``(n_series, n_boot)``) of daily PnL series ``daily`` (``(n_series, n_days)``)
    resampling whole calendar months with replacement.  The draws depend only on ``seed`` and the number of
    months, so calling it with the same seed on any set of series on the same calendar gives paired samples."""
    X = np.atleast_2d(np.asarray(daily, dtype=float))
    nm = int(months.max()) + 1
    cnt = np.bincount(months, minlength=nm).astype(float)
    ix = np.random.default_rng(seed).integers(0, nm, size=(n_boot, nm))
    N = cnt[ix].sum(1)
    out = np.empty((X.shape[0], n_boot))
    for a in range(0, X.shape[0], chunk):
        Xa = X[a:a + chunk]
        S1 = np.stack([np.bincount(months, weights=x, minlength=nm) for x in Xa])
        S2 = np.stack([np.bincount(months, weights=x * x, minlength=nm) for x in Xa])
        s1, s2 = S1[:, ix].sum(-1), S2[:, ix].sum(-1)
        mean = s1 / N
        var = (s2 - N * mean ** 2) / (N - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:a + chunk] = np.where(var > 0, mean / np.sqrt(var) * np.sqrt(days_per_year), np.nan)
    return out


def paired_differences(daily: Mapping[str, np.ndarray], boot: Mapping[str, np.ndarray], pairs: Sequence[tuple[str, str]],
                       *, days_per_year: float = 365.0, level: float = 0.95) -> pd.DataFrame:
    """``a - b`` Sharpe differences of daily PnL series with paired bootstrap intervals and ``P(Δ > 0)``."""
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for a, b in pairs:
        d = boot[a] - boot[b]
        d = d[np.isfinite(d)]
        rows.append({"comparison": f"{a} − {b}", "Δ Sharpe": float(_sharpe(daily[a], days_per_year) - _sharpe(daily[b], days_per_year)),
                     "CI low": np.quantile(d, lo) if len(d) else np.nan, "CI high": np.quantile(d, hi) if len(d) else np.nan,
                     "P(Δ > 0)": float((d > 0).mean()) if len(d) else np.nan})
    return pd.DataFrame(rows).set_index("comparison")

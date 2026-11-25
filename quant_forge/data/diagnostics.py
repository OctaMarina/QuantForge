"""Sampling diagnostics of bar frames: achieved bars per day and bar durations, by asset and over time.

A bar's *duration* is the time between its end and the previous bar's end inside the same segment (the sampling
interval; for time bars it is the bar length), so gaps that split segments (missing klines, outages) are not
counted as long bars.  The achieved frequency of an asset in a period is the number of bars ending in the period
divided by the number of days of the period during which the asset had bars (between its first and last bar).

    from quant_forge.data import sampling_summary, sampling_by_period
    sampling_summary(bars, target_bars_per_day=6)          # one row per asset
    sampling_by_period(bars, freq="YS")                     # one row per (asset, year)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["bar_durations", "sampling_summary", "sampling_by_period"]

_DAY_US = 86_400 * 10 ** 6
_PERIOD_CODES = {"YS": "Y", "QS": "Q", "MS": "M", "W": "W", "D": "D", "Y": "Y", "Q": "Q", "M": "M"}


def bar_durations(bars: pd.DataFrame) -> pd.Series:
    """Duration in hours of every bar (NaN for the first bar of a segment), indexed like ``bars``."""
    end = bars["bar_end_us"].to_numpy(np.int64)
    d = np.r_[np.nan, np.diff(end) / 3.6e9]
    new_seg = np.zeros(len(bars), dtype=bool)
    if len(bars):
        new_seg[0] = True
    if "segment_id" in bars.columns:
        s = bars["segment_id"].to_numpy()
        new_seg[1:] |= s[1:] != s[:-1]
    if "bar_seq" in bars.columns:
        q = bars["bar_seq"].to_numpy()
        new_seg[1:] |= q[1:] != q[:-1] + 1
    if "complete" in bars.columns:
        c = bars["complete"].to_numpy().astype(bool)
        new_seg |= ~c
        new_seg[1:] |= ~c[:-1]
    d[new_seg] = np.nan
    return pd.Series(d, index=bars.index, name="duration_h")


def sampling_summary(bars: Mapping[str, pd.DataFrame], target_bars_per_day: float | None = None,
                     quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """One row per asset: first / last bar, bars, achieved bars per day (and its ratio to the target), number of
    segments, duration quantiles in hours and the share of bars lasting more than 4x the nominal length."""
    rows = []
    for a, b in bars.items():
        if not len(b):
            continue
        dur = bar_durations(b)
        days = (b["bar_end_us"].iloc[-1] - b["bar_start_us"].iloc[0]) / _DAY_US
        row = {"asset": a, "first bar": b.index[0], "last bar": b.index[-1], "bars": len(b),
               "bars/day": len(b) / days if days > 0 else np.nan,
               "segments": int(dur.isna().sum())}
        if target_bars_per_day:
            row["ratio to target"] = row["bars/day"] / target_bars_per_day
            row["share > 4x nominal"] = float((dur > 4 * 24 / target_bars_per_day).mean())
        for qv in quantiles:
            row[f"dur p{int(round(qv * 100)):02d} (h)"] = dur.quantile(qv)
        rows.append(row)
    return pd.DataFrame(rows).set_index("asset")


def sampling_by_period(bars: Mapping[str, pd.DataFrame], freq: str = "YS") -> pd.DataFrame:
    """One row per (asset, period): bars ending in the period, active days, bars per day and duration quantiles."""
    rows = []
    for a, b in bars.items():
        if not len(b):
            continue
        dur = bar_durations(b)
        first, last = b["bar_start_us"].iloc[0], b["bar_end_us"].iloc[-1]
        per = b.index.tz_convert("UTC").tz_localize(None).to_period(_PERIOD_CODES.get(freq, freq))
        g = pd.DataFrame({"period": per, "dur": dur.to_numpy()}).groupby("period")
        for p, gg in g:
            p0 = p.start_time.tz_localize("UTC").value // 1000
            p1 = (p.end_time.tz_localize("UTC").value + 1) // 1000
            active = (min(p1, last) - max(p0, first)) / _DAY_US
            rows.append({"asset": a, "period": p.start_time.tz_localize("UTC"), "bars": len(gg),
                         "active days": active, "bars/day": len(gg) / active if active > 0 else np.nan,
                         "dur p10 (h)": gg["dur"].quantile(0.1), "dur p50 (h)": gg["dur"].quantile(0.5),
                         "dur p90 (h)": gg["dur"].quantile(0.9)})
    return pd.DataFrame(rows)

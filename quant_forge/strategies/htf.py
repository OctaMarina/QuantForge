"""Conservative higher-timeframe (HTF) green-streak confirmation (TIS strategy).

A HTF candle is rebuilt from lower-timeframe bars whose *end* lies in a UTC
epoch-aligned bucket.  Its close is revealed only once the bucket has fully
ended AND every contributing bar was available, never at the last intraday bar.
Buckets that miss a boundary by more than the tolerance, or missing calendar
buckets, reset the streak.  This is deliberately stricter than TradingView's
``request.security(..., lookahead_off)`` mapping.

Deviation from the reference notebook (a bug correction, see ``tests/strategies/test_htf_correction.py``):
confirmation times are made monotone with a running maximum before the binary search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def htf_duration_us(htf: str) -> int:
    duration = pd.Timedelta(htf)
    micros = int(duration // pd.Timedelta(1, "us"))
    if micros <= 0:
        raise ValueError("htf must be a positive fixed timeframe, e.g. '1D' or '4h'")
    return micros


def higher_timeframe_momentum(idx, open_, close, start_us, end_us, ready_us,
                              streak_length: int, htf: str, boundary_tolerance_hours: float):
    """Map confirmed HTF green streaks back onto the bars ``idx`` of one segment.

    Returns ``(green_htf, htf_streak)`` aligned with ``idx``.
    """
    idx = np.asarray(idx, dtype=int)
    green_htf = np.zeros(len(idx), dtype=bool)
    htf_streak = np.zeros(len(idx), dtype=np.int32)
    if len(idx) == 0:
        return green_htf, htf_streak
    duration_us = htf_duration_us(htf)

    period = (end_us[idx].astype(np.int64) - 1) // duration_us
    unique_periods, first_idx = np.unique(period, return_index=True)
    last_idx = np.r_[first_idx[1:] - 1, len(idx) - 1]

    confirmations, confirmed_streaks = [], []
    streak = 0
    previous_period = None
    tolerance_us = int(boundary_tolerance_hours * 3_600_000_000)
    for key, a, b in zip(unique_periods, first_idx, last_idx):
        rows = idx[a:b + 1]
        period_start = int(key * duration_us)
        period_end = period_start + duration_us

        if previous_period is not None and int(key) != previous_period + 1:
            streak = 0  # a missing calendar bucket breaks the streak
        previous_period = int(key)

        covers_start = start_us[rows[0]] <= period_start + tolerance_us
        covers_end = end_us[rows[-1]] >= period_end - tolerance_us
        if covers_start and covers_end:
            streak = streak + 1 if close[rows[-1]] > open_[rows[0]] else 0
        else:
            streak = 0

        confirmations.append(max(period_end, int(np.max(ready_us[rows]))))
        confirmed_streaks.append(streak)

    # CORRECTION vs the reference notebook: `searchsorted` needs a sorted array, but when bars become
    # available out of order (a later bar available before an earlier one) the confirmation times of
    # consecutive buckets are not monotone and the reference result becomes undefined (it depends on the
    # array length and the numpy version).  A running maximum publishes a bucket no earlier than its
    # predecessors, which is the intended semantics.  It changes NOTHING when confirmations are already
    # monotone (e.g. bar_available_us == bar_end_us, as in the BTCUSDT dataset).
    confirmations = np.maximum.accumulate(np.asarray(confirmations, dtype=np.int64))
    confirmed_streaks = np.asarray(confirmed_streaks, dtype=np.int32)
    known = np.searchsorted(confirmations, ready_us[idx], side="right") - 1
    good = known >= 0
    htf_streak[good] = confirmed_streaks[known[good]]
    green_htf[good] = htf_streak[good] >= streak_length
    return green_htf, htf_streak

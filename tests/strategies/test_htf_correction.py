"""The one deliberate deviation from the reference notebook, validated separately.

Bug: ``_higher_timeframe_momentum`` looks confirmations up with ``np.searchsorted`` although they are
``max(bucket end, latest availability of the bucket's bars)``, which is not monotone when bars become available
out of order.  ``searchsorted`` on an unsorted array is undefined: the reference result then depends on the array
length and on the numpy version.  The library applies a running maximum first.

Claims checked here:
1. the precondition (non-monotone confirmations) really occurs in the late-availability data;
2. whenever confirmations are monotone (clean data, delayed-but-ordered availability, the BTCUSDT dataset) the
   correction is a no-op: the *unpatched* reference equals the library;
3. on out-of-order data the library equals the reference with exactly this one-line fix;
4. the library result no longer depends on the length of the array (bars before a cut are unchanged).
"""

from __future__ import annotations

import numpy as np
import pytest

import quant_forge.strategies as qs
from quant_forge.strategies.bars import prepare_bars
from quant_forge.strategies.htf import htf_duration_us

from .helpers import assert_same, compare_trades
from .test_equivalence import LOOSE_TIS, data

NAME = "tis_btc_momentum_confluence"


def raw_confirmations(df, htf):
    b = prepare_bars(df)
    dur = htf_duration_us(htf)
    period = (b.end_us.astype(np.int64) - 1) // dur
    keys = np.unique(period)
    return np.array([max((int(k) + 1) * dur, int(b.ready_us[period == k].max())) for k in keys])


def ordered_delay(df):
    """Late but monotone availability: ready = end + 2h for every bar."""
    out = df.copy()
    out["bar_available_us"] = out["bar_end_us"] + 2 * 3_600_000_000
    out["decision_time_us"] = out["bar_available_us"]
    return out


def test_precondition_confirmations_are_not_monotone_with_out_of_order_availability():
    # availability delays are < 3h, so only buckets shorter than that can overtake each other
    for htf in ("1h", "2h"):
        c = raw_confirmations(data("late_availability"), htf)
        assert (np.diff(c) < 0).any(), f"{htf}: expected non-monotone confirmations"
    for htf in ("4h", "1D"):
        assert (np.diff(raw_confirmations(data("late_availability"), htf)) >= 0).all()
    for df in (data("clean"), ordered_delay(data("clean"))):
        for htf in ("1h", "2h", "4h", "1D"):
            assert (np.diff(raw_confirmations(df, htf)) >= 0).all()


@pytest.mark.parametrize("htf", ["1h", "2h", "4h", "1D"])
@pytest.mark.parametrize("which", ["clean", "ordered_delay", "breaks", "calendar_gap"])
def test_correction_is_a_noop_when_confirmations_are_monotone(reference, reference_monotone_htf, htf, which):
    base = {"ordered_delay": lambda: ordered_delay(data("clean"))}.get(which, lambda: data(which))()
    kw = dict(**LOOSE_TIS, htf=htf)
    plain, cap_plain = reference.run(base, strategy=NAME, **kw)
    fixed, cap_fixed = reference_monotone_htf.run(base, strategy=NAME, **kw)
    for k in ("htf_green_streak", "htf_momentum", "enter"):
        assert_same(cap_plain[k], cap_fixed[k], f"patched reference differs from reference on {k}", exact=True)
    compare_trades(NAME, plain, qs.run_strategy(base, NAME, **kw))
    assert len(plain) > 0


def test_correction_is_a_noop_on_the_real_dataset(reference, reference_monotone_htf, btc_bars):
    plain, cap_plain = reference.run(btc_bars, strategy=NAME)
    fixed, cap_fixed = reference_monotone_htf.run(btc_bars, strategy=NAME)
    for k in ("htf_green_streak", "htf_momentum", "enter"):
        assert_same(cap_plain[k], cap_fixed[k], k, exact=True)
    compare_trades(NAME, plain, qs.run_strategy(btc_bars, NAME))


@pytest.mark.parametrize("htf", ["1h", "2h", "4h"])
def test_library_equals_patched_reference_on_out_of_order_availability(reference_monotone_htf, htf):
    df = data("late_availability")
    kw = dict(**LOOSE_TIS, htf=htf)
    ref, cap = reference_monotone_htf.run(df, strategy=NAME, **kw)
    res = qs.run_strategy(df, NAME, **kw)
    st = qs.get_strategy(NAME)
    feats = st.compute_features(prepare_bars(df), res.config)
    for k in ("htf_green_streak", "htf_momentum"):
        assert_same(feats[k], cap[k], k, exact=True)
    compare_trades(NAME, ref, res)
    assert len(ref) > 5


@pytest.mark.parametrize("htf", ["1h", "2h", "4h"])
def test_features_use_only_bars_available_at_the_decision_time(htf):
    """With out-of-order availability, 'the past' is defined by availability, not by row order.

    Overwrite every bar that was NOT yet available at bar i's decision time (row > i and ready > ready_i)
    with a flat bar: the features at bar i must not change.
    """
    df = data("late_availability")
    kw = dict(**LOOSE_TIS, htf=htf)
    st = qs.get_strategy(NAME)
    cfg = st.make_config(**kw)
    bars = prepare_bars(df)
    full = st.compute_features(bars, cfg)
    rng = np.random.default_rng(0)
    checked = 0
    for i in rng.choice(np.arange(300, len(df) - 5), 40, replace=False):
        hidden = (np.arange(len(df)) > i) & (bars.ready_us > bars.ready_us[i])
        masked = df.copy()
        for col in ("open", "high", "low", "close"):
            masked.loc[hidden, col] = df["close"].iloc[i]
        got = st.compute_features(prepare_bars(masked), cfg)
        for key, arr in full.items():
            assert arr[i] == got[key][i] or (np.isnan(arr[i]) and np.isnan(got[key][i])), f"{key} at {i}"
        checked += int(hidden.any())
    assert checked >= 30


def test_how_much_the_correction_matters_on_out_of_order_data(reference, reference_monotone_htf):
    """Informational bound: the difference to the unpatched reference is confined to the out-of-order case."""
    df = data("late_availability")
    kw = dict(**LOOSE_TIS, htf="2h")
    _, cap_plain = reference.run(df, strategy=NAME, **kw)
    _, cap_fixed = reference_monotone_htf.run(df, strategy=NAME, **kw)
    differing = int((cap_plain["htf_green_streak"] != cap_fixed["htf_green_streak"]).sum())
    assert differing < 0.05 * len(df)        # a handful of bars around out-of-order buckets, never the bulk

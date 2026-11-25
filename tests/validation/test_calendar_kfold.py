import numpy as np
import pandas as pd
import pytest

from quant_forge.validation import CalendarPurgedKFold, PurgedKFold, calendar_windows, expanding_split, to_int64_ns
from tests.synthetic import make_trade_panel


def _brute(t0, t1, n_splits, frac, cover_all=True):
    cuts = np.quantile(t0, np.linspace(0, 1, n_splits + 1)).astype("int64")
    h = int(frac * (t0.max() - t0.min()))
    out = []
    for k in range(n_splits):
        if cover_all:
            test = [i for i in range(len(t0)) if (k == 0 or cuts[k] <= t0[i]) and (k == n_splits - 1 or t0[i] < cuts[k + 1])]
        else:
            test = [i for i in range(len(t0)) if cuts[k] <= t0[i] and (t0[i] < cuts[k + 1] or (k == n_splits - 1 and t0[i] == cuts[k + 1]))]
        if not test:
            continue
        ws, we = min(t0[i] for i in test), max(t1[i] for i in test)
        train = [i for i in range(len(t0)) if i not in test and not (t0[i] <= we and t1[i] >= ws)
                 and not (we < t0[i] <= we + h)]
        out.append((train, test))
    return out


@pytest.mark.parametrize("cover_all", [True, False])
def test_calendar_kfold_matches_brute_force_on_panel(cover_all):
    tr = make_trade_panel(300, seed=1)
    t0, t1 = to_int64_ns(tr["decision_time"]), to_int64_ns(tr["exit_time"])
    got = [(a.tolist(), b.tolist()) for a, b in CalendarPurgedKFold(4, 0.02, cover_all).split(t0, t1)]
    assert got == _brute(t0, t1, 4, 0.02, cover_all)


def test_legacy_edges_can_leave_the_first_event_untested():
    """float64 quantiles of int64 ns round: the legacy rule may miss the extreme events, the default never does."""
    tr = make_trade_panel(500, seed=4)
    t0, t1 = to_int64_ns(tr["decision_time"]), to_int64_ns(tr["exit_time"])
    tested_new = np.concatenate([b for _, b in CalendarPurgedKFold(5, 0.01).split(t0, t1)])
    tested_old = np.concatenate([b for _, b in CalendarPurgedKFold(5, 0.01, cover_all=False).split(t0, t1)])
    assert sorted(tested_new.tolist()) == list(range(len(tr)))
    assert set(tested_old) <= set(tested_new)


def test_calendar_kfold_invariants():
    tr = make_trade_panel(500, seed=4)
    cv = CalendarPurgedKFold(5, 0.01)
    folds = list(cv.split_with_diagnostics(tr["decision_time"], tr["exit_time"]))
    t0, t1 = to_int64_ns(tr["decision_time"]), to_int64_ns(tr["exit_time"])
    assert sorted(np.concatenate([f.test for f in folds]).tolist()) == list(range(len(tr)))
    for f in folds:
        parts = np.concatenate([f.train, f.test, f.purged, f.embargoed])
        assert sorted(parts.tolist()) == list(range(len(tr)))                         # disjoint and complete
        tr_ = f.train
        assert not np.any((t0[tr_] <= f.window_end) & (t1[tr_] >= f.window_start))      # no overlap leaks
    with pytest.raises(ValueError):
        list(cv.split(t0, t0 - 1))


def test_calendar_rule_agrees_with_purged_kfold_single_series():
    """On one series the purge decision equals PurgedKFold's for the same test window."""
    tr = make_trade_panel(200, seed=7)
    t0 = pd.DatetimeIndex(tr["decision_time"])
    t1 = pd.Series(pd.DatetimeIndex(tr["exit_time"]), index=t0)
    lib = PurgedKFold(n_splits=4, samples_info_sets=t1, pct_embargo=0.0)
    a0, a1 = to_int64_ns(t0), to_int64_ns(t1)
    for f in lib.split_with_diagnostics(pd.DataFrame(index=t0)):
        ws, we = a0[f.test].min(), a1[f.test].max()
        mine = ~((a0 <= we) & (a1 >= ws))
        mine[f.test] = False
        np.testing.assert_array_equal(np.flatnonzero(mine), f.train)


def test_calendar_windows_and_expanding_split():
    w = calendar_windows("2019-03-01", end="2026-11-19", years=1)
    assert len(w) == 8 and w[0][0] == pd.Timestamp("2019-03-01", tz="UTC") and w[-1][1] == pd.Timestamp("2026-11-19", tz="UTC")
    assert all(a[1] == b[0] for a, b in zip(w[:-1], w[1:]))
    assert len(calendar_windows("2020-01-01", end="2020-12-31", months=3)) == 4
    tr = make_trade_panel(300)
    train, test = expanding_split(tr, *w[1], train_mask=tr["asset"].ne("A1"))
    assert (train["exit_time"] < w[1][0]).all() and not train["asset"].eq("A1").any()
    assert test["decision_time"].between(w[1][0], w[1][1], inclusive="left").all()

"""PurgedKFold: hand-calculated folds, boundary cases, validation, brute-force agreement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_forge.validation import PurgedFold, PurgedKFold
from quant_forge.validation.purged_kfold import _embargo_count

from .helpers import (brute_force_split, count_overlaps, day, folds, kfold_bounds, make_info, make_X)

# --------------------------------------------------------------------------------------------------
# Hand-calculated example A: 6 events, starts 0..5, non-monotone ends, k=3 -> test {0,1},{2,3},{4,5}
#   fold 0: window [0, 3]  -> purge 2 (inside), 3 (start == window end: shared endpoint)       train {4,5}
#   fold 1: window [2, 4]  -> purge 1 (end 3 >= 2, crosses start), 4 (start == window end)      train {0,5}
#                             event 0 (end 1 < 2) survives
#   fold 2: window [4, 6]  -> purge 3 (end == 4 == window start: shared endpoint)               train {0,1,2}
#                             window end is event 4's end (6), NOT the last row's end (5)
# --------------------------------------------------------------------------------------------------
A = make_info(range(6), [1, 3, 3, 4, 6, 5])


def test_example_a_zero_embargo():
    cv = PurgedKFold(3, samples_info_sets=A)
    assert folds(cv, A) == [([4, 5], [0, 1]), ([0, 5], [2, 3]), ([0, 1, 2], [4, 5])]


def test_example_a_positive_embargo():
    # pct 0.34 -> floor(6 * 0.34) = 2
    #   fold 0: p = searchsorted(starts, 3, left) = 3, embargo {3, 4}; 4 is embargo-only         train {5}
    #   fold 1: p = 4, embargo {4, 5}; 5 is embargo-only                                         train {0}
    #   fold 2: window end 6 is after every start, p = 6 = n: nothing to embargo (clipping)      train {0,1,2}
    cv = PurgedKFold(3, samples_info_sets=A, pct_embargo=0.34)
    assert folds(cv, A) == [([5], [0, 1]), ([0], [2, 3]), ([0, 1, 2], [4, 5])]
    details = list(cv.split_with_diagnostics(make_X(A)))
    assert [d.purged.tolist() for d in details] == [[2, 3], [1, 4], [3]]
    assert [d.embargoed.tolist() for d in details] == [[4], [5], []]
    assert details[2].test_start == day(4) and details[2].test_end == day(6)   # max end, not last row's end


def test_zero_embargo_still_purges_and_embargo_never_readmits():
    zero = folds(PurgedKFold(3, samples_info_sets=A), A)
    emb = folds(PurgedKFold(3, samples_info_sets=A, pct_embargo=0.34), A)
    assert all(set(e[0]) <= set(z[0]) for z, e in zip(zero, emb))
    assert zero[0][0] != [2, 3, 4, 5]


# --- overlapping intervals on both sides, containment ---------------------------------------------
def test_overlap_on_both_sides_and_containment():
    # 8 events, k=4 -> test pairs; starts 0..7. Test fold 1 = {2,3}.
    # event 0 is long: [0, 6] contains the whole window of fold 1 (containment, train event contains test)
    # event 1: [1, 2]  ends exactly at the window start 2 (left shared endpoint)
    # test events 2: [2, 3], 3: [3, 5]  -> window [2, 5]
    # event 4: [4, 4] zero duration inside the window (right side, contained by test event 3)
    # event 5: [5, 5] zero duration at the window end (right shared endpoint)
    # event 6: [6, 7] starts after the window end; event 7: [7, 7]
    info = make_info(range(8), [6, 2, 3, 5, 4, 5, 7, 7])
    cv = PurgedKFold(4, samples_info_sets=info)
    f = folds(cv, info)
    assert f[1] == ([6, 7], [2, 3])         # 0 (contains), 1 (shared endpoint), 4 (contained), 5 (shared) purged
    # fold 0: test {0,1}, window [0, 6]: purges everything starting <= 6 -> events 2..6; only 7 survives
    assert f[0] == ([7], [0, 1])
    # fold 3: test {6,7}, window [6, 7]: event 0 ends at 6 == window start (shared) -> purged;
    # 1..5 end before 6 -> kept
    assert f[3] == ([1, 2, 3, 4, 5], [6, 7])


def test_test_event_containing_train_events():
    # the single long event in the test fold (event 2) contains later short events 3,4 (right side)
    info = make_info(range(6), [0, 1, 5, 3, 4, 5])      # fold 1 test = {2,3}: window [2, 5]
    f = folds(PurgedKFold(3, samples_info_sets=info), info)
    assert f[1] == ([0, 1], [2, 3])
    # events 4 (inside) and 5 (start == window end 5) are purged


def test_zero_duration_events_never_overlap_distinct_neighbours():
    info = make_info(range(5), range(5))                 # every event is a single instant
    f = folds(PurgedKFold(5, samples_info_sets=info), info)
    assert f == [([1, 2, 3, 4], [0]), ([0, 2, 3, 4], [1]), ([0, 1, 3, 4], [2]),
                 ([0, 1, 2, 4], [3]), ([0, 1, 2, 3], [4])]


def test_embargo_convention_when_max_end_equals_last_test_start():
    # zero-duration events: window end == start of the last test event, so p is that test position and the
    # embargo window [p, p + m) overlaps the test fold; only m - 1 later observations are removed.
    info = make_info(range(5), range(5))
    # pct 0.4 -> m = 2: fold i removes only i+1 (clipped at the end)
    f = folds(PurgedKFold(5, samples_info_sets=info, pct_embargo=0.4), info)
    assert f == [([2, 3, 4], [0]), ([0, 3, 4], [1]), ([0, 1, 4], [2]), ([0, 1, 2], [3]), ([0, 1, 2, 3], [4])]
    # m = 1 removes nothing extra for zero-duration events
    f1 = folds(PurgedKFold(5, samples_info_sets=info, pct_embargo=0.2), info)
    assert f1 == folds(PurgedKFold(5, samples_info_sets=info), info)


def test_embargo_counts_event_starting_exactly_at_window_end():
    # fold 0 test {0,1}: window [0, 2]; event 2 starts exactly at 2: it is position p, purged by the closed
    # rule and the first embargoed slot, so with m = 2 only position 3 is additionally embargoed.
    info = make_info(range(6), [0, 2, 2, 3, 9, 9])
    cv = PurgedKFold(3, samples_info_sets=info, pct_embargo=0.34)       # m = 2
    d = next(iter(cv.split_with_diagnostics(make_X(info))))
    assert d.purged.tolist() == [2] and d.embargoed.tolist() == [3] and d.train.tolist() == [4, 5]


def test_embargo_clipping_with_valid_training_set():
    info = make_info(range(10), range(10))
    cv = PurgedKFold(5, samples_info_sets=info, pct_embargo=0.5)          # m = 5, tests {0,1},{2,3},...
    f = folds(cv, info)
    # fold 4 (last): window end = 9 -> p = 9, embargo {9..13} clipped to {9} (test) -> train 0..7
    assert f[4] == ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    # fold 3: window [6, 7], p = 7 (start == end of last test event): embargo {7..11} -> {8, 9}
    assert f[3] == ([0, 1, 2, 3, 4, 5], [6, 7])


def test_embargo_count_formula():
    assert _embargo_count(100, 0.0) == 0
    assert _embargo_count(100, 0.07) == 7
    assert _embargo_count(100, 0.29) == 29           # 100 * 0.29 == 28.999999999999996 in binary floating point
    assert _embargo_count(6, 1 / 3) == 2
    assert _embargo_count(99, 0.01) == 0             # floor(0.99)
    assert _embargo_count(10, 0.55) == 5             # floor(5.5)
    assert _embargo_count(60, 0.05) == 3


# --- irregular timestamps, tz-aware, non-monotonic ends ----------------------------------------------
def test_irregular_timestamps_non_monotone_ends_tz_aware():
    start = pd.DatetimeIndex(["2024-03-01 09:30:00", "2024-03-01 09:31:07", "2024-03-01 09:45:00",
                              "2024-03-01 10:02:30", "2024-03-01 13:00:00", "2024-03-02 09:30:00"], tz="UTC")
    end = pd.DatetimeIndex(["2024-03-01 10:30:00", "2024-03-01 09:40:00", "2024-03-01 09:45:00",
                            "2024-03-01 13:00:00", "2024-03-01 13:00:00", "2024-03-02 09:30:01"], tz="UTC")
    info = pd.Series(end, index=start)
    # k=3: tests {0,1}, {2,3}, {4,5}
    # fold 0 window [09:30, 10:30]: purge 2 (09:45) and 3 (10:02:30) -> train {4, 5}
    # fold 1 window [09:45, 13:00]: event 0 ends 10:30 >= 09:45 -> purged, event 1 ends 09:40 < 09:45 -> kept,
    #   event 4 starts 13:00 == window end -> purged (shared endpoint); event 5 survives -> train {1, 5}
    # fold 2 window [13:00, next day 09:30:01]: event 3 ends exactly 13:00 (shared) -> purged,
    #   0,1,2 end before 13:00 -> kept -> train {0, 1, 2}
    assert folds(PurgedKFold(3, samples_info_sets=info), info) == [([4, 5], [0, 1]), ([1, 5], [2, 3]),
                                                                    ([0, 1, 2], [4, 5])]


# --- test fold sizes / coverage ----------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(6, 3), (10, 3), (11, 4), (7, 7), (23, 5), (100, 10)])
def test_fold_sizes_and_exact_coverage(n, k):
    info = make_info(range(n), range(n))        # no overlap, so the test folds are visible directly
    cv = PurgedKFold(k, samples_info_sets=info)
    tests = [te.tolist() for _, te in cv.split(make_X(info))]
    assert [(t[0], t[-1] + 1) for t in tests] == kfold_bounds(n, k)
    assert sorted(i for t in tests for i in t) == list(range(n))     # each observation tested exactly once


# --- random agreement with the independent brute-force reference --------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_agrees_with_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 45))
    k = int(rng.integers(2, min(8, n) + 1))
    # coarse integer grid -> many ties between ends and later starts, zero durations, equal ends
    starts = np.sort(rng.choice(np.arange(0, n * 3), size=n, replace=False))
    ends = starts + rng.choice([0, 0, 1, 2, 3, 5, 8, 20], size=n)
    info = make_info(starts, ends)
    pct = float(rng.choice([0.0, 0.0, 0.05, 0.1, 0.2]))
    cv = PurgedKFold(k, samples_info_sets=info, pct_embargo=pct)
    try:
        got = folds(cv, info)
    except ValueError as exc:
        assert "no training observations" in str(exc)           # the brute force produces an empty train too
        assert any(not tr for tr, _ in brute_force_split(info, k, pct))
        return
    expected = brute_force_split(info, k, pct)
    assert got == expected
    for tr, te in got:
        assert count_overlaps(info, tr, te) == 0                # no leaking pair remains
        assert not set(tr) & set(te)


def test_embargo_excludes_exactly_the_documented_positions():
    rng = np.random.default_rng(7)
    n = 40
    starts = np.arange(n) * 2
    ends = starts + rng.integers(0, 9, size=n)
    info = make_info(starts, ends)
    cv = PurgedKFold(4, samples_info_sets=info, pct_embargo=0.1)         # m = 4
    for d in cv.split_with_diagnostics(make_X(info)):
        p = int(np.searchsorted(info.index.to_numpy(), np.datetime64(d.test_end.to_datetime64()), "left"))
        window = set(range(p, min(p + 4, n)))
        excluded = set(d.purged) | set(d.embargoed) | set(d.test)
        assert window <= excluded
        assert set(d.embargoed) == window - set(d.test) - set(d.purged)


# --- structural properties ----------------------------------------------------------------------------
def test_partition_deterministic_and_dtype():
    rng = np.random.default_rng(3)
    n = 50
    starts = np.cumsum(rng.integers(1, 4, size=n))
    info = make_info(starts, starts + rng.integers(0, 12, size=n))
    cv = PurgedKFold(5, samples_info_sets=info, pct_embargo=0.04)
    X = make_X(info)
    first = [(tr.copy(), te.copy()) for tr, te in cv.split(X)]
    second = list(cv.split(X))
    for (a, b), (c, d) in zip(first, second):
        assert np.array_equal(a, c) and np.array_equal(b, d)
        assert a.dtype.kind == "i" and b.dtype.kind == "i"
        assert np.all(np.diff(a) > 0) and np.all(np.diff(b) > 0)
        assert not set(a) & set(b)
        X.iloc[a], X.iloc[b]                                   # usable with .iloc
    for det in cv.split_with_diagnostics(X):
        parts = [det.train, det.test, det.purged, det.embargoed]
        assert sorted(np.concatenate(parts).tolist()) == list(range(n))   # categories partition the sample
        assert isinstance(det, PurgedFold)
    assert cv.get_n_splits() == cv.get_n_splits(X, groups=None) == 5


def test_does_not_mutate_inputs_and_is_a_snapshot():
    info = A.copy()
    X = make_X(info)
    info_before, X_before = info.copy(), X.copy()
    cv = PurgedKFold(3, samples_info_sets=info, pct_embargo=0.34)
    expected = folds(cv, A)
    list(cv.split(X))
    pd.testing.assert_series_equal(info, info_before)
    pd.testing.assert_frame_equal(X, X_before)
    info.iloc[:] = info.iloc[0]                                 # caller mutates afterwards
    assert folds(cv, A) == expected


def test_y_accepted_and_checked():
    X = make_X(A)
    cv = PurgedKFold(3, samples_info_sets=A)
    assert len(list(cv.split(X, pd.Series(np.zeros(6), index=A.index)))) == 3
    assert len(list(cv.split(X, np.zeros(6)))) == 3
    with pytest.raises(ValueError, match="y has"):
        cv.split(X, np.zeros(5))
    with pytest.raises(ValueError, match="y.index is not aligned"):
        cv.split(X, pd.Series(np.zeros(6), index=A.index + pd.Timedelta(days=1)))


def test_mixed_datetime_units_are_compared_by_instant():
    ns = make_info(range(6), [1, 3, 3, 4, 6, 5])
    us_index = ns.index.as_unit("us")
    us_info = pd.Series(ns.to_numpy().astype("datetime64[us]"), index=us_index)
    ref = folds(PurgedKFold(3, samples_info_sets=ns), ns)
    assert folds(PurgedKFold(3, samples_info_sets=us_info), us_info) == ref
    assert [(tr.tolist(), te.tolist()) for tr, te in
            PurgedKFold(3, samples_info_sets=ns).split(make_X(us_info))] == ref     # X in another unit


def test_large_input_runs_without_pairwise_matrix():
    n = 200_000
    starts = np.arange(n, dtype="int64") * 60
    ends = starts + (np.arange(n) % 17) * 60
    idx = pd.to_datetime(starts, unit="s")
    info = pd.Series(pd.to_datetime(ends, unit="s"), index=idx)
    cv = PurgedKFold(5, samples_info_sets=info, pct_embargo=0.001)
    X = pd.DataFrame({"a": np.zeros(n)}, index=idx)
    for d in cv.split_with_diagnostics(X):
        s, e = idx.to_numpy()[d.train], info.to_numpy()[d.train]
        assert not np.any((s <= d.test_end.to_datetime64()) & (e >= d.test_start.to_datetime64()))
        assert len(d.test) == n // 5


# --- invalid input -------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [1, 0, -3, 2.5, True, "3", None])
def test_invalid_n_splits(bad):
    with pytest.raises(ValueError, match="n_splits"):
        PurgedKFold(bad, samples_info_sets=A)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan"), float("inf"), "0.1", None, True])
def test_invalid_pct_embargo(bad):
    with pytest.raises(ValueError, match="pct_embargo"):
        PurgedKFold(3, samples_info_sets=A, pct_embargo=bad)


def test_samples_info_sets_is_required_keyword():
    with pytest.raises(TypeError):
        PurgedKFold(3)                                                          # noqa
    with pytest.raises(TypeError):
        PurgedKFold(3, A)                                                       # noqa  (keyword-only)


def test_n_splits_larger_than_samples():
    with pytest.raises(ValueError, match="greater than the number of samples"):
        PurgedKFold(7, samples_info_sets=A)


def test_info_sets_must_be_series():
    with pytest.raises(TypeError, match="Series"):
        PurgedKFold(3, samples_info_sets=A.to_frame())
    with pytest.raises(TypeError, match="Series"):
        PurgedKFold(3, samples_info_sets=list(A))


def test_index_must_be_datetimeindex():
    s = pd.Series(A.to_numpy(), index=range(6))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        PurgedKFold(3, samples_info_sets=s)


def test_multiindex_rejected_as_multiasset():
    mi = pd.MultiIndex.from_arrays([A.index, ["BTC"] * 6], names=["t", "asset"])
    with pytest.raises(ValueError, match="multi-asset"):
        PurgedKFold(3, samples_info_sets=pd.Series(A.to_numpy(), index=mi))


def test_duplicate_timestamps_rejected():
    info = make_info([0, 1, 1, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="duplicated"):
        PurgedKFold(3, samples_info_sets=info)


def test_unsorted_rejected_not_sorted():
    info = make_info([0, 2, 1, 3, 4, 5], [1, 3, 2, 4, 5, 6])
    with pytest.raises(ValueError, match="not chronological"):
        PurgedKFold(3, samples_info_sets=info)


def test_nat_rejected():
    idx = pd.DatetimeIndex([day(0), pd.NaT, day(2), day(3)])
    with pytest.raises(ValueError, match="NaT"):
        PurgedKFold(2, samples_info_sets=pd.Series([day(1), day(2), day(3), day(4)], index=idx))
    ends = A.copy()
    ends.iloc[2] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        PurgedKFold(3, samples_info_sets=ends)


def test_end_before_start_rejected():
    info = make_info(range(6), [1, 3, 1, 4, 6, 5])       # event 2 starts at 2, ends at 1
    with pytest.raises(ValueError, match="end before they start"):
        PurgedKFold(3, samples_info_sets=info)


def test_values_must_be_datetimes():
    with pytest.raises(TypeError, match="datetimes"):
        PurgedKFold(3, samples_info_sets=pd.Series(np.arange(6), index=A.index))


def test_timezone_mismatches():
    aware_idx = make_info(range(6), range(6), tz="UTC")
    mixed = pd.Series(A.to_numpy(), index=aware_idx.index)                      # naive ends, aware index
    with pytest.raises(ValueError, match="Incompatible timezones"):
        PurgedKFold(3, samples_info_sets=mixed)
    with pytest.raises(ValueError, match="Incompatible timezones"):
        PurgedKFold(3, samples_info_sets=aware_idx.copy().dt.tz_convert("Europe/Bucharest"))
    # X in another timezone / naive
    cv = PurgedKFold(3, samples_info_sets=aware_idx)
    with pytest.raises(ValueError, match="Incompatible timezones"):
        cv.split(make_X(aware_idx).tz_localize(None))
    with pytest.raises(ValueError, match="Incompatible timezones"):
        cv.split(make_X(aware_idx).tz_convert("Europe/Bucharest"))


def test_alignment_errors():
    cv = PurgedKFold(3, samples_info_sets=A)
    X = make_X(A)
    with pytest.raises(ValueError, match="entries"):
        cv.split(X.iloc[:5])
    with pytest.raises(ValueError, match="not aligned"):
        cv.split(X.set_axis(A.index + pd.Timedelta(hours=1)))
    reordered = X.iloc[[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError, match="not aligned"):
        cv.split(reordered)
    with pytest.raises(TypeError, match="cannot be verified"):
        cv.split(X.to_numpy())
    with pytest.raises(TypeError, match="DatetimeIndex"):
        cv.split(X.reset_index(drop=True))
    with pytest.raises(TypeError):
        cv.split(None)


def test_validation_happens_before_first_fold():
    cv = PurgedKFold(3, samples_info_sets=A)
    with pytest.raises(ValueError):
        cv.split(make_X(A).iloc[:4])            # raises on the call itself, not on first iteration


def test_groups_rejected():
    cv = PurgedKFold(3, samples_info_sets=A)
    with pytest.raises(ValueError, match="groups"):
        cv.split(make_X(A), groups=np.arange(6))
    with pytest.raises(ValueError, match="groups"):
        cv.get_n_splits(groups=np.arange(6))


def test_multiasset_X_rejected():
    X = make_X(A)
    X.index = pd.MultiIndex.from_arrays([A.index, ["BTC"] * 6])
    with pytest.raises(TypeError, match="multi-asset"):
        PurgedKFold(3, samples_info_sets=A).split(X)


def test_empty_training_fold_from_purging():
    # every event reaches the end of the sample: whatever fold is tested, nothing is left to train on
    info = make_info(range(4), [9, 9, 9, 9])
    with pytest.raises(ValueError, match=r"Fold 0 has no training observations"):
        list(PurgedKFold(2, samples_info_sets=info).split(make_X(info)))


def test_empty_training_fold_from_embargo():
    info = make_info(range(4), range(4))
    cv = PurgedKFold(2, samples_info_sets=info, pct_embargo=0.5)       # m = 2 leaves fold 0 train {3}
    assert folds(cv, info)[0] == ([3], [0, 1])                          # p = 1 -> {1, 2} embargoed
    big = PurgedKFold(2, samples_info_sets=info, pct_embargo=0.75)     # m = 3 -> {1, 2, 3}: nothing left
    with pytest.raises(ValueError, match="no training observations left"):
        list(big.split(make_X(info)))


def test_repr_is_compact():
    r = repr(PurgedKFold(3, samples_info_sets=A, pct_embargo=0.1))
    assert r.startswith("PurgedKFold(n_splits=3, pct_embargo=0.1") and "\n" not in r

# `quant_forge.validation` — purged cross-validation and walk-forward windows

Events (trades, triple-barrier labels) carry an information interval `[start, end]`. Ordinary K-fold leaks because a
training event can overlap a test event in time. This package implements purging and embargo (López de Prado, *AFML*
ch. 7) for one series and for a panel of assets, plus walk-forward windows.

| Object | Use it for |
|---|---|
| `PurgedKFold` | one series of events (unique, sorted timestamps); scikit-learn cross-validator |
| `CalendarPurgedKFold` | a panel of events of many assets (repeated / interleaved timestamps); common calendar folds |
| `calendar_windows`, `expanding_split` | outer walk-forward evaluation |

## `PurgedKFold` — one series

```python
from quant_forge.validation import PurgedKFold
from sklearn.model_selection import cross_validate

# t1: pd.Series, index = event start (chronological, unique DatetimeIndex == X.index), values = event end
cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.01)
cross_validate(pipeline, X, y, cv=cv)                          # scikit-learn's cv= argument
for train, test in cv.split(X): X.iloc[train], X.iloc[test]    # sorted positional integer arrays
for fold in cv.split_with_diagnostics(X): fold.purged, fold.embargoed   # why each observation was dropped
```

Demonstration and audit on synthetic overlapping events: `research/cross-validation/test_purged_kfold_quantforge.ipynb`.

**Rules.** Test folds are contiguous, chronological and sized exactly like `KFold(shuffle=False)`; nothing is shuffled and
every observation is tested once. Training is everything outside the test fold (before *and* after it) minus:

* **Purged** events: closed interval `[start, end]` intersecting the test window `[earliest test start, max test end]`
  (max over the fold, not the last row). Sharing an endpoint counts as overlap; containment and zero-duration events are covered.
* **Embargoed** events: `embargo_count = floor(n_samples * pct_embargo)` (fraction of the *full* input, not of the test
  fold, not a time duration). With `p = searchsorted(starts, max_test_end, side="left")`, positions `p … p+embargo_count-1`
  (clipped at the end) leave the training set. Purging is applied independently, so zero embargo still purges.
* A fold left without training observations raises `ValueError`.

**Input contract.** `samples_info_sets` is a `pd.Series` (index: start, values: end) with a chronological, unique,
NaT-free `DatetimeIndex` equal to `X.index`; end times need not be sorted but must be `>=` their start and share the index's
timezone. Unsorted input is rejected, never sorted. `groups`, `MultiIndex`/multi-asset input, duplicated timestamps and
`X` without an index are rejected explicitly – use `CalendarPurgedKFold` for panels.

**Subsets.** The splitter is bound to the exact events it was built from. For a training window or the inner loop of
nested validation build a new `PurgedKFold` from `samples_info_sets.loc[subset.index]`.

**From the book vs. choices.** From AFML §7.4 (Snippets 7.1–7.3): interval purging against the test window with the
maximum test end, the `searchsorted` embargo position, embargo as a fraction of the sample, training on both sides of
the test fold. QuantForge's choices: closed intervals, purging independent of the embargo, `floor` with a relative `1e-9`
tolerance (`100 * 0.29` is 29), strict input validation, no sklearn dependency, `split_with_diagnostics`.

## `CalendarPurgedKFold` — a panel of events

```python
from quant_forge.validation import CalendarPurgedKFold

cv = CalendarPurgedKFold(n_splits=4, embargo_frac=0.01)
folds = list(cv.split(train["decision_time"], train["exit_time"]))      # [(train_idx, test_idx), ...] positional
for f in cv.split_with_diagnostics(t0, t1): f.train, f.test, f.purged, f.embargoed, f.window_start, f.window_end
```

* Fold boundaries are calendar dates – the quantiles of the decision times – identical for all assets.
* A training event is **purged** when `[t0, t1]` intersects `[min t0, max t1]` of the validation fold (same closed-interval
  rule as `PurgedKFold`; a test checks that the two agree on a single series), and **embargoed** when its decision falls
  within `embargo_frac × (calendar span)` after that window.
* Inputs: any datetime-like arrays (tz-aware Series, numpy `datetime64`, int64 ns) in any order.
* `cover_all=True` (default): the first fold is open to the left and the last to the right, so every event is tested once.
  `cover_all=False` reproduces the research notebook, whose outer edges were float-rounded quantiles of int64 ns and could
  leave the events at the very first / last decision time untested.

## Walk-forward windows

```python
from quant_forge.validation import calendar_windows, expanding_split

windows = calendar_windows("2019-03-01", end="2026-11-19", years=1)   # [(w0, w1), ...]; last window ends at `end`
train, test = expanding_split(trades, *windows[0], train_mask=trades["asset"].isin(SEEN))
```

The model of a window trains on events that **ended** before `w0` (no training label reaches into the window) and is
evaluated on events **decided** in `[w0, w1)`. `months=` gives shorter windows, `n=` fixes the number of starts.

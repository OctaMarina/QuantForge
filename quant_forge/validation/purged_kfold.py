"""Purged K-Fold cross-validation with embargo for observations that carry an information interval.

Methodology: Marcos Lopez de Prado, *Advances in Financial Machine Learning* (2018), chapter 7
(sections 7.4.1-7.4.3, snippets 7.1-7.3).  Each observation ``i`` is an *event* whose label/features use
information from ``start_i`` (the index entry) to ``end_i`` (the value of ``samples_info_sets``).  Ordinary
K-Fold leaks because a training event can share information with a test event; purging and embargo remove
such events from the training set.

What is taken from the book and what is an implementation choice is spelled out in :class:`PurgedKFold`.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = ["PurgedKFold", "PurgedFold"]

_UNIT_ORDER = {"s": 0, "ms": 1, "us": 2, "ns": 3}


@dataclass(frozen=True)
class PurgedFold:
    """One fold with the reason every non-training observation was excluded.

    All index arrays are sorted, positional (``np.intp``) and pairwise disjoint; together they cover
    ``0 .. n_samples - 1`` exactly once.  Category precedence: ``test`` first, then ``purged``
    (interval overlap with the test window), then ``embargoed`` (inside the embargo window but with no
    interval overlap), and ``train`` is everything left.

    ``test_start`` / ``test_end`` are the earliest test start and the *maximum* test end, i.e. the test
    information window used for purging and for locating the embargo.
    """

    train: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    embargoed: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class PurgedKFold:
    """K-Fold cross-validator with purging and embargo (scikit-learn cross-validator interface).

    Parameters
    ----------
    n_splits : int, default 3
        Number of folds, ``>= 2`` and ``<= n_samples``.
    samples_info_sets : pandas.Series
        Event intervals supplied by the caller.  **Index**: event/information start time.
        **Values**: event/information end time (e.g. the barrier touch time of a triple-barrier label, or the
        exit time of a strategy trade).  Required, keyword-only.  For this first version the index must be a
        chronological (strictly increasing), unique, NaT-free ``DatetimeIndex`` and must equal ``X.index``
        exactly when :meth:`split` is called; end times need not be sorted but must not precede their start
        and must have the same timezone-awareness (and timezone) as the index.  The Series is copied at
        construction, so later changes by the caller do not affect the splitter.
    pct_embargo : float, default 0.0
        Embargo as a fraction of the **full input sample count**, in ``[0, 1)``:
        ``embargo_count = floor(n_samples * pct_embargo)`` (a ``1e-9`` relative tolerance absorbs binary
        floating point noise, so ``100 * 0.29`` gives 29 and not 28).  It is a number of *observations*,
        not a wall-clock duration and not a share of the test fold.

    Splitting rules
    ---------------
    1. *Test folds* are contiguous and chronological with the sizes of ``KFold(n_splits, shuffle=False)``
       (the first ``n % k`` folds get one extra observation).  Nothing is shuffled; every observation is in
       exactly one test fold.
    2. *Candidates for training* are all observations outside the test fold, before **and** after it.
    3. *Purging*: the test information window is ``[min test start, max test end]`` (the maximum over the
       whole fold, not the end of the last test row).  A candidate is purged when its closed interval
       ``[start, end]`` intersects that window: ``start <= window_end and end >= window_start``.  Sharing an
       endpoint counts as overlap, containment (either direction) and zero-duration events are handled.
    4. *Embargo* (Snippet 7.3 convention): ``p = searchsorted(starts, window_end, side="left")`` is the
       position of the first event whose start is ``>= window_end``; positions ``p .. p + embargo_count - 1``
       (clipped to the sample) are excluded from training.  Purging is applied independently of, and in
       addition to, the embargo, so the embargo can never re-admit an overlapping event.
    5. Test indices are removed from the training set explicitly.

    Boundary behaviour of the embargo window (inherited from the book's convention)
    -------------------------------------------------------------------------------
    * The window starts *at* ``p``.  If an event starts exactly at ``window_end`` it is position ``p`` itself
      (it is purged anyway by the closed-interval rule) and counts as the first embargoed position, so only
      ``embargo_count - 1`` further observations are removed.
    * If the maximum test end equals the start of the last test event (e.g. zero-duration events), ``p``
      is the last test position and the window overlaps the test fold, so only ``embargo_count - 1``
      observations after the fold are removed.  In particular ``embargo_count == 1`` removes nothing extra
      in that case.
    * If ``window_end`` is later than every start (``p == n_samples``) or the window runs past the last
      observation, it is clipped and nothing beyond the sample is excluded.
    * The embargo only looks forward from the test window; observations *before* the test fold are only
      subject to purging, as in the book.

    Raises ``ValueError`` when a fold has no training observations left.  With lazy iteration the error is
    raised when that fold is reached; inputs, alignment and parameters are validated before the first fold.

    Not supported (rejected explicitly): ``groups``, multi-asset / ``MultiIndex`` / duplicated-timestamp
    inputs, unsorted inputs (never sorted silently) and ``X`` without an ``.index`` (alignment could not be
    verified).  Model fitting, sample weights, tuning, feature selection and scoring are left to the
    caller.

    Validating a subset
    -------------------
    The splitter is bound to the exact events it was built from.  To cross-validate a subset (a training
    window, the training part of an outer fold in nested validation, ...) build a **new** ``PurgedKFold`` from
    ``samples_info_sets.loc[subset.index]``; do not reuse one built for the full sample, and note that
    ``pct_embargo`` then refers to the size of the subset.

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.01)   # doctest: +SKIP
    >>> cross_validate(pipeline, X, y, cv=cv)                                   # doctest: +SKIP
    """

    def __init__(self, n_splits: int = 3, *, samples_info_sets: pd.Series, pct_embargo: float = 0.0):
        self.n_splits = _check_n_splits(n_splits)
        self.pct_embargo = _check_pct_embargo(pct_embargo)
        index, ends = _check_info_sets(samples_info_sets)
        if len(index) < self.n_splits:
            raise ValueError(f"Cannot have n_splits={self.n_splits} greater than the number of samples: "
                             f"{len(index)}.")
        self.samples_info_sets = samples_info_sets.copy()      # snapshot; never the caller's object
        self._unit = index.unit
        self._start_index = index
        self._end_index = ends
        self._starts = index.asi8
        self._ends = ends.asi8
        self._n = len(index)

    # ---- scikit-learn interface -------------------------------------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of folds (``X``, ``y`` unused; ``groups`` must be ``None``)."""
        _check_no_groups(groups)
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` as sorted positional integer arrays.

        ``X`` must expose an ``index`` (DataFrame/Series) equal to ``samples_info_sets.index``.  Use the
        arrays with ``.iloc`` or pass the splitter as ``cv=`` to scikit-learn.
        """
        # not a generator function itself: inputs are validated when split() is called, not at first next()
        return ((fold.train, fold.test) for fold in self.split_with_diagnostics(X, y, groups))

    def split_with_diagnostics(self, X, y=None, groups=None) -> Iterator[PurgedFold]:
        """Like :meth:`split`, but yield :class:`PurgedFold` objects that also say which observations were
        purged and which were embargoed (this is the single code path behind :meth:`split`)."""
        _check_no_groups(groups)
        self._check_aligned(X, y)
        n, k = self._n, self.n_splits
        embargo_count = _embargo_count(n, self.pct_embargo)
        sizes = np.full(k, n // k, dtype=np.intp)
        sizes[: n % k] += 1
        bounds = np.concatenate(([0], np.cumsum(sizes)))
        return self._iter_folds(bounds, embargo_count)

    def __repr__(self) -> str:
        return (f"PurgedKFold(n_splits={self.n_splits}, pct_embargo={self.pct_embargo}, "
                f"samples_info_sets=<{self._n} events, {self._start_index[0]} .. {self._start_index[-1]}>)")

    # ---- internals --------------------------------------------------------------------------------
    def _iter_folds(self, bounds: np.ndarray, embargo_count: int) -> Iterator[PurgedFold]:
        n, starts, ends = self._n, self._starts, self._ends
        positions = np.arange(n, dtype=np.intp)
        for fold_no in range(self.n_splits):
            a, b = int(bounds[fold_no]), int(bounds[fold_no + 1])
            test = positions[a:b]
            window_start = starts[a]                              # earliest test start (index is sorted)
            last = a + int(np.argmax(ends[a:b]))                  # row holding the maximum test end
            window_end = ends[last]
            is_test = np.zeros(n, dtype=bool)
            is_test[a:b] = True
            # closed-interval overlap with the test information window (vectorised, O(n) per fold)
            overlaps = (starts <= window_end) & (ends >= window_start) & ~is_test
            # Snippet 7.3 embargo: embargo_count positions starting at the insertion point of the max end
            in_embargo = np.zeros(n, dtype=bool)
            if embargo_count:
                p = int(np.searchsorted(starts, window_end, side="left"))
                in_embargo[p: min(p + embargo_count, n)] = True
            embargoed_mask = in_embargo & ~is_test & ~overlaps
            train_mask = ~(is_test | overlaps | embargoed_mask)
            if not train_mask.any():
                raise ValueError(
                    f"Fold {fold_no} has no training observations left after purging {int(overlaps.sum())} "
                    f"overlapping and embargoing {int(embargoed_mask.sum())} observations "
                    f"(n_samples={n}, n_splits={self.n_splits}, embargo_count={embargo_count}). Use more "
                    f"samples, fewer splits, a smaller pct_embargo or shorter event intervals.")
            yield PurgedFold(train=positions[train_mask], test=test, purged=positions[overlaps],
                             embargoed=positions[embargoed_mask],
                             test_start=self._start_index[a], test_end=self._end_index[last])

    def _check_aligned(self, X, y) -> None:
        if X is None:
            raise TypeError("PurgedKFold.split needs X with a DatetimeIndex equal to samples_info_sets.index.")
        if not hasattr(X, "index"):
            raise TypeError(
                f"X must be a pandas DataFrame/Series whose index equals samples_info_sets.index; got "
                f"{type(X).__name__}, for which alignment with the event intervals cannot be verified. "
                f"Wrap array data in a DataFrame indexed by the event start times.")
        _check_same_index(X.index, self._start_index, "X.index")
        if y is not None:
            if len(y) != self._n:
                raise ValueError(f"y has {len(y)} samples but samples_info_sets has {self._n}.")
            if isinstance(getattr(y, "index", None), pd.Index):
                _check_same_index(y.index, self._start_index, "y.index")


# ---- validation helpers ---------------------------------------------------------------------------
def _check_n_splits(n_splits) -> int:
    if isinstance(n_splits, bool) or not isinstance(n_splits, numbers.Integral):
        raise ValueError(f"n_splits must be an integer, got {n_splits!r}.")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
    return int(n_splits)


def _check_pct_embargo(pct) -> float:
    if isinstance(pct, bool) or not isinstance(pct, numbers.Real):
        raise ValueError(f"pct_embargo must be a real number in [0, 1), got {pct!r}.")
    pct = float(pct)
    if not (0.0 <= pct < 1.0):                      # also rejects NaN
        raise ValueError(f"pct_embargo must be in [0, 1), got {pct}.")
    return pct


def _embargo_count(n_samples: int, pct_embargo: float) -> int:
    """``floor(n_samples * pct_embargo)`` with a relative 1e-9 tolerance against float noise."""
    x = n_samples * pct_embargo
    nearest = round(x)
    if abs(x - nearest) <= 1e-9 * max(1.0, x):
        return int(nearest)
    return int(math.floor(x))


def _check_no_groups(groups) -> None:
    if groups is not None:
        raise ValueError("PurgedKFold does not support `groups`: grouped / multi-asset validation is not "
                         "implemented. Pass groups=None and one chronological, single-asset event series.")


def _common_unit(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> str:
    return a.unit if _UNIT_ORDER[a.unit] >= _UNIT_ORDER[b.unit] else b.unit


def _check_info_sets(info) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if not isinstance(info, pd.Series):
        raise TypeError(f"samples_info_sets must be a pandas Series (index: event start, values: event end), "
                        f"got {type(info).__name__}.")
    idx = info.index
    if isinstance(idx, pd.MultiIndex):
        raise ValueError("samples_info_sets has a MultiIndex: multi-asset / panel inputs are not supported. "
                         "Provide one chronological DatetimeIndex for a single asset.")
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"samples_info_sets.index must be a DatetimeIndex, got {type(idx).__name__}.")
    if len(idx) == 0:
        raise ValueError("samples_info_sets is empty.")
    if idx.hasnans:
        raise ValueError("samples_info_sets.index contains missing timestamps (NaT).")
    if not pd.api.types.is_datetime64_any_dtype(info.dtype):
        raise TypeError(f"samples_info_sets values must be datetimes (event end times), got dtype {info.dtype}.")
    ends = pd.DatetimeIndex(info)
    if ends.hasnans:
        raise ValueError("samples_info_sets contains missing event end times (NaT).")
    if (idx.tz is None) != (ends.tz is None):
        raise ValueError(f"Incompatible timezones: index tz={idx.tz}, end-time tz={ends.tz}. Both must be "
                         f"timezone-naive or both timezone-aware.")
    if idx.tz is not None and str(idx.tz) != str(ends.tz):
        raise ValueError(f"Incompatible timezones: index tz={idx.tz}, end-time tz={ends.tz}; convert them to "
                         f"the same timezone first.")
    if not idx.is_unique:
        raise ValueError("samples_info_sets.index has duplicated timestamps (multi-asset or stacked events "
                         "are not supported; start times must be unique).")
    if not idx.is_monotonic_increasing:
        raise ValueError("samples_info_sets.index is not chronological; it is rejected rather than sorted "
                         "silently. Sort X, y and samples_info_sets together first.")
    unit = _common_unit(idx, ends)
    idx, ends = idx.as_unit(unit), ends.as_unit(unit)
    bad = ends < idx
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{int(bad.sum())} event(s) end before they start; first at position {first} "
                         f"(start {idx[first]}, end {ends[first]}).")
    return idx, ends


def _check_same_index(other, start_index: pd.DatetimeIndex, name: str) -> None:
    if isinstance(other, pd.MultiIndex) or not isinstance(other, pd.DatetimeIndex):
        raise TypeError(f"{name} must be a DatetimeIndex equal to samples_info_sets.index, got "
                        f"{type(other).__name__} (multi-asset inputs are not supported).")
    if len(other) != len(start_index):
        raise ValueError(f"{name} has {len(other)} entries but samples_info_sets has {len(start_index)}.")
    if (other.tz is None) != (start_index.tz is None) or (other.tz is not None
                                                           and str(other.tz) != str(start_index.tz)):
        raise ValueError(f"Incompatible timezones: {name} tz={other.tz}, samples_info_sets.index "
                         f"tz={start_index.tz}.")
    unit = _common_unit(other, start_index)
    a, b = other.as_unit(unit).asi8, start_index.as_unit(unit).asi8
    if not np.array_equal(a, b):
        first = int(np.flatnonzero(a != b)[0])
        raise ValueError(f"{name} is not aligned with samples_info_sets.index: first mismatch at position "
                         f"{first} ({other[first]} vs {start_index[first]}). The splitter is bound to the "
                         f"exact events it was built from; build a new PurgedKFold for a subset.")

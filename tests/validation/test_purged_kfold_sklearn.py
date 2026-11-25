"""PurgedKFold used through scikit-learn's public cross-validation API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone                                                    # noqa: E402
from sklearn.linear_model import LogisticRegression                               # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline                                             # noqa: E402
from sklearn.preprocessing import StandardScaler                                  # noqa: E402

from quant_forge.validation import PurgedKFold                                    # noqa: E402

from .helpers import make_info                                                    # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 90
    starts = np.cumsum(rng.integers(1, 5, size=n))
    info = make_info(starts, starts + rng.integers(0, 15, size=n))
    # a drifting feature, so a scaler fitted on different rows gives visibly different means
    X = pd.DataFrame({"a": np.arange(n) / 10 + rng.normal(size=n), "b": rng.normal(size=n)}, index=info.index)
    y = pd.Series(rng.integers(0, 2, size=n), index=info.index)
    return X, y, info


def pipeline():
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])


@pytest.mark.parametrize("n,k", [(90, 5), (91, 4), (13, 13)])
def test_test_folds_match_kfold_without_shuffle(n, k):
    info = make_info(range(n), range(n))
    X = pd.DataFrame({"a": np.arange(n)}, index=info.index)
    ours = [te for _, te in PurgedKFold(k, samples_info_sets=info).split(X)]
    ref = [te for _, te in KFold(k, shuffle=False).split(X)]
    assert len(ours) == len(ref) == k
    for a, b in zip(ours, ref):
        assert np.array_equal(a, b)


def test_cross_validate_with_pipeline_fits_preprocessing_inside_each_fold(data):
    X, y, info = data
    cv = PurgedKFold(5, samples_info_sets=info, pct_embargo=0.02)
    res = cross_validate(pipeline(), X, y, cv=cv, scoring="accuracy", return_estimator=True)
    splits = list(cv.split(X))
    assert len(res["test_score"]) == cv.get_n_splits() == 5
    assert np.all(np.isfinite(res["test_score"]))
    for est, (train, test) in zip(res["estimator"], splits):
        scaler = est.named_steps["scale"]
        assert np.allclose(scaler.mean_, X.iloc[train].mean().to_numpy())          # fitted on this fold's train
        assert not np.allclose(scaler.mean_, X.mean().to_numpy())                  # not on the whole sample
        assert scaler.n_samples_seen_ == len(train)


def test_cross_val_score_and_predict_accept_the_splitter(data):
    X, y, info = data
    cv = PurgedKFold(4, samples_info_sets=info)
    scores = cross_val_score(pipeline(), X, y, cv=cv)
    assert scores.shape == (4,)
    pred = cross_val_predict(pipeline(), X, y, cv=cv)           # needs the test folds to partition the sample
    assert pred.shape == (len(X),)


def test_sklearn_sees_same_splits_as_direct_call(data):
    X, y, info = data
    cv = PurgedKFold(3, samples_info_sets=info, pct_embargo=0.05)
    direct = [(tr.tolist(), te.tolist()) for tr, te in cv.split(X, y)]
    seen = []

    class Recorder(LogisticRegression):
        def fit(self, Xf, yf, *a, **k):
            seen.append(tuple(Xf.index))
            return super().fit(Xf, yf, *a, **k)

    cross_validate(Pipeline([("clf", Recorder())]), X, y, cv=cv)
    assert [list(s) for s in seen] == [list(X.index[tr]) for tr, _ in direct]


def test_splitter_can_be_cloned_with_estimator_and_reused(data):
    X, y, info = data
    cv = PurgedKFold(3, samples_info_sets=info)
    a = cross_val_score(clone(pipeline()), X, y, cv=cv)
    b = cross_val_score(clone(pipeline()), X, y, cv=cv)
    assert np.array_equal(a, b)                                  # deterministic, reusable object

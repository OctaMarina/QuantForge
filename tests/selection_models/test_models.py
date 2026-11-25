import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from quant_forge.models import (PlattCalibrator, default_model_specs, fit_predict, oof_prior, pca_preprocessor,
                                positive_proba, tune_purged)
from quant_forge.validation import CalendarPurgedKFold
from tests.synthetic import make_trade_panel


def _data():
    tr = make_trade_panel(600, seed=3)
    X = tr[[c for c in tr.columns if c.startswith(("f_", "t_"))]].copy()
    X.iloc[::17, 2] = np.nan
    folds = list(CalendarPurgedKFold(4, 0.01).split(tr["decision_time"], tr["exit_time"]))
    return X, tr["y"].to_numpy(), folds


def _naive_oof(spec, params, X, y, folds, pca=False):
    oof = np.full(len(y), np.nan)
    for tr, va in folds:
        Xa, Xb = X.iloc[tr], X.iloc[va]
        if pca:
            pp = pca_preprocessor().fit(Xa)
            Xa, Xb = pd.DataFrame(pp.transform(Xa)), pd.DataFrame(pp.transform(Xb))
        oof[va] = positive_proba(spec.pipeline(params, 1).fit(Xa, y[tr]), Xb)
    return oof


@pytest.mark.parametrize("name,pca", [("LR", False), ("LR", True), ("RF", False), ("HGB", False)])
def test_tuning_equals_refitting_the_pipeline(name, pca):
    X, y, folds = _data()
    spec = default_model_specs(seed=0)[name]
    spec = type(spec)(spec.name, spec.preprocessor, spec.estimator, spec.grid[:2], spec.seed)
    res = tune_purged(spec, X, y, folds, pca=pca, n_jobs=1)
    for params, score in res.scores:
        oof = _naive_oof(spec, params, X, y, folds, pca)
        ok = ~np.isnan(oof)
        ll = -(y[ok] * np.log(np.clip(oof[ok], 1e-6, 1 - 1e-6)) + (1 - y[ok]) * np.log(1 - np.clip(oof[ok], 1e-6, 1 - 1e-6))).mean()
        assert score == pytest.approx(ll, abs=1e-12)
    assert res.score == min(s for _, s in res.scores)
    p, est, pp = fit_predict(spec, res.params, X, y, X.iloc[:10], pca=pca, n_jobs=1)
    assert p.shape == (10,) and ((pp is not None) == pca)


def test_default_grids_are_reproducible():
    a, b = default_model_specs(seed=0), default_model_specs(seed=0)
    assert a["RF"].grid == b["RF"].grid and a["HGB"].grid == b["HGB"].grid
    assert [g["max_depth"] for g in a["RF"].grid] == [None, 6, None, 6]
    assert [g["C"] for g in a["LR"].grid] == [0.003, 0.01, 0.03, 0.1, 0.3]
    assert default_model_specs(seed=1)["HGB"].grid != a["HGB"].grid


def test_oof_prior_and_platt():
    X, y, folds = _data()
    prior = oof_prior(y, folds)
    tr, va = folds[0]
    assert np.all(prior[va] == y[tr].mean())
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    yy = (rng.uniform(size=p.size) < p).astype(int)
    a, b = PlattCalibrator().fit(p, yy).coefficients
    assert a == pytest.approx(1.0, abs=0.1) and b == pytest.approx(0.0, abs=0.1)

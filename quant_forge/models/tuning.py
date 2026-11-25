"""Hyper-parameter search with purged cross-validation and log-loss scoring (AFML §9.2-9.4).

:func:`tune_purged` evaluates every grid point of a :class:`~quant_forge.models.ModelSpec` on the given folds,
collecting out-of-fold (OOF) probabilities, and keeps the point with the lowest OOF log loss (ties: first).
The preprocessor (and the optional PCA step) is fitted once per fold and shared by all grid points - the
result equals refitting ``spec.pipeline(params)`` on every fold.

The OOF probabilities of the winner are returned because they are the honest inputs of everything that is
learned *after* model choice (probability calibration, thresholds, bet-sizing constants).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..evaluation.metrics import log_loss_vector
from .factories import ModelSpec, pca_preprocessor, positive_proba

__all__ = ["TuningResult", "tune_purged", "fit_predict", "oof_prior"]


@dataclass
class TuningResult:
    """Best grid point: OOF log loss ``score``, ``params``, ``oof`` probabilities (NaN where never validated)
    and ``scores`` = ``[(params, score), ...]`` for every grid point."""

    score: float
    params: dict
    oof: np.ndarray
    scores: list = field(default_factory=list)


def _fold_inputs(spec: ModelSpec, X: pd.DataFrame, folds, pca: bool):
    data = []
    for tr, va in folds:
        Xa, Xb = X.iloc[tr], X.iloc[va]
        with warnings.catch_warnings():          # e.g. n_quantiles > n_samples on small folds
            warnings.simplefilter("ignore")
            if pca:
                pp = pca_preprocessor().fit(Xa)
                Xa, Xb = pd.DataFrame(pp.transform(Xa)), pd.DataFrame(pp.transform(Xb))
            pre = spec.preprocessor()
            Za = pre.fit_transform(Xa)
            data.append((tr, va, Za, pre.transform(Xb)))
    return data


def tune_purged(spec: ModelSpec, X: pd.DataFrame, y, folds: Sequence, *, columns: Sequence[str] | None = None,
                pca: bool = False, n_jobs: int | None = None, grid: Sequence[dict] | None = None) -> TuningResult:
    """Grid search of ``spec`` on ``(train, validation)`` index pairs ``folds`` by OOF log loss.

    columns  feature subset (default: all columns of ``X``); pca  insert :func:`pca_preprocessor` per fold.
    """
    y = np.asarray(y)
    Xc = X[list(columns)] if columns is not None else X
    data = _fold_inputs(spec, Xc, folds, pca)
    best, scores = None, []
    for params in (spec.grid if grid is None else grid):
        oof = np.full(len(y), np.nan)
        for tr, va, Za, Zb in data:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof[va] = positive_proba(spec.make_estimator(params, n_jobs).fit(Za, y[tr]), Zb)
        ok = ~np.isnan(oof)
        score = log_loss_vector(oof[ok], y[ok]).mean()
        scores.append((params, score))
        if best is None or score < best[0]:
            best = (score, params, oof)
    return TuningResult(score=best[0], params=best[1], oof=best[2], scores=scores)


def fit_predict(spec: ModelSpec, params: dict, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, *,
                pca: bool = False, n_jobs: int | None = None):
    """Refit ``spec`` with ``params`` on all training rows and predict ``X_test``.

    Returns ``(probabilities, fitted pipeline, fitted PCA preprocessor or None)``.
    """
    pp = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pca:
            pp = pca_preprocessor().fit(X_train)
            X_train, X_test = pd.DataFrame(pp.transform(X_train)), pd.DataFrame(pp.transform(X_test))
        est = spec.pipeline(params, n_jobs).fit(X_train, np.asarray(y_train))
        return positive_proba(est, X_test), est, pp


def oof_prior(y, folds) -> np.ndarray:
    """OOF predictions of the constant model (training-fold win rate), the reference every model must beat."""
    y = np.asarray(y)
    out = np.full(len(y), np.nan)
    for tr, va in folds:
        out[va] = y[tr].mean()
    return out

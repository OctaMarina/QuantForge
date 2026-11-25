"""Model specifications for meta-labeling: logistic regression, random forest, histogram gradient boosting.

A :class:`ModelSpec` splits a scikit-learn pipeline into a parameter-independent **preprocessor** (imputation,
quantile transform) and a parametrised **estimator**.  Hyper-parameter search
(:func:`quant_forge.models.tune_purged`) fits the preprocessor once per fold and reuses it for every candidate,
which gives exactly the predictions of refitting the whole pipeline at a fraction of the cost.
``spec.pipeline(**params)`` returns the equivalent ordinary :class:`~sklearn.pipeline.Pipeline`.

Defaults follow AFML ch. 6/9 recommendations for financial labels: entropy criterion, balanced subsample class
weights and a minimum leaf weight for forests (AFML §6.4), small learning rates and shallow trees for boosting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["ModelSpec", "median_imputer", "positive_proba", "make_lr", "make_rf", "make_hgb", "lr_spec", "rf_spec",
           "hgb_spec", "pca_preprocessor", "default_model_specs"]


def median_imputer():
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy="median", keep_empty_features=True)


def positive_proba(estimator, X) -> np.ndarray:
    """Probability of class 1 from a fitted classifier."""
    return estimator.predict_proba(X)[:, list(estimator.classes_).index(1)]


def _lr_pre():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    return make_pipeline(median_imputer(), QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                                               subsample=10 ** 9))


def _lr_est(C, n_jobs=None, seed=0):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, max_iter=3000)


def _imputer_pre():
    from sklearn.pipeline import make_pipeline

    return make_pipeline(median_imputer())


def _rf_est(max_features, mwfl, max_depth, n_jobs=None, seed=0, n_estimators=150):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_features=max_features,
                                  min_weight_fraction_leaf=mwfl, max_depth=max_depth,
                                  class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)


def _hgb_est(lr, depth, msl, l2, mf, n_jobs=None, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=150, learning_rate=lr, max_depth=depth, min_samples_leaf=msl,
                                          l2_regularization=l2, max_features=mf, early_stopping=False,
                                          random_state=seed)


def pca_preprocessor(variance: float = 0.95):
    """Imputation, standardisation and PCA keeping ``variance`` of the variance (AFML §8.4.2)."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(median_imputer(), StandardScaler(), PCA(n_components=variance, svd_solver="full"))


@dataclass(frozen=True)
class ModelSpec:
    """A model family: ``preprocessor()`` (no hyper-parameters), ``estimator(**params, n_jobs, seed)`` and a grid."""

    name: str
    preprocessor: Callable[[], Any]
    estimator: Callable[..., Any]
    grid: list = field(default_factory=list)
    seed: int = 0

    def make_estimator(self, params: dict, n_jobs: int | None = None):
        return self.estimator(**params, n_jobs=n_jobs, seed=self.seed)

    def pipeline(self, params: dict | None = None, n_jobs: int | None = None):
        """Ordinary sklearn pipeline ``preprocessor steps + estimator``."""
        from sklearn.pipeline import Pipeline

        pre = self.preprocessor()
        return Pipeline(list(pre.steps) + [("estimator", self.make_estimator(params or {}, n_jobs))])


def lr_spec(grid=None, seed: int = 0) -> ModelSpec:
    """Logistic regression on median-imputed, normal-quantile-transformed features; grid over ``C``."""
    return ModelSpec("LR", _lr_pre, _lr_est, list(grid) if grid is not None else
                     [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3)], seed)


def rf_spec(grid, seed: int = 0) -> ModelSpec:
    """Random forest (entropy, balanced subsample); params ``max_features``, ``mwfl``, ``max_depth``."""
    return ModelSpec("RF", _imputer_pre, _rf_est, list(grid), seed)


def hgb_spec(grid, seed: int = 0) -> ModelSpec:
    """Histogram gradient boosting (150 iterations, no early stopping); params ``lr``, ``depth``, ``msl``, ``l2``, ``mf``."""
    return ModelSpec("HGB", _imputer_pre, _hgb_est, list(grid), seed)


def default_model_specs(seed: int = 0, n_random: int = 4) -> dict[str, ModelSpec]:
    """``{"LR", "RF", "HGB"}`` with the grids of the meta-labeling research notebooks.

    LR: ``C`` in {0.003 ... 0.3}.  RF and HGB: ``n_random`` random draws from small discrete spaces, drawn in a
    fixed order from one ``default_rng(seed)`` stream (RF first), so the grids are reproducible.
    """
    rng = np.random.default_rng(seed)

    def pick(opts, n):
        return [opts[i] for i in rng.integers(0, len(opts), n)]

    depths = [None, 6] * (n_random // 2) + [None] * (n_random % 2)
    rf_grid = [{"max_features": mf, "mwfl": w, "max_depth": d} for mf, w, d in
               zip(pick(["sqrt", 0.3], n_random), pick([0.01, 0.02, 0.05], n_random), depths)]
    hgb_grid = [{"lr": lr, "depth": d, "msl": m, "l2": l2, "mf": mf} for lr, d, m, l2, mf in
                zip(pick([0.03, 0.1], n_random), pick([2, 3], n_random), pick([40, 80, 160], n_random),
                    pick([1.0, 10.0], n_random), pick([0.3, 0.6], n_random))]
    return {"LR": lr_spec(seed=seed), "RF": rf_spec(rf_grid, seed), "HGB": hgb_spec(hgb_grid, seed)}


# ---- convenience constructors returning full pipelines ------------------------------------------------------------
def make_lr(C: float):
    return lr_spec().pipeline({"C": C})


def make_rf(max_features, mwfl, max_depth, n_jobs=None, seed: int = 0, n_estimators: int = 150):
    from sklearn.pipeline import make_pipeline

    return make_pipeline(median_imputer(), _rf_est(max_features, mwfl, max_depth, n_jobs, seed, n_estimators))


def make_hgb(lr, depth, msl, l2, mf, seed: int = 0):
    from sklearn.pipeline import make_pipeline

    return make_pipeline(median_imputer(), _hgb_est(lr, depth, msl, l2, mf, None, seed))

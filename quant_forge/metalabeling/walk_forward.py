"""Walk-forward meta-labeling: everything that is learned is fitted inside each training window.

For one ``(strategy, window [w0, w1))`` :func:`fit_window` uses the training trades (exited before ``w0``,
optionally restricted to ``config.train_assets``) and

1. keeps features with ``>= min_coverage`` non-missing values and more than one distinct value;
2. builds purged inner folds on common calendar dates (:class:`~quant_forge.validation.CalendarPurgedKFold`);
3. clusters redundant features, adds placebo features and runs the jointly-permuted clustered MDA; clusters
   above the largest placebo and positive on ``inner_splits - 1`` folds are *selected*; clustered MDI is kept as
   the in-sample counterpart;
4. tunes every model family on the full and on the selected feature set (plus a PCA variant of the families in
   ``pca_models``) by inner out-of-fold log loss; the best ``(model, feature set)`` becomes ``META``;
5. refits every combination on all training trades, calibrates it with Platt scaling on its inner OOF
   probabilities and predicts the window's trades (all assets);
6. fits the bet-sizing constants of ``META`` from its calibrated OOF probabilities.

:func:`run_walk_forward` runs all ``(strategy, window)`` pairs in parallel processes with an optional disk
cache; :func:`collect_predictions` stacks the predictions with trade information for evaluation.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd

from ..evaluation.metrics import log_loss_vector
from ..feature_selection import clustered_mda, clustered_mdi, make_placebos, redundancy_clusters, select_clusters
from ..models import PlattCalibrator, default_model_specs, fit_predict, oof_prior, tune_purged
from ..sizing import fit_sizing_constants
from ..validation.calendar_kfold import CalendarPurgedKFold, to_int64_ns

__all__ = ["MetaLabelingConfig", "WindowFit", "fit_window", "run_walk_forward", "collect_predictions", "window_key"]


@dataclass(frozen=True)
class MetaLabelingConfig:
    """Parameters of :func:`fit_window` (defaults = the research notebook ``meta_labeling_economic_value``)."""

    min_train: int = 200                 # minimum training trades of a (strategy, window)
    min_coverage: float = 0.70           # minimum share of non-missing values of a feature in training
    inner_splits: int = 4                # calendar folds of the inner purged K-fold
    embargo_frac: float = 0.01           # embargo as a share of the training span
    cover_all_events: bool = True        # False = research-notebook fold edges (see CalendarPurgedKFold)
    cluster_max_dist: float = 0.30       # redundancy clusters: 1 - |rho| cut
    n_placebo: int = 5
    n_perm: int = 3                      # permutations per cluster and fold in the MDA
    seed: int = 0
    train_assets: tuple | None = None    # assets the model may train on (None = all)
    model_names: tuple = ("LR", "RF", "HGB")
    pca_models: tuple = ("LR",)
    vol_clip: tuple = (0.25, 4.0)
    vol_column: str = "v_rvol_30d"
    label_column: str = "y"
    return_column: str = "trade_return"
    decision_column: str = "decision_time"
    exit_column: str = "exit_time"
    asset_column: str = "asset"
    feature_selection: bool = True       # False = no clusters / placebos / MDA / MDI: every model on all usable features

    def cache_key(self) -> dict:
        d = dataclasses.asdict(self)
        d["train_assets"] = list(self.train_assets) if self.train_assets is not None else None
        if d["feature_selection"]:        # default: keep the keys of caches written before the option existed
            del d["feature_selection"]
        return d

    def required_columns(self) -> list[str]:
        return [self.asset_column, self.decision_column, self.exit_column, self.label_column, self.return_column,
                self.vol_column]


class WindowFit(NamedTuple):
    """``predictions``: one row per test trade (``row`` = index in the trade table, ``p_<combo>_raw``, ``p_<combo>``,
    ``p`` = calibrated META probability, ``prior``, ``window``, ``meta``); ``diagnostics``: what was learned."""

    predictions: pd.DataFrame
    diagnostics: dict


def window_key(name: str, w0: pd.Timestamp) -> str:
    return f"{name}_{w0.strftime('%Y%m%d')}"


def fit_window(g: pd.DataFrame, w0: pd.Timestamp, w1: pd.Timestamp, candidates: Sequence[str],
               config: MetaLabelingConfig = MetaLabelingConfig(), *, name: str = "", n_jobs: int | None = 1,
               specs: Mapping | None = None) -> WindowFit | None:
    """Fit the whole meta-labeling system for one window on the trades ``g`` of one strategy (all assets).

    Returns ``None`` when there are fewer than ``min_train`` training trades, a single class, or no test trade.
    """
    cfg = config
    specs = specs or default_model_specs(cfg.seed)
    specs = {k: specs[k] for k in cfg.model_names}
    rng = np.random.default_rng(cfg.seed + int(w0.value // 10 ** 9) % 1000)
    tr_mask = g[cfg.exit_column] < w0
    if cfg.train_assets is not None:
        tr_mask &= g[cfg.asset_column].isin(cfg.train_assets)
    train = g.loc[tr_mask]
    test = g.loc[(g[cfg.decision_column] >= w0) & (g[cfg.decision_column] < w1)]
    if len(train) < cfg.min_train or train[cfg.label_column].nunique() < 2 or test.empty:
        return None
    y = train[cfg.label_column].to_numpy()
    r_tr = train[cfg.return_column].to_numpy()
    cov = train[list(candidates)].notna().mean()
    usable = [c for c in candidates if cov[c] >= cfg.min_coverage and train[c].nunique(dropna=True) > 1]
    X, Xt = train[usable], test[usable]
    folds = list(CalendarPurgedKFold(cfg.inner_splits, cfg.embargo_frac, cfg.cover_all_events).split(
        to_int64_ns(train[cfg.decision_column]), to_int64_ns(train[cfg.exit_column])))
    diag = {"strategy": name, "window": w0, "n_train": len(train), "n_test": len(test), "n_usable": len(usable),
            "fold_sizes": [(len(a), len(b)) for a, b in folds]}

    # 1. clusters, placebos, clustered MDA (selection) and MDI ---------------------------------------------------
    if cfg.feature_selection:
        Zi = X.fillna(X.median())
        groups = redundancy_clusters(Zi, cfg.cluster_max_dist)
        P = make_placebos(Zi, cfg.n_placebo, rng)
        groups_p = {**groups, **{c: [c] for c in P.columns}}
        mda = clustered_mda(pd.concat([Zi, P], axis=1), y, folds, groups_p, n_perm=cfg.n_perm, rng=rng, n_jobs=n_jobs)
        sel = select_clusters(mda, groups, list(P.columns), min_positive_folds=cfg.inner_splits - 1)
        sel_cols = sel.columns
        diag.update({"groups": groups, "mda": mda, "mdi": clustered_mdi(Zi, y, groups, n_jobs=n_jobs, seed=cfg.seed),
                     "selected": sel.selected, "placebo_max": sel.placebo_max,
                     "mda_base_ll": float(mda["_base"].mean()), "mda_prior_ll": float(mda["_prior"].mean())})
    else:
        sel_cols = []
        diag.update({"groups": {}, "mda": None, "mdi": {}, "selected": [], "placebo_max": np.nan,
                     "mda_base_ll": np.nan, "mda_prior_ll": np.nan})

    # 2. tuning per (model, feature set) on the same purged folds ------------------------------------------------
    prior_oof = oof_prior(y, folds)
    ok = ~np.isnan(prior_oof)
    prior_ll = log_loss_vector(prior_oof[ok], y[ok]).mean()
    combos = {}
    for mname, spec in specs.items():
        combos[f"{mname}-FULL"] = (spec, tune_purged(spec, X, y, folds, columns=usable, n_jobs=n_jobs), usable, False)
        if sel_cols:
            combos[f"{mname}-SEL"] = (spec, tune_purged(spec, X, y, folds, columns=sel_cols, n_jobs=n_jobs), sel_cols, False)
        if mname in cfg.pca_models:
            combos[f"{mname}-PCA"] = (spec, tune_purged(spec, X, y, folds, columns=usable, pca=True, n_jobs=n_jobs), usable, True)
    tuning = {n: {"inner_ll": t.score, "params": t.params, "n_features": len(cols)} for n, (_, t, cols, _) in combos.items()}
    meta_name = min(tuning, key=lambda n: tuning[n]["inner_ll"])
    diag.update({"tuning": tuning, "prior_inner_ll": prior_ll, "meta": meta_name,
                 "meta_beats_prior": tuning[meta_name]["inner_ll"] < prior_ll})

    # 3. refit, Platt calibration on inner OOF, prediction of the window -----------------------------------------
    pred = pd.DataFrame({"row": test.index})
    for cname, (spec, t, cols, pca) in combos.items():
        p_raw, _, pp = fit_predict(spec, t.params, X[cols], y, Xt[cols], pca=pca, n_jobs=n_jobs)
        if pp is not None:
            tuning[cname]["n_features"] = int(pp[-1].n_components_)
        m = ~np.isnan(t.oof)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = PlattCalibrator().fit(t.oof[m], y[m])
            p_cal, oof_cal = cal.predict(p_raw), cal.predict(t.oof[m])
        pred[f"p_{cname}_raw"], pred[f"p_{cname}"] = p_raw, p_cal
        if cname == meta_name:
            diag["constants"] = fit_sizing_constants(oof_cal, y[m], r_tr[m], train[cfg.vol_column].to_numpy()[m],
                                                     vol_clip=cfg.vol_clip)
            diag["calib_oof_ll"] = float(log_loss_vector(oof_cal, y[m]).mean())
            diag["raw_oof_ll"] = float(log_loss_vector(t.oof[m], y[m]).mean())
            diag["platt"] = cal.coefficients
    pred["p"] = pred[f"p_{meta_name}"]
    pred["prior"], pred["window"], pred["meta"] = y.mean(), w0, meta_name
    return WindowFit(pred, diag)


def _run_task(name, w0, w1, g, cands, config, n_jobs, cache_root):
    from ..utils.cache import DiskCache

    cache = DiskCache(cache_root) if cache_root is not None else None
    key = window_key(name, w0)
    if cache is not None and key in cache:
        return cache.load(key)
    res = fit_window(g, w0, w1, cands, config, name=name, n_jobs=n_jobs)
    if cache is not None:
        cache.save(key, res)
    return res


def run_walk_forward(trades: pd.DataFrame, strategies: Sequence[str], windows: Sequence[tuple],
                     candidates: Mapping[str, Sequence[str]], config: MetaLabelingConfig = MetaLabelingConfig(), *,
                     cache=None, n_jobs: int = 4, inner_n_jobs: int | None = 5, strategy_column: str = "strategy",
                     verbose: bool = True) -> dict:
    """Fit every ``(strategy, window)``; returns ``{(strategy, w0): WindowFit}`` (skipped pairs omitted).

    cache         a :class:`~quant_forge.utils.DiskCache` (one pickle per pair) or ``None``;
    n_jobs        parallel processes over pairs; inner_n_jobs  threads of the forests inside each process.
    Each process receives only the columns it needs.  Results do not depend on ``n_jobs``.
    """
    tasks = [(s, w0, w1) for s in strategies for w0, w1 in windows]
    cache_root = str(cache.root) if cache is not None else None
    todo = [t for t in tasks if cache is None or window_key(t[0], t[1]) not in cache]
    t0 = time.time()
    frames = {}
    for s in strategies:
        cols = list(dict.fromkeys(config.required_columns() + list(candidates[s])))
        frames[s] = trades.loc[trades[strategy_column].eq(s), cols]
    args = [(s, w0, w1, frames[s], list(candidates[s]), config, inner_n_jobs, cache_root) for s, w0, w1 in tasks]
    if n_jobs == 1 or len(todo) <= 1:
        results = [_run_task(*a) for a in args]
    else:
        from joblib import Parallel, delayed

        # computed pairs first (largest training sets first), cached ones are cheap loads
        order = sorted(range(len(args)), key=lambda i: (window_key(tasks[i][0], tasks[i][1]) in (cache or {}),
                                                        -int(tasks[i][1].value)))
        out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run_task)(*args[i]) for i in order)
        results = [None] * len(args)
        for i, r in zip(order, out):
            results[i] = r
    res = {(s, w0): r for (s, w0, _), r in zip(tasks, results) if r is not None}
    if verbose:
        skipped = [(s, str(w0.date())) for (s, w0, _), r in zip(tasks, results) if r is None]
        print(f"walk-forward: {len(tasks)} pairs, {len(todo)} computed, {time.time() - t0:.0f} s; "
              f"{len(res)} fitted; skipped (too few training trades): {skipped}")
    return res


def collect_predictions(results: Mapping, trades: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Stack the window predictions and join ``columns`` of the trade table (by ``row``)."""
    pred = pd.concat([r.predictions for r in results.values()], ignore_index=True)
    return pred.join(trades[list(columns)], on="row")

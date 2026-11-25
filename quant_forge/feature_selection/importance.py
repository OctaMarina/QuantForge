"""Clustered feature importance with placebo features (AFML ch. 8; MLAM ch. 6).

* :func:`make_placebos` - null features: circularly shifted copies of real columns (same marginal distribution
  and autocorrelation, no relation to the labels);
* :func:`clustered_mda` - out-of-sample mean decrease in accuracy on purged folds: the columns of a cluster
  are permuted *jointly* (one row permutation for the whole cluster) and the importance is the increase in
  validation log loss (AFML §8.3.2, MLAM §6.5.2);
* :func:`clustered_mdi` - in-sample mean decrease in impurity of a forest with ``max_features = 1``
  (AFML Snippet 8.2), summed over the members of each cluster (MLAM §6.5.1);
* :func:`select_clusters` - keep the clusters whose mean MDA exceeds the largest placebo MDA and is positive
  on at least ``min_positive_folds`` folds;
* :func:`pca_eigen_rank_check` - AFML §8.4.2: weighted Kendall tau between the MDI of a forest trained on the
  principal components and their inverse eigen-rank, against a label-permutation null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..evaluation.metrics import log_loss_vector
from ..models.factories import make_rf, positive_proba

__all__ = ["make_placebos", "clustered_mda", "clustered_mdi", "ClusterSelection", "select_clusters",
           "pca_eigen_rank_check"]


def make_placebos(Z: pd.DataFrame, n: int, rng: np.random.Generator, *, prefix: str = "placebo_",
                  min_shift: float = 0.2, max_shift: float = 0.8) -> pd.DataFrame:
    """``n`` placebo columns: real columns (drawn without replacement) rolled by 20-80 % of the sample."""
    src = rng.choice(list(Z.columns), size=n, replace=False)
    shifts = rng.integers(int(min_shift * len(Z)), int(max_shift * len(Z)), size=n)
    return pd.DataFrame({f"{prefix}{i}": np.roll(Z[s].to_numpy(), k) for i, (s, k) in enumerate(zip(src, shifts))},
                        index=Z.index)


def clustered_mda(Z: pd.DataFrame, y, folds: Sequence, groups: Mapping[str, Sequence[str]], *, n_perm: int = 3,
                  rng: np.random.Generator, estimator: Callable[[], object] | None = None, n_jobs: int | None = None,
                  max_rows_per_predict: int = 250_000) -> pd.DataFrame:
    """Jointly-permuted cluster MDA on purged folds.

    Returns one row per fold with ``_base`` (validation log loss of the unpermuted model), ``_prior``
    (log loss of the training win rate) and, per cluster, the mean increase in log loss over ``n_perm``
    permutations.  ``estimator`` builds the evaluation model (default: random forest, 150 trees, entropy,
    ``max_features="sqrt"``, ``min_weight_fraction_leaf=0.02``).  Permuted copies are predicted in chunks of at
    most ``max_rows_per_predict`` rows (memory bound; results do not depend on it).
    """
    y = np.asarray(y)
    estimator = estimator or (lambda: make_rf("sqrt", 0.02, None, n_jobs))
    cols = list(Z.columns)
    pos = {c: i for i, c in enumerate(cols)}
    rows = []
    for k, (tr, va) in enumerate(folds):
        est = estimator().fit(Z.iloc[tr], y[tr])
        A0 = Z.iloc[va].to_numpy()
        yv = y[va]
        base = log_loss_vector(positive_proba(est, Z.iloc[va]), yv).mean()
        names, losses, pending, pending_names = [], {}, [], []

        def flush():
            if not pending:
                return
            P = positive_proba(est, pd.DataFrame(np.vstack(pending), columns=cols)).reshape(len(pending), len(A0))
            for (g, j), p in zip(pending_names, P):
                losses.setdefault(g, []).append(log_loss_vector(p, yv).mean() - base)
            pending.clear()
            pending_names.clear()

        for g, members in groups.items():
            ci = [pos[c] for c in members if c in pos]
            if not ci:
                continue
            for j in range(n_perm):
                A = A0.copy()
                A[:, ci] = A0[rng.permutation(len(A0))][:, ci]
                pending.append(A)
                pending_names.append((g, j))
                if len(pending) * len(A0) >= max_rows_per_predict:
                    flush()
            names.append(g)
        flush()
        r = {"fold": k, "_base": base,
             "_prior": log_loss_vector(np.full(len(yv), y[tr].mean()), yv).mean()}
        for g in names:
            r[g] = float(np.mean(losses[g]))
        rows.append(r)
    return pd.DataFrame(rows).set_index("fold")


def clustered_mdi(Z: pd.DataFrame, y, groups: Mapping[str, Sequence[str]], *, n_jobs: int | None = None,
                  seed: int = 0, n_estimators: int = 200, min_weight_fraction_leaf: float = 0.02) -> dict[str, float]:
    """Normalised MDI (zeros of unused features ignored per tree, AFML Snippet 8.2) summed per cluster."""
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_features=1,
                                min_weight_fraction_leaf=min_weight_fraction_leaf, class_weight="balanced_subsample",
                                n_jobs=n_jobs, random_state=seed).fit(Z.fillna(Z.median()), np.asarray(y))
    per_tree = pd.DataFrame([t.feature_importances_ for t in rf.estimators_], columns=Z.columns).replace(0, np.nan)
    m = per_tree.mean()
    m = m / m.sum()
    return {g: float(m[[c for c in members if c in m.index]].sum()) for g, members in groups.items()}


@dataclass
class ClusterSelection:
    """Outcome of :func:`select_clusters`: selected cluster names, their member columns, the placebo bar and the
    per-cluster mean importance / number of positive folds."""

    selected: list
    columns: list
    placebo_max: float
    mean_importance: pd.Series
    positive_folds: pd.Series


def select_clusters(mda: pd.DataFrame, groups: Mapping[str, Sequence[str]], placebo_columns: Sequence[str], *,
                    min_positive_folds: int) -> ClusterSelection:
    """Clusters with mean MDA above ``max(0, max placebo MDA)`` and positive MDA on ``>= min_positive_folds`` folds."""
    imp = mda.drop(columns=[c for c in ("_base", "_prior") if c in mda.columns])
    mean_imp, pos_folds = imp.mean(), (imp > 0).sum()
    placebo_max = max(0.0, mean_imp[list(placebo_columns)].max()) if len(placebo_columns) else 0.0
    selected = [g for g in groups if mean_imp[g] > placebo_max and pos_folds[g] >= min_positive_folds]
    return ClusterSelection(selected, [c for g in selected for c in groups[g]], placebo_max, mean_imp, pos_folds)


def pca_eigen_rank_check(Z: pd.DataFrame, y, *, n_null: int = 10, seed: int = 0, variance: float = 0.95,
                         n_jobs: int | None = -1) -> dict:
    """Weighted Kendall tau between PC-forest MDI and inverse eigen-rank, with a label-permutation null."""
    from scipy.stats import weightedtau
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    Zs = StandardScaler().fit_transform(Z)
    pca = PCA(n_components=variance, svd_solver="full").fit(Zs)
    Pz = pd.DataFrame(pca.transform(Zs))

    def pc_mdi(yy, s):
        rf = RandomForestClassifier(n_estimators=200, criterion="entropy", max_features=1, min_weight_fraction_leaf=0.02,
                                    class_weight="balanced_subsample", n_jobs=n_jobs, random_state=s).fit(Pz, yy)
        m = pd.DataFrame([t.feature_importances_ for t in rf.estimators_]).replace(0, np.nan).mean()
        return (m / m.sum()).to_numpy()

    inv_rank = 1.0 / np.arange(1, pca.n_components_ + 1)
    tau = weightedtau(pc_mdi(y, seed), inv_rank)[0]
    rng = np.random.default_rng(seed)
    null = [weightedtau(pc_mdi(rng.permutation(y), seed + 1 + i), inv_rank)[0] for i in range(n_null)]
    return {"n_components": int(pca.n_components_), "pc1_variance": float(pca.explained_variance_ratio_[0]),
            "tau": float(tau), "null_mean": float(np.mean(null)), "null_p95": float(np.quantile(null, 0.95))}

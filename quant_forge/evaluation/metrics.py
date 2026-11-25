"""Probability-quality metrics for meta-models (log loss vs a prior, AUC, rank IC, calibration).

All functions take plain arrays or a prediction table with one row per event.  Confidence intervals use a
**monthly block bootstrap**: months are resampled with replacement and the statistic is recomputed from the
per-month sums, which keeps the dependence between events of the same month.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["EPS", "log_loss_vector", "auc_safe", "delta_log_loss_ci", "information_coefficient",
           "reliability_table", "probability_quality_table"]

EPS = 1e-6


def log_loss_vector(p, y, eps: float = EPS) -> np.ndarray:
    """Per-event binary log loss with probabilities clipped to ``[eps, 1 - eps]``."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc_safe(y, p) -> float:
    """ROC AUC, NaN when only one class is present."""
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def delta_log_loss_ci(df: pd.DataFrame, col: str, *, prior_col: str = "prior", label_col: str = "y",
                      month_col: str = "month", n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean log-loss difference ``model - prior`` with a paired monthly block-bootstrap CI.

    Returns ``(mean, low, high)``; negative = the model beats the prior.
    """
    y = df[label_col].to_numpy()
    d = log_loss_vector(df[col].to_numpy(), y) - log_loss_vector(df[prior_col].to_numpy(), y)
    mi = pd.factorize(df[month_col])[0]
    nm = mi.max() + 1
    s, c = np.bincount(mi, weights=d, minlength=nm), np.bincount(mi, minlength=nm)
    ix = np.random.default_rng(seed).integers(0, nm, size=(n_boot, nm))
    boot = s[ix].sum(1) / c[ix].sum(1)
    return d.mean(), np.quantile(boot, alpha / 2), np.quantile(boot, 1 - alpha / 2)


def information_coefficient(df: pd.DataFrame, col: str = "p", target: str = "trade_return",
                            group_cols: Sequence[str] = ("strategy", "window")) -> float:
    """Pooled rank correlation between ``col`` and ``target`` with ranks taken within ``group_cols``
    (so level differences between strategies / windows do not count as skill)."""
    grouped = df.groupby(list(group_cols))
    rp = grouped[col].rank(pct=True) - 0.5
    rt = grouped[target].rank(pct=True) - 0.5
    return (rp * rt).sum() / np.sqrt((rp ** 2).sum() * (rt ** 2).sum())


def reliability_table(df: pd.DataFrame, col: str, *, label_col: str = "y", bins: int = 10):
    """Equal-count bins of ``col``: ``(table[n, mean_p, win_rate], expected calibration error)``."""
    q = pd.qcut(df[col].rank(method="first"), bins, labels=False)
    t = df.groupby(q).agg(n=(label_col, "size"), mean_p=(col, "mean"), win_rate=(label_col, "mean"))
    ece = (t["n"] / t["n"].sum() * (t["mean_p"] - t["win_rate"]).abs()).sum()
    return t, ece


def probability_quality_table(df: pd.DataFrame, columns: Mapping[str, str], *, label_col: str = "y",
                              return_col: str = "trade_return", prior_col: str = "prior", month_col: str = "month",
                              group_cols: Sequence[str] = ("strategy", "window"), n_boot: int = 2000,
                              seed: int = 0) -> pd.DataFrame:
    """One row per ``{label: probability column}``: n, Δ log loss vs prior [CI], AUC, Brier, IC(return), IC(label)."""
    from sklearn.metrics import brier_score_loss

    rows = []
    for label, col in columns.items():
        sub = df.loc[df[col].notna()]
        m, lo, hi = delta_log_loss_ci(sub, col, prior_col=prior_col, label_col=label_col, month_col=month_col,
                                      n_boot=n_boot, seed=seed)
        rows.append({"combo": label, "n": len(sub), "Δ log loss vs prior": m, "CI low": lo, "CI high": hi,
                     "AUC": auc_safe(sub[label_col], sub[col]),
                     "Brier": brier_score_loss(sub[label_col], np.clip(sub[col], 0, 1)),
                     "IC(return)": information_coefficient(sub, col, return_col, group_cols),
                     "IC(label)": information_coefficient(sub, col, label_col, group_cols)})
    return pd.DataFrame(rows).set_index("combo")

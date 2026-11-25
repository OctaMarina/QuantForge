"""Tables summarising a walk-forward meta-labeling run (diagnostics -> DataFrames)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["window_summary", "feature_selection_records", "family_selection_table", "sizing_constants_table",
           "probability_columns"]


def window_summary(diagnostics: Mapping) -> pd.DataFrame:
    """One row per ``(strategy, window)``: sizes, clusters, selection, inner log losses and the META choice."""
    rows = []
    for (s, w0), d in diagnostics.items():
        rows.append({"strategy": s, "window": w0.date(), "n_train": d["n_train"], "n_test": d["n_test"],
                     "usable": d["n_usable"], "clusters": len(d["groups"]), "selected": len(d["selected"]),
                     "sel_features": sum(len(d["groups"][g]) for g in d["selected"]), "placebo_max": d["placebo_max"],
                     "RF inner ll − prior": d["mda_base_ll"] - d["mda_prior_ll"], "META": d["meta"],
                     "META inner ll − prior": d["tuning"][d["meta"]]["inner_ll"] - d["prior_inner_ll"],
                     "calibrated inner ll − prior": d["calib_oof_ll"] - d["prior_inner_ll"]})
    return pd.DataFrame(rows).set_index(["strategy", "window"])


def feature_selection_records(diagnostics: Mapping, family: Mapping[str, str]) -> pd.DataFrame:
    """One row per (strategy, window, feature): its cluster's MDA, positive folds, selection, cluster size and
    cluster MDI per member; placebo features are included with family ``PLACEBO``."""
    rows = []
    for (s, w0), d in diagnostics.items():
        imp = d["mda"].drop(columns=["_base", "_prior"])
        mean_imp, pos = imp.mean(), (imp > 0).sum()
        for g, members in d["groups"].items():
            for c in members:
                rows.append({"strategy": s, "window": w0, "feature": c, "family": family.get(c, "?"),
                             "cluster_mda": mean_imp[g], "folds_positive": int(pos[g]), "selected": g in d["selected"],
                             "cluster_size": len(members), "cluster_mdi": d["mdi"].get(g, np.nan) / max(len(members), 1)})
        for c in [c for c in imp.columns if c.startswith("placebo_")]:
            rows.append({"strategy": s, "window": w0, "feature": c, "family": "PLACEBO", "cluster_mda": mean_imp[c],
                         "folds_positive": int(pos[c]), "selected": False, "cluster_size": 1, "cluster_mdi": np.nan})
    return pd.DataFrame(rows)


def family_selection_table(records: pd.DataFrame) -> pd.DataFrame:
    """Per family and strategy: selection share, mean cluster MDA and share of rows positive on >= 3 folds."""
    return records.groupby(["family", "strategy"]).agg(
        sel_share=("selected", "mean"), mean_mda=("cluster_mda", "mean"),
        share_folds_ge3=("folds_positive", lambda x: (x >= 3).mean())).unstack("strategy")


def sizing_constants_table(diagnostics: Mapping) -> pd.DataFrame:
    """Training-side sizing constants per window (prior, p_be, tau, shares of OOF p above p_be and 1/2, norms)."""
    rows = []
    for (s, w0), d in diagnostics.items():
        k = d["constants"]
        rows.append({"strategy": s, "window": w0.date(), "prior": k.prior, "p_be": k.p_be, "tau": k.tau,
                     "tau quantile": k.tau_quantile, "train share p ≥ p_be": float(np.mean(k.oof_sorted >= k.p_be)),
                     "train share p ≥ 0.5": float(np.mean(k.oof_sorted >= 0.5)), "norm P": k.norm["P"],
                     "norm P05": k.norm["P05"]})
    return pd.DataFrame(rows)


def probability_columns(pred: pd.DataFrame) -> dict[str, str]:
    """``{combo: column}`` for the calibrated combo probabilities of a prediction table, plus ``META -> p``."""
    combos = sorted({c[2:] for c in pred.columns if c.startswith("p_") and not c.endswith("_raw") and c != "p_"} - {""},
                    key=lambda x: (x.split("-")[0], x))
    return {**{c: f"p_{c}" for c in combos}, "META": "p"}

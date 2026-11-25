"""Bet sizing from a meta-model probability (AFML ch. 10, ch. 15) and the rule set of the research notebooks.

Primitives
----------
* :func:`afml_bet_size` - AFML Snippet 10.1 with a reference probability ``p0``:
  ``z = (p - p0) / sqrt(p (1 - p))``, size ``2 Phi(z) - 1`` when ``z > 0`` else 0.  The book uses ``p0 = 1/2``;
  for meta-labeling with asymmetric payouts the economically neutral point is the break-even win rate
  ``p_be = L / (W + L)`` (AFML §15.2-15.3), below which the expected value of the trade is negative.
* :func:`discretize_size` - AFML Snippet 10.3 (round to steps).
* :func:`ecdf` - empirical CDF of a training distribution evaluated at new points (point-in-time ranks).
* :func:`break_even_win_rate` - ``L / (W + L)`` from net trade returns.

Rule set (:data:`SIZING_RULES`)
-------------------------------
``U`` uniform, ``D`` inverse volatility ``clip(median_train(vol) / vol, 0.25, 4)``, ``F_be`` / ``F_tau`` binary
filters at ``p_be`` / at the threshold maximising training OOF PnL, ``R = 0.5 + ECDF_train(p)``,
``P`` = AFML size centred at ``p_be``, ``P05`` the literal rule (``p0 = 1/2``), ``Pd`` = ``P`` discretised by
0.1, and the products ``DxF_tau``, ``DxF_be``, ``DxR``, ``DxP`` (meta-model on top of inverse volatility).

Every constant comes from the training trades of a window (:func:`fit_sizing_constants`), and each rule is
divided by its training mean (:func:`normalized_sizes`) so that all rules have the same expected exposure per
signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..evaluation.metrics import EPS

__all__ = ["afml_bet_size", "discretize_size", "ecdf", "break_even_win_rate", "SIZING_RULES", "SizingConstants",
           "fit_sizing_constants", "rule_sizes", "normalized_sizes"]

SIZING_RULES = ("U", "D", "F_tau", "F_be", "R", "P", "P05", "Pd", "DxF_tau", "DxF_be", "DxR", "DxP")


def afml_bet_size(p, p0: float = 0.5) -> np.ndarray:
    """AFML Snippet 10.1 for the "trade" class with reference probability ``p0`` (0 when ``p <= p0``)."""
    from scipy.stats import norm

    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    z = (p - p0) / np.sqrt(p * (1 - p))
    return np.where(z > 0, 2 * norm.cdf(z) - 1, 0.0)


def discretize_size(m, step: float = 0.1) -> np.ndarray:
    """AFML Snippet 10.3: round sizes to multiples of ``step``."""
    return np.round(np.asarray(m, dtype=float) / step) * step


def ecdf(sorted_reference: np.ndarray, x) -> np.ndarray:
    """Share of the (sorted) reference sample ``<= x``."""
    return np.searchsorted(sorted_reference, np.asarray(x, dtype=float), side="right") / len(sorted_reference)


def break_even_win_rate(returns, labels=None) -> tuple[float, float, float]:
    """``(p_be, avg_win W, avg_loss L)`` with ``p_be = L / (W + L)`` (0.5 when undefined)."""
    r = np.asarray(returns, dtype=float)
    y = (r > 0).astype(int) if labels is None else np.asarray(labels)
    wins, losses = r[y == 1], -r[y == 0]
    W = wins.mean() if len(wins) else 0.0
    L = losses.mean() if len(losses) else 0.0
    return (float(L / (W + L)) if (W + L) > 0 else 0.5), float(W), float(L)


@dataclass
class SizingConstants:
    """Everything a sizing rule needs, estimated on training trades only.

    prior         training win rate;  p_be, avg_win, avg_loss  break-even win rate and payouts;
    vol_med       median training volatility (inverse-volatility reference);
    oof_sorted    sorted training OOF probabilities (for ECDF ranks);
    tau           filter threshold maximising training OOF PnL; tau_quantile its quantile level;
    norm          training mean of every raw rule (the normalisation that equalises expected exposure);
    vol_clip      clip of the inverse-volatility multiplier.
    """

    prior: float
    p_be: float
    avg_win: float
    avg_loss: float
    vol_med: float
    oof_sorted: np.ndarray
    tau: float = np.nan
    tau_quantile: float = np.nan
    norm: dict = field(default_factory=dict)
    vol_clip: tuple = (0.25, 4.0)

    def __getitem__(self, key):   # dict-style access, convenient in reports
        return getattr(self, key)


def rule_sizes(p, vol, k: SizingConstants) -> dict[str, np.ndarray]:
    """Un-normalised sizes of every rule in :data:`SIZING_RULES` from probabilities ``p`` and volatilities ``vol``."""
    p = np.asarray(p, dtype=float)
    d = np.clip(k.vol_med / np.asarray(vol, dtype=float), *k.vol_clip)
    d = np.where(np.isfinite(d), d, 1.0)
    s = {"U": np.ones(len(p)), "D": d,
         "F_tau": (p >= k.tau).astype(float), "F_be": (p >= k.p_be).astype(float),
         "R": 0.5 + ecdf(k.oof_sorted, p), "P": afml_bet_size(p, k.p_be), "P05": afml_bet_size(p, 0.5),
         "Pd": discretize_size(afml_bet_size(p, k.p_be))}
    s["DxF_tau"], s["DxF_be"], s["DxR"], s["DxP"] = d * s["F_tau"], d * s["F_be"], d * s["R"], d * s["P"]
    return s


def fit_sizing_constants(oof_p, y, returns, vol, *, vol_clip=(0.25, 4.0),
                         tau_quantiles=np.linspace(0, 0.7, 15)) -> SizingConstants:
    """Sizing constants from training trades: OOF probabilities, labels, net returns and volatilities.

    The filter threshold ``tau`` is the quantile of the OOF probabilities (levels ``tau_quantiles``, accept-all
    included) that maximises the summed training return of the accepted trades (ties: lowest).
    """
    oof_p, y, r = np.asarray(oof_p, dtype=float), np.asarray(y), np.asarray(returns, dtype=float)
    p_be, W, L = break_even_win_rate(r, y)
    k = SizingConstants(prior=float(y.mean()), p_be=p_be, avg_win=W, avg_loss=L, vol_med=float(np.nanmedian(vol)),
                        oof_sorted=np.sort(oof_p), vol_clip=tuple(vol_clip))
    cands = np.quantile(oof_p, tau_quantiles)
    pnl = [r[oof_p >= c].sum() for c in cands]
    best = int(np.argmax(pnl))
    k.tau, k.tau_quantile = float(cands[best]), float(np.asarray(tau_quantiles)[best])
    k.norm = {rule: float(np.mean(v)) for rule, v in rule_sizes(oof_p, vol, k).items()}
    return k


def normalized_sizes(pred: pd.DataFrame, constants: dict, *, key_cols=("strategy", "window"), p_col: str = "p",
                     vol_col: str = "v_rvol_30d", cap: float | None = 10.0, rules=SIZING_RULES,
                     return_uncapped: bool = False):
    """Sizes of every rule for a prediction table, each rule divided by its training mean, capped at ``cap``.

    ``constants`` maps a key tuple (values of ``key_cols``, e.g. ``(strategy, window_start)``) to the
    :class:`SizingConstants` fitted for that model.  Rows without constants get size 0.
    """
    keys = list(key_cols)
    out = {r: np.zeros(len(pred)) for r in rules}
    for key, k in constants.items():
        m = np.ones(len(pred), dtype=bool)
        for c, v in zip(keys, key if isinstance(key, tuple) else (key,)):
            m &= (pred[c] == v).to_numpy()
        raw = rule_sizes(pred.loc[m, p_col].to_numpy(), pred.loc[m, vol_col].to_numpy(), k)
        for r in rules:
            out[r][m] = raw[r] / k.norm[r] if k.norm[r] > 0 else 0.0
    uncapped = pd.DataFrame(out, index=pred.index)
    capped = uncapped.clip(upper=cap) if cap is not None else uncapped
    return (capped, uncapped) if return_uncapped else capped

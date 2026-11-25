"""Probability calibration.

:class:`PlattCalibrator` fits ``P(y = 1) = sigmoid(a * logit(p) + b)`` on out-of-fold probabilities (practically
unregularised logistic regression).  Bet-sizing rules read the probability literally (AFML §10.3; Joubert 2022
§4), so a model that ranks well but is mis-scaled must be calibrated before sizing.
"""

from __future__ import annotations

import numpy as np

from ..evaluation.metrics import EPS

__all__ = ["PlattCalibrator", "logit"]


def logit(p, eps: float = EPS) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(q / (1 - q))


class PlattCalibrator:
    """``PlattCalibrator().fit(oof_p, y).predict(p)``."""

    def __init__(self, eps: float = EPS, C: float = 1e6, max_iter: int = 1000):
        self.eps, self.C, self.max_iter = eps, C, max_iter

    def fit(self, p, y) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(logit(p, self.eps)[:, None], np.asarray(y))
        return self

    def predict(self, p) -> np.ndarray:
        return self.model_.predict_proba(logit(p, self.eps)[:, None])[:, 1]

    @property
    def coefficients(self) -> tuple[float, float]:
        """``(slope a, intercept b)``; ``(1, 0)`` means the input was already calibrated."""
        return float(self.model_.coef_[0, 0]), float(self.model_.intercept_[0])

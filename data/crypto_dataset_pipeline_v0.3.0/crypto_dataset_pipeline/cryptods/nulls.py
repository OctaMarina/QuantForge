"""Null-reason vocabulary.

A feature value that cannot be computed correctly is null (never 0, never imputed).  Each feature
column is paired with an ``int8`` reason-code array during computation; the codes are compressed
into run-length records in ``quality.parquet`` so the main dataset stays narrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID = 0
REASONS = {
    0: "valid",
    1: "not_available_in_period",
    2: "not_applicable",
    3: "missing_source_data",
    4: "unknown_publication_time",
    5: "unknown_historical_vintage",
    6: "insufficient_history",
    7: "stale_input",
    8: "feed_gap",
    9: "insufficient_book_depth",
    10: "unsupported_granularity",
    11: "invalid_regression",
    12: "unverified_formula",
    13: "zero_denominator",
    14: "nonpositive_log_input",
    15: "incomplete_coverage",
    16: "source_unreachable",
    17: "unsupported_instrument",
    18: "not_finalized",
    19: "invalid_quote",
    20: "no_observation",
    21: "not_evaluated",
}
CODE = {v: k for k, v in REASONS.items()}


def code(name: str) -> int:
    return CODE[name]


class Col:
    """A feature column under construction: float values + reason codes (0 = valid)."""

    __slots__ = ("values", "reason")

    def __init__(self, n: int, default_reason: str = "missing_source_data", dtype=np.float64):
        self.values = np.full(n, np.nan, dtype=dtype)
        self.reason = np.full(n, CODE[default_reason], dtype=np.int8)

    @classmethod
    def from_arrays(cls, values, reason):
        c = cls.__new__(cls)
        c.values = np.asarray(values, dtype=np.float64)
        c.reason = np.asarray(reason, dtype=np.int8)
        return c

    def set_valid(self, mask, vals):
        mask = np.asarray(mask, dtype=bool)
        self.values[mask] = np.asarray(vals)[mask] if np.ndim(vals) else vals
        self.reason[mask] = VALID

    def invalidate(self, mask, reason: str):
        mask = np.asarray(mask, dtype=bool)
        self.values[mask] = np.nan
        self.reason[mask] = CODE[reason]

    def finalize(self):
        """Any non-finite value still flagged valid is a bug upstream -> mark explicitly."""
        bad = (self.reason == VALID) & ~np.isfinite(self.values)
        if bad.any():
            self.reason[bad] = CODE["invalid_regression"]
            self.values[bad] = np.nan
        self.values[self.reason != VALID] = np.nan
        return self


def all_null(n: int, reason: str) -> Col:
    return Col(n, reason)


def compress_reasons(feature: str, reason: np.ndarray, row_key: np.ndarray) -> pd.DataFrame:
    """Run-length encode reason codes along rows (only non-valid runs are stored)."""
    reason = np.asarray(reason)
    n = reason.size
    if n == 0:
        return pd.DataFrame(columns=["feature", "reason", "first_row", "last_row", "n_rows"])
    change = np.flatnonzero(np.diff(reason)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]]) - 1
    codes = reason[starts]
    keep = codes != VALID
    starts, ends, codes = starts[keep], ends[keep], codes[keep]
    return pd.DataFrame({
        "feature": feature,
        "reason": [REASONS[int(c)] for c in codes],
        "first_row": np.asarray(row_key)[starts],
        "last_row": np.asarray(row_key)[ends],
        "n_rows": (ends - starts + 1).astype(np.int64),
    })

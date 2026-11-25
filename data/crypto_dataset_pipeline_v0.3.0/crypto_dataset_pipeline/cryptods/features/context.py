"""Feature computation context shared by all families."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..nulls import Col
from ..registry import FeatureSpec


@dataclass
class Ctx:
    cfg: dict
    spec: object                      # InstrumentSpec
    bar_type: str
    bars: pd.DataFrame                # all bars incl. warm-up, sorted by bar_seq (float view + ints)
    decision_us: np.ndarray           # decision time per bar row
    ref_us: np.ndarray                # market-time reference (bar_end) per row
    px1s: pd.DataFrame | None = None  # sec_start_us, last_ts_us, price  (from the same trade stream)
    gaps: list = field(default_factory=list)          # [(start_us, end_us)] unrecoverable trade gaps
    data_start_us: int | None = None  # first time covered by the trade stream
    tables: dict = field(default_factory=dict)        # normalised auxiliary / external tables
    out: dict = field(default_factory=dict)           # name -> Col
    specs: list = field(default_factory=list)         # FeatureSpec
    quality: dict = field(default_factory=dict)       # family -> notes

    @property
    def n(self) -> int:
        return len(self.bars)

    def add(self, spec: FeatureSpec, col: Col):
        if spec.name in self.out:
            raise ValueError(f"duplicate feature {spec.name}")
        self.out[spec.name] = col.finalize()
        self.specs.append(spec)

    def gap_overlaps(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """True where a known trade-feed gap intersects the closed interval [lo, hi]."""
        lo = np.asarray(lo, dtype=np.int64)
        hi = np.asarray(hi, dtype=np.int64)
        bad = np.zeros(lo.shape, dtype=bool)
        for (a, b) in self.gaps:
            bad |= (lo <= b) & (hi >= a)
        return bad

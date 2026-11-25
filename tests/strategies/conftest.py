"""Shared fixtures: synthetic bar generator, break injection and the reference oracle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE_FILE = HERE / "reference" / "primary_strategies_reference.py"
# The original notebook was moved to research/meta-labeling/legacy/; accept either location.
_NOTEBOOK_CANDIDATES = [ROOT / "research/meta-labeling/legacy/primary_strategies.ipynb",
                        ROOT / "research/meta-labeling/primary_strategies.ipynb"]
NOTEBOOK = next((p for p in _NOTEBOOK_CANDIDATES if p.exists()), _NOTEBOOK_CANDIDATES[0])
BTC_PARQUET = (ROOT / "data/crypto_dataset_pipeline/data/datasets/"
               "BTCUSDT_spot_volume_2650_25pd_20210901-20260901_99bf72d5/volume.parquet")

# Lines of the reference source where the signal arrays are complete.  Capture hooks are
# injected right before them; the assertion below makes a changed reference fail loudly.
_HOOKS = {
    "rsi_tis": (
        "    fee = fee_bps / 10_000.0\n    slip_usd_per_coin = slippage_ticks * tick_size\n",
        "    _CAPTURE.clear()\n"
        "    _CAPTURE.update(enter=enter.copy(), exit_ma=exit_ma.copy(), exit_tail=exit_tail.copy(),\n"
        "                    rsi_value=rsi_values.copy(), sma_trend=sma_values.copy(), atr_value=atr_values.copy(),\n"
        "                    green_streak=streak_values.copy(), htf_green_streak=htf_streak_values.copy(),\n"
        "                    htf_momentum=htf_momentum.copy())\n"),
    "supertrend": (
        "    # Pine `time` is bar-start time, and its backtest window gates entries only.\n",
        "    _CAPTURE.clear()\n"
        "    _CAPTURE.update(longs=longs.copy(), shorts=shorts.copy(), st_flip=flips.copy(),\n"
        "                    signal_score=scores.copy(), st_band=st_bands.copy(), st_direction=st_dirs.copy(),\n"
        "                    atr_value=atr_vals.copy(), st_adx=adx_vals.copy(), st_atr_ratio=atr_ratio.copy(),\n"
        "                    st_regime=regimes.copy(), st_adapt_multiplier=adapted_mult.copy(),\n"
        "                    st_ema=trend_ema.copy(), st_vol_ma=vol_ma.copy(),\n"
        "                    sl_distance=sl_distances.copy(), tp_distance=tp_distances.copy())\n"),
    "donchian": (
        "    fee, slip = fee_bps / 10000.0, slippage_ticks * tick_size\n",
        "    _CAPTURE.clear()\n"
        "    _CAPTURE.update(longs=longs.copy(), shorts=shorts.copy(), **{k: a.copy() for k, a in arrays.items()})\n"),
}


HTF_LOOKUP = '    known = np.searchsorted(confirmations, ready_us[idx], side="right") - 1\n'
HTF_LOOKUP_FIXED = ('    confirmations = np.maximum.accumulate(confirmations)  # test-side copy of the library correction\n'
                    + HTF_LOOKUP)


class Reference:
    """The frozen notebook implementation, with read-only capture of its signal arrays.

    ``monotone_htf=True`` applies the single documented correction of the library (running maximum of the
    HTF confirmation times) so that the remaining behaviour can still be compared exactly.
    """

    def __init__(self, monotone_htf: bool = False):
        source = REFERENCE_FILE.read_text()
        if monotone_htf:
            assert source.count(HTF_LOOKUP) == 1, "HTF lookup line not found exactly once in the reference"
            source = source.replace(HTF_LOOKUP, HTF_LOOKUP_FIXED)
        for key, (anchor, hook) in _HOOKS.items():
            assert source.count(anchor) == 1, f"reference hook anchor for {key!r} not found exactly once"
            source = source.replace(anchor, hook + anchor)
        namespace: dict = {"_CAPTURE": {}, "__name__": "primary_strategies_reference"}
        exec(compile(source, str(REFERENCE_FILE), "exec"), namespace)
        self._ns = namespace

    def run(self, df, **kwargs):
        out = self._ns["strategy_events"](df, **kwargs)
        return out, dict(self._ns["_CAPTURE"])


@pytest.fixture(scope="session")
def reference() -> Reference:
    return Reference()


@pytest.fixture(scope="session")
def reference_monotone_htf() -> Reference:
    return Reference(monotone_htf=True)


# ----------------------------------------------------------------------------- data
def make_bars(n: int = 4000, seed: int = 0, mean_minutes: float = 20.0,
              start: str = "2024-01-01", with_extras: bool = True) -> pd.DataFrame:
    """Contiguous volume-bar-like series with trending/ranging regimes and volatility clusters."""
    rng = np.random.default_rng(seed)
    duration = (rng.exponential(mean_minutes, n).clip(2, 120) * 60e6).astype(np.int64)
    t0 = int(pd.Timestamp(start, tz="UTC").value // 1000)
    bar_end = t0 + np.cumsum(duration)
    bar_start = np.r_[t0, bar_end[:-1]]
    regime_len = 250
    drift = np.repeat(rng.normal(0, 0.0012, n // regime_len + 1), regime_len)[:n]
    vol = np.repeat(rng.uniform(0.0006, 0.003, n // 90 + 1), 90)[:n]
    ret = drift + vol * rng.standard_normal(n)
    close = np.round(50_000 * np.exp(np.cumsum(ret)), 1)
    open_ = np.r_[close[0], close[:-1]] * (1 + 0.0002 * rng.standard_normal(n))
    open_ = np.round(open_, 1)
    spread = vol * np.abs(rng.standard_normal(n)) * 0.6
    high = np.round(np.maximum(open_, close) * (1 + spread), 1)
    low = np.round(np.minimum(open_, close) * (1 - spread), 1)
    df = pd.DataFrame({
        "bar_start_us": bar_start, "bar_end_us": bar_end,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(3.0, 0.5, n) * (1 + 3 * vol / vol.mean()),
    })
    if with_extras:
        df["first_trade_ts_us"] = bar_start + rng.integers(0, 5_000_000, n)
        df["bar_available_us"] = bar_end + rng.integers(0, 20_000_000, n)
        df["decision_time_us"] = df["bar_available_us"]
        df["complete"] = 1
        df["bar_seq"] = np.arange(n)
        df["segment_id"] = 0
        df["series_id"] = "synthetic"
        df["bar_type"] = "volume"
        df["bar_id"] = [f"synthetic:{i}" for i in range(n)]
        df["aggr_imbalance_notional"] = rng.normal(0, 1, n)   # an unrelated bar-level column
    return df


def inject_breaks(df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Add every kind of data break the strategies must treat as a segment boundary."""
    out = df.copy()
    n = len(out)
    at = lambda f: int(n * f)  # noqa: E731
    out.loc[at(0.15):, "bar_seq"] += 3                                   # bar_seq gap
    out.loc[at(0.30):, "segment_id"] += 1                                # declared segment change
    out.loc[at(0.45), "complete"] = 0                                    # incomplete bar
    out.loc[at(0.55), "close"] = np.nan                                  # non-finite price
    bad = at(0.65)
    out.loc[bad, "high"] = out.loc[bad, "low"] * 0.99                    # inconsistent OHLC
    ov = at(0.75)
    out.loc[ov, "bar_start_us"] = out.loc[ov - 1, "bar_end_us"] - 1      # overlap with previous bar
    out.loc[at(0.85), "volume"] = -1.0                                   # invalid volume
    return out


def drop_calendar_gap(df: pd.DataFrame, frac: float = 0.5, hours: float = 40.0) -> pd.DataFrame:
    """Remove bars covering `hours` and renumber bar_seq so the hole looks adjacent."""
    out = df.copy()
    a = int(len(out) * frac)
    t = out["bar_start_us"].iloc[a]
    keep = ~((out["bar_start_us"] >= t) & (out["bar_start_us"] < t + hours * 3.6e9))
    out = out[keep].reset_index(drop=True)
    out["bar_seq"] = np.arange(len(out))
    return out


@pytest.fixture(scope="session")
def bars() -> pd.DataFrame:
    return make_bars(4000, seed=1)


@pytest.fixture(scope="session")
def btc_bars() -> pd.DataFrame:
    if not BTC_PARQUET.exists():
        pytest.skip(f"real dataset not available: {BTC_PARQUET}")
    d = pd.read_parquet(BTC_PARQUET)
    d = d.set_index(pd.to_datetime(d["bar_end_us"], unit="us", utc=True)).sort_index().rename_axis("datetime")
    return d[~d.index.duplicated(keep="first")]

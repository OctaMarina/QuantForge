"""Comparison helpers between the library and the reference notebook implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def us(series) -> np.ndarray:
    """tz-aware timestamps -> int64 microseconds (independent of the datetime unit)."""
    s = pd.Series(pd.to_datetime(series, utc=True))
    return s.astype("datetime64[us, UTC]").astype("int64").to_numpy()


def _plain(x):
    if isinstance(x, pd.Series) and (str(x.dtype).startswith(("Int", "Float", "boolean"))):
        return x.to_numpy(dtype=float)
    return np.asarray(x)


def assert_same(a, b, what: str, exact: bool = False, rtol: float = 1e-12) -> None:
    a, b = _plain(a), _plain(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    if a.dtype.kind in "fc" and not exact:
        np.testing.assert_allclose(a.astype(float), b.astype(float), rtol=rtol, atol=1e-12,
                                   equal_nan=True, err_msg=what)
    else:
        np.testing.assert_array_equal(a, b, err_msg=what)


# reference column -> library column (same quantity, renamed or unchanged)
COMMON_TRADE_MAP = {
    "trade_id": "trade_id", "entry_side": "side", "entry_price": "entry_price", "exit_price": "exit_price",
    "exit_reason": "exit_reason", "trade_return_gross": "trade_return_gross", "trade_return": "trade_return",
    "trade_return_pct": "trade_return_pct", "label": "label", "pnl_usd": "pnl_usd", "fees_usd": "fees_usd",
    "position_qty": "position_qty", "position_notional_usd": "position_notional_usd",
    "position_exposure_pct": "position_exposure_pct", "entry_equity_usd": "entry_equity_usd",
    "exit_equity_usd": "exit_equity_usd", "holding_hours": "holding_hours",
}
TIME_MAP = {"signal_time": "decision_time", "entry_time": "entry_time", "exit_time": "exit_time"}

EXTRA_TRADE_MAP = {
    "donchian_breakout_continuation": {
        "holding_bars": "holding_bars", "signal_bar_index": "signal_bar_index",
        "entry_bar_index": "entry_bar_index", "exit_signal_bar_index": "exit_signal_bar_index",
        "exit_bar_index": "exit_bar_index", "equity_path_complete": "equity_path_complete",
        "signal_close_before_available": "signal_close_before_available"},
    "supertrend_regime_confluence": {
        "sl_price": "sl_price", "tp_price": "tp_price", "ambiguous_intrabar": "ambiguous_intrabar",
        "exit_time_is_bar_end_proxy": "exit_time_is_bar_end_proxy",
        "trailing_activated": "trailing_activated", "position_sizing_mode": "position_sizing_mode"},
}

# reference feature column on the trade row -> library feature column
FEATURE_MAP = {
    "rsi_cross_above": {"rsi_value": "rsi_value"},
    "tis_btc_momentum_confluence": {"sma_trend": "sma_trend", "atr_value": "atr_value", "green_streak": "green_streak",
                                    "htf_green_streak": "htf_green_streak", "htf_momentum": "htf_momentum"},
    "donchian_breakout_continuation": {
        "dc_high": "dc_high", "dc_low": "dc_low", "atr_value": "atr_value", "adx_value": "adx_value",
        "plus_di": "plus_di", "minus_di": "minus_di", "ema200": "ema_trend",
        "long_breakout_atr": "long_breakout_atr", "short_breakout_atr": "short_breakout_atr"},
    "supertrend_regime_confluence": {
        "atr_value": "atr_value", "st_band": "st_band", "st_direction": "st_direction", "st_regime": "st_regime",
        "st_atr_ratio": "st_atr_ratio", "st_adx": "st_adx", "st_ema": "st_ema", "st_vol_ma": "st_vol_ma",
        "st_adapt_multiplier": "st_adapt_multiplier", "signal_score": "signal_score",
        "sl_distance": "sl_distance", "tp_distance": "tp_distance"},
}

# reference result.attrs -> library diagnostics
ATTR_MAP = {
    "censored_positions": "censored_positions", "discarded_pending_orders": "n_discarded_pending_orders",
    "total_opened_positions": "total_opened_positions", "remaining_realized_equity_usd": "remaining_realized_equity_usd",
    "unrealized_censored_pnl_usd": "unrealized_censored_pnl_usd",
}


def compare_trades(strategy: str, ref: pd.DataFrame, res) -> None:
    new = res.closed_trades_frame()
    assert len(ref) == len(new), f"closed trades: reference {len(ref)} vs library {len(new)}"
    for r, n in COMMON_TRADE_MAP.items():
        assert_same(ref[r], new[n], f"{strategy}.{r}")
    for r, n in TIME_MAP.items():
        assert_same(us(ref[r]), us(new[n]), f"{strategy}.{r}", exact=True)
    if strategy == "donchian_breakout_continuation":
        assert_same(us(ref["exit_signal_time"]), us(new["exit_decision_time"]), "exit_signal_time", exact=True)
    for r, n in EXTRA_TRADE_MAP.get(strategy, {}).items():
        assert_same(ref[r], new[n], f"{strategy}.{r}")
    for r, n in FEATURE_MAP[strategy].items():
        assert_same(ref[r], new[n], f"{strategy}.feature.{r}")


def compare_diagnostics(ref: pd.DataFrame, res) -> None:
    for r, n in ATTR_MAP.items():
        if r in ref.attrs:
            a, b = ref.attrs[r], res.diagnostics[n]
            assert np.isclose(a, b, rtol=1e-12, atol=1e-9), f"{r}: reference {a} vs library {b}"

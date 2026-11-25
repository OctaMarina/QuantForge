# `quant_forge.features` — point-in-time feature engineering

```python
from quant_forge import features as qf

feats = qf.bar_features(bars, bars_per_day=6, btc="BTC", n_jobs=1)      # {asset: DataFrame}
sel = qf.select_ffd_order({a: f["_log_close"] for a, f in feats.items()}, train_end=pd.Timestamp("2019-03-01", tz="UTC"))
feats = qf.add_ffd_features(feats, sel.pooled)                           # ffd_log_{close,open,volume}[_z]
feats = qf.add_raw_levels(feats, bars)                                   # raw_open, raw_close, ... (diagnostics only)
cols = qf.model_columns(feats["BTC"])                                    # everything except "_" auxiliaries
```

Every feature at bar `t` uses bars `≤ t` (tested by truncation in `tests/features`). Horizons are in **days** and are
converted with `bars_per_day` (6 for 4-hour bars, 1 for daily bars).

## Bar features

### Single asset — `asset.py`

`asset_features(bars, bars_per_day=6, families=("V", "T", "F", "CAL"), aux=True)` (or `volatility_features`,
`trend_features`, `flow_features`, `calendar_features` for one family). Input: one bar frame from
`quant_forge.data.klines_to_bars` (OHLC, `volume`, `quote_volume`, `n_trades`, `taker_buy_base`, bar-end index).

| Family | Columns |
|---|---|
| V volatility | `v_rvol_{1,3,7,30,90}d` (realised vol per bar), `v_rvol_ratio_1d_30d`, `_3d_30d`, `_7d_90d`, `v_rvol_pct_1y`, `v_bb_width`, `v_atr_rel`, `v_range_rel`, `v_range_rel_6`, `v_parkinson_7d`, `v_gk_7d`, `v_vol_of_vol_30d`, `v_rvol_30d` (30-day vol, the inverse-volatility sizing input) |
| T trend | `t_ret_{1,3,7,14,30,60,90}d` and `_z` (scaled by 30-day vol), `t_ema{20,50,100,200}_dist_atr`, `t_sma{200,50}d_dist`, `t_dd_{30,90,180}d`, `t_up_*`, `t_eff_{10,20,40}`, `t_rsi{7,14,28}`, `t_pos_range_{30,90}d`, `t_bars_since_high_90d`, `t_skew_30d`, `t_kurt_30d`, `t_acf1_30d`, `t_green_share_7d` |
| F flow | `f_taker_imb_{1,6,18,42}`, `f_qvol_rel_{7,30,90}d`, `f_ntrades_rel_30d`, `f_vol_z_20`, `f_amihud_7d`, `f_corr_ret_vol_20`, `f_qvol_trend_7d_30d` |
| CAL | `cal_hour_sin/cos`, `cal_dow_sin/cos`, `cal_weekend` |
| auxiliary | `_r` (log return), `_above_sma50d`, `_above_sma200d`, `_log_close`, `_log_open`, `_log_volume` |

### Panel — `market.py`

* `market_state_features(feats, bars_per_day, btc="BTC")` → one frame on the union of bar times (family M):
  equal-weighted index returns `m_ret_{1,7,30,90,365}d[_z]`, `m_rvol_7d/30d`, `m_rvol_ratio_7d_90d`, `m_rvol_pct_1y`,
  `m_dd_180d`, breadth `m_breadth_sma{50d,200d}`, dispersion, `m_alt_vs_btc_30d`, BTC state `b_*`, `m_n_assets`, and
  `m_avg_corr_30d` (mean pairwise correlation over 30 days, computed once per day and forward-filled).
* `cross_sectional_features(feats, bars_per_day, rank_columns=DEFAULT_RANK_COLUMNS)` → per asset: R family
  (`x_rs_{7,30,90}d` vs the cross-sectional median, `x_corr_mkt_30d`, `x_beta_30d`, `x_rank_ret_30d`) and X family
  (`xr_<name>` = percentile rank among assets with data at the same time). Each panel is ranked once.
* `panel_features(feats, ...)` appends R, X and M to every asset frame; `bar_features` = `asset_features` for every
  asset + `panel_features`.

Only assets with a bar at `t` enter the panel aggregates at `t`. The order of assets in the input mapping only matters at
floating-point rounding level.

## Bar features in elapsed time — `elapsed.py` (any bar type)

`asset.py` counts horizons in bars (`days * bars_per_day`), exact only for time bars. `elapsed.py` measures every horizon in
elapsed time, so the same definitions apply to time, volume and dollar bars at any frequency:

```python
feats = qf.elapsed_bar_features(bars, btc="BTC", max_staleness="3D", n_jobs=8)   # {asset: V, T, F, CAL, M, R, X}
```

* a d-day window holds the bars whose end lies in `(t − d, t]`; sums over it (squared returns, volume, trades) do not depend on
  how often the series is sampled; returns over d days use the last bar ended at or before `t − d`;
* moving averages are duration-weighted (time averages of the price path), calendar features use the bar end (hour with
  minutes);
* panel families are computed at each asset's own bar ends from the other assets' **latest bars ended at or before t** (as-of
  alignment), dropping assets whose last bar is older than `max_staleness`. With synchronous time bars this is the usual
  cross-section.

Columns: `v_vol_{1,7,90}d`, `v_rvol_30d` (inverse-volatility input), `v_vol_ratio_1d_30d`, `v_vol_ratio_7d_90d`,
`v_range_1d`, `t_ret_{1,7,30,90}d` (+ `_z`), `t_dd_{30,90}d`, `t_up_30d`, `t_pos_range_30d`, `t_sma{50,200}d_dist`,
`t_eff_7d`, `f_taker_imb_{1,7}d`, `f_qvol_rel_1d_30d`, `f_qvol_rel_7d_30d`, `f_qvol_rel_30d_365d`, `f_ntrades_rel_1d_30d`,
`f_amihud_7d`, `cal_*`, `m_n_assets`, `m_ret_{1,7,30}d`, `m_vol_7d`, `m_rvol_30d`, `m_breadth_sma{50,200}d`,
`m_dispersion_7d`, `m_alt_vs_btc_30d`, `b_ret_7d_z`, `b_ret_30d_z`, `b_vol_ratio_1d_30d`, `b_dd_90d`, `x_rs_{7,30}d`,
`x_rank_ret_30d`, `xr_vol_30d`, `xr_qvol_rel_1d_30d`, `xr_ret_7d_z`, plus `raw_close` and the auxiliaries
`_above_sma{50,200}d`. On time bars without missing klines the per-asset columns equal their bar-count counterparts in
`asset.py` (tested); tests also truncate asynchronous panels after a time and check that nothing before it changes.

## Fractional differentiation — `fracdiff.py` (AFML ch. 5)

| Function | Purpose |
|---|---|
| `ffd_weights(d, thresh=1e-4, max_size=None)` | fixed-width window weights (oldest … newest), identical to `mlfinpy.util.frac_diff.get_weights_ffd` |
| `frac_diff_ffd(x, d, thresh)` | causal FFD by convolution → `(values, window width)`; NaN until the window is full |
| `adf_statistic(x, min_obs=200, maxlag=1, regression="c")` | ADF statistic and 5 % critical value (statsmodels) |
| `select_ffd_order(series, train_end, grid, thresh, min_obs, pooled_quantile=0.9)` | per asset the smallest `d` rejecting a unit root **on data before `train_end`**; fallback (median) for assets without history; one pooled `d` (90th percentile); trade-off table (share rejecting, median ADF, retained memory = corr with the input, window length) |
| `add_ffd_features(feats, d, sources=DEFAULT_FFD_SOURCES, z_window_days=365, z_min_days=60)` | FFD of `_log_close/_log_open/_log_volume` and a trailing z-score version (`*_z`) |

Choosing `d` on a training span only (and one pooled `d` for a pooled model) avoids look-ahead. With `d < 1` the series
keeps a slowly moving level, so the cross-asset scale still differs: the trailing z-score makes it poolable.

## Trade-level features — `trades.py`

Work on a trade table (one row per closed trade with `strategy`, `asset`, `side`, `decision_time`, `entry_time`,
`exit_time`, `trade_return`, `y`).

* `track_record_features(trades)` (family S): `s_swr_{30,100}` / `s_mr_{30,100}` (win rate / mean clipped return of the
  last N closed trades of the strategy), `s_awr_100` (all strategies), `s_wr_asset_{10,30}` (strategy × asset),
  `s_open_positions`, `s_signals_24h`, `s_closed_30d` (normalised by `m_n_assets`). Only trades with
  `exit_time < decision_time` count; a rate needs `N // 2` closed trades.
* `orient_by_side(trades, directional=DIRECTIONAL_PREFIXES, centered=CENTERED_PREFIXES)`: direction-carrying features are
  multiplied by `side` (centred ones first become `x − 0.5`, or `x` for `lib_*` ratios), so "in favour of the trade" has
  one sign for longs and shorts. No-op for long-only data.
* `relative_levels(trades, feature_names, close_column="raw_close")`: the strategy's price-level features become
  `lib_<name>_rel = level / close − 1` (distances such as ATR: `/ close`), with the raw level kept as `lib_<name>_level`
  for diagnostics; `LEVEL_FEATURES` lists which features are levels.
* `feature_family(column)`, `feature_families(columns)`, `FAMILY_PREFIXES`, `FAMILY_ORDER`.

## Kernels — `kernels.py`

* `wilder_mean(values, n)` – Wilder smoothing seeded by the mean of the first `n` finite values; later NaNs carry the
  previous value (numba-accelerated when installed, identical results). This differs on purpose from
  `quant_forge.strategies.indicators.wilder_rma`, which must restart after a NaN to match TradingView.
* `rsi(close, n)` – RSI on a pandas Series (50 for a flat market).
* `bars_since_rolling_max(x, window)` – vectorised equivalent of
  `x.rolling(window).apply(lambda v: len(v) - 1 - np.argmax(v), raw=True)`.

## Performance

40 assets × ~16 000 4h bars: about 10 s for all bar features (per-asset 4 s, market 3 s, cross-section 3 s). The
average-correlation feature dominates the market part and is computed with pandas' own pairwise-complete correlation so
that results match a straightforward implementation exactly.

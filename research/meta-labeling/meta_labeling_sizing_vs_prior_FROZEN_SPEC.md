# Frozen specification — meta-model sizing vs prior (written before running the notebook)

Date: 2026-11-19. Strategies: RSI_CROSS_ABOVE, DONCHIAN_BREAKOUT_CONTINUATION, TIS_BTC_MOMENTUM_CONFLUENCE,
SUPERTREND_REGIME_CONFLUENCE (library defaults, fee 5 bps/fill, slippage 0). Label y = 1[trade_return > 0].
Bars: Binance spot 4h klines (identical to cryptods time bars), last bar end 2026-11-19 00:00 UTC.

## What is already known (and therefore NOT clean)
- SEEN = the 20 assets used in every earlier analysis (BTC ETH BNB SOL XRP ADA AVAX BCH DOGE DOT LINK LTC UNI AAVE ATOM ETC FIL
  NEAR TRX XLM). Periods A (decided < 2021-09-01) and B (2021-09-01 .. 2026-08-31) were used in earlier experiments;
  the signal reported in the notebook `meta_labeling_beating_the_prior` follow-up was found there.
- Clean evidence therefore comes from: (1) **NEW assets** = 20 assets never used in any analysis
  (ALGO VET THETA SAND MANA AXS ICP HBAR CRV GRT ENJ CHZ ZEC DASH XTZ COMP SNX IOTA ZIL QTUM), over all periods;
  (2) **period C** (decided >= 2026-09-01), all assets. Results on SEEN assets in A/B are reported but labelled "seen".

## Features (own-asset only, point-in-time, all computed from bars up to the signal bar)
- Strategy library features: RSI [rsi_value]; DONCHIAN [adx_value, di_diff, long_breakout_atr, ema_distance, atr_rel];
  TIS [green_streak, htf_green_streak]; SUPERTREND [st_regime, st_atr_ratio, st_adx, st_adapt_multiplier, signal_score, side].
- FT (trend): t_ret_1d_z, t_ret_7d_z, t_ret_30d_z, t_ema50_dist_atr, t_up_90d, t_eff_20, t_rsi14.
- FV (volatility): v_rvol_ratio_1d_30d, v_bb_width, v_range_rel.
- Directional features multiplied by `side`; t_rsi14 centered at 0.5 then multiplied by `side`.

## Model M1 (the only frozen model)
Per strategy, logistic regression L2, C = 0.01, median imputation + QuantileTransformer(200, normal).
Refit at every window on all SEEN-asset trades of that strategy exited before the window start. Windows: 6 months from
2019-03-01 up to 2026-09-01, plus the final window 2026-09-01 .. 2026-11-19. The same fitted model predicts the SEEN
trades and the NEW-asset trades of the window (transfer to unseen assets; NEW trades are never used for training).
Reference prior for log loss: training win rate of the strategy.

## Sizing rules (all exposure-normalised to mean 1 within the evaluated scope)
- U: uniform (equivalent to a constant probability).
- O ("prior-asset"): rank rule applied to p_own = (wins + 20 * prior) / (n + 20), the shrunk win rate of the same strategy
  on the same asset over its own trades exited before the window start.
- D1: inverse volatility, clip(median_train(v_rvol_30d of that strategy) / v_rvol_30d, 0.25, 4).
- R: m = 0.5 + percentile rank of p inside (strategy, window, evaluated universe).
- P: m = clip(p / prior, 0.25, 4).
- R x D1.
Kelly and binary filters are NOT part of the frozen system (earlier tests showed them dominated by exposure concentration).

## Hypotheses and decision rules (fixed in advance)
- H1 (ordering): pooled rank IC between p and trade_return, inside (strategy, window), on **NEW assets**, month-block
  bootstrap 95 % CI excludes 0. Expected about +0.05.
- H2 (sizing vs uniform): on NEW assets, R has a higher Sharpe than U (monthly PnL by exit month); permutation test p < 0.05
  (shuffling sizes inside strategy x window). Expected +0.05 .. +0.13.
- H3 (vs prior): paired Sharpe difference R(M1) - R(O) on NEW assets has a bootstrap CI above 0. Expected: CI includes 0.
- H4 (incremental over volatility sizing): paired Sharpe difference R x D1 - D1 on NEW assets has a CI above 0.
  Expected: CI includes 0.
- Period C is reported but is too short for decisions; it is re-runnable later by moving DATA_END.

## Exploratory section (not frozen, labelled as such)
Feature-set and model-class ablations and the relationship between log loss and economic value are run on all data for
description only.

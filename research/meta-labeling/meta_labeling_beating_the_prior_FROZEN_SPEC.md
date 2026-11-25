# Frozen system specification (written before any look at period A or C outcomes)

Frozen: 2026-11-19, after development on period B only (trades decided >= 2021-09-01 and exited < 2026-09-01,
walk-forward test windows 2023-09-01 .. 2026-09-01, training restricted to B).

## Data
- Binance spot 4h klines, 20 assets (identical to the cryptods time bars on the overlap), first 30 days after listing dropped,
  last bar end 2026-11-19 00:00 UTC. bar_seq from the 4h grid (gaps split segments).
- 4 strategies (RSI_CROSS_ABOVE, DONCHIAN_BREAKOUT_CONTINUATION, TIS_BTC_MOMENTUM_CONFLUENCE, SUPERTREND_REGIME_CONFLUENCE),
  library defaults, fee 5 bps/fill, slippage 0. Label y = 1[trade_return > 0].

## Probability layer (meta-model), pooled across the 4 strategies
- S1: logistic regression on [strategy one-hot (4), v_rvol_30d]   (v_rvol_30d = sqrt(sum of squared 4h log returns, 180 bars))
- S2: S1 + [t_rsi14 * is_TIS, t_ema50_dist_atr * is_TIS]
- Preprocessing: median imputation, QuantileTransformer(200, normal) fitted on training; L2 logistic.
- C chosen per training set by 3 expanding, purged inner time folds from {0.003, 0.01, 0.03, 0.1, 0.3};
  if no C beats the training prior on the inner folds -> predict the prior (abstain).
- Prior (reference) = training win rate of the trade's strategy (constant within a test window).

## Decision layer
- D0 base: every trade, fixed notional.
- D1 inverse-vol sizing: notional x clip(median_train(v_rvol_30d of that strategy) / v_rvol_30d, 0.25, 4). No model.
- D2 filter: take the trade iff p >= prior.
- D3 inverse-vol sizing x filter.

## Evaluation (each run once)
- Probability: per-trade log loss minus prior log loss, month-block bootstrap 95 % CI, per strategy and all.
- Economics: Sharpe of monthly PnL (exit month), difference vs D0 with month-block bootstrap; filter lift vs random
  selection with the same acceptance count per strategy.
- Period A (pristine): test windows of 6 months starting 2019-03-01, 2019-09-01, 2020-03-01, 2020-09-01, 2021-03-01
  (trades decided in the window), training = all trades exited before the window start.
- Period C (pristine, fresh data): trades decided 2026-09-01 .. 2026-11-19 and closed by then, training = all trades
  exited before 2026-09-01.
- Period B (development, biased by selection) reported for reference.

## Pre-registered expectations
- If the volatility effect is real: D3 and D2 improve Sharpe vs D0 and beat random filters in A; log loss gains small
  (around -0.002) and possibly not significant. TIS is the strategy most likely to show a log-loss gain.
- Main risk: the effect is a market-volatility regime effect (V7 ~ V1 on B), with few independent episodes;
  2021 (high volatility, strong trends) is the adverse case.

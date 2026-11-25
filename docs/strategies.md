# `quant_forge.strategies` — primary strategies

Four primary (rule-based) strategies with an inspectable specification, signal generation, execution accounting and
results that keep decision-time features apart from outcomes.

```python
import quant_forge.strategies as qs

list(qs.Strategies)    # RSI_CROSS_ABOVE, DONCHIAN_BREAKOUT_CONTINUATION (alias STRATEGY_432_BTC),
                       # TIS_BTC_MOMENTUM_CONFLUENCE, SUPERTREND_REGIME_CONFLUENCE
print(qs.Strategies.DONCHIAN_BREAKOUT_CONTINUATION.spec.describe())   # inputs, indicators, rules, warm-up, ...

S = qs.Strategies      # members are str subclasses: plain strings such as "rsi_cross_above" still work
result = qs.run_strategy(bars, S.STRATEGY_432_BTC, fee_bps=5, context_columns=["aggr_imbalance_notional"])
trades = result.closed_trades_frame()       # closed trades + decision-time features + returns + labels
result.model_input_columns                  # known at decision time -> eligible ML inputs
result.post_decision_columns                # execution/outcome columns, never model inputs

candidates = S.RSI_CROSS_ABOVE.run(bars, evaluate=False)   # events + features, no outcomes
```

## Input

`bars` is a DataFrame of OHLC bars with microsecond timestamps (`bar_start_us`, `bar_end_us`; optional
`first_trade_ts_us`, `bar_available_us`, `decision_time_us`, `complete`, `bar_seq`, `segment_id`; SuperTrend also needs a
volume column). It is never modified. `quant_forge.data.klines_to_bars` produces exactly this format.

* A bar is *valid* when OHLC is finite and consistent and `complete` (if present) is truthy.
* Two bars are *adjacent* when both are valid, `bar_seq` increases by one and `segment_id` is unchanged; a maximal run of
  adjacent bars is a *segment*. Indicators restart in every segment; positions and pending orders do not survive a break.
* The decision time of a signal is `max(bar_end_us, bar_available_us, decision_time_us)` of the signal bar.

## Layout and pipeline

`quant_forge/strategies/implementations/` holds the strategies (one module each: config, spec, features, signals,
position handling); the framework (pipeline, results, indicators, execution accounting, enum, registry) is in
`quant_forge/strategies/`.

Pipeline of every strategy (`strategies/base.py`): `prepare_bars -> compute_features -> generate_signals -> simulate ->
assemble`. A `StrategyResult` holds:

| attribute | content |
|---|---|
| `events` | candidate events (rule signals) with stable `event_id`, source-bar references and a `status` (executed, skipped_position_open, skipped_cooldown, order_discarded_data_break, order_unfilled_end_of_data, ...) |
| `features` | decision-time features of the events (+ requested `context_columns`, copied from the signal bar) |
| `trades` | executed entries with decision / entry / exit times kept apart; `closed`, `censored_data_break` or `open_at_end` — only closed trades carry a label |
| `diagnostics`, `metadata` | counters (data breaks, censoring, discarded orders, equity-path completeness) and the configuration/assumptions of the run |

Configuration: every strategy has a frozen config dataclass (`qs.Strategies.X.config_class`); keyword overrides are
validated (`run_strategy(bars, S.RSI_CROSS_ABOVE, rsi_length=10, fee_bps=5)`). Costs: `fee_bps` per fill and
`slippage_ticks × tick_size` adverse slippage (`CostModel`). Execution: `next_open` (default) or `signal_close`.

## Using strategy output as a meta-labeling dataset

`closed_trades_frame()` tags every column with a role (`attrs["column_roles"]`): `feature` (decision time, model input),
`reference`, `execution`, `outcome`. Bar-level features computed elsewhere (e.g. `quant_forge.features`) are attached with
`context_columns` – they are copied from the signal bar, so nothing after the decision enters. For many assets and
strategies at once use `quant_forge.metalabeling.build_trade_panel`.

## Behaviour relative to the reference notebook

The strategies reproduce `../research/meta-labeling` (`strategy_events`): feature arrays,
signals, fills, exit reasons, returns and labels are compared with a frozen copy of it in `tests/strategies`.
Intentional differences:

* **HTF confirmation correction (TIS strategy).** The reference binary-searched confirmation times that are not
  monotone when bars become available out of order, which made its result depend on the array length and the numpy
  version. The library takes a running maximum first. It changes nothing when confirmations are monotone (e.g.
  `bar_available_us == bar_end_us`, as in the BTCUSDT dataset). Validated in `tests/strategies/test_htf_correction.py`.
* Naming only: `signal_time -> decision_time`, `entry_side -> side`, Donchian `ema200 -> ema_trend` and
  `dc_adx_min/dc_equity_pct/ema200_length -> adx_min/equity_pct/ema_length`, SuperTrend `st_risk_pct -> risk_pct`.
* `net return` is computed as `pnl / entry notional` for every strategy (the reference used `gross - fees/notional` for
  RSI/TIS): equal up to 1e-16.
* Results contain what the strategy produced, not the input columns; bar columns used as model inputs are requested with
  `context_columns`. Non-closed trades and non-executed events are reported instead of being dropped.
* Preserved quirks (documented in each spec): SuperTrend uses a zero-filled-warm-up ADX that differs from the Donchian
  ADX, EMAs are not masked during warm-up, HTF candles of volume bars are approximate.

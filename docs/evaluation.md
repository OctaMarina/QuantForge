# `quant_forge.evaluation` — probability quality and economics

## Probability quality — `metrics.py`

A prediction table has one row per event with the probability column(s), the label `y`, the training prior `prior`,
`trade_return`, a `month` label and grouping columns (`strategy`, `window`).

| Function | Output |
|---|---|
| `log_loss_vector(p, y, eps=1e-6)` | per-event log loss |
| `delta_log_loss_ci(df, col, prior_col="prior", month_col="month", n_boot=2000, seed=0)` | mean `logloss(model) − logloss(prior)` and a **paired monthly block-bootstrap** 95 % CI (negative = beats the prior) |
| `auc_safe(y, p)` | ROC AUC, NaN with one class |
| `information_coefficient(df, col="p", target="trade_return", group_cols=("strategy", "window"))` | pooled rank correlation with ranks taken within groups (level differences between strategies/windows are not skill) |
| `reliability_table(df, col, bins=10)` | `(table[n, mean_p, win_rate], ECE)` over equal-count bins |
| `probability_quality_table(df, {label: column})` | n, Δ log loss [CI], AUC, Brier, IC(return), IC(label) per probability column |

AUC and IC measure ranking; Δ log loss against the win-rate prior measures whether the probability is better than
knowing the strategy's base rate. They often disagree: a model can rank well (AUC 0.6) and still lose to the prior in log
loss.

## Economics — `economics.py`

```python
from quant_forge.evaluation import evaluate_sizing, paired_sharpe_table, verdict

ev = evaluate_sizing(trades, sizes, perm_rules=("P", "R"), notional=10_000, n_boot=2000, n_perm=500, seed=0)
ev.table          # one row per rule (columns below)
ev.monthly["P"]   # monthly PnL;  ev.boot["P"]: bootstrap Sharpe samples (paired across rules);  ev.budget: K
tab, M, B = ev    # tuple-style unpacking
paired_sharpe_table(ev, [("P", "U"), ("DxP", "D")])   # Δ Sharpe, CI, P(Δ > 0)
```

Conventions:

* PnL of a trade = `notional × size × trade_return`, booked in its **exit month**; every calendar month between the first
  and last exit counts. Sharpe = monthly mean / sd × √12.
* **Paired monthly block bootstrap**: the same resampled months for every rule, so Sharpe differences have honest CIs.
* **Permutation test** (`perm_rules`): sizes are shuffled within `perm_group_cols` (default strategy × window), keeping the
  size distribution and destroying the information; p = share of shuffles whose Sharpe gain over `U` ≥ observed.
* Budget `K` = time-weighted 95th percentile of the gross open exposure of `U` (≥ one notional).

`ev.table` columns: `Sharpe`, `Δ vs U` with `CI U lo/hi`, `Δ vs D` with `CI D lo/hi` (when `D` is evaluated), `p perm`,
`PnL (kUSD)`, `return on K (ann.)`, `bps / notional`, `max DD / K`, `exposure / K`, `traded`, `mean open pos.`,
`max open pos.`.

Building blocks: `monthly_pnl`, `annualized_sharpe` (vectorised over rows), `exposure_path` (time-weighted gross exposure
and concurrency), `max_drawdown`, `verdict(row)` ("yes" / "no (worse)" / "not distinguishable from 0" from a CI).

Random draws: bootstrap months first, then permutations in rule order – the same `seed` reproduces every number, and
evaluations of the same scope share their bootstrap samples (reuse `ev.boot` instead of recomputing).

## Portfolios on a common calendar — `portfolio.py`

`economics.py` books a trade's PnL in its exit month, which is fine for one trade list. To compare trade lists whose holding
periods differ by orders of magnitude (the same strategy on 30-minute and on daily bars) every trade is marked to market at
each UTC day end instead, on one calendar shared by all series.

```python
from quant_forge import data as qd, evaluation as qe, sizing as qz

cal = pd.date_range("2019-03-01", "2026-11-18", freq="D", tz="UTC")
closes = qe.daily_close_panel(qd.load_bars(assets, qd.BarConfig("time", 1), end=end, warmup_days=0), cal)
td = qe.expand_trade_days(trades, closes, fee_bps=5)            # once per trade list
w = qz.capital_constrained_sizes(trades["entry_time"], trades["exit_time"], sizes, budget=20 * 10_000, notional=10_000)
pnl = qe.daily_pnl(td, w, notional=10_000)                        # (n_days,); 2-D sizes -> one row per rule
tab = qe.portfolio_table({"rule": pnl}, budget=200_000, trades={"rule": trades}, sizes={"rule": w})
boot = qe.bootstrap_sharpe(np.vstack([pnl_a, pnl_b]), qe.month_labels(cal), n_boot=2000, seed=0)
qe.paired_differences({"a": pnl_a, "b": pnl_b}, {"a": boot[0], "b": boot[1]}, [("a", "b")])
```

| Function | Purpose |
|---|---|
| `daily_close_panel(daily_bars, calendar=None)` | close at each UTC day end per asset from 1-day bars (forward-filled) |
| `expand_trade_days(trades, closes, fee_bps=...)` | `TradeDays`: one row per (trade, day held) with the PnL per unit of entry notional; entry fee on the entry day, exit fee on the exit day, exit price from `trade_return_gross`; `unit_return()` sums back to the net trade return exactly |
| `daily_pnl(td, sizes, notional=..., group=None)` | daily PnL of one or several sizing vectors, optionally split by an integer group label |
| `portfolio_table(daily, budget=..., trades=..., sizes=...)` | net PnL, return on the budget per year, Sharpe of daily PnL (x sqrt(365)), max drawdown / budget, trades taken, turnover (entry + exit notional per year / budget), mean exposure, open positions |
| `bootstrap_sharpe(daily, months, n_boot, seed)` | Sharpe of many series under the same monthly block-bootstrap draws (the draws depend only on `seed` and the number of months, so any two series on the same calendar are paired) |
| `paired_differences(daily, boot, pairs)` | `a − b` Sharpe differences with percentile intervals and `P(Δ > 0)` |
| `month_labels(calendar)`, `max_drawdown_daily(x)` | helpers |

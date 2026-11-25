# `quant_forge.sizing` — bet sizing for meta-labeled trades

```python
from quant_forge.sizing import (afml_bet_size, fit_sizing_constants, rule_sizes, normalized_sizes,
                                capital_constrained_sizes, SIZING_RULES)

k = fit_sizing_constants(oof_p, y_train, r_train, vol_train)   # constants from TRAINING trades only
raw = rule_sizes(p_test, vol_test, k)                          # {"U": ..., "D": ..., "P": ..., ...}
sizes = {r: raw[r] / k.norm[r] for r in SIZING_RULES}          # equal expected exposure per signal
```

## Primitives

* `afml_bet_size(p, p0=0.5)` – AFML Snippet 10.1 for the "trade" class: `z = (p − p0) / sqrt(p(1 − p))`, size
  `2Φ(z) − 1` if `z > 0`, else 0. With meta-labeling, class 0 means "no trade", so the literal rule (`p0 = ½`) only trades
  when `p > ½` and almost never trades strategies that win less than half of the time but have large winners. Centring at
  the **break-even win rate** `p_be = L / (W + L)` (AFML §15.2–15.3; zero expected value) fixes this.
* `break_even_win_rate(returns, labels=None)` → `(p_be, W, L)`.
* `discretize_size(m, step=0.1)` – AFML Snippet 10.3.
* `ecdf(sorted_reference, x)` – percentile of new probabilities in the training distribution (point-in-time ranks).

## Rule set (`SIZING_RULES`)

| Rule | Size | Question |
|---|---|---|
| `U` | 1 | the primary strategy as is |
| `D` | `clip(median_train(σ) / σ, 0.25, 4)` | inverse volatility, no model |
| `F_be` | `1[p ≥ p_be]` | binary filter at break-even |
| `F_tau` | `1[p ≥ τ]` | filter with the threshold maximising training OOF PnL |
| `R` | `0.5 + ECDF_train(p)` | rank sizing without filtering |
| `P` | `afml_bet_size(p, p_be)` | AFML sizing centred at break-even |
| `P05` | `afml_bet_size(p, 0.5)` | the literal book rule |
| `Pd` | `P` discretised by 0.1 | AFML §10.5 |
| `DxF_tau`, `DxF_be`, `DxR`, `DxP` | products with `D` | what the model adds on top of inverse volatility |

## Constants — `fit_sizing_constants(oof_p, y, returns, vol, vol_clip=(0.25, 4.0))`

Returns a `SizingConstants` with `prior`, `p_be`, `avg_win`, `avg_loss`, `vol_med`, `oof_sorted`, `tau`, `tau_quantile`
and `norm` (training mean of every rule). Use calibrated **out-of-fold** probabilities of the training trades, so the
constants exist at the moment a trade is placed. Dividing a rule by its `norm` gives every rule the same expected exposure
per signal (a filter that skips trades puts more into the trades it takes).

`normalized_sizes(pred, constants, key_cols=("strategy", "window"), cap=10.0)` applies this to a prediction table where
each `(strategy, window)` has its own constants, with a position limit `cap`; `return_uncapped=True` also returns the sizes
before the cap.

## Capital budget — `capital_constrained_sizes(entry_time, exit_time, sizes, budget, notional=10_000)`

Equal *expected* exposure does not bound *realised* exposure: signals cluster in time. The replay processes trades by
entry time and cuts each new size to the capital not yet committed (`(budget − open exposure) / notional`); capital is
freed at exit. Point-in-time (uses only earlier entries and exits).

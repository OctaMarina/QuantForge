# QuantForge

Systematic-trading research library (import name `quant_forge`): primary strategies, point-in-time features,
purged validation, meta-labeling, bet sizing and economic evaluation for crypto markets.

```bash
pip install -e ".[dev]"          # library + pytest + scikit-learn, scipy, statsmodels, joblib (all tests)
pip install -e ".[research]"     # + pyarrow, matplotlib, catboost, numba, notebook tooling
pytest                           # tests/ (synthetic data)
```

| Package | Purpose | Guide |
|---|---|---|
| `quant_forge.data` | time bars from Binance klines, volume / dollar bars from the aggTrades archive (`load_bars`, `BarConfig`), sampling diagnostics; research universes | [docs/data.md](docs/data.md) |
| `quant_forge.features` | bar features (V, T, F, CAL, M, R, X) in bar counts or elapsed time (asynchronous panels), fractional differentiation, trade-level features | [docs/features.md](docs/features.md) |
| `quant_forge.strategies` | the four primary strategies (specs, signals, execution, results) | [docs/strategies.md](docs/strategies.md) |
| `quant_forge.validation` | `PurgedKFold`, `CalendarPurgedKFold` (panels), walk-forward windows | [docs/validation.md](docs/validation.md) |
| `quant_forge.feature_selection` | clustering, placebos, clustered MDA / MDI, selection | [docs/feature_selection.md](docs/feature_selection.md) |
| `quant_forge.models` | LR / RF / HGB specs, purged tuning by log loss, Platt calibration | [docs/models.md](docs/models.md) |
| `quant_forge.sizing` | AFML bet sizing, rule set, training constants, capital budget | [docs/sizing.md](docs/sizing.md) |
| `quant_forge.evaluation` | Δ log loss / AUC / IC / calibration; Sharpe, bootstrap, permutation, exposure; daily mark-to-market portfolios | [docs/evaluation.md](docs/evaluation.md) |
| `quant_forge.metalabeling` | trade panels, walk-forward meta-labeling worker (parallel, cached), reports | [docs/metalabeling.md](docs/metalabeling.md) |
| `quant_forge.utils` | `DiskCache`, hashing helpers | [docs/README.md](docs/README.md#quant_forgeutils) |

Start with [docs/README.md](docs/README.md) (conventions and an end-to-end example).

```python
import quant_forge.strategies as qs
from quant_forge import data as qd, features as qf

bars = qd.load_binance_bars(["BTC", "ETH"], end="2026-11-19", interval="4h")
feats = qf.bar_features(bars, bars_per_day=6)
result = qs.run_strategy(bars["BTC"].join(feats["BTC"][["v_rvol_30d"]]), qs.Strategies.RSI_CROSS_ABOVE,
                         fee_bps=5, context_columns=["v_rvol_30d"])
result.closed_trades_frame()      # closed trades + decision-time features + returns + labels
```

Research notebooks using the library: `research/meta-labeling` (strategies
vs the reference notebook), `research/cross-validation/test_purged_kfold_quantforge.ipynb` (purged K-fold audit),
`research/meta-labeling` (full meta-labeling and bet-sizing study),
`research/bar_types/bar_type_frequency_meta_labeling.ipynb` (bar type and sampling frequency vs the primary strategies and
the value of meta-labeling; event bars built by `research/bar_types/scripts/build_event_bars.py`).

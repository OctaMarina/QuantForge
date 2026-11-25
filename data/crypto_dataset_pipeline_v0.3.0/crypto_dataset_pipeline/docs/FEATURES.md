# Registrul feature-urilor (generat)

Generat cu `python crypto_dataset.py registry --config configs/btc_demo.yaml` (configurația demonstrativă BTC; nu conține valori de date). Numele coloanelor care conțin parametri (ferestre, praguri, benzi, mărimi) se schimbă odată cu configurația; registrul JSON scris la fiecare build este sursa autoritativă pentru acel build.

## returns

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `ret_bars_1` | log(P_t / P_(t-1 bars)), P = previous-tick close on the bar clock | log return | volume bars | bar close | observed | da | implemented_verified | insufficient_history, feed_gap |
| `ret_bars_5` | log(P_t / P_(t-5 bars)), P = previous-tick close on the bar clock | log return | volume bars | bar close | observed | da | implemented_verified | insufficient_history, feed_gap |
| `ret_bars_20` | log(P_t / P_(t-20 bars)), P = previous-tick close on the bar clock | log return | volume bars | bar close | observed | da | implemented_verified | insufficient_history, feed_gap |
| `ret_5min` | log(P_t / P(t-5min)); P_t = bar close (previous-tick for empty time bars); P(x) = last trade strictly before the latest 1-second grid point <= x | log return | calendar time | bar close; reference age <= 1min | observed | da | implemented_verified | insufficient_history, stale_input, feed_gap |
| `ret_5min_ref_age_s` | (t-h) - timestamp of the reference trade | seconds | calendar time | bar close | observed | da | implemented_verified | insufficient_history |
| `ret_1h` | log(P_t / P(t-1h)); P_t = bar close (previous-tick for empty time bars); P(x) = last trade strictly before the latest 1-second grid point <= x | log return | calendar time | bar close; reference age <= 1min | observed | da | implemented_verified | insufficient_history, stale_input, feed_gap |
| `ret_1h_ref_age_s` | (t-h) - timestamp of the reference trade | seconds | calendar time | bar close | observed | da | implemented_verified | insufficient_history |
| `ret_4h` | log(P_t / P(t-4h)); P_t = bar close (previous-tick for empty time bars); P(x) = last trade strictly before the latest 1-second grid point <= x | log return | calendar time | bar close; reference age <= 1min | observed | da | implemented_verified | insufficient_history, stale_input, feed_gap |
| `ret_4h_ref_age_s` | (t-h) - timestamp of the reference trade | seconds | calendar time | bar close | observed | da | implemented_verified | insufficient_history |
| `ret_1d` | log(P_t / P(t-1d)); P_t = bar close (previous-tick for empty time bars); P(x) = last trade strictly before the latest 1-second grid point <= x | log return | calendar time | bar close; reference age <= 1min | observed | da | implemented_verified | insufficient_history, stale_input, feed_gap |
| `ret_1d_ref_age_s` | (t-h) - timestamp of the reference trade | seconds | calendar time | bar close | observed | da | implemented_verified | insufficient_history |
| `ret_7d` | log(P_t / P(t-7d)); P_t = bar close (previous-tick for empty time bars); P(x) = last trade strictly before the latest 1-second grid point <= x | log return | calendar time | bar close; reference age <= 1min | observed | da | implemented_verified | insufficient_history, stale_input, feed_gap |
| `ret_7d_ref_age_s` | (t-h) - timestamp of the reference trade | seconds | calendar time | bar close | observed | da | implemented_verified | insufficient_history |
| `ret_30d` | log(P_t / P(t-30d)); P_t = bar close (previous-tick for empty time bars); P(x) = last trade strictly before the latest 1-second grid point <= x | log return | calendar time | bar close; reference age <= 1min | observed | da | implemented_verified | insufficient_history, stale_input, feed_gap |
| `ret_30d_ref_age_s` | (t-h) - timestamp of the reference trade | seconds | calendar time | bar close | observed | da | implemented_verified | insufficient_history |

Note:
- Bar-count horizon; on volume/notional bars this is NOT a calendar horizon.

## realized_vol

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `rv_1h` | sum_(g in (G-1h, G]) r_g^2 | log-return^2 | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_1h` | sqrt(rv_1h) | log-return | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_ann_1h` | sqrt(rv_1h * T_year / T_W), T_year=365d | annualized log-return (fraction) | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rv_1d` | sum_(g in (G-1d, G]) r_g^2 | log-return^2 | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_1d` | sqrt(rv_1d) | log-return | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_ann_1d` | sqrt(rv_1d * T_year / T_W), T_year=365d | annualized log-return (fraction) | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rv_7d` | sum_(g in (G-7d, G]) r_g^2 | log-return^2 | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_7d` | sqrt(rv_7d) | log-return | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_ann_7d` | sqrt(rv_7d * T_year / T_W), T_year=365d | annualized log-return (fraction) | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rv_30d` | sum_(g in (G-30d, G]) r_g^2 | log-return^2 | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_30d` | sqrt(rv_30d) | log-return | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rvol_ann_30d` | sqrt(rv_30d * T_year / T_W), T_year=365d | annualized log-return (fraction) | calendar grid 1min | bar close (grid point <= bar end); grid price age <= 5min | estimated | da | implemented_verified | insufficient_history, stale_input |
| `rv_bars_20` | sum of squared log close-to-close returns over last 20 bars | log-return^2 | volume bars | bar close | estimated | da | implemented_verified | insufficient_history |
| `rv_bars_100` | sum of squared log close-to-close returns over last 100 bars | log-return^2 | volume bars | bar close | estimated | da | implemented_verified | insufficient_history |

Note:
- Not annualised: a bar count has no fixed calendar length.
- Previous-tick sampling on the grid (P(g) = last trade strictly before g). Estimates quadratic variation of log price incl. microstructure noise at this frequency; cannot observe jumps' timing within a grid step.

## duration

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `bar_duration_s` | time bars: interval length; event bars: close(prev bar or segment origin) -> close(this bar) | seconds | calendar time | bar close | observed | da | implemented_verified | feed_gap |
| `trade_span_s` | last trade ts - first trade ts in bar | seconds | calendar time | bar close | observed | da | implemented_verified | not_applicable, feed_gap |

## intensity

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `avg_trade_size` | volume / n_trades (executions) | BTC | bar | bar close | observed | da | implemented_verified | not_applicable |
| `trade_intensity_per_s` | trade_intensity / bar_duration_s | executions/s | bar | bar close | observed | da | implemented_verified | feed_gap, zero_denominator |
| `volume_per_s` | volume / bar_duration_s | BTC/s | bar | bar close | observed | da | implemented_verified | feed_gap, zero_denominator |
| `notional_per_s` | notional / bar_duration_s | USDT/s | bar | bar close | observed | da | implemented_verified | feed_gap, zero_denominator |

Note:
- Denominator = bar_duration_s; null (zero_denominator) for an instantaneous event bar - no epsilon is added.

## flow_aggressor

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `aggr_buy_volume` | sum of qty with buyer as aggressor | BTC | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |
| `aggr_sell_volume` | sum of qty with seller as aggressor | BTC | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |
| `aggr_signed_volume` | buy - sell (observed aggressor) | BTC | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |
| `aggr_classified_frac_volume` | (buy + sell) / total | fraction | bar | bar close | observed | da | implemented_verified | not_applicable |
| `aggr_imbalance_volume` | (V_buy - V_sell) / (V_buy + V_sell) | ratio in [-1,1] | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage, zero_denominator |
| `aggr_buy_notional` | sum of qty with buyer as aggressor | USDT | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |
| `aggr_sell_notional` | sum of qty with seller as aggressor | USDT | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |
| `aggr_signed_notional` | buy - sell (observed aggressor) | USDT | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |
| `aggr_classified_frac_notional` | (buy + sell) / total | fraction | bar | bar close | observed | da | implemented_verified | not_applicable |
| `aggr_imbalance_notional` | (V_buy - V_sell) / (V_buy + V_sell) | ratio in [-1,1] | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage, zero_denominator |

Note:
- Binance isBuyerMaker mapping: true -> -1 (seller aggressor), false -> +1.

## flow_tick_rule

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `tick_imbalance_volume` | (V_tick+ - V_tick-) / (V_tick+ + V_tick-), tick rule b_i | ratio in [-1,1] | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage, zero_denominator |
| `tick_signed_volume` | V_tick+ - V_tick- | BTC | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |
| `tick_imbalance_notional` | (V_tick+ - V_tick-) / (V_tick+ + V_tick-), tick rule b_i | ratio in [-1,1] | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage, zero_denominator |
| `tick_signed_notional` | V_tick+ - V_tick- | USDT | bar | bar close | observed | da | implemented_verified | feed_gap, incomplete_coverage |

Referință: AFML (2018) ch. 19 'Microstructural Features' (Kyle / Hasbrouck lambdas, tick rule). Primary text not accessible in this build (O'Reilly preview only, checked 2026-10-20); regression form = task specification section 5.6-5.8.

Note:
- Trades before the first price change of a segment have unknown tick sign -> bar imbalance null (incomplete_coverage).

## lambda

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `kyle_lambda_trades_tick` | OLS dp_i (quote) = lambda * b_i q_i (base) + e (no intercept), trades of the bar; sign=tick rule — *literal tick-rule* | USDT per BTC | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `kyle_tstat_trades_tick` | OLS dp_i (quote) = lambda * b_i q_i (base) + e (no intercept), trades of the bar; sign=tick rule; t = lambda / sqrt(s^2/Sxx), s^2 = RSS/(n-1) — *literal tick-rule* | t-statistic | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `kyle_n_trades_tick` | observations used | count | bar | bar close | observed | da | implemented_verified | feed_gap |
| `kyle_lambda_trades_aggr` | OLS dp_i (quote) = lambda * b_i q_i (base) + e (no intercept), trades of the bar; sign=observed aggressor — *observed-aggressor variant* | USDT per BTC | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `kyle_tstat_trades_aggr` | OLS dp_i (quote) = lambda * b_i q_i (base) + e (no intercept), trades of the bar; sign=observed aggressor; t = lambda / sqrt(s^2/Sxx), s^2 = RSS/(n-1) — *observed-aggressor variant* | t-statistic | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `kyle_n_trades_aggr` | observations used | count | bar | bar close | observed | da | implemented_verified | feed_gap |
| `hasbrouck_lambda_trades_tick` | OLS log(p_i/p_(i-1)) = lambda * b_i sqrt(p_i q_i) + e (no intercept), trades of the bar; sign=tick rule — *literal tick-rule* | log-return per sqrt(USDT) | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `hasbrouck_tstat_trades_tick` | OLS log(p_i/p_(i-1)) = lambda * b_i sqrt(p_i q_i) + e (no intercept), trades of the bar; sign=tick rule; t = lambda / sqrt(s^2/Sxx), s^2 = RSS/(n-1) — *literal tick-rule* | t-statistic | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `hasbrouck_n_trades_tick` | observations used | count | bar | bar close | observed | da | implemented_verified | feed_gap |
| `hasbrouck_lambda_trades_aggr` | OLS log(p_i/p_(i-1)) = lambda * b_i sqrt(p_i q_i) + e (no intercept), trades of the bar; sign=observed aggressor — *observed-aggressor variant* | log-return per sqrt(USDT) | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `hasbrouck_tstat_trades_aggr` | OLS log(p_i/p_(i-1)) = lambda * b_i sqrt(p_i q_i) + e (no intercept), trades of the bar; sign=observed aggressor; t = lambda / sqrt(s^2/Sxx), s^2 = RSS/(n-1) — *observed-aggressor variant* | t-statistic | trades within the bar | bar close | estimated | da | implemented_verified | insufficient_history, invalid_regression, feed_gap |
| `hasbrouck_n_trades_aggr` | observations used | count | bar | bar close | observed | da | implemented_verified | feed_gap |
| `kyle_lambda_bars50_aggr` | rolling OLS over last 50 bars: (P_B - P_(B-1)) = lambda * sum_(i in B) b_i q_i, no intercept; regressor from trade-level signed volume (not sign(close-open)*volume) — *observed-aggressor* | USDT per BTC | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |
| `kyle_tstat_bars50_aggr` | rolling OLS over last 50 bars: (P_B - P_(B-1)) = lambda * sum_(i in B) b_i q_i, no intercept; regressor from trade-level signed volume (not sign(close-open)*volume); t = coef/sqrt(var) — *observed-aggressor* | t-statistic | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |
| `hasbrouck_lambda_bars50_aggr` | rolling OLS over last 50 bars: log(P_B/P_(B-1)) = lambda * sum_(i in B) b_i sqrt(p_i q_i), no intercept (sum of per-trade root notionals, not root of summed notional) — *observed-aggressor* | log-return per sqrt(USDT) | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |
| `hasbrouck_tstat_bars50_aggr` | rolling OLS over last 50 bars: log(P_B/P_(B-1)) = lambda * sum_(i in B) b_i sqrt(p_i q_i), no intercept (sum of per-trade root notionals, not root of summed notional); t = coef/sqrt(var) — *observed-aggressor* | t-statistic | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |
| `kyle_lambda_bars50_tick` | rolling OLS over last 50 bars: (P_B - P_(B-1)) = lambda * sum_(i in B) b_i q_i, no intercept; regressor from trade-level signed volume (not sign(close-open)*volume) — *tick rule* | USDT per BTC | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |
| `kyle_tstat_bars50_tick` | rolling OLS over last 50 bars: (P_B - P_(B-1)) = lambda * sum_(i in B) b_i q_i, no intercept; regressor from trade-level signed volume (not sign(close-open)*volume); t = coef/sqrt(var) — *tick rule* | t-statistic | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |
| `hasbrouck_lambda_bars50_tick` | rolling OLS over last 50 bars: log(P_B/P_(B-1)) = lambda * sum_(i in B) b_i sqrt(p_i q_i), no intercept (sum of per-trade root notionals, not root of summed notional) — *tick rule* | log-return per sqrt(USDT) | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |
| `hasbrouck_tstat_bars50_tick` | rolling OLS over last 50 bars: log(P_B/P_(B-1)) = lambda * sum_(i in B) b_i sqrt(p_i q_i), no intercept (sum of per-trade root notionals, not root of summed notional); t = coef/sqrt(var) — *tick rule* | t-statistic | volume bars | bar close | estimated | da | implemented_verified | insufficient_history, feed_gap, invalid_regression |

Referință: AFML (2018) ch. 19 'Microstructural Features' (Kyle / Hasbrouck lambdas, tick rule). Primary text not accessible in this build (O'Reilly preview only, checked 2026-10-20); regression form = task specification section 5.6-5.8.

Note:
- Construction choice for the bar-aggregated form (task 5.6/5.7); the AFML bar-level definition could not be checked against the primary text.
- First trade after a segment start has no previous price and is excluded; dp of the first trade of a bar uses the previous trade of the earlier bar.

## structural_sadf

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `sadf_w200_tau30_p1_c` | max over all start points t0 (>= 30 effective obs, window 200) of ADF_(t0,t) = b/SE(b); dy_i = a + b y_(i-1) + sum_(j<=1) phi_j dy_(i-j) + e — *windowed backward SADF at endpoint t (PSY: BSADF_t; AFML: SADF_t with start points limited to W)* | t-statistic (right tail = explosive) | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |
| `qadf_q95_w200_tau30_p1_c` | type-7 empirical 0.95-quantile of {ADF_(t0,t)} (same set as SADF) | t-statistic | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |
| `cadf_q95_w200_tau30_p1_c` | mean of ADF_(t0,t) >= QADF_t (inclusive tail; ties at the quantile included) | t-statistic | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |
| `sadf_n_starts_w200_tau30_p1_c` | number of ADF start points evaluated | count | volume bars | bar close (evaluated at endpoint t only) | observed | da | implemented_verified | insufficient_history |

Referință: AFML (2018) ch. 17 'Structural Breaks'.  Primary text NOT accessible here (O'Reilly preview ends before 17.3, checked 2026-10-20).  Form = task specification; cross-checked against MizarLabs sadf/sdfc source (implementation, not authority) and statsmodels recursive_olsresiduals.

Note:
- AFML tail convention (>= vs >) not verified against the primary text.

## structural_sdfc

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `sdfc_w200_trim15` | max over candidates tau of delta/SE(delta) in dy_i = delta y_(i-1) 1{i>=tau} + e_i (no intercept, no lags) | t-statistic | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |
| `sdfc_argmax_bars_ago_w200` | t - argmax_tau DFC_tau (bars), known at t | bars | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history |

Referință: AFML (2018) ch. 17 'Structural Breaks'.  Primary text NOT accessible here (O'Reilly preview ends before 17.3, checked 2026-10-20).  Form = task specification; cross-checked against MizarLabs sadf/sdfc source (implementation, not authority) and statsmodels recursive_olsresiduals.

Note:
- Dummy uses i >= tau (MizarLabs and secondary sources); the task's 1{i>tau} is the same family shifted by one index.  Value is assigned to the endpoint t, never to the candidate date.

## structural_csw

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `csw_up_w200` | sup_n S_(n,t), S_(n,t)=(y_t-y_n)/(sigma_t sqrt(t-n)), sigma_t^2=(t-1)^-1 sum_(i=2..t) dy_i^2, window-relative indices | statistic | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |
| `csw_down_w200` | sup_n -S_(n,t), S_(n,t)=(y_t-y_n)/(sigma_t sqrt(t-n)), sigma_t^2=(t-1)^-1 sum_(i=2..t) dy_i^2, window-relative indices | statistic | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |
| `csw_two_w200` | sup_n \|S_(n,t)\|, S_(n,t)=(y_t-y_n)/(sigma_t sqrt(t-n)), sigma_t^2=(t-1)^-1 sum_(i=2..t) dy_i^2, window-relative indices | statistic | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |

Referință: AFML (2018) ch. 17 'Structural Breaks'.  Primary text NOT accessible here (O'Reilly preview ends before 17.3, checked 2026-10-20).  Form = task specification; cross-checked against MizarLabs sadf/sdfc source (implementation, not authority) and statsmodels recursive_olsresiduals.

Note:
- No critical boundary is emitted (the b_alpha constant was not verified for this sampling).

## structural_bde

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `bde_cusum_w200_p1` | S_t = sum_j w_j / sd(w), recursive residuals of r_i = c + sum_(j<=1) phi_j r_(i-j) + e over the last 200 bars — *windowed BDE; regression configured as AR(1) with intercept on bar log returns (test configuration)* | statistic (~N(0, n_w) under H0) | volume bars | bar close (evaluated at endpoint t only) | test_statistic | da | implemented_verified | insufficient_history, feed_gap, invalid_regression, unverified_formula |
| `bde_n_resid_w200_p1` | number of recursive residuals | count | volume bars | bar close (evaluated at endpoint t only) | observed | da | implemented_verified | insufficient_history |

Referință: AFML (2018) ch. 17 'Structural Breaks'.  Primary text NOT accessible here (O'Reilly preview ends before 17.3, checked 2026-10-20).  Form = task specification; cross-checked against MizarLabs sadf/sdfc source (implementation, not authority) and statsmodels recursive_olsresiduals.

Note:
- sd(w) uses ddof=1 over residuals available at t (prefix), never the future sample.  AFML's exact standardisation (denominator, sigma_eps in f_t) not verified against the book.

## premium

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `premium_raw_coinbase_btc_1m` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_fx_coinbase_btc_1m` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_raw_coinbase_btc_1d` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_raw_coinbase_btc_1d_z90` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_raw_coinbase_btc_1d_coverage` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_raw_coinbase_eth_1m` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_fx_coinbase_eth_1m` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_raw_coinbase_eth_1d` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_raw_coinbase_eth_1d_z90` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |
| `premium_raw_coinbase_eth_1d_coverage` | see FORMULAS.md (premium) | log ratio | 1-minute candles | minute end (market-time replay) | observed | da | implemented_untested_real | missing_source_data, stale_input, source_unreachable |

## trend

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `trend_30d_volscaled` | log(C_d/C_(d-30)) / (s_30 sqrt(30)) | ratio | UTC day | day end | observed | da | implemented_untested_real | insufficient_history, zero_denominator, missing_source_data |

## basis

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `basis_last_perp_spot_1m` | log(P_perp/P_spot) | log ratio | 1-minute candles | minute end | observed | da | implemented_untested_real | missing_source_data, stale_input |
| `basis_mark_index_1m` | log(mark/index) | log ratio | 1-minute candles | minute end | observed | da | implemented_untested_real | missing_source_data, stale_input |
| `premium_index_binance_1m` | exchange premium index | fraction | 1-minute candles | minute end | observed | da | implemented_untested_real | missing_source_data, stale_input |

## funding

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `funding_rate_last` | realised funding | fraction | funding events | funding time | observed | da | implemented_untested_real | missing_source_data, stale_input |
| `funding_rate_last_age_s` | realised funding | fraction | funding events | funding time | observed | da | implemented_untested_real | missing_source_data, stale_input |
| `funding_interval_h_observed` | realised funding | fraction | funding events | funding time | observed | da | implemented_untested_real | missing_source_data, stale_input |

## etf_flows

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `etf_btc_net_flow_total_usd` | Farside 'Total' column for the latest US trading session known (not summed with the fund rows) | USD | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_net_flow_total_usd_age_s` | decision - availability | seconds | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_net_flow_total_usd_economic_age_d` | decision date - economic date | days | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_net_flow_total_usd_z60` | causal z vs previous 60 observations | z-score | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_sum_reported_funds_usd` | sum of fund cells reported for the session (differs from Total when cells are blank) | USD | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_sum_reported_funds_usd_age_s` | decision - availability | seconds | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_sum_reported_funds_usd_economic_age_d` | decision date - economic date | days | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_n_funds_reported` | number of fund cells with a numeric value for the session | count | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_n_funds_reported_age_s` | decision - availability | seconds | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_btc_n_funds_reported_economic_age_d` | decision date - economic date | days | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_net_flow_total_usd` | Farside 'Total' column for the latest US trading session known (not summed with the fund rows) | USD | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_net_flow_total_usd_age_s` | decision - availability | seconds | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_net_flow_total_usd_economic_age_d` | decision date - economic date | days | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_net_flow_total_usd_z60` | causal z vs previous 60 observations | z-score | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_sum_reported_funds_usd` | sum of fund cells reported for the session (differs from Total when cells are blank) | USD | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_sum_reported_funds_usd_age_s` | decision - availability | seconds | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_sum_reported_funds_usd_economic_age_d` | decision date - economic date | days | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_n_funds_reported` | number of fund cells with a numeric value for the session | count | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_n_funds_reported_age_s` | decision - availability | seconds | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `etf_eth_n_funds_reported_economic_age_d` | decision date - economic date | days | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 4d | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |

Note:
- Compare with n_funds_reported to judge completeness.
- Weekends/holidays have no row (never zero); pre-launch funds blank -> not zero; 'Total' as published.

## fear_greed

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `fear_greed_value` | Alternative.me Fear & Greed value for the latest economic day known | index 0-100 | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 36h | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `fear_greed_value_age_s` | decision - availability | seconds | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 36h | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `fear_greed_value_economic_age_d` | decision date - economic date | days | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 36h | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |
| `fear_greed_value_change_prev` | latest value - previous economic observation (both known) | index 0-100 | publication events | first verifiable availability (live poll received_at or archive capture time); age <= 36h | observed | da | implemented_untested_real | unknown_publication_time, stale_input, insufficient_history, missing_source_data |

Note:
- Bitcoin-centric market context, not coin-specific sentiment; API timestamp is the economic day, not a publication time; price/volume components overlap with market data.

## implied_vol

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `dvol_btc` | DVOL close of the last closed candle | annualised implied vol, percentage points (per Deribit methodology article; API docs give no unit) | DVOL candles res 60 | candle start + resolution (conservative) | observed | da | implemented_untested_real | missing_source_data, stale_input, not_available_in_period |
| `dvol_btc_change_1h` | DVOL(t) - DVOL(t-1h), both as-of known values | percentage points | DVOL candles res 60 | candle start + resolution (conservative) | observed | da | implemented_untested_real | missing_source_data, stale_input, not_available_in_period |
| `dvol_btc_change_1d` | DVOL(t) - DVOL(t-1d), both as-of known values | percentage points | DVOL candles res 60 | candle start + resolution (conservative) | observed | da | implemented_untested_real | missing_source_data, stale_input, not_available_in_period |
| `dvol_btc_minus_trailing_rvol30d_pp` | DVOL (forward 30d IV, pp) - 100 * trailing 30d realised vol (annualised, 1-min grid) | percentage points | decision | decision | constructed | da | implemented_untested_real | missing_source_data, stale_input, not_applicable |
| `dvol_eth` | DVOL close of the last closed candle | annualised implied vol, percentage points (per Deribit methodology article; API docs give no unit) | DVOL candles res 60 | candle start + resolution (conservative) | observed | da | implemented_untested_real | missing_source_data, stale_input, not_available_in_period |
| `dvol_eth_change_1h` | DVOL(t) - DVOL(t-1h), both as-of known values | percentage points | DVOL candles res 60 | candle start + resolution (conservative) | observed | da | implemented_untested_real | missing_source_data, stale_input, not_available_in_period |
| `dvol_eth_change_1d` | DVOL(t) - DVOL(t-1d), both as-of known values | percentage points | DVOL candles res 60 | candle start + resolution (conservative) | observed | da | implemented_untested_real | missing_source_data, stale_input, not_available_in_period |

Note:
- Descriptive IV - trailing RV gap; horizons differ (forward vs trailing); not a rigorously estimated variance risk premium.
- Market context for other assets; it is not the implied volatility of BTC.

## options_skew

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `skew25d_30d_btc` | see options.py docstring | IV points (venue unit; Deribit: percentage points) | 1-minute snapshots | snapshot time (quotes strictly earlier); quotes <= 60s, snapshot <= 5min | estimated | da | unavailable_from_current_source | not_available_in_period, insufficient_book_depth, no_observation, stale_input |
| `iv_put25d_30d_btc` | see options.py docstring | IV points (venue unit; Deribit: percentage points) | 1-minute snapshots | snapshot time (quotes strictly earlier); quotes <= 60s, snapshot <= 5min | estimated | da | unavailable_from_current_source | not_available_in_period, insufficient_book_depth, no_observation, stale_input |
| `iv_call25d_30d_btc` | see options.py docstring | IV points (venue unit; Deribit: percentage points) | 1-minute snapshots | snapshot time (quotes strictly earlier); quotes <= 60s, snapshot <= 5min | estimated | da | unavailable_from_current_source | not_available_in_period, insufficient_book_depth, no_observation, stale_input |

## liquidations

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `binance_reported_sampled_liq_long_qty_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_short_qty_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_count_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_imbalance_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_intensity_per_min_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_notional_exec_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_surprise_pctile_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_long_qty_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_short_qty_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_count_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_imbalance_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_intensity_per_min_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_notional_exec_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `binance_reported_sampled_liq_surprise_pctile_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_long_qty_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_short_qty_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_count_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_imbalance_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_intensity_per_min_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_notional_at_bankruptcy_px_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_surprise_pctile_5min` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_long_qty_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_short_qty_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_count_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_imbalance_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_intensity_per_min_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_notional_at_bankruptcy_px_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |
| `bybit_liq_surprise_pctile_1h` | see liquidations.py | varies | calendar windows | event time (market-time) or receipt time (observed mode) | observed | da | unavailable_from_current_source | feed_gap, not_available_in_period, insufficient_history |

Note:
- Binance stream is sampled (latest liquidation per symbol per 1000 ms).
- Bybit price is the bankruptcy price.

## l2_book

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `spread` | a - b (best ask - best bid) at decision | USDT | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | stale_input, feed_gap, not_available_in_period |
| `spread_bps` | 1e4 (a - b) / m, m = (a+b)/2 | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | stale_input, feed_gap, not_available_in_period |
| `mid` | (a+b)/2 at decision | USDT | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | stale_input, not_available_in_period |
| `book_age_s` | decision - exchange time of last applied book message | seconds | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | not_available_in_period |
| `twa_spread_bps` | (1/\|B\|) int_B spread_bps(t) dt; requires valid book over the whole bar | bps | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `book_valid_time_frac` | valid synced book time / bar duration | fraction | bar interval | bar close | observed | da | unavailable_from_current_source | not_available_in_period |
| `depth_bid_10bps` | sum p_i q_i over bid levels with p >= m(1-10/1e4) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `depth_ask_10bps` | sum p_i q_i over ask levels with p <= m(1+10/1e4) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `depth_imbalance_10bps` | (D_bid - D_ask) / (D_bid + D_ask) | ratio | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, zero_denominator, not_available_in_period |
| `twa_depth_bid_10bps` | (1/\|B\|) int_B D_bid,10(t) dt; only when the band is covered for the whole bar | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `twa_depth_ask_10bps` | (1/\|B\|) int_B D_ask,10(t) dt; only when the band is covered for the whole bar | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_bid_25bps` | sum p_i q_i over bid levels with p >= m(1-25/1e4) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `depth_ask_25bps` | sum p_i q_i over ask levels with p <= m(1+25/1e4) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `depth_imbalance_25bps` | (D_bid - D_ask) / (D_bid + D_ask) | ratio | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, zero_denominator, not_available_in_period |
| `twa_depth_bid_25bps` | (1/\|B\|) int_B D_bid,25(t) dt; only when the band is covered for the whole bar | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `twa_depth_ask_25bps` | (1/\|B\|) int_B D_ask,25(t) dt; only when the band is covered for the whole bar | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_bid_50bps` | sum p_i q_i over bid levels with p >= m(1-50/1e4) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `depth_ask_50bps` | sum p_i q_i over ask levels with p <= m(1+50/1e4) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `depth_imbalance_50bps` | (D_bid - D_ask) / (D_bid + D_ask) | ratio | decision time | decision time (book messages with ts < decision); book age <= 2s | observed | da | unavailable_from_current_source | insufficient_book_depth, zero_denominator, not_available_in_period |
| `twa_depth_bid_50bps` | (1/\|B\|) int_B D_bid,50(t) dt; only when the band is covered for the whole bar | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `twa_depth_ask_50bps` | (1/\|B\|) int_B D_ask,50(t) dt; only when the band is covered for the whole bar | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |

Note:
- Null when the band extends beyond the known-complete part of the book.

## l2_withdrawal_refill

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `liq_added_bid_10bps` | sum of same-price size increases inside the band during the bar (net observed, incl. new orders) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_withdrawn_bid_10bps` | sum of same-price size decreases inside the band during the bar (cancels AND executions) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_change_band_shift_bid_10bps` | D(end) - D(start) - (added - withdrawn): mechanical change from the moving mid | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_added_ask_10bps` | sum of same-price size increases inside the band during the bar (net observed, incl. new orders) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_withdrawn_ask_10bps` | sum of same-price size decreases inside the band during the bar (cancels AND executions) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_change_band_shift_ask_10bps` | D(end) - D(start) - (added - withdrawn): mechanical change from the moving mid | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_added_bid_25bps` | sum of same-price size increases inside the band during the bar (net observed, incl. new orders) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_withdrawn_bid_25bps` | sum of same-price size decreases inside the band during the bar (cancels AND executions) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_change_band_shift_bid_25bps` | D(end) - D(start) - (added - withdrawn): mechanical change from the moving mid | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_added_ask_25bps` | sum of same-price size increases inside the band during the bar (net observed, incl. new orders) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_withdrawn_ask_25bps` | sum of same-price size decreases inside the band during the bar (cancels AND executions) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_change_band_shift_ask_25bps` | D(end) - D(start) - (added - withdrawn): mechanical change from the moving mid | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_added_bid_50bps` | sum of same-price size increases inside the band during the bar (net observed, incl. new orders) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_withdrawn_bid_50bps` | sum of same-price size decreases inside the band during the bar (cancels AND executions) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_change_band_shift_bid_50bps` | D(end) - D(start) - (added - withdrawn): mechanical change from the moving mid | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_added_ask_50bps` | sum of same-price size increases inside the band during the bar (net observed, incl. new orders) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `liq_withdrawn_ask_50bps` | sum of same-price size decreases inside the band during the bar (cancels AND executions) | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `depth_change_band_shift_ask_50bps` | D(end) - D(start) - (added - withdrawn): mechanical change from the moving mid | USDT notional | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |

Note:
- Aggregated L2 cannot separate simultaneous cancels and new orders: this is net observed refill.
- Not a cancellation rate: executions also reduce displayed size.

## execution_cost

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `cost_buy_0.1BTC_bps` | 1e4 (VWAP_ask(q)/m - 1); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `cost_sell_0.1BTC_bps` | 1e4 (1 - VWAP_bid(q)/m); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `hypo_roundtrip_cost_0.1BTC_current_book_bps` | cost_buy + cost_sell on the SAME current book (hypothetical; the future exit cost is unknown) | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, not_available_in_period |
| `cost_buy_1BTC_bps` | 1e4 (VWAP_ask(q)/m - 1); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `cost_sell_1BTC_bps` | 1e4 (1 - VWAP_bid(q)/m); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `hypo_roundtrip_cost_1BTC_current_book_bps` | cost_buy + cost_sell on the SAME current book (hypothetical; the future exit cost is unknown) | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, not_available_in_period |
| `cost_buy_10BTC_bps` | 1e4 (VWAP_ask(q)/m - 1); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `cost_sell_10BTC_bps` | 1e4 (1 - VWAP_bid(q)/m); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `hypo_roundtrip_cost_10BTC_current_book_bps` | cost_buy + cost_sell on the SAME current book (hypothetical; the future exit cost is unknown) | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, not_available_in_period |
| `cost_buy_10000USDT_bps` | 1e4 (VWAP_ask(q)/m - 1); q = 10000 USDT / m (reference notional; actual spend = VWAP*q); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `cost_sell_10000USDT_bps` | 1e4 (1 - VWAP_bid(q)/m); q = 10000 USDT / m (reference notional; actual spend = VWAP*q); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `hypo_roundtrip_cost_10000USDT_current_book_bps` | cost_buy + cost_sell on the SAME current book (hypothetical; the future exit cost is unknown) | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, not_available_in_period |
| `cost_buy_100000USDT_bps` | 1e4 (VWAP_ask(q)/m - 1); q = 100000 USDT / m (reference notional; actual spend = VWAP*q); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `cost_sell_100000USDT_bps` | 1e4 (1 - VWAP_bid(q)/m); q = 100000 USDT / m (reference notional; actual spend = VWAP*q); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `hypo_roundtrip_cost_100000USDT_current_book_bps` | cost_buy + cost_sell on the SAME current book (hypothetical; the future exit cost is unknown) | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, not_available_in_period |
| `cost_buy_1000000USDT_bps` | 1e4 (VWAP_ask(q)/m - 1); q = 1000000 USDT / m (reference notional; actual spend = VWAP*q); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `cost_sell_1000000USDT_bps` | 1e4 (1 - VWAP_bid(q)/m); q = 1000000 USDT / m (reference notional; actual spend = VWAP*q); includes spread crossing | bps vs mid | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, stale_input, not_available_in_period |
| `hypo_roundtrip_cost_1000000USDT_current_book_bps` | cost_buy + cost_sell on the SAME current book (hypothetical; the future exit cost is unknown) | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | estimated | da | unavailable_from_current_source | insufficient_book_depth, not_available_in_period |

Note:
- Static walk of the displayed book at decision; excludes latency, queue position, hidden liquidity and impact beyond observed depth. No half-spread is added on top.

## l2_concentration

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `wall_dist_bid` | distance mid -> notional-weighted price of the nearest eligible concentration | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | constructed | da | unavailable_from_current_source | insufficient_history, no_observation, insufficient_book_depth, not_available_in_period |
| `wall_size_bid` | notional of that concentration (contiguous eligible bins) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | constructed | da | unavailable_from_current_source | insufficient_history, no_observation, insufficient_book_depth, not_available_in_period |
| `wall_age_bid` | observed age so far of the same concentration (overlap matching) | seconds | decision time | decision time (book messages with ts < decision); book age <= 2s | constructed | da | unavailable_from_current_source | insufficient_history, no_observation, insufficient_book_depth, not_available_in_period |
| `wall_dist_ask` | distance mid -> notional-weighted price of the nearest eligible concentration | bps | decision time | decision time (book messages with ts < decision); book age <= 2s | constructed | da | unavailable_from_current_source | insufficient_history, no_observation, insufficient_book_depth, not_available_in_period |
| `wall_size_ask` | notional of that concentration (contiguous eligible bins) | USDT notional | decision time | decision time (book messages with ts < decision); book age <= 2s | constructed | da | unavailable_from_current_source | insufficient_history, no_observation, insufficient_book_depth, not_available_in_period |
| `wall_age_ask` | observed age so far of the same concentration (overlap matching) | seconds | decision time | decision time (book messages with ts < decision); book age <= 2s | constructed | da | unavailable_from_current_source | insufficient_history, no_observation, insufficient_book_depth, not_available_in_period |

Note:
- Fixed absolute price bins of None quote units; threshold mode snapshot_median_multiple (snapshot_median_multiple: bin > 5.0 x median bin of the current side; history_quantile: bin > 0.99-quantile of strictly earlier samples within 1h); within 100 bps; contiguous bins merged; age resolution = sample period; price levels only - no order identity.

## l2_flow_interaction

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `buy_pressure_10bps` | aggressive buy notional in bar / ask depth within 10 bps at bar start | ratio | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `sell_pressure_10bps` | aggressive sell notional in bar / bid depth within 10 bps at bar start | ratio | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `mid_move_bps` | 1e4 (m_end / m_start - 1) over the bar | bps | bar interval | bar close | observed | da | unavailable_from_current_source | incomplete_coverage, not_available_in_period |
| `impact_per_buy_pressure_10bps` | mid_move_bps / buy_pressure_10bps (low \|value\| with high pressure = absorbed flow) | bps per unit of depth consumed | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `impact_per_sell_pressure_10bps` | mid_move_bps / sell_pressure_10bps (low \|value\| with high pressure = absorbed flow) | bps per unit of depth consumed | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `buy_pressure_25bps` | aggressive buy notional in bar / ask depth within 25 bps at bar start | ratio | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `sell_pressure_25bps` | aggressive sell notional in bar / bid depth within 25 bps at bar start | ratio | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `impact_per_buy_pressure_25bps` | mid_move_bps / buy_pressure_25bps (low \|value\| with high pressure = absorbed flow) | bps per unit of depth consumed | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `impact_per_sell_pressure_25bps` | mid_move_bps / sell_pressure_25bps (low \|value\| with high pressure = absorbed flow) | bps per unit of depth consumed | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `buy_pressure_50bps` | aggressive buy notional in bar / ask depth within 50 bps at bar start | ratio | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `sell_pressure_50bps` | aggressive sell notional in bar / bid depth within 50 bps at bar start | ratio | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `impact_per_buy_pressure_50bps` | mid_move_bps / buy_pressure_50bps (low \|value\| with high pressure = absorbed flow) | bps per unit of depth consumed | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |
| `impact_per_sell_pressure_50bps` | mid_move_bps / sell_pressure_50bps (low \|value\| with high pressure = absorbed flow) | bps per unit of depth consumed | bar interval | bar close | constructed | da | unavailable_from_current_source | incomplete_coverage, zero_denominator, not_available_in_period |

Note:
- Constructed 'absorption' proxy defined only by this formula; uses no later price reversal and claims no participant intent.

## quotes_spread

| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |
|---|---|---|---|---|---|---|---|---|
| `quote_spread` | see quotes.py | varies | decision time | decision time (quotes with ts < decision); quote age <= 2s | observed | da | implemented_verified | not_available_in_period, stale_input, invalid_quote |
| `quote_spread_bps` | see quotes.py | varies | decision time | decision time (quotes with ts < decision); quote age <= 2s | observed | da | implemented_verified | not_available_in_period, stale_input, invalid_quote |
| `quote_age_s` | see quotes.py | varies | decision time | decision time (quotes with ts < decision); quote age <= 2s | observed | da | implemented_verified | not_available_in_period, stale_input, invalid_quote |
| `quote_twa_spread_bps` | see quotes.py | varies | decision time | decision time (quotes with ts < decision); quote age <= 2s | observed | da | implemented_verified | not_available_in_period, stale_input, invalid_quote |
| `quote_valid_time_frac` | see quotes.py | varies | decision time | decision time (quotes with ts < decision); quote age <= 2s | observed | da | implemented_verified | not_available_in_period, stale_input, invalid_quote |

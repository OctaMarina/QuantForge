# Rezultatele testelor (rulate efectiv)

Comandă: `python -m pytest tests -v` · Python 3.13.16 · Linux · 2026-10-26 · fixture-urile reale prezente.

**Rezultat: 182 passed in 135.88s**

Fallback fără numba (`NUMBA_DISABLE_JIT=1 pytest tests/test_bars.py tests/test_structural.py tests/test_flow.py tests/test_information_bars.py`): **96 passed in 61.49s**

## tests/test_bars.py

construcția barelor: OHLC [t,t+Δ), interval gol, tranzacția de prag, prag exact fără float, conservare, ordine la timestamp identic, invarianță batch/restart, goluri, recuperare, deduplicare, precizie, isBuyerMaker, prag adaptiv, depășire int64 refuzată

- `test_time_bar_ohlc_half_open` — PASSED
- `test_empty_interval_is_zero_volume_null_ohlc` — PASSED
- `test_volume_threshold_crossing_trade_included` — PASSED
- `test_exact_threshold_without_float_error` — PASSED
- `test_notional_bars_and_conservation` — PASSED
- `test_same_timestamp_ordering_and_unique_ids` — PASSED
- `test_batch_and_restart_invariance[time]` — PASSED
- `test_batch_and_restart_invariance[volume]` — PASSED
- `test_batch_and_restart_invariance[notional]` — PASSED
- `test_gap_invalidates_event_segment_and_time_bars` — PASSED
- `test_gap_recovery_equals_uninterrupted` — PASSED
- `test_dedupe_identity` — PASSED
- `test_precision_errors` — PASSED
- `test_is_buyer_maker_mapping_and_units` — PASSED
- `test_timestamp_regression_rejected` — PASSED
- `test_adaptive_threshold_uses_only_prior_complete_days` — PASSED
- `test_int64_overflow_is_refused_not_wrapped` — PASSED

## tests/test_book.py

carnet L2: reguli Binance spot/futures, benzi incrementale = brute force, acoperire, cost VWAP, regula tie, concentrări și persistență, fixture-uri reale Binance USD-M și Tardis Deribit, buffering înainte de snapshot

- `test_binance_spot_sync_rules` — PASSED
- `test_binance_futures_pu_rule` — PASSED
- `test_incremental_bands_equal_bruteforce_property` — PASSED
- `test_band_coverage_beyond_snapshot_is_null` — PASSED
- `test_cost_walk_vwap_and_no_double_spread` — PASSED
- `test_engine_tie_rule_twa_and_withdrawal_identity` — PASSED
- `test_walls_detection_age_and_reset` — PASSED
- `test_real_binance_futures_depth_fixture` — PASSED
- `test_real_tardis_deribit_incremental_inverse_fixture` — PASSED
- `test_binance_buffered_diffs_applied_after_snapshot` — PASSED

## tests/test_book_features.py

feature-uri L2 la nivel de dataset: spread, cost (cu/fără comision), presiune, adâncime insuficientă, absență = null

- `test_book_features_compute_costs_spread_and_pressure` — PASSED
- `test_book_absent_means_null_not_zero` — PASSED

## tests/test_config_validation.py

validări de configurație: simbol vs. base+quote, DVOL max_age ≥ rezoluție, opțiuni information bars (RMS, E0[T] fix, per_type) și zile lipsă, avertismente, identitatea seriei = exact parametrii care schimbă barele (id-uri 0.2.0 păstrate unde nu s-a schimbat nimic)

- `test_symbol_must_match_base_and_quote` — PASSED
- `test_dvol_max_age_must_cover_resolution` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.information.sign=majority]` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.information.ewma_span_bars=0]` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.information.expected_ticks_bounds=[100,10]]` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.information.min_expected_imbalance_ratio=1.5]` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.information.guard_max_bars_per_day=0]` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.on_unconfirmed_missing_day=skip]` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.archive_publication_lag_days=-1]` — PASSED
- `test_information_and_missing_day_options_are_validated[bars.types=[dollar_imbalance]]` — PASSED
- `test_warnings` — PASSED
- `test_guards_and_missing_day_policy_do_not_change_series_identity_but_stabilisers_do` — PASSED
- `test_rms_and_per_type_validation[bad0]` — PASSED
- `test_rms_and_per_type_validation[bad1]` — PASSED
- `test_rms_and_per_type_validation[bad2]` — PASSED
- `test_rms_and_per_type_validation[bad3]` — PASSED
- `test_rms_and_per_type_validation[bad4]` — PASSED
- `test_rms_and_per_type_validation[bad5]` — PASSED
- `test_rms_and_per_type_validation[bad6]` — PASSED
- `test_rms_and_per_type_validation[bad7]` — PASSED
- `test_rms_and_per_type_validation[bad8]` — PASSED
- `test_rms_and_per_type_validation[bad9]` — PASSED
- `test_rms_and_per_type_validation[bad10]` — PASSED
- `test_per_type_overrides_and_bounds_only_for_runs` — PASSED
- `test_series_identity_contains_exactly_the_parameters_that_change_the_bars` — PASSED

## tests/test_crossmarket.py

premium brut/FX, OHLC intrabar invizibil, minut lipsă nefolosit, as-of fără viitor, premium zilnic + z90, trend 30d, funding

- `test_premium_raw_and_fx_manual` — PASSED
- `test_intrabar_ohlc_not_visible` — PASSED
- `test_missing_minute_never_filled` — PASSED
- `test_asof_never_uses_future_even_if_closer` — PASSED
- `test_daily_premium_coverage_and_z_excludes_current` — PASSED
- `test_trend_30d_manual_and_zero_vol` — PASSED
- `test_funding_interval_observed_not_assumed` — PASSED
- `test_causal_zscore_excludes_current_and_skips_missing` — PASSED

## tests/test_external.py

Farside (fără dublarea Total, lipsă ≠ 0), ETF strict/research, publicare întârziată și revizii, Fear & Greed, DVOL, skew 25Δ, lichidări (zero observat vs. necunoscut, convenții de semn)

- `test_farside_parse_no_double_total_and_missing_not_zero` — PASSED
- `test_etf_strict_null_without_publication_evidence` — PASSED
- `test_delayed_publication_and_revision_vintages` — PASSED
- `test_fear_greed_parse_and_poll_vintage` — PASSED
- `test_dvol_normalisation_and_availability` — PASSED
- `test_option_name_parse` — PASSED
- `test_skew_25d_30d_manual` — PASSED
- `test_liquidation_mappings_coverage_and_zero_vs_unknown` — PASSED
- `test_live_liquidation_side_conventions` — PASSED

## tests/test_flow.py

tick rule, identitatea buy+sell, imbalance, Kyle/Hasbrouck vs statsmodels, eroare standard vs varianță, suma rădăcinilor

- `test_tick_rule_initialisation_and_carry` — PASSED
- `test_aggressor_identity_and_imbalance_bounds` — PASSED
- `test_imbalance_zero_denominator_and_partial_classification` — PASSED
- `test_kyle_hasbrouck_trade_level_vs_statsmodels` — PASSED
- `test_hasbrouck_bar_regressor_is_sum_of_roots` — PASSED
- `test_bar_level_rolling_lambda_vs_statsmodels` — PASSED

## tests/test_information_bars.py

information-driven bars (AFML §2.3.2 și variantele declarate): exemple calculate de mână (TIB, VRB, b0 peste graniță, warm-up, prag RMS), egalitate exactă cu o referință independentă (6 tipuri × 2 semne × stabilizatoare / RMS / E0[T] fix), EWMA = pandas adjust=True, E[T] ≈ H pe semne independente, RMS nu colapsează unde AFML colapsează, E0[T] fix fără derivă, invarianță prefix / perturbare viitoare / batch / restart, gol → warm-up nou, stări 0.2.0 încărcabile

- `test_tick_imbalance_hand_computed` — PASSED
- `test_volume_runs_hand_computed` — PASSED
- `test_warmup_without_classified_tick_repeats` — PASSED
- `test_equals_independent_reference[tick_imbalance-tick_rule-False]` — PASSED
- `test_equals_independent_reference[tick_imbalance-tick_rule-True]` — PASSED
- `test_equals_independent_reference[tick_imbalance-aggressor-False]` — PASSED
- `test_equals_independent_reference[tick_imbalance-aggressor-True]` — PASSED
- `test_equals_independent_reference[volume_imbalance-tick_rule-False]` — PASSED
- `test_equals_independent_reference[volume_imbalance-tick_rule-True]` — PASSED
- `test_equals_independent_reference[volume_imbalance-aggressor-False]` — PASSED
- `test_equals_independent_reference[volume_imbalance-aggressor-True]` — PASSED
- `test_equals_independent_reference[notional_imbalance-tick_rule-False]` — PASSED
- `test_equals_independent_reference[notional_imbalance-tick_rule-True]` — PASSED
- `test_equals_independent_reference[notional_imbalance-aggressor-False]` — PASSED
- `test_equals_independent_reference[notional_imbalance-aggressor-True]` — PASSED
- `test_equals_independent_reference[tick_runs-tick_rule-False]` — PASSED
- `test_equals_independent_reference[tick_runs-tick_rule-True]` — PASSED
- `test_equals_independent_reference[tick_runs-aggressor-False]` — PASSED
- `test_equals_independent_reference[tick_runs-aggressor-True]` — PASSED
- `test_equals_independent_reference[volume_runs-tick_rule-False]` — PASSED
- `test_equals_independent_reference[volume_runs-tick_rule-True]` — PASSED
- `test_equals_independent_reference[volume_runs-aggressor-False]` — PASSED
- `test_equals_independent_reference[volume_runs-aggressor-True]` — PASSED
- `test_equals_independent_reference[notional_runs-tick_rule-False]` — PASSED
- `test_equals_independent_reference[notional_runs-tick_rule-True]` — PASSED
- `test_equals_independent_reference[notional_runs-aggressor-False]` — PASSED
- `test_equals_independent_reference[notional_runs-aggressor-True]` — PASSED
- `test_equals_independent_reference[tick_imbalance-tick_rule-rms]` — PASSED
- `test_equals_independent_reference[tick_imbalance-aggressor-rms]` — PASSED
- `test_equals_independent_reference[volume_imbalance-tick_rule-rms]` — PASSED
- `test_equals_independent_reference[volume_imbalance-aggressor-rms]` — PASSED
- `test_equals_independent_reference[notional_imbalance-tick_rule-rms]` — PASSED
- `test_equals_independent_reference[notional_imbalance-aggressor-rms]` — PASSED
- `test_ewma_is_pandas_adjusted_ewm` — PASSED
- `test_prefix_and_future_perturbation_invariance[tick_imbalance]` — PASSED
- `test_prefix_and_future_perturbation_invariance[volume_imbalance]` — PASSED
- `test_prefix_and_future_perturbation_invariance[notional_imbalance]` — PASSED
- `test_prefix_and_future_perturbation_invariance[tick_runs]` — PASSED
- `test_prefix_and_future_perturbation_invariance[volume_runs]` — PASSED
- `test_prefix_and_future_perturbation_invariance[notional_runs]` — PASSED
- `test_prefix_and_future_perturbation_invariance[tick_imbalance_rms]` — PASSED
- `test_prefix_and_future_perturbation_invariance[volume_imbalance_rms]` — PASSED
- `test_prefix_and_future_perturbation_invariance[notional_imbalance_rms]` — PASSED
- `test_batch_and_restart_invariance[tick_imbalance]` — PASSED
- `test_batch_and_restart_invariance[volume_imbalance]` — PASSED
- `test_batch_and_restart_invariance[notional_imbalance]` — PASSED
- `test_batch_and_restart_invariance[tick_runs]` — PASSED
- `test_batch_and_restart_invariance[volume_runs]` — PASSED
- `test_batch_and_restart_invariance[notional_runs]` — PASSED
- `test_batch_and_restart_invariance[tick_imbalance_rms]` — PASSED
- `test_batch_and_restart_invariance[volume_imbalance_rms]` — PASSED
- `test_batch_and_restart_invariance[notional_imbalance_rms]` — PASSED
- `test_gap_restarts_expectations_with_a_warmup_bar` — PASSED
- `test_information_types_have_their_own_series_identity` — PASSED
- `test_rms_tick_imbalance_hand_computed` — PASSED
- `test_rms_expected_length_is_the_horizon_for_independent_signs[tick_imbalance-0.85-1.4]` — PASSED
- `test_rms_expected_length_is_the_horizon_for_independent_signs[volume_imbalance-0.85-2.5]` — PASSED
- `test_rms_gap_restarts_and_old_state_files_load` — PASSED
- `test_rms_does_not_collapse_where_afml_does` — PASSED
- `test_fixed_expected_ticks_equals_reference_with_constant_E0T[tick_runs]` — PASSED
- `test_fixed_expected_ticks_equals_reference_with_constant_E0T[volume_runs]` — PASSED
- `test_fixed_expected_ticks_equals_reference_with_constant_E0T[volume_imbalance]` — PASSED
- `test_fixed_expected_ticks_removes_the_drift_of_runs_bars` — PASSED

## tests/test_lookahead.py

pe date reale: prefix invariance, perturbare viitoare, granițe stabile la alt export, identitatea seriei, paritate replay live, manifest/calitate, modul observed, hash config; information bars (runs literale, imbalance stabilizate, imbalance RMS) prefix + perturbare; colapsul imbalance literal oprit de gardă

- `test_prefix_invariance` — PASSED
- `test_future_perturbation` — PASSED
- `test_boundaries_stable_across_export_periods` — PASSED
- `test_origin_is_part_of_series_identity` — PASSED
- `test_replay_live_parity` — PASSED
- `test_manifest_and_quality_outputs` — PASSED
- `test_observed_availability_mode_uses_receipt_times` — PASSED
- `test_config_hash_ignores_storage_paths` — PASSED
- `test_information_bars_prefix_and_future_perturbation[runs_literal]` — PASSED
- `test_information_bars_prefix_and_future_perturbation[imbalance_stabilised]` — PASSED
- `test_information_bars_prefix_and_future_perturbation[imbalance_rms]` — PASSED
- `test_literal_imbalance_bars_collapse_on_real_trades_and_are_halted` — PASSED

## tests/test_pipeline_archive.py

calea arhivei: fără reset la miezul nopții, update incremental = build integral; zi lipsă neconfirmată → oprire și completare ulterioară identică; 404 confirmat → segment nou și bare incomplete; politica gap + avertisment de gol învechit; 404 recent neconfirmat; information bars în pipeline (incremental = integral, QA, parametri per tip, RMS fără colaps); garda de colaps; coduri de ieșire CLI; lag de publicare

- `test_no_reset_at_day_boundary_and_incremental_equals_oneshot` — PASSED
- `test_unconfirmed_missing_day_stops_instead_of_creating_a_permanent_gap` — PASSED
- `test_missing_day_gap_segment_and_incomplete_time_bars` — PASSED
- `test_gap_policy_for_unconfirmed_days_and_stale_gap_warning` — PASSED
- `test_recent_404_is_not_a_confirmed_missing_day` — PASSED
- `test_information_bars_in_the_pipeline` — PASSED
- `test_collapse_guard_halts_only_the_runaway_series` — PASSED
- `test_cli_exit_codes` — PASSED
- `test_days_inside_the_publication_lag_are_not_an_error` — PASSED
- `test_rms_imbalance_and_per_type_parameters_in_the_pipeline` — PASSED

## tests/test_qa.py

comparație trades→1m vs klines native

- `test_kline_crosscheck_exact_and_detects_differences` — PASSED

## tests/test_quotes.py

spread din quote-uri, inclusiv pe cotațiile reale Binance BTCUSDT

- `test_quote_spread_manual_tie_and_twa` — PASSED
- `test_real_binance_quotes_and_trades_fixture` — PASSED

## tests/test_scripts.py

scripturile de rulare pe o arhivă sintetică: build clasic → calibrare pe lună pilot → build cu parametrii calibrați (bare/zi la ±25% de țintă) → raport

- `test_calibrate_build_report` — PASSED

## tests/test_sources.py

adaptoare: CHECKSUM, cache, revizii, antete futures, listare S3, paginare Coinbase, continuare DVOL, vintage-uri Wayback

- `test_archive_fetch_checksum_cache_and_revision` — PASSED
- `test_futures_header_and_klines` — PASSED
- `test_s3_listing_pagination` — PASSED
- `test_coinbase_pagination_and_missing_minutes` — PASSED
- `test_deribit_dvol_continuation` — PASSED
- `test_wayback_captures_as_vintages` — PASSED

## tests/test_structural.py

SADF/QADF/CADF, SDFC, CSW, BDE vs referințe; cauzalitate la endpoint; fără atribuire retroactivă; rupturi de lanț

- `test_sadf_qadf_cadf_vs_bruteforce[0]` — PASSED
- `test_sadf_qadf_cadf_vs_bruteforce[1]` — PASSED
- `test_sadf_qadf_cadf_vs_bruteforce[2]` — PASSED
- `test_sadf_detects_explosive_segment_right_tail` — PASSED
- `test_sdfc_vs_bruteforce` — PASSED
- `test_csw_vs_bruteforce` — PASSED
- `test_bde_vs_statsmodels_recursive_residuals` — PASSED
- `test_endpoint_causality_future_perturbation` — PASSED
- `test_sdfc_not_assigned_to_candidate_date` — PASSED
- `test_chain_breaks_restart_windows` — PASSED

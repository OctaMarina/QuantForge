# Rezultatele testelor (rulate efectiv)

Comandă: `python -m pytest tests -v` · Python 3.13.16 · Linux · 2026-10-20 · fixture-urile reale prezente.

**Rezultat: 80 passed in 25.15s**

Fallback fără numba (`NUMBA_DISABLE_JIT=1 pytest tests/test_bars.py tests/test_structural.py tests/test_flow.py`): **32 passed in 7.40s**

## tests/test_bars.py

construcția barelor: OHLC [t,t+Δ), interval gol, tranzacția de prag, prag exact fără float, conservare, ordine la timestamp identic, invarianță batch/restart, goluri, recuperare, deduplicare, precizie, isBuyerMaker, prag adaptiv

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

## tests/test_lookahead.py

pe date reale: prefix invariance, perturbare viitoare, granițe stabile la alt export, identitatea seriei, paritate replay live, manifest/calitate, modul observed, hash config

- `test_prefix_invariance` — PASSED
- `test_future_perturbation` — PASSED
- `test_boundaries_stable_across_export_periods` — PASSED
- `test_origin_is_part_of_series_identity` — PASSED
- `test_replay_live_parity` — PASSED
- `test_manifest_and_quality_outputs` — PASSED
- `test_observed_availability_mode_uses_receipt_times` — PASSED
- `test_config_hash_ignores_storage_paths` — PASSED

## tests/test_pipeline_archive.py

calea arhivei: fără reset la miezul nopții, update incremental = build integral, zi lipsă → segment nou și bare incomplete

- `test_no_reset_at_day_boundary_and_incremental_equals_oneshot` — PASSED
- `test_missing_day_gap_segment_and_incomplete_time_bars` — PASSED

## tests/test_qa.py

comparație trades→1m vs klines native

- `test_kline_crosscheck_exact_and_detects_differences` — PASSED

## tests/test_quotes.py

spread din quote-uri, inclusiv pe cotațiile reale Binance BTCUSDT

- `test_quote_spread_manual_tie_and_twa` — PASSED
- `test_real_binance_quotes_and_trades_fixture` — PASSED

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

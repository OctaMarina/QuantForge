"""Feature registry: every emitted column maps to exactly one definition (pattern) with formula, unit,
sources, clock, parameters, availability, validity mask, implementation, tests and status.

Clock legend:  AVAIL = availability clock (received_time, ns); a window W=(t-w, t] ends at eval_time t
               (inclusive) and, for trades, at the bar's closing trade (cursor). BAR = events of the bar only.
Status legend: implemented | implemented_diagnostic (kept for audit/diagnostics, not recommended as model input)
               | blocked_by_data (definition kept, inputs unavailable/invalid on the history — see validity map)
               | not_implemented (inventory only)
"""
from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import asdict, dataclass, field

W = "{w}"


@dataclass
class FeatureDef:
    pattern: str
    family: str
    priority: str
    formula: str
    unit: str
    sources: str
    clock: str
    params: str
    validity: str
    impl: str
    tests: str
    status: str = "implemented"
    notes: str = ""

    PH = {"w": r"[0-9]+(?:ms|s|m|h)", "h": r"[0-9]+(?:ms|s|m|h)", "g": r"[0-9]+(?:ms|s|m|h)", "b": r"[0-9p]+bps",
          "k": r"[0-9]+", "q": r"[0-9p]+", "d": r"[0-9]+", "r": r"[0-9]+", "l": r"[0-9]+", "n": r"[0-9]+"}

    def regex(self):
        out = []
        for tok in re.split(r"(\{[a-z]\})", self.pattern):
            if len(tok) == 3 and tok[0] == "{" and tok[2] == "}":
                out.append(self.PH[tok[1]])
            else:
                out.append(re.escape(tok))
        return re.compile("^" + "".join(out) + "$")


TR = "features.trades.compute_trade_features"
SL = "features.slow"
BK = "book.core._emit_query"
DS = "dataset"

REGISTRY: list[FeatureDef] = []


def add(*a, **k):
    REGISTRY.append(FeatureDef(*a, **k))


# ------------------------------------------------------------------------------------------- metadata
for c, f in [("bar_id", "sequential id within bar set"), ("decision_time_ns", "time bars: T_{k+1}; event bars: recv of closing trade"),
             ("eval_time_ns", "as-of time (inclusive) for non-trade streams; time bars T_{k+1}-1ns"),
             ("t_start_ns", "time bar grid start / first trade recv"), ("t_end_ns", "time bar grid end / last trade recv"),
             ("t_open_recv_ns", "recv of first trade in bar"), ("t_close_recv_ns", "recv of last trade in bar"),
             ("m_open_ms", "trade_time (market clock) of first trade"), ("m_close_ms", "trade_time of last trade"),
             ("first_seq", "canonical position of first trade (-1 empty)"), ("last_seq", "trade cursor (canonical position)"),
             ("first_trade_id", "exchange trade_id of first trade"), ("last_trade_id", "exchange trade_id of last trade"),
             ("seg", "trade-continuity segment id"), ("close_reason", "0 threshold,1 gap partial,2 end partial,3 threshold change,4 daily reset,5 time,6 fresh-start partial"),
             ("complete", "1 complete bar, 0 partial"), ("threshold_int", "active threshold (e3 BTC / e5 USDT / 1e-2 bp / ns)"),
             ("feed_complete", "1 if the trade feed is verified complete over the bar")]:
    add(c, "meta", "P0", f, "-", "canonical trades", "AVAIL", "bar spec", "-", "bars.engines / dataset.meta_columns",
        "test_bars_*", notes="provenance/identity of the decision event")

# ------------------------------------------------------------------------------------------- bar intrinsic
for c, f, u in [("bar_open", "first trade price", "USDT/BTC"), ("bar_high", "max trade price", "USDT/BTC"),
                ("bar_low", "min trade price", "USDT/BTC"), ("bar_close", "last trade price", "USDT/BTC"),
                ("bar_n_trades", "N", "count"), ("bar_volume_btc", "sum q", "BTC"), ("bar_notional_usdt", "sum p*q", "USDT"),
                ("bar_vwap", "sum pq / sum q", "USDT/BTC"), ("bar_return_log", "log(C_k / C_{k-1}) (same segment)", "log"),
                ("bar_oc_log", "log(C/O)", "log"), ("bar_range_log", "log(H/L)", "log"),
                ("bar_close_location", "(C-L)/(H-L); H=L -> null", "[0,1]"), ("bar_tfi_btc", "sum eps q", "BTC"),
                ("bar_tfi_usdt", "sum eps p q", "USDT"), ("bar_tfi_ratio", "tfi/V", "[-1,1]"),
                ("bar_count_imbalance", "(N_buy-N_sell)/N", "[-1,1]"), ("bar_mean_trade_size_btc", "V/N", "BTC"),
                ("bar_duration_s", "time: Δ; event: recv(last)-recv(first)", "s"),
                ("bar_volume_rate_btc_s", "V/duration", "BTC/s"), ("bar_notional_rate_usdt_s", "D/duration", "USDT/s"),
                ("bar_overshoot_btc", "V - V* (complete volume bars)", "BTC"), ("bar_threshold_btc", "V*", "BTC"),
                ("bar_overshoot_usdt", "D - D* (complete dollar bars)", "USDT"), ("bar_threshold_usdt", "D*", "USDT"),
                ("bar_threshold_bps", "CUSUM b (h=log(1+b/1e4))", "bp"),
                ("cusum_trigger_side", "+1 S+>=h, -1 S-<=-h", "-"), ("cusum_pre_reset", "triggered accumulator before reset", "log"),
                ("cusum_opposite_state", "opposite accumulator at trigger", "log"), ("cusum_filter_overshoot", "|S|-h", "log"),
                ("cusum_threshold_log", "h", "log")]:
    add(c, "F1/bar", "P0", f, u, "canonical trades", "BAR", "bar spec", "OHLC null if bar empty; partial bars flagged by `complete`",
        "dataset.bar_features", "test_bars_conservation, test_cusum_*")

# ------------------------------------------------------------------------------------------- F1 / F3 trades
add("last_trade_price", "F1", "P0", "price of trade at cursor", "USDT/BTC", "trades", "AVAIL", "-", "null before first trade", TR, "test_trade_prefix_invariance")
add("last_trade_age_ms", "F1", "P0", "t - recv(cursor trade)", "ms", "trades", "AVAIL", "-", "-", TR, "")
add("tw_state_{w}", "F3", "P0", "window state: 0 valid,1 warmup,2 source_gap", "code", "trades + trades_gaps", "AVAIL", "w", "-", TR, "test_trade_gap_mask")
for pat, f, u in [("trade_count_{w}", "N_W", "count"), ("volume_{w}_btc", "V_W=sum q", "BTC"),
                  ("notional_{w}_usdt", "D_W=sum p q", "USDT"), ("trade_rate_{w}_per_s", "N_W/w", "1/s"),
                  ("volume_rate_{w}_btc_s", "V_W/w", "BTC/s"), ("notional_rate_{w}_usdt_s", "D_W/w", "USDT/s"),
                  ("tfi_{w}_btc", "sum eps q", "BTC"), ("tfi_{w}_usdt", "sum eps p q", "USDT"),
                  ("tfi_ratio_{w}", "tfi_btc/V (V>0)", "[-1,1]"), ("count_imbalance_{w}", "(N_buy-N_sell)/N", "[-1,1]"),
                  ("mean_trade_size_{w}_btc", "V/N", "BTC"), ("size_concentration_{w}", "sum (q/V)^2", "[0,1]"),
                  ("large_trade_share_{w}", "sum d 1{d>u}/D, u=p95 of previous UTC day trade notionals", "[0,1]"),
                  ("signed_large_flow_{w}_usdt", "sum eps d 1{d>u}", "USDT")]:
    add(pat, "F3", "P0", f, u, "trades(price,quantity,is_buyer_maker)", "AVAIL", "w in trades.flow_windows_s",
        "tw_state_<w>=0; ratio null if denominator 0", TR, "test_trade_features_bruteforce, test_trade_prefix_invariance")
for pat, f in [("vwap_{w}", "sum pq/sum q"), ("vwap_distance_{w}_bps", "1e4(P_last/VWAP-1), P=last trade (fresh<=10s)"),
               ("range_log_{w}", "log(H/L) over trades in W"), ("close_location_{w}", "(P-L)/(H-L)"),
               ("drawdown_from_high_{w}", "log(P/H_W)"), ("rebound_from_low_{w}", "log(P/L_W)")]:
    add(pat, "F1", "P0", f, "see formula", "trades", "AVAIL", "w in flow_windows_s", "tw_state=0, N>0, last price fresh",
        TR, "test_trade_features_bruteforce")
add("ret_trade_{w}", "F1", "P0", "log(P(t)/P(t-w)), P = last trade price as-of, both ends fresh (<=10s) with no trade gap between price and endpoint; gaps inside the window do not matter", "log",
    "trades", "AVAIL", "w in return_windows_s", "tw_state=0 and both ends fresh", TR, "test_trade_features_bruteforce")
add("path_efficiency_trade_{w}_grid1s", "F1", "P0", "log(P_t/P_{t-w}) / sum |Δ log P_j| on 1s as-of grid", "[-1,1]",
    "trades", "AVAIL(grid)", "w in path_eff_windows_s", "no gap in grid window; denominator>0", TR, "")
for nm in ("p50", "p90", "p99"):
    add(f"trade_size_{nm}_{{w}}_btc", "F3", "P0", f"{nm} of trade quantities in W (linear interpolation)", "BTC", "trades",
        "AVAIL", "w in quantile_windows_s", "tw_state=0", TR, "test_trade_features_bruteforce")
    add(f"trade_size_{nm}_{{w}}_usdt", "F3", "P0", f"{nm} of trade notionals in W", "USDT", "trades", "AVAIL",
        "w in quantile_windows_s", "tw_state=0", TR, "")
for pat, f, pri in [("rv_trade_{w}_grid{g}", "RV=sum r_j^2, r on as-of trade-price grid g, last w/g returns ending at last grid point<=t", "P0"),
                    ("rv_neg_trade_{w}_grid{g}", "sum r^2 1{r<0}", "P1"), ("rv_pos_trade_{w}_grid{g}", "sum r^2 1{r>0}", "P1"),
                    ("rv_asym_trade_{w}_grid{g}", "(RV+-RV-)/RV", "P1"), ("rv_n_returns_{w}_grid{g}", "number of grid returns", "P0"),
                    ("bv_trade_{w}_grid{g}", "(pi/2) n/(n-1) sum |r_j||r_{j-1}| (n>=30)", "P1"),
                    ("jump_trade_{w}_grid{g}", "max(RV-BV,0) — proxy, not a shock label", "P1"),
                    ("max_abs_ret_trade_{w}_grid{g}", "max |r_j|", "P1")]:
    add(pat, "F2", pri, f, "log^2 / log", "trades", "AVAIL(grid)", "grids {100ms:5s,30s;1s:30s..1h;1m:4h,24h}",
        "complete grid window without trade gap", TR, "test_rv_identities")
add("sign_acf_lag{l}_n{n}", "F3", "P1", "corr(eps_i, eps_{i-l}) over last N trades, centred, pair variances", "[-1,1]",
    "trades", "AVAIL(trade count)", "N=1000, l in {1,10}, >=100 pairs", "same continuity segment; variance>0", TR, "")
for c, f in [("run_length_trades", "consecutive same-sign trades ending at cursor"), ("run_sign", "sign of current run"),
             ("run_volume_btc", "volume of current run"), ("run_duration_s", "recv span of current run"),
             ("run_censored", "1 if the run started before the loaded history chunk")]:
    add(c, "F3", "P1", f, "-", "trades", "AVAIL", "-", "run broken by continuity gaps", TR, "")
add("burstiness_fano_{w}_cell1s", "F3", "P1", "Var(N_j)/Mean(N_j) over complete 1s cells in W (>=30 cells)", "-",
    "trades", "AVAIL", "w in burst_windows_s", "tw_state=0", TR, "")
add("notional_rate_ratio_{w}_vs_24h", "F3", "P0", "(D_W/w)/(D_24h/86400)", "ratio", "trades", "AVAIL", "w", "24h window valid", TR, "")
add("notional_15m_vs_slot_median_28d", "F10", "P1", "D_15m / median over prior 28 days of D in same 15-min UTC slot (>=8 days)",
    "ratio", "trades", "AVAIL", "28 days, min 8", "null in warm-up", "features.slow.seasonal_ratio", "")

# ------------------------------------------------------------------------------------------- F7 mark/index/funding
add("mark_state", "F7", "P0", "0 valid,1 none,2 stale (market age > 3s)", "code", "mark_price", "AVAIL", "mark_fast_age_s=3", "-", SL, "")
add("mark_recv_age_ms", "F7", "P0", "t - recv(last mark row)", "ms", "mark_price", "AVAIL", "-", "-", SL, "")
add("mark_mkt_age_ms", "F7", "P0", "t - event_time(last mark row)", "ms", "mark_price", "AVAIL", "-", "-", SL, "")
add("mark_price", "F7", "P0", "as-of mark", "USDT/BTC", "mark_price", "AVAIL", "slow age 10s", "stale -> null", SL, "")
add("index_price", "F7", "P0", "as-of index", "USDT/BTC", "mark_price", "AVAIL", "slow age 10s", "stale -> null", SL, "")
add("basis_mark_index_bps", "F7", "P0", "1e4(M/I-1)", "bp", "mark_price", "AVAIL", "fast age 3s", "stale -> null", SL, "test_basis_identity")
add("basis_mark_index_coverage_{w}", "F7", "P0", "fresh-time fraction of W", "[0,1]", "mark_price", "AVAIL", "w", "-", SL, "")
add("tw_basis_mark_index_bps_{w}", "F7", "P0", "time-weighted mean of 1e4(M/I-1) over W (exact step integral)", "bp", "mark_price",
    "AVAIL", "w in basis_windows_s; validity until event+10s", "coverage=1", SL, "test_tw_piecewise")
add("std_basis_mark_index_bps_{w}", "F7", "P0", "time-weighted std", "bp", "mark_price", "AVAIL", "w", "coverage=1", SL, "")
add("basis_mark_index_change_{w}_bps", "F7", "P0", "B(t)-B(t-w)", "bp", "mark_price", "AVAIL", "w", "both ends fresh", SL, "")
add("ret_index_{w}", "F1/F7", "P0", "log(I(t)/I(t-w))", "log", "mark_price", "AVAIL", "w", "both ends fresh (3s)", SL, "")
add("ret_mark_{w}", "F7", "P1", "log(M(t)/M(t-w))", "log", "mark_price", "AVAIL", "w", "both ends fresh", SL, "")
add("ret_trade_minus_index_{w}", "F7", "P0", "r^trade_W - r^index_W (trade variant of perp-index return difference)", "log",
    "trades+mark_price", "AVAIL", "w", "both valid", SL, "", notes="mid variant needs a valid book (see book family)")
add("funding_rate_asof", "F7", "P0", "last published funding rate (fraction/period)", "fraction", "mark_price.funding_rate", "AVAIL",
    "slow age 10s", "stale -> null", SL, "")
add("funding_rate_age_ms", "F7", "P0", "recv age of funding observation", "ms", "mark_price", "AVAIL", "-", "-", SL, "")
add("funding_change_{w}", "F7", "P0", "f(t)-f(t-w)", "fraction", "mark_price", "AVAIL", "w in funding_change_windows_s", "-", SL, "")
add("funding_cycle_changed_{w}", "F7", "P0", "next_funding_time(t) != next_funding_time(t-w)", "0/1", "mark_price", "AVAIL", "w", "-", SL, "")
add("time_to_funding_s", "F7", "P0", "(T_next - t)/1s; expired -> null", "s", "mark_price.next_funding_time", "AVAIL", "-", "-", SL, "")
add("funding_interval_s_observed", "F7", "P0", "difference between consecutive distinct next_funding_time values seen", "s",
    "mark_price", "AVAIL", "-", "null until two cycles observed", SL, "")
add("funding_phase_sin", "F7", "P0", "sin(2π(1 - ttf/interval))", "-", "mark_price", "AVAIL", "-", "interval known", SL, "")
add("funding_phase_cos", "F7", "P0", "cos(2π(1 - ttf/interval))", "-", "mark_price", "AVAIL", "-", "interval known", SL, "")
add("funding_rz_7d", "F7", "P1", "(f - median_7d)/(1.4826 MAD_7d) on prior 1-min as-of grid", "z", "mark_price", "AVAIL",
    "7 days, >=3.5 days", "warm-up -> null; MAD=0 -> null", SL, "")
add("settle_est_index_bps_p2", "F7/F12", "P2", "1e4(E/I-1), E=estimated_settle_price", "bp", "mark_price", "AVAIL", "-",
    "semantics on perpetual unconfirmed", SL, "", status="implemented_diagnostic")

# ------------------------------------------------------------------------------------------- F8 OI
for c, f, u in [("oi_state", "0 valid,1 none known,2 reference older than 15 min,3 too old because a newer point is rejected or in quarantine", "code"),
                ("oi_btc_asof", "OI of the latest-timestamp point among versions received <= t", "BTC"),
                ("oi_usdt_asof", "sum_open_interest_value of that point", "USDT"),
                ("oi_reference_age_s", "t - timestamp of reference point", "s"),
                ("oi_recv_age_s", "t - receipt of reference version", "s"),
                ("oi_reference_ts_ns", "timestamp of reference point", "ns"),
                ("oi_btc_known_delayed", "same as oi_btc_asof without the age limit (delayed context, different definition)", "BTC"),
                ("oi_reference_change_btc", "O(τ1)-O(τ0), two most recent distinct points known at t", "BTC"),
                ("oi_reference_dtau_s", "τ1-τ0", "s"), ("oi_change_rate_per_s", "Δlog O/Δτ (0<Δτ<=900s)", "1/s"),
                ("oi_value_price_implied", "P*=OI_value/OI", "USDT/BTC"),
                ("oi_level_rz_7d", "robust z of log O vs prior 2016 distinct 5-min points", "z"),
                ("oi_change_rz_7d", "robust z of 5-min Δlog O vs prior 2016 points", "z")]:
    add(c, "F8", "P0" if "rz" not in c else "P1", f, u, "open_interest", "AVAIL (versions by first receipt)", "oi_max_ref_age_s=900",
        "oi_state", SL + ".oi_features", "test_oi_no_lookahead")
for pat, f in [("oi_log_change_known_{w}", "log(O_known(t)/O_known(t-w))"), ("oi_ref_dtau_known_{w}_s", "τ_t - τ_{t-w}"),
               ("oi_points_received_{w}", "number of OI versions received in W"), ("tw_oi_known_{w}_btc", "TW mean of known OI over W")]:
    add(pat, "F8", "P0", f, "-", "open_interest", "AVAIL", "w in oi_windows_s", "oi_state=0", SL + ".oi_features", "test_oi_no_lookahead")

# ------------------------------------------------------------------------------------------- F9 liquidations
add("liq_state_{w}", "F9", "P0", "0 valid,1 warmup,2 hour without liquidation file,3 market-stream reception gap", "code",
    "liquidations + mark_price(reception)", "AVAIL", "policy=unknown", "-", SL + ".liq_features", "")
add("liq_inferred_zero_{w}", "F9", "P0", "1 if missing-file hours were treated as zero (policy=inferred_zero only)", "0/1",
    "liquidations", "AVAIL", "opt-in policy", "-", SL + ".liq_features", "")
for pat, f, u in [("liq_notional_short_{w}_usdt", "sum avg_price*filled_qty, side=BUY (short liquidations)", "USDT"),
                  ("liq_notional_long_{w}_usdt", "same, side=SELL (long liquidations)", "USDT"),
                  ("liq_count_{w}", "deduplicated notifications", "count"), ("liq_rate_{w}_per_s", "count/w", "1/s"),
                  ("liq_signed_notional_{w}_usdt", "L_short - L_long", "USDT"),
                  ("liq_imbalance_{w}", "(Ls-Ll)/(Ls+Ll); 0 with liq_any=0 on verified feed", "[-1,1]"),
                  ("liq_any_{w}", "count>0", "0/1"), ("log1p_liq_notional_{w}", "log1p(Ls+Ll)", "log USDT"),
                  ("liq_notional_rate_{w}_usdt_s", "(Ls+Ll)/w", "USDT/s"), ("liq_peak_1s_{w}_usdt", "max notional in a 1s cell", "USDT")]:
    add(pat, "F9", "P0", f, u, "liquidations (observed notifications, not exchange total)", "AVAIL", "w in liq_windows_s",
        "liq_state_<w>=0", SL + ".liq_features", "")
add("liq_burst_ratio_2m_15m", "F9", "P0", "rate_2m/rate_15m", "ratio", "liquidations", "AVAIL", "-", "denominator>0", SL, "")
for nm in ("buy_short_liq", "sell_long_liq"):
    add(f"time_since_{nm}_s", "F9", "P0", "t - recv of last notification of that side; null if an hour without file or a shared stream gap lies since that notification", "s", "liquidations", "AVAIL", "-",
        "censored flag", SL, "")
    add(f"time_since_{nm}_censored", "F9", "P0", "1 if no prior notification in history", "0/1", "liquidations", "AVAIL", "-", "-", SL, "")

# ------------------------------------------------------------------------------------------- F10 ticker / calendar
for c, f, u in [("ticker_state", "0 valid,1 none,2 stale (>10s market age)", "code"), ("ticker_mkt_age_ms", "t-event_time", "ms"),
                ("ticker_return", "price_change_percent/100", "fraction"),
                ("ticker_return_consistency", "last/open-1 - pct/100 (control)", "fraction"),
                ("ticker_range_log", "log(high/low)", "log"),
                ("ticker_range_position_trade", "(P-low)/(high-low), P=last trade (not clamped)", "-"),
                ("ticker_vwap_distance_trade_bps", "1e4(P/weighted_avg-1)", "bp"),
                ("ticker_window_s", "statistics_close-statistics_open", "s"),
                ("ticker_quote_rate_usdt_s", "quote_volume/T_stats", "USDT/s"),
                ("ticker_avg_fill_size_btc", "base_volume/total_trades", "BTC"),
                ("ticker_base_volume_24h_btc", "base_asset_volume (rolling, never summed)", "BTC")]:
    add(c, "F10", "P0", f, u, "ticker", "AVAIL", "ticker_max_age_s=10", "ticker_state", SL + ".ticker_features", "")
for c in ("hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend"):
    add(c, "F10", "P1", "UTC calendar of eval time", "-", "-", "AVAIL", "-", "-", SL + ".calendar_features", "")

# ------------------------------------------------------------------------------------------- F4/F5/F6/F7 book
BKV = "book_state=0 (anchored, chained, not crossed, fresh <=1s market age); band/levels inside certified domain"
add("book_state", "F4", "P0", "0 valid,1 unanchored,2 stale,3 crossed,4 awaiting anchor connection,5 synthetic-anchor settle,6 empty side,7 top outside certified domain; book_mode=rebuilt only: 9 no anchor, top exact by protocol certificate, 10 no anchor, top not certified (features null)",
    "code", "orderbook", "AVAIL (effective application time)", "max_age_ms=1000", "-", BK, "test_book_*")
for c, f, u in [("book_recv_age_ms", "t - effective time of last applied message", "ms"),
                ("book_mkt_age_ms", "t - event_time of last applied message", "ms"),
                ("best_bid", "b1", "USDT/BTC"), ("best_ask", "a1", "USDT/BTC"), ("mid", "(a1+b1)/2", "USDT/BTC"),
                ("spread_bps", "1e4 (a1-b1)/m", "bp"), ("qty_bid1_btc", "q^b_1", "BTC"), ("qty_ask1_btc", "q^a_1", "BTC"),
                ("qi_1", "(q^b_1-q^a_1)/(q^b_1+q^a_1)", "[-1,1]"),
                ("weighted_mid_offset_bps", "1e4 (wm-m)/m, wm=(a1 qb1 + b1 qa1)/(qb1+qa1) = s/2 QI1 identity", "bp"),
                ("gap_ask12_bps", "1e4 (a2-a1)/m", "bp"), ("gap_bid12_bps", "1e4 (b1-b2)/m", "bp"),
                ("depth_bid_top{k}_btc", "sum of K best occupied bid levels", "BTC"), ("depth_ask_top{k}_btc", "same ask", "BTC"),
                ("qi_top{k}", "(D^b_K-D^a_K)/(D^b_K+D^a_K)", "[-1,1]"),
                ("depth_bid_{b}_btc", "sum q^b(p), p >= m(1-δ/1e4)", "BTC"), ("depth_ask_{b}_btc", "sum q^a(p), p <= m(1+δ/1e4)", "BTC"),
                ("notional_bid_{b}_usdt", "sum p q^b(p) in band (exact notional)", "USDT"),
                ("notional_ask_{b}_usdt", "sum p q^a(p) in band", "USDT"),
                ("qi_{b}", "(D^b_δ - D^a_δ)/(D^b_δ + D^a_δ); both zero -> null", "[-1,1]"),
                ("conc_bid_{b}", "sum (q/D)^2 in band (levels, not participants)", "[0,1]"), ("conc_ask_{b}", "same ask", "[0,1]"),
                ("cost_buy_{q}btc_bps", "1e4(P_buy(Q)/m-1), walk asks; insufficient certified depth -> null", "bp"),
                ("cost_sell_{q}btc_bps", "1e4(1-P_sell(Q)/m)", "bp"),
                ("cost_buy_{d}usdt_atmid_bps", "Q=D*/m, cost as above (exposure fixed at mid)", "bp"),
                ("cost_sell_{d}usdt_atmid_bps", "same, sell", "bp"),
                ("cert_bid_floor_dist_bps", "1e4(1 - cert_bid_lo/m): certified depth below mid", "bp"),
                ("cert_ask_ceiling_dist_bps", "1e4(cert_ask_hi/m - 1)", "bp")]:
    add(c, "F4" if not c.startswith("cost") else "F6", "P0" if not c.startswith(("conc", "gap")) else "P1", f, u, "orderbook (reconstructed)",
        "AVAIL", "bands {1,5,10,25,50}bps; K {5,10,20}; Q {0.1,1,5} BTC; D* {1e4,1e5,1e6}", BKV, BK, "test_book_vs_reference, test_book_identities")
for c, f, u, pri in [("book_coverage_{w}", "valid fraction of W", "[0,1]", "P0"),
                     ("ofi_{w}_btc", "sum_n e_n (Cont-Kukanov-Stoikov, level 1, complete messages, never across anchors/gaps)", "BTC", "P0"),
                     ("ofi_rate_{w}_btc_s", "OFI/w", "BTC/s", "P0"),
                     ("ofi_norm_{w}", "OFI / TW((qb1+qa1)/2)", "-", "P0"),
                     ("mlofi_l{r}_{w}_btc", "OFI at rank r (ranks recomputed per message)", "BTC", "P1"),
                     ("mlofi_norm_l{r}_{w}", "MLOFI_r / TW depth at rank r", "-", "P1"),
                     ("tw_spread_bps_{w}", "time-weighted mean spread (exact)", "bp", "P0"),
                     ("tw_qi_1_{w}", "TW mean QI1", "[-1,1]", "P0"), ("tw_depth1_{w}_btc", "TW mean (qb1+qa1)/2", "BTC", "P0"),
                     ("tw_depth_bid_{b}_{w}_btc", "TW mean band depth", "BTC", "P0"), ("tw_depth_ask_{b}_{w}_btc", "TW mean band depth", "BTC", "P0"),
                     ("max_spread_bps_{w}", "max spread over valid states in W", "bp", "P1"),
                     ("min_depth1_{w}_btc", "min (qb1+qa1)/2 in W", "BTC", "P1"),
                     ("min_depth_bid_{b}_{w}_btc", "min band depth in W", "BTC", "P1"), ("min_depth_ask_{b}_{w}_btc", "same ask", "BTC", "P1"),
                     ("add_bid_{b}_{w}_btc", "sum max(Δq,0) at fixed price in band (previous mid) — net visible adds", "BTC", "P1"),
                     ("rem_bid_{b}_{w}_btc", "sum max(-Δq,0) — net visible reductions", "BTC", "P1"),
                     ("add_ask_{b}_{w}_btc", "same ask", "BTC", "P1"), ("rem_ask_{b}_{w}_btc", "same ask", "BTC", "P1"),
                     ("book_msg_count_{w}", "applied messages (feed control, not alpha)", "count", "P1"),
                     ("book_level_changes_{w}", "level rows applied", "count", "P1")]:
    add(c, "F5" if ("ofi" in c or "add" in c or "rem" in c) else "F4", pri, f, u, "orderbook (reconstructed)", "AVAIL",
        "windows {1s,5s,30s,2m,15m}; mlofi {5s,30s}; min/max <=2m", BKV + "; strict coverage=1", BK,
        "test_tw_integrals, test_ofi_formula, test_book_partition")
for c, f in [("basis_mid_index_bps", "1e4(m/I-1)"), ("basis_mid_mark_bps", "1e4(m/M-1) (diagnostic; m/I=(m/M)(M/I))"),
             ("basis_mark_recv_age_ms", "recv age of mark used for basis"), ("basis_mark_mkt_age_ms", "market age of mark"),
             ("basis_coverage_{w}", "fraction of W with valid book and fresh mark"),
             ("tw_basis_mid_index_bps_{w}", "TW mean of 1e4(m/I-1) (integrated at every book/mark change)"),
             ("std_basis_mid_index_bps_{w}", "TW std"), ("tw_basis_mid_mark_bps_{w}", "TW mean of 1e4(m/M-1)"),
             ("std_basis_mid_mark_bps_{w}", "TW std")]:
    add(c, "F7", "P0", f, "bp", "orderbook + mark_price", "AVAIL", "mark_max_age_ms=3000; windows {30s,2m,15m,1h}", BKV, BK, "test_basis_identity")
for c, f, u in [("eff_spread_qw_bps_{w}", "q-weighted mean of 2e4 eps (p - m_-)/m_-, m_- = last valid quote with recv<recv_i and txn<trade_time", "bp"),
                ("eff_spread_dw_bps_{w}", "notional-weighted", "bp"), ("eff_spread_n_{w}", "matched trades", "count"),
                ("markout_{h}_qw_bps_{w}", "q-weighted mean of 1e4 eps log(m_{i+h}/m_-), only markouts matured in W", "bp"),
                ("markout_{h}_n_{w}", "matured markouts in W", "count")]:
    add(c, "F6", "P1", f, u, "trades + orderbook", "AVAIL (maturity time)", "es max quote age 1s; h {1,5,30}s",
        BKV + "; markout counted at maturity only", BK, "test_markout_maturation")

# ------------------------------------------------------------------------------------------- F11 combinations
for c, f in [("pressure_buy_5bps_{w}", "V_buy_W / TW(D^a_5bps)"), ("pressure_sell_5bps_{w}", "V_sell_W / TW(D^b_5bps)"),
             ("pressure_net_5bps_{w}", "P_buy - P_sell"), ("tfi_over_tw_depth1_{w}", "TFI_W / TW depth1"),
             ("tfi_ratio_x_ofi_norm_{w}", "TFI_ratio x OFI_norm"), ("tfi_ratio_x_ret_trade_{w}", "TFI_ratio x r_trade"),
             ("funding_x_oi_level_rz", "f x z_OI"), ("funding_x_oi_log_change_1h", "f x Δlog O (1h known)"),
             ("turnover_to_oi_{w}", "V_W / TW(O_known)"), ("liq_to_trade_notional_{w}", "(Ls+Ll)/D_W"),
             ("liq_to_oi_{w}", "(Ls+Ll)/OI_value"), ("liq_short_to_ask_depth_10bps_{w}", "Ls / ask notional depth 10bps"),
             ("liq_long_to_bid_depth_10bps_{w}", "Ll / bid notional depth 10bps"),
             ("activity_vs_ticker24h_{w}", "(D_W/w)/(quote_volume/T_stats)"), ("rv_ratio_2m_1h_grid1s", "(RV_2m/120)/(RV_1h/3600)"),
             ("liq_any_2m_x_rv_ratio", "liq_any_2m x rv_ratio")]:
    add(c, "F6/F11", "P0" if c.startswith(("pressure", "turnover", "liq_to")) else "P1", f, "ratio", "components", "AVAIL",
        "-", "valid iff all components valid", "dataset.combos", "")

# ------------------------------------------------------------------------------------------- inventory (not implemented)
INVENTORY = [
    dict(item="Sign-oriented features s_k x {TFI, OFI, QI, basis, cost, funding} (F11)", priority="P0/P1",
         dependency="primary signal s_k (labeling/strategy stage)", status="not_implemented (out of scope: targets/signals)"),
    dict(item="Flow sensitivity λ (rolling OLS of 1s mid returns on signed flow) (F6)", priority="P1",
         dependency="valid mid on 1s grid (book) + trades", status="not_implemented; mid grid available only where book valid"),
    dict(item="Basis persistence ΔB_j = a + φ B_{j-1}, half-life (F7)", priority="P1", dependency="1-min basis grid, >=100 pairs",
         status="not_implemented"),
    dict(item="Time-weighted quantiles of spread/depth; fraction of time depth below past quantile (F4/F5)", priority="P1",
         dependency="state history (ring)", status="not_implemented (min/max implemented)"),
    dict(item="Depth resilience / recovery episodes with censoring (F5/F12)", priority="P1/P2",
         dependency="continuous valid book, episode definitions from past", status="not_implemented"),
    dict(item="Amihud intraday proxy (F6)", priority="P2", dependency="1-min cells", status="not_implemented"),
    dict(item="Micro-price calibrated (Stoikov) (F12)", priority="P1/P2", dependency="calibration on train only", status="not_implemented"),
    dict(item="Hawkes / conditional intensities (F12)", priority="P2", dependency="intensity model", status="not_implemented"),
    dict(item="Volume-bucket absolute imbalance (VPIN-like) (F12)", priority="P2", dependency="volume buckets", status="not_implemented"),
    dict(item="Official Premium Index reconstruction (F12)", priority="P2", dependency="impact margin notional history, valid deep book",
         status="not_implemented"),
    dict(item="Dynamic impact / VAR (F12)", priority="P2", dependency="past-only estimation", status="not_implemented"),
    dict(item="Sign entropy / complexity (F12)", priority="P2", dependency="encoding choices", status="not_implemented"),
    dict(item="Learned regimes (F12)", priority="P2", dependency="trained model, online filtering", status="not_implemented"),
    dict(item="L2 tensor / OFI sequences (F12)", priority="P1", dependency="valid book + train normalisation",
         status="not_implemented (100ms grid of top-of-book is produced)"),
    dict(item="Market-clock time bars (secondary, idealised) (§3.2)", priority="P1", dependency="finalisation lag policy",
         status="not_implemented (market times kept as columns)"),
    dict(item="Trade-fragmenting bar variant (§7.1)", priority="P2", dependency="-", status="not_implemented (main convention: no fragmentation)"),
    dict(item="CUSUM on reconstructed mid / on 100ms-1s price grid (§7.5)", priority="P1", dependency="valid book for mid variant",
         status="not_implemented"),
]


def match(col: str):
    hits = [f for f in REGISTRY if f.regex().match(col)]
    return hits


def check_columns(cols):
    unreg = [c for c in cols if not match(c)]
    return unreg


def write_docs(path_md, path_csv):
    rows = [asdict(f) for f in REGISTRY]
    with open(path_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Feature registry", "", "Generated from `chdds/registry.py` (single source of truth). Patterns: `{w}` window, "
             "`{b}` band, `{k}` levels, `{q}` quantity, `{d}` notional, `{h}` horizon, `{g}` grid, `{r}` rank, `{l}` lag, `{n}` count.",
             "", "Clock AVAIL = availability (received_time); windows are (t-w, t] at the bar's eval time; trades also bounded by the "
             "bar's closing-trade cursor.", "",
             "| pattern | family | prio | formula | unit | sources | clock | params | validity | implementation | tests | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in REGISTRY:
        lines.append("| `" + f.pattern + "` | " + " | ".join([f.family, f.priority, f.formula.replace("|", "\\|"), f.unit,
                                                             f.sources, f.clock, f.params, f.validity.replace("|", "\\|"),
                                                             f.impl, f.tests, f.status]) + " |")
    lines += ["", "## Inventory of extensions not implemented (dependencies and stage)", "",
              "| item | priority | dependency | status |", "|---|---|---|---|"]
    for it in INVENTORY:
        lines.append(f"| {it['item']} | {it['priority']} | {it['dependency']} | {it['status']} |")
    open(path_md, "w").write("\n".join(lines) + "\n")

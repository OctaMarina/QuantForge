"""Single source of truth for the delivery status of every feature family (used by the registry
export and docs/STATUS.md).  Status vocabulary (task section 1):

* implemented_verified               - implemented and checked on real data and/or reference implementations
* implemented_untested_real          - implemented and unit-tested on documented layouts, never run on real data here
* unavailable_from_current_source    - the free sources do not provide the needed history
* not_implemented
"""

FAMILY_STATUS = {
    "bars": ("implemented_verified", "Binance trades archive", "verified on real ETHUSDT trades fixture (dedup of 70 "
             "duplicate rows, contiguous ids) + exact unit tests; archive itself not reachable from the build host"),
    "returns": ("implemented_verified", "trades", "real fixture + hand-computed tests"),
    "realized_vol": ("implemented_verified", "trades", "real fixture + tests"),
    "duration": ("implemented_verified", "trades", "real fixture"),
    "intensity": ("implemented_verified", "trades", "real fixture"),
    "flow_aggressor": ("implemented_verified", "trades (isBuyerMaker)", "mapping verified in Binance docs; real fixture"),
    "flow_tick_rule": ("implemented_verified", "trades", "hand tests incl. initialisation/state carry"),
    "lambda": ("implemented_verified", "trades", "equal to statsmodels OLS (coef, t); AFML primary text not accessible"),
    "structural_sadf": ("implemented_verified", "bars", "equal to brute-force OLS references; AFML primary text not accessible"),
    "structural_sdfc": ("implemented_verified", "bars", "equal to brute-force OLS reference; AFML text not accessible"),
    "structural_csw": ("implemented_verified", "bars", "equal to naive reference; AFML text not accessible"),
    "structural_bde": ("implemented_verified", "bars", "equal to statsmodels recursive_olsresiduals rcusum"),
    "premium": ("implemented_untested_real", "Coinbase candles + Binance klines + Coinbase USDT-USD", "sources unreachable here"),
    "trend": ("implemented_untested_real", "Binance 1d klines", "archive unreachable here"),
    "basis": ("implemented_untested_real", "Binance USD-M klines (last, mark, index, premium index)", "archive unreachable here"),
    "funding": ("implemented_untested_real", "Binance /fapi/v1/fundingRate", "REST unreachable here"),
    "etf_flows": ("implemented_untested_real", "Farside + vintages (live polls / Wayback)",
                  "historical values without publication evidence stay null in strict mode"),
    "fear_greed": ("implemented_untested_real", "Alternative.me + vintages", "strict null without publication evidence"),
    "implied_vol": ("implemented_untested_real", "Deribit DVOL", "history depth not verifiable from here"),
    "options_skew": ("unavailable_from_current_source", "Deribit tickers (live) / Tardis options_chain (1st of month)",
                     "no free continuous historical option chain; live collection implemented, untested on real data"),
    "liquidations": ("unavailable_from_current_source", "Binance forceOrder (sampled) / Bybit allLiquidation (live)",
                     "no free history; live collection + replay implemented, untested on real data"),
    "l2_book": ("unavailable_from_current_source", "Binance diff depth + snapshots (live) / Tardis",
                "no free continuous history; reconstruction verified on synthetic + real fixtures"),
    "l2_withdrawal_refill": ("unavailable_from_current_source", "L2", "as l2_book"),
    "l2_concentration": ("unavailable_from_current_source", "L2", "as l2_book"),
    "l2_flow_interaction": ("unavailable_from_current_source", "L2 + trades", "as l2_book"),
    "execution_cost": ("unavailable_from_current_source", "L2", "as l2_book"),
    "quotes_spread": ("implemented_verified", "bookTicker (live) / USD-M bookTicker archive / Tardis quotes",
                      "verified on real Binance BTCUSDT quotes fixture; spot has no free quote history"),
}

NOT_IMPLEMENTED = {
    "usd_notional_bars": "needs a contemporaneous causal per-trade USD conversion; refused by config validation",
    "native_kline_time_bar_fallback": "optional in the task; refused by config validation",
    "hac_lambda_variants": "only classical OLS standard errors are provided (no silent HAC substitution)",
    "csw_critical_boundaries": "b_alpha calibration not verified for this sampling; not emitted",
}


def apply(specs):
    for s in specs:
        st = FAMILY_STATUS.get(s.family)
        if st:
            s.status = st[0]
    return specs

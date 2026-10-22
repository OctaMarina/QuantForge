"""Configuration: defaults, YAML loading, CLI overrides, validation and a stable hash.

All defaults are *reasonable starting points*, not optimised values.  Each one is explained in
``docs/CONFIG.md``.  The configuration hash identifies a dataset build; the *bar-series* identity is
a separate hash that only covers the parameters which determine bar boundaries (instrument, bar
type/threshold, bar_origin, trade source, gap policy) so that changing the export period or
warm-up never changes boundaries.
"""
from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import yaml

from . import timeutil as tu

DEFAULTS: dict = {
    "instrument": {
        "exchange": "binance",
        "market_type": "spot",          # spot | perpetual
        "base": "BTC",
        "quote": "USDT",
        "symbol": None,                  # derived: BTCUSDT
        # Exact scaled-integer representation.  Must be >= the decimals the venue ever used.
        "price_decimals": None,          # None -> instrument registry default
        "qty_decimals": None,
        "contract": {"type": "linear", "contract_size": None, "contract_size_currency": None},
    },
    "period": {
        "start": "2024-06-01T00:00:00Z",   # inclusive, UTC
        "end": "2024-06-08T00:00:00Z",     # exclusive, UTC
        # Event bars accumulate from bar_origin; boundaries depend only on (origin, threshold, data).
        "bar_origin": "2024-05-25T00:00:00Z",
    },
    "warmup": {
        # Extra history downloaded before period.start for windows (filtered out of the export).
        "intraday_days": 31,             # trades/klines needed for 30d calendar returns etc.
        "daily_days": 200,               # daily external series (90d z-score + 30d trend + margin)
        "policy": "null_until_warm",     # missing warm-up -> local null, never row deletion
    },
    "bars": {
        "types": ["time", "volume", "notional"],
        "trade_source": "trades",        # trades (individual executions) | aggTrades
        "time": {"interval": "5min", "synthetic_fill": False},
        "volume": {"threshold": "100", "unit": "base"},             # 100 BTC per bar
        "notional": {"threshold": "10000000", "currency": "quote"},  # 10M USDT (quote notional)
        "usd_notional": {"enabled": False, "fx_source": "coinbase:USDT-USD", "threshold": "10000000"},
        "adaptive": {"enabled": False, "lookback_days": 30, "target_bars_per_day": 300},
        "gap_policy": "invalidate_and_restart_segment",   # | halt
        "max_ts_regression_us": 0,       # tolerated non-monotonic exchange timestamps (0 = none)
        "native_kline_fallback": False,  # time bars from native klines when trades are missing
    },
    "timing": {
        "replay_mode": "market_time",    # market_time | observed_availability
        "bar_finalization_delay": "0ms", # added to bar_end -> bar_available_at (market-time replay)
        "decision_latency": "0ms",       # bar_available_at -> decision_time
        "cross_stream_tie": "exclude_equal",  # other-stream obs with ts == decision are NOT known
        "strictness": "strict",          # strict | research
        "live_finalization_grace": "2s", # live: wait this long after interval end before closing
        "research_assumed_lags": {       # ONLY used in research mode; never point-in-time certified
            "etf_flows": "27h",          # session date 00:00 UTC + 27h = next day 03:00 UTC
            "fear_greed": "3h",
        },
    },
    "features": {
        "returns": {
            "calendar_horizons": ["5min", "1h", "4h", "1d", "7d", "30d"],
            "bar_horizons": [1, 5, 20],
            "ref_tolerance": "1min",     # max age of the as-of reference price at t-h
        },
        "realized_vol": {
            "grid": "1min",
            "windows": ["1h", "1d", "7d", "30d"],
            "year": "365d",
            "max_grid_staleness": "5min",
            "bar_windows": [20, 100],
        },
        "flow": {"min_trades_regression": 20, "bar_windows": [50]},
        "structural": {
            "enabled": True,
            "series": "log_close",
            "window_bars": 200,          # finite window W (rolling/windowed variants)
            "min_length": 30,            # tau: minimum effective sample length after lags
            "adf_lags": 1,
            "adf_constant": True,
            "adf_trend": False,
            "quantile_q": 0.95,
            "sdfc_trim": 0.15,
            "csw_window": 200,
            "bde_window": 200,
            "bde_lags": 1,               # BDE regression: r_t = c + phi_1 r_{t-1} + e
            "eval_every": 1,             # evaluate every k bars (carry-forward with age otherwise)
            "formula_gate": "task_spec", # task_spec | primary_verified_only  (see FORMULAS.md)
        },
        "premium": {
            "usd_venue": "coinbase",
            "context_assets": ["BTC", "ETH"],
            "fx_product": "USDT-USD",
            "minute_tolerance": "0s",    # FX/prices must be in the same closed minute
            "daily_min_coverage": 0.9,
            "z_window_days": 90,
            "z_min_obs": 60,
            "daily_finalization_delay": "0s",
        },
        "trend": {"days": 30, "ddof": 1},
        "etf": {"z_window_sessions": 60, "z_min_obs": 40, "max_age": "4d"},
        "dvol": {"currencies": ["BTC", "ETH"], "resolution": "60", "max_age": "5min"},
        "skew": {"target_days": 30, "delta": 0.25, "iv_source": "bid_ask_mid",
                 "max_quote_age": "60s", "max_snapshot_age": "5min", "currency": "BTC"},
        "basis": {"max_age": "2min"},
        "funding": {"max_age": "9h"},
        "liquidations": {"windows": ["5min", "1h"], "surprise_lookback": "30d", "surprise_min_obs": 500},
        "fear_greed": {"max_age": "36h"},
        "book": {
            "bands_bps": [10, 25, 50],
            "sizes_base": ["0.1", "1", "10"],
            "sizes_quote": ["10000", "100000", "1000000"],
            "taker_fee_bps": None,       # None -> total cost null; set explicitly to include a fee
            "max_book_age": "2s",
            "max_quote_age": "2s",
            "wall": {"bin_size": None, "max_distance_bps": 100,
                     "threshold_mode": "snapshot_median_multiple", "median_multiple": 5.0,
                     "quantile": 0.99, "absolute_notional": None,
                     "history_window": "1h", "sample_every": "10s", "min_history_samples": 60,
                     "track_every": "1s", "match_tolerance_bins": 1},
        },
    },
    "storage": {
        "root": "./data",
        "max_download_gb": 50,
        "request_timeout_s": 30,
        "retries": 5,
        "backoff_s": 1.0,
        "rate_limit_per_s": {"binance": 10, "coinbase": 5, "deribit": 10, "alternative_me": 1,
                              "farside": 0.2, "wayback": 1, "tardis": 2, "bybit": 5},
        "csv_export": False,
        "omit_all_null_columns": False,  # explicit option; recorded in manifest
        "qa_kline_crosscheck": True,     # internal 1m bars vs venue klines (exact-match report)
        "keep_normalized_trades": False,
    },
    "api_keys": {"tardis_env": "TARDIS_API_KEY"},  # read from environment only, never stored
    "live": {
        "streams": ["trades", "depth", "bookTicker", "forceOrder", "bybit_liquidations",
                    "deribit_dvol", "deribit_options", "fear_greed_poll", "farside_poll"],
        "depth_speed": "100ms",
        "rotate": "1h",
        "poll_every": {"fear_greed": "15min", "farside": "15min"},
        "deribit_option_expiry_window_days": [7, 60],
    },
}


class Cfg(dict):
    """dict with attribute access (read-only convenience)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(d: dict, dotted: str, value):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    try:
        value = yaml.safe_load(value) if isinstance(value, str) else value
    except yaml.YAMLError:
        pass
    cur[keys[-1]] = value


def load(path: str | None = None, overrides: list[str] | None = None) -> Cfg:
    user = {}
    if path:
        with open(path) as fh:
            user = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULTS, user)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} must be key.path=value")
        k, v = ov.split("=", 1)
        set_dotted(cfg, k.strip(), v.strip())
    validate(cfg)
    return Cfg(cfg)


def validate(cfg: dict):
    ins = cfg["instrument"]
    if ins["market_type"] not in ("spot", "perpetual"):
        raise ValueError("instrument.market_type must be spot or perpetual")
    if not ins.get("symbol"):
        ins["symbol"] = f"{ins['base']}{ins['quote']}".upper()
    p = cfg["period"]
    s, e, o = tu.parse_time(p["start"]), tu.parse_time(p["end"]), tu.parse_time(p["bar_origin"])
    if not s < e:
        raise ValueError("period.start must be < period.end")
    if o > s:
        raise ValueError("period.bar_origin must be <= period.start (event bars accumulate from the origin)")
    b = cfg["bars"]
    for t in b["types"]:
        if t not in ("time", "volume", "notional"):
            raise ValueError(f"unknown bar type {t}")
    if b["trade_source"] not in ("trades", "aggTrades"):
        raise ValueError("bars.trade_source must be trades or aggTrades")
    if b["gap_policy"] not in ("invalidate_and_restart_segment", "halt"):
        raise ValueError("bars.gap_policy must be invalidate_and_restart_segment or halt")
    if Decimal(str(b["volume"]["threshold"])) <= 0 or Decimal(str(b["notional"]["threshold"])) <= 0:
        raise ValueError("bar thresholds must be positive")
    if b["notional"]["currency"] not in ("quote",):
        raise ValueError("bars.notional.currency must be 'quote' (USD bars use bars.usd_notional)")
    tu.parse_duration(b["time"]["interval"])
    if b["usd_notional"]["enabled"]:
        raise ValueError("bars.usd_notional is NOT implemented: USD-notional bars need a contemporaneous causal "
                         "USD conversion per trade; use quote-notional bars (bars.notional) instead")
    if b["native_kline_fallback"]:
        raise ValueError("bars.native_kline_fallback is NOT implemented in this version")
    t = cfg["timing"]
    if t["replay_mode"] not in ("market_time", "observed_availability"):
        raise ValueError("timing.replay_mode must be market_time or observed_availability")
    if t["strictness"] not in ("strict", "research"):
        raise ValueError("timing.strictness must be strict or research")
    if t["cross_stream_tie"] not in ("exclude_equal", "include_equal"):
        raise ValueError("timing.cross_stream_tie must be exclude_equal or include_equal")
    st = cfg["features"]["structural"]
    if st["min_length"] <= st["adf_lags"] + 2:
        raise ValueError("structural.min_length must exceed adf_lags + 2")
    if st["window_bars"] < st["min_length"]:
        raise ValueError("structural.window_bars must be >= min_length")
    if not (0 < st["quantile_q"] < 1):
        raise ValueError("structural.quantile_q must be in (0,1)")
    if not (0 < st["sdfc_trim"] < 0.5):
        raise ValueError("structural.sdfc_trim must be in (0, 0.5)")
    if st["formula_gate"] not in ("task_spec", "primary_verified_only"):
        raise ValueError("structural.formula_gate must be task_spec or primary_verified_only")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


NON_SEMANTIC = {"storage": ("root", "rate_limit_per_s", "request_timeout_s", "retries", "backoff_s", "max_download_gb")}


def config_hash(cfg: dict) -> str:
    """Hash of everything that can change dataset CONTENT (paths, rate limits, live collection and API
    key settings are excluded, so moving the storage root does not create a new build identity)."""
    c = copy.deepcopy(dict(cfg))
    c.pop("api_keys", None)
    c.pop("live", None)
    for sec, keys in NON_SEMANTIC.items():
        for k in keys:
            c.get(sec, {}).pop(k, None)
    return hashlib.sha256(_canon(c).encode()).hexdigest()[:16]


def series_identity(cfg: dict, bar_type: str) -> dict:
    """Parameters that determine bar boundaries. Export period / warm-up are deliberately excluded."""
    ins = cfg["instrument"]
    b = cfg["bars"]
    ident = {
        "exchange": ins["exchange"], "market_type": ins["market_type"], "symbol": ins["symbol"],
        "bar_type": bar_type, "trade_source": b["trade_source"], "gap_policy": b["gap_policy"],
        "price_decimals": ins.get("price_decimals"), "qty_decimals": ins.get("qty_decimals"),
    }
    if bar_type == "time":
        ident["interval_us"] = tu.parse_duration(b["time"]["interval"])
        ident["synthetic_fill"] = bool(b["time"]["synthetic_fill"])
    else:
        ident["bar_origin_us"] = tu.parse_time(cfg["period"]["bar_origin"])
        ident["adaptive"] = b["adaptive"] if b["adaptive"]["enabled"] else False
        if bar_type == "volume":
            ident["threshold"] = str(Decimal(str(b["volume"]["threshold"])).normalize())
            ident["unit"] = b["volume"]["unit"]
        else:
            ident["threshold"] = str(Decimal(str(b["notional"]["threshold"])).normalize())
            ident["currency"] = ins["quote"]
    return ident


def series_id(cfg: dict, bar_type: str) -> str:
    ident = series_identity(cfg, bar_type)
    h = hashlib.sha256(_canon(ident).encode()).hexdigest()[:12]
    return f"{cfg['instrument']['exchange']}_{cfg['instrument']['market_type']}_{cfg['instrument']['symbol']}_{bar_type}_{h}"


def dump(cfg: dict, path: Path):
    path.write_text(yaml.safe_dump(json.loads(_canon(cfg)), sort_keys=False))

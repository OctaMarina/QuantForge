"""ETF flows (Farside), Fear & Greed, DVOL, vintages / delayed publication, skew, liquidations."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import mk_trades
from helpers import make_cfg, build_bars, ctx_for
from cryptods.features import external, options as opt, liquidations as liq
from cryptods.nulls import VALID, CODE
from cryptods.sources.publications import parse_farside, farside_daily, parse_fng, vintages_from_snapshots
from cryptods.sources.deribit import normalize_dvol, parse_option_name

D = 86_400_000_000
H = 3_600_000_000

# Synthetic page reproducing the VERIFIED Farside layout (title US$m, fund columns + Total, Fee row,
# parentheses for negatives, '-' for not-yet-launched funds, summary rows at the bottom).
FARSIDE_HTML = """<html><body><h1>Bitcoin ETF Flow (US$m)</h1><table>
<tr><th></th><th>IBIT</th><th>FBTC</th><th>GBTC</th><th>Total</th></tr>
<tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>1.50%</td><td></td></tr>
<tr><td>01 Oct 2026</td><td>195.6</td><td>(60.7)</td><td>-</td><td>134.9</td></tr>
<tr><td>02 Oct 2026</td><td>158.2</td><td>29.3</td><td>0.0</td><td>187.5</td></tr>
<tr><td>05 Oct 2026</td><td>1,069.9</td><td>(74.5)</td><td>(15.0)</td><td>980.4</td></tr>
<tr><td>Total</td><td>1,423.7</td><td>(105.9)</td><td>(15.0)</td><td>1,302.8</td></tr>
<tr><td>Average</td><td>474.6</td><td>(35.3)</td><td>(5.0)</td><td>434.3</td></tr>
<tr><td>Maximum</td><td>1,069.9</td><td>29.3</td><td>0.0</td><td>980.4</td></tr>
<tr><td>Minimum</td><td>158.2</td><td>(74.5)</td><td>(15.0)</td><td>134.9</td></tr>
</table></body></html>"""


def test_farside_parse_no_double_total_and_missing_not_zero():
    long = parse_farside(FARSIDE_HTML.encode(), "BTC")
    assert set(long.economic_date) == {"2026-10-01", "2026-10-02", "2026-10-05"}   # no weekend rows, no summary rows
    d = farside_daily(long).set_index("economic_date")
    assert d.loc["2026-10-01", "total_musd"] == pytest.approx(134.9)
    assert d.loc["2026-10-01", "sum_funds_musd"] == pytest.approx(195.6 - 60.7)   # '-' is not 0
    assert d.loc["2026-10-01", "n_funds_reported"] == 2
    assert d.loc["2026-10-05", "total_musd"] == pytest.approx(980.4)               # thousands separator
    assert d.total_musd.sum() == pytest.approx(134.9 + 187.5 + 980.4)              # 'Total' row not added


def _ctx(spec, tmp_path, decisions, tables, strict="strict"):
    t = mk_trades(spec, range(1, len(decisions) + 1), decisions, ["100"] * len(decisions), ["10"] * len(decisions))
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=1.0)
    return ctx_for(make_cfg(timing={"strictness": strict}), spec, bars, tables=tables)


def test_etf_strict_null_without_publication_evidence(spec, tmp_path):
    d = farside_daily(parse_farside(FARSIDE_HTML.encode(), "BTC"))
    d["available_at_us"] = np.nan
    t0 = pd.Timestamp("2026-10-06", tz="UTC").value // 1000
    ctx = _ctx(spec, tmp_path, [t0], {"etf": {"BTC": d}})
    external.compute(ctx)
    c = ctx.out["etf_btc_net_flow_total_usd"]
    assert c.reason[0] == CODE["unknown_publication_time"] and np.isnan(c.values[0])
    # research mode: an explicitly labelled assumed-lag variant exists, excluded from strict output
    ctx2 = _ctx(spec, tmp_path / "r", [t0 + 4 * H], {"etf": {"BTC": d}}, strict="research")
    external.compute(ctx2)
    r = ctx2.out["etf_btc_net_flow_total_usd__research_assumed_lag"]
    spec_r = next(s for s in ctx2.specs if s.name.endswith("__research_assumed_lag") and s.name.startswith("etf_btc_net"))
    assert r.values[0] == pytest.approx(980.4e6) and spec_r.strict_eligible is False


def test_delayed_publication_and_revision_vintages(spec, tmp_path):
    """Value for 2026-10-05 first seen at P1 = 2026-10-06 02:00; revised at P2 = 2026-10-07 01:00."""
    base = farside_daily(parse_farside(FARSIDE_HTML.encode(), "BTC"))
    early = base[base.economic_date < "2026-10-05"]
    p0 = pd.Timestamp("2026-10-05 01:00", tz="UTC").value // 1000
    p1 = pd.Timestamp("2026-10-06 02:00", tz="UTC").value // 1000
    p2 = pd.Timestamp("2026-10-07 01:00", tz="UTC").value // 1000
    rev = base.copy()
    rev.loc[rev.economic_date == "2026-10-05", "total_musd"] = 1000.0
    v = vintages_from_snapshots([(early, p0), (base, p1), (rev, p2)], ["total_musd", "sum_funds_musd", "n_funds_reported"],
                                ["economic_date"])
    assert (v[v.economic_date == "2026-10-05"].version.tolist() == [1, 2])
    dec = [p1 - 1, p1 + 1, p2 - 1, p2 + 1]
    ctx = _ctx(spec, tmp_path, dec, {"etf": {"BTC": v}})
    external.compute(ctx)
    c = ctx.out["etf_btc_net_flow_total_usd"]
    assert c.values[0] == pytest.approx(187.5e6)      # before publication: previous session's value
    assert c.values[1] == pytest.approx(980.4e6)      # first vintage
    assert c.values[2] == pytest.approx(980.4e6)      # revision not yet known
    assert c.values[3] == pytest.approx(1000.0e6)     # revision known
    age = ctx.out["etf_btc_net_flow_total_usd_age_s"]
    assert age.values[1] == pytest.approx(1e-6)


def test_fear_greed_parse_and_poll_vintage(spec, tmp_path):
    js = {"name": "Fear and Greed Index", "data": [
        {"value": "40", "value_classification": "Fear", "timestamp": "1760918400", "time_until_update": "3600"},
        {"value": "55", "value_classification": "Neutral", "timestamp": "1760832000"}], "metadata": {"error": None}}
    f = parse_fng(json.dumps(js).encode())
    assert list(f.value) == [55.0, 40.0] and f.economic_date.iloc[1] == "2025-10-20"
    rec = 1760918400_000_000 + 2 * H
    v = vintages_from_snapshots([(f, rec)], ["value"], ["economic_date"])
    ctx = _ctx(spec, tmp_path, [rec - 1, rec + 1], {"fng": v})
    external.compute(ctx)
    c = ctx.out["fear_greed_value"]
    assert c.reason[0] != VALID and c.values[1] == 40.0


def test_dvol_normalisation_and_availability(spec, tmp_path):
    rows = [[1_700_000_000_000, 50.1, 51, 50, 50.5], [1_700_000_060_000, 50.5, 52, 50, 51.0]]
    d = normalize_dvol(rows, "BTC", "60")
    assert (d.period_end_us - d.period_start_us == 60_000_000).all() and not d.unit_suspicious.any()
    t = 1_700_000_060_000_000 + 30_000_000      # inside the 2nd candle -> only the 1st is closed
    ctx = _ctx(spec, tmp_path, [t], {"dvol": {"BTC": d}})
    external.compute(ctx)
    assert ctx.out["dvol_btc"].values[0] == 50.5


def test_option_name_parse():
    cur, exp, k, typ = parse_option_name("BTC-27DEC24-100000-C")
    assert (cur, k, typ) == ("BTC", 100000.0, "C")
    assert pd.Timestamp(exp, unit="us", tz="UTC") == pd.Timestamp("2024-12-27 08:00", tz="UTC")


def _chain(s_us, expiries_days, put_iv=60.0, call_iv=50.0):
    rows = []
    for ed in expiries_days:
        exp = s_us + int(ed * D)
        for typ, deltas, iv0 in (("put", [-0.40, -0.30, -0.20, -0.10], put_iv), ("call", [0.10, 0.20, 0.30, 0.40], call_iv)):
            for dl in deltas:
                iv = iv0 + 10 * abs(dl) + ed / 10   # smile + term structure
                rows.append({"ts_us": s_us - 1_000_000, "instrument": f"{typ}{ed}{dl}", "expiry_us": exp, "strike": 1,
                             "type": typ, "bid_iv": iv - 0.5, "ask_iv": iv + 0.5, "mark_iv": iv, "delta": dl,
                             "underlying_price": 100, "bid_price": 0.01, "ask_price": 0.02, "state": "open"})
    return pd.DataFrame(rows)


def test_skew_25d_30d_manual():
    s = 1_700_000_000_000_000
    q = _chain(s, [20, 40])
    cfg = make_cfg()["features"]["skew"]
    r = opt.skew_at(q, s, cfg)
    assert r["ok"]
    iv_p = {e: 60 + 10 * 0.25 + e / 10 for e in (20, 40)}   # linear in delta between -0.30 and -0.20
    iv_c = {e: 50 + 10 * 0.25 + e / 10 for e in (20, 40)}
    tau = {e: e / 365 for e in (20, 40)}
    tt = 30 / 365

    def cm(iv):
        w1, w2 = iv[20] ** 2 * tau[20], iv[40] ** 2 * tau[40]
        return np.sqrt((w1 + (w2 - w1) * (tt - tau[20]) / (tau[40] - tau[20])) / tt)
    assert r["iv_put"] == pytest.approx(cm(iv_p)) and r["iv_call"] == pytest.approx(cm(iv_c))
    assert r["skew"] == pytest.approx(cm(iv_p) - cm(iv_c))
    # no extrapolation in tenor: only expiries beyond 30d -> null
    assert not opt.skew_at(_chain(s, [40, 60]), s, cfg)["ok"]
    # stale quotes are ignored
    stale = q.assign(ts_us=s - 10 * 60_000_000)
    assert not opt.skew_at(stale, s, cfg)["ok"]


def test_liquidation_mappings_coverage_and_zero_vs_unknown(spec, tmp_path):
    t0 = 1_700_000_000_000_000
    M = 60_000_000
    ev_bin = pd.DataFrame({"ts_us": [t0 + 10 * M, t0 + 11 * M], "received_at_us": [t0 + 10 * M + 5, t0 + 11 * M + 5],
                           "liquidated_position": ["long", "short"], "qty": [2.0, 1.0], "price": [100.0, 101.0]})
    tables = {"liquidations": {"binance": {"events": ev_bin, "uptime": [(t0, t0 + 30 * M)]},
                               "bybit": {"events": ev_bin.iloc[:0], "uptime": [(t0, t0 + 20 * M)]}}}
    dec = [t0 + 12 * M, t0 + 25 * M, t0 + 40 * M]
    ctx = _ctx(spec, tmp_path, dec, tables)
    liq.compute(ctx)
    lq = ctx.out["binance_reported_sampled_liq_long_qty_5min"]
    assert lq.values[0] == 2.0 and ctx.out["binance_reported_sampled_liq_short_qty_5min"].values[0] == 1.0
    assert lq.values[1] == 0.0                     # covered window without events -> observed zero
    assert lq.reason[2] == CODE["feed_gap"]        # collector down -> unknown, not zero
    assert ctx.out["binance_reported_sampled_liq_imbalance_5min"].values[0] == pytest.approx(1 / 3)
    assert ctx.out["binance_reported_sampled_liq_notional_exec_5min"].values[0] == pytest.approx(2 * 100 + 101)
    by = ctx.out["bybit_liq_long_qty_5min"]
    assert by.values[0] == 0.0 and by.reason[1] == CODE["feed_gap"]
    assert "bybit_liq_notional_at_bankruptcy_px_5min" in ctx.out


def test_live_liquidation_side_conventions(tmp_path):
    """Binance forceOrder S is the ORDER side (SELL closes a long); Bybit S is the POSITION side."""
    from cryptods.storage import JsonlGzWriter
    from cryptods.live import replay
    w = JsonlGzWriter(tmp_path, "binance_um_forceOrder", 3600_000_000)
    w.write(1_700_000_000_000_000, {"e": "forceOrder", "E": 1, "o": {"s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC",
            "q": "0.5", "p": "99", "ap": "100", "X": "FILLED", "l": "0.5", "z": "0.5", "T": 1_700_000_000_000}})
    w.close()
    w = JsonlGzWriter(tmp_path, "bybit_allLiquidation", 3600_000_000)
    w.write(1_700_000_000_000_000, {"topic": "allLiquidation.BTCUSDT", "type": "snapshot", "ts": 1,
                                    "data": [{"T": 1_700_000_000_000, "s": "BTCUSDT", "S": "Buy", "v": "0.2", "p": "95"}]})
    w.close()
    out = replay.liquidations(tmp_path)
    assert out["binance"]["events"].liquidated_position.iloc[0] == "long"
    assert out["binance"]["events"].price.iloc[0] == 100.0          # average execution price
    assert out["bybit"]["events"].liquidated_position.iloc[0] == "long"
    assert out["bybit"]["events"].price.iloc[0] == 95.0             # bankruptcy price

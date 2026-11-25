"""End-to-end absence-of-lookahead checks on REAL trades (Binance ETHUSDT 2020-08-14, fixture).

1. prefix invariance         - rows built on data up to a cutoff are identical after more data arrives
2. future perturbation       - drastic changes after the cutoff leave earlier rows unchanged
3. delayed publication       - see test_external.test_delayed_publication_and_revision_vintages
4. intrabar OHLC             - see test_crossmarket.test_intrabar_ohlc_not_visible
5. adaptive thresholds       - see test_bars.test_adaptive_threshold_uses_only_prior_complete_days
6. SDFC retrospection        - see test_structural.test_sdfc_not_assigned_to_candidate_date
7. replay / live parity      - live-log replay of the same trades == offline build (here)
8. staleness / asynchrony    - see test_crossmarket.test_asof_never_uses_future_even_if_closer /
                               test_missing_minute_never_filled and the DVOL check below
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from cryptods import config as cfgmod
from cryptods.pipeline import Pipeline
from cryptods.sample import load_nautilus_trades_csv

SAMPLE = ROOT / "configs" / "sample_real_ethusdt.yaml"
TYPES = ("time", "volume", "notional")
CUT = pd.Timestamp("2020-08-14 12:30", tz="UTC").value // 1000


def run(tmp, df, extra=()):
    cfg = cfgmod.load(str(SAMPLE), [f"storage.root={tmp}"] + list(extra))
    p = Pipeline(cfg, offline=True)
    res = p.build(allow_fetch=False, live_trades=df)
    out = Path(res["out_dir"])
    return {bt: pd.read_parquet(out / f"{bt}.parquet") for bt in TYPES}, res


@pytest.fixture(scope="module")
def trades(tmp_path_factory):
    real = ROOT / "tests" / "fixtures" / "real" / "binance_ethusdt-trades.csv"
    if not real.exists():
        pytest.skip("real fixtures missing: run scripts/fetch_real_fixtures.sh")
    cfg = cfgmod.load(str(SAMPLE))
    from cryptods.instruments import from_config
    df, _ = load_nautilus_trades_csv(real, from_config(cfg))
    return df


@pytest.fixture(scope="module")
def full(trades, tmp_path_factory):
    return run(tmp_path_factory.mktemp("full"), trades)


def assert_rows_identical(a: pd.DataFrame, b: pd.DataFrame, mask_b_rows=None):
    """Every row of ``a`` exists in ``b`` (bar_id) with identical values in every column."""
    a = a.set_index("bar_id")
    b = b.set_index("bar_id")
    missing = a.index.difference(b.index)
    assert len(missing) == 0, f"rows missing after more data: {list(missing)[:5]}"
    bb = b.loc[a.index, a.columns]
    pd.testing.assert_frame_equal(a, bb, check_exact=True, check_dtype=False)


def test_prefix_invariance(trades, full, tmp_path):
    cut, _ = run(tmp_path, trades[trades.ts_us < CUT])
    for bt in TYPES:
        assert len(cut[bt]) > 10
        assert_rows_identical(cut[bt], full[0][bt])


def test_future_perturbation(trades, full, tmp_path):
    pert = trades.copy()
    late = pert.ts_us >= CUT
    pert.loc[late, "price_i"] = pert.loc[late, "price_i"] * 3 // 2
    pert.loc[late, "qty_i"] = pert.loc[late, "qty_i"] * 10
    pert["notional_i"] = pert.price_i * pert.qty_i
    pert.loc[late, "aggressor"] = -pert.loc[late, "aggressor"]
    p, _ = run(tmp_path, pert)
    for bt in TYPES:
        f = full[0][bt]
        before = f[f.decision_time_us < CUT]
        assert_rows_identical(before, p[bt])
        after = f[f.decision_time_us > CUT + 3600_000_000]
        assert not after.close.equals(p[bt].set_index("bar_id").loc[after.bar_id, "close"].reset_index(drop=True))


def test_boundaries_stable_across_export_periods(trades, full, tmp_path):
    other, res = run(tmp_path, trades, ["period.start=2020-08-14T12:00:00Z"])
    for bt in TYPES:
        assert len(other[bt]) < len(full[0][bt])
        # identical bar ids and bar columns; features may differ only by warm-up (none here: same data)
        assert_rows_identical(other[bt], full[0][bt])


def test_origin_is_part_of_series_identity():
    cfg = cfgmod.load(str(SAMPLE))
    cfg2 = cfgmod.load(str(SAMPLE), ["period.bar_origin=2020-08-14T09:00:00Z"])
    cfg3 = cfgmod.load(str(SAMPLE), ["period.start=2020-08-14T11:00:00Z"])
    assert cfgmod.series_id(cfg, "volume") != cfgmod.series_id(cfg2, "volume")
    assert cfgmod.series_id(cfg, "volume") == cfgmod.series_id(cfg3, "volume")   # export period excluded
    assert cfgmod.series_id(cfg, "time") == cfgmod.series_id(cfg2, "time")       # time bars: epoch grid


def test_replay_live_parity(trades, full, tmp_path):
    """Write the same trades as live WebSocket logs (Binance trade payloads, microsecond timeUnit),
    replay them through the live path, and require identical datasets."""
    from cryptods.storage import JsonlGzWriter
    from cryptods.live import replay
    from cryptods.instruments import from_config
    w = JsonlGzWriter(tmp_path, "binance_spot_trades", 3600_000_000)
    ps, qs = 100, 10 ** 8
    for r in trades.itertuples(index=False):
        payload = {"e": "trade", "E": int(r.ts_us), "s": "ETHUSDT", "t": int(r.trade_id),
                   "p": f"{r.price_i / ps:.2f}", "q": f"{r.qty_i / qs:.8f}", "T": int(r.ts_us), "m": bool(r.aggressor < 0)}
        w.write(int(r.ts_us) + 150_000, payload)   # received 150 ms after the trade
    w.close()
    cfg = cfgmod.load(str(SAMPLE))
    live_df = replay.trades(tmp_path, from_config(cfg), futures=False)
    assert len(live_df) == len(trades)
    lv, _ = run(tmp_path / "out", live_df)
    for bt in TYPES:
        pd.testing.assert_frame_equal(lv[bt], full[0][bt], check_exact=True)


def test_manifest_and_quality_outputs(full):
    _, res = full
    out = Path(res["out_dir"])
    import json
    m = json.loads((out / "manifest.json").read_text())
    assert m["strictness"] == "strict" and m["replay_mode"] == "market_time"
    assert "no measured live latency" in m["temporal_evidence_level"]
    for bt in TYPES:
        q = pd.read_parquet(out / f"{bt}_quality.parquet")
        cov = pd.read_parquet(out / f"{bt}_coverage.parquet")
        assert {"feature", "reason", "first_row", "last_row", "n_rows"} <= set(q.columns)
        assert {"feature", "month", "valid_share", "internal_null_runs"} <= set(cov.columns)
        ds = full[0][bt]
        # requested families are kept even when entirely null (no silent omission)
        for c in ("etf_btc_net_flow_total_usd", "dvol_btc", "skew25d_30d_eth", "spread_bps",   # sample: skew on ETH
                  "bybit_liq_long_qty_5min", "premium_fx_coinbase_btc_1m"):
            assert c in ds.columns and ds[c].isna().all()
        # timing columns
        assert (ds.decision_time_us >= ds.bar_available_us).all() and (ds.bar_available_us >= ds.bar_end_us).all()


def test_observed_availability_mode_uses_receipt_times(trades, tmp_path):
    """Level-2 temporal evidence: with receipt logs, a bar is available when its closing trade was
    received (event bars) / at max(end + grace, last receipt) (time bars) - never earlier."""
    from cryptods.storage import JsonlGzWriter
    from cryptods.live import replay
    from cryptods.instruments import from_config
    w = JsonlGzWriter(tmp_path, "binance_spot_trades", 3600_000_000)
    sub = trades.iloc[:20000]
    for r in sub.itertuples(index=False):
        w.write(int(r.ts_us) + 250_000, {"e": "trade", "t": int(r.trade_id), "p": f"{r.price_i / 100:.2f}",
                                         "q": f"{r.qty_i / 1e8:.8f}", "T": int(r.ts_us), "m": bool(r.aggressor < 0)})
    w.close()
    cfg = cfgmod.load(str(SAMPLE))
    live_df = replay.trades(tmp_path, from_config(cfg), futures=False)
    obs, _ = run(tmp_path / "o", live_df, ["timing.replay_mode=observed_availability"])
    mkt, _ = run(tmp_path / "m", live_df)
    v_o, v_m = obs["volume"].set_index("bar_id"), mkt["volume"].set_index("bar_id")
    assert ((v_o.decision_time_us - v_m.decision_time_us) == 250_000).all()
    t_o = obs["time"]
    assert (t_o.bar_available_us >= t_o.bar_end_us + 2_000_000).all()      # live finalization grace
    # archives cannot claim observed availability
    with pytest.raises(ValueError):
        run(tmp_path / "x", trades.drop(columns=[c for c in trades.columns if c == "received_at_us"]),
            ["timing.replay_mode=observed_availability"])


def test_config_hash_ignores_storage_paths():
    a = cfgmod.load(str(SAMPLE), ["storage.root=/x"])
    b = cfgmod.load(str(SAMPLE), ["storage.root=/y", "storage.retries=1"])
    c = cfgmod.load(str(SAMPLE), ["bars.volume.threshold=1999"])
    assert cfgmod.config_hash(a) == cfgmod.config_hash(b) != cfgmod.config_hash(c)


# ------------------------------------------------------------------ information-driven bars on real trades
# 5 hours of trades: a 1000-tick warm-up (the default 10000 would leave ~7 bars)
IB_RUNS = ["bars.types=[tick_runs,volume_runs,notional_runs]", "bars.information.warmup_ticks=1000"]
IB_IMB = ["bars.types=[tick_imbalance,volume_imbalance,notional_imbalance]", "bars.information.warmup_ticks=1000",
          "bars.information.expected_ticks_bounds=[200,5000]", "bars.information.min_expected_imbalance_ratio=0.1"]


IB_BASE = ["bars.information.per_type={}", "bars.information.imbalance_threshold=afml",
           "bars.information.sign=tick_rule"]          # neutral starting point, independent of the sample config


def run_ib(tmp, df, sets):
    cfg = cfgmod.load(str(SAMPLE), [f"storage.root={tmp}"] + IB_BASE + sets)
    res = Pipeline(cfg, offline=True).build(allow_fetch=False, live_trades=df)
    out = Path(res["out_dir"])
    return {bt: pd.read_parquet(out / f"{bt}.parquet") for bt in cfg["bars"]["types"]
            if (out / f"{bt}.parquet").exists()}, res


IB_RMS = ["bars.types=[tick_imbalance,volume_imbalance,notional_imbalance]", "bars.information.sign=aggressor",
          "bars.information.imbalance_threshold=rms", "bars.information.rms_horizon_ticks=1000",
          "bars.information.warmup_ticks=1000"]


@pytest.mark.parametrize("sets", [IB_RUNS, IB_IMB, IB_RMS], ids=["runs_literal", "imbalance_stabilised", "imbalance_rms"])
def test_information_bars_prefix_and_future_perturbation(trades, tmp_path, sets):
    full, res = run_ib(tmp_path / "f", trades, sets)
    assert res["status"] == "complete"
    cut, _ = run_ib(tmp_path / "c", trades[trades.ts_us < CUT], sets)
    pert = trades.copy()
    late = pert.ts_us >= CUT
    pert.loc[late, "qty_i"] = pert.loc[late, "qty_i"] * 10
    pert["notional_i"] = pert.price_i * pert.qty_i
    pert.loc[late, "aggressor"] = -pert.loc[late, "aggressor"]
    p, _ = run_ib(tmp_path / "p", pert, sets)
    for bt in full:
        assert len(full[bt]) > 40, bt
        assert len(cut[bt]) > 15, bt
        assert_rows_identical(cut[bt], full[bt])
        before = full[bt][full[bt].decision_time_us < CUT]
        assert_rows_identical(before, p[bt])
        q = res["series"][bt]["information_bar_qa"]
        assert q["ticks_per_bar"]["median"] > 20 and "warning" not in q


def test_literal_imbalance_bars_collapse_on_real_trades_and_are_halted(trades, tmp_path):
    """Documented finding (docs/INFORMATION_BARS.md): literal AFML imbalance bars on real ETHUSDT trades
    collapse into ~1-tick bars; the default guard halts them instead of exporting them silently."""
    _, res = run_ib(tmp_path, trades, ["bars.types=[volume_imbalance,tick_runs]", "bars.information.warmup_ticks=1000"])
    assert res["status"] == "incomplete" and set(res["halted_series"]) == {"volume_imbalance"}
    assert "collapsed" in res["halted_series"]["volume_imbalance"]
    assert res["series"]["tick_runs"]["rows"] > 40

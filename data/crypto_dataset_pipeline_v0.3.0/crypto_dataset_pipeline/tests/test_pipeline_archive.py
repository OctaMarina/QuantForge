"""Archive path end-to-end with daily files in the documented Binance layout (written to the raw cache):
no reset at day boundaries, incremental update == one-shot build, missing-day policy (stop vs confirmed gap)."""
import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cryptods import config as cfgmod
from cryptods.pipeline import Pipeline
from cryptods.sources import binance as bn
from cryptods.storage import RawStore

D0 = date(2024, 3, 1)
DAY_US = 86_400_000_000


def day_csv(day_idx, start_id, n, rng):
    t0 = int(pd.Timestamp(D0 + timedelta(days=day_idx), tz="UTC").value // 1_000_000)  # ms
    ts = t0 + np.sort(rng.integers(0, 86_400_000, n))
    px = 60000 + np.cumsum(rng.integers(-3, 4, n)) * 0.01
    q = rng.integers(1, 2000, n) / 1e4
    ibm = rng.random(n) < 0.5
    lines = [f"{start_id + i},{px[i]:.2f},{q[i]:.8f},{px[i] * q[i]:.8f},{ts[i]},{'True' if ibm[i] else 'False'},True"
             for i in range(n)]
    return "\n".join(lines) + "\n"


def write_days(root, days, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    raw = RawStore(root)
    sid = 1
    texts = {}
    for k in range(3):
        texts[k] = (day_csv(k, sid, n, rng), sid)
        sid += n
    for k in days:
        d = D0 + timedelta(days=k)
        rel = bn.archive_rel("spot", "daily", "trades", "BTCUSDT", d)
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr(f"BTCUSDT-trades-{d}.csv", texts[k][0])
        raw.put("binance", rel, b.getvalue(), {"checksum_verified": True, "note": "synthetic test file"})


def cfg_for(root, end_day=3, extra=()):
    return cfgmod.load(None, [f"storage.root={root}", "period.start=2024-03-01T00:00:00Z",
                              f"period.end=2024-03-0{1 + end_day}T00:00:00Z", "period.bar_origin=2024-03-01T00:00:00Z",
                              "warmup.intraday_days=0", "warmup.daily_days=0", "bars.volume.threshold=5",
                              "bars.notional.threshold=300000", "bars.time.interval=1h",
                              "features.structural.window_bars=40", "features.structural.min_length=15",
                              "features.structural.csw_window=40", "features.structural.bde_window=40"] + list(extra))


def build(root, end_day=3, extra=()):
    p = Pipeline(cfg_for(root, end_day, extra), offline=True)
    res = p.build(allow_fetch=False)
    out = Path(res["out_dir"])
    return {bt: pd.read_parquet(out / f"{bt}.parquet") for bt in ("time", "volume", "notional")
            if (out / f"{bt}.parquet").exists()}, res


def confirm_404(root, k):
    """What `download` records after a 404 for a day older than the publication lag."""
    rel = bn.archive_rel("spot", "daily", "trades", "BTCUSDT", D0 + timedelta(days=k))
    path = RawStore(root).path("binance", rel + ".notfound.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": 404, "checked_at": "2024-03-10T00:00:00Z"}))


def test_no_reset_at_day_boundary_and_incremental_equals_oneshot(tmp_path):
    write_days(tmp_path / "a", [0, 1, 2])
    full, res = build(tmp_path / "a")
    v = full["volume"]
    day = lambda us: us // DAY_US
    crossing = v[day(v.first_trade_ts_us) != day(v.last_trade_ts_us)]
    assert len(crossing) >= 1                       # a bar spans midnight: accumulation not reset
    assert (full["time"].bar_status == "closed").sum() == 72 and len(full["time"]) == 72
    assert res["missing_trade_days"] == []
    # incremental: days 0-1 first, then day 2 arrives (same storage, restartable state)
    write_days(tmp_path / "b", [0, 1])
    part, _ = build(tmp_path / "b")
    write_days(tmp_path / "b", [2])
    inc, res2 = build(tmp_path / "b")
    for bt in full:
        pd.testing.assert_frame_equal(inc[bt], full[bt], check_exact=True)
        # rows published by the partial build are unchanged after the update
        a = part[bt].set_index("bar_id")
        pd.testing.assert_frame_equal(a, inc[bt].set_index("bar_id").loc[a.index, a.columns], check_exact=True)
    inv = res2["series"]["volume"]["prefix_invariance_vs_previous_build"]
    assert inv["cells_different"] == 0 and inv["rows_compared"] > 0 and inv["previous_rows_missing_now"] == 0


def test_unconfirmed_missing_day_stops_instead_of_creating_a_permanent_gap(tmp_path):
    """A day that is merely absent from the cache (download failed / not run) must not be frozen into the
    bar state as a gap: the build stops before it, and once the day is obtained the result equals a build
    that never had the problem."""
    write_days(tmp_path / "ref", [0, 1, 2])
    ref, _ = build(tmp_path / "ref")
    write_days(tmp_path / "s", [0, 2])              # day 1 not in the cache, no 404 evidence
    out, res = build(tmp_path / "s")
    assert res["status"] == "incomplete"
    assert res["stopped_before_day"]["day"] == "2024-03-02"
    assert res["stopped_before_day"]["reason"] == "not_in_cache"
    assert res["missing_trade_days"] == []
    v = out["volume"]
    assert set(v.segment_id) == {0} and res["series"]["volume"]["gaps"] == []
    assert (v.bar_end_us < pd.Timestamp("2024-03-02", tz="UTC").value // 1000).all()
    write_days(tmp_path / "s", [1])                 # the missing day is obtained later
    out2, res2 = build(tmp_path / "s")
    assert res2["status"] == "complete" and res2["stopped_before_day"] is None
    for bt in ref:
        pd.testing.assert_frame_equal(out2[bt], ref[bt], check_exact=True)


def test_missing_day_gap_segment_and_incomplete_time_bars(tmp_path):
    write_days(tmp_path, [0, 2])                    # day 1 missing in the archive ...
    confirm_404(tmp_path, 1)                        # ... and confirmed (HTTP 404 after the publication lag)
    out, res = build(tmp_path)
    assert res["missing_trade_days"] == [{"day": "2024-03-02", "reason": "confirmed_missing"}]
    assert res["status"] == "complete"
    t = out["time"]
    d1 = t[(t.bar_start_us >= DAY_US * (pd.Timestamp("2024-03-02", tz="UTC").value // 1000 // DAY_US))
           & (t.bar_start_us < (pd.Timestamp("2024-03-03", tz="UTC").value // 1000))]
    assert len(d1) == 24 and (d1.bar_status == "incomplete_interval").all() and d1.close.isna().all()
    assert d1.volume.isna().all()                  # unknown, not zero
    v = out["volume"]
    assert set(v.segment_id) == {0, 1}
    seg = res["series"]["volume"]["segments"]
    assert seg[-1]["reason"] == "trade_id_gap"
    assert res["series"]["volume"]["invalidated_bars"] == 1
    # returns never bridge the gap on the bar clock
    first_after = v[v.segment_id == 1].iloc[0]
    assert np.isnan(first_after.ret_bars_1)


def test_gap_policy_for_unconfirmed_days_and_stale_gap_warning(tmp_path):
    write_days(tmp_path, [0, 2])
    out, res = build(tmp_path, extra=["bars.on_unconfirmed_missing_day=gap"])
    assert res["missing_trade_days"] == [{"day": "2024-03-02", "reason": "not_in_cache"}]
    assert set(out["volume"].segment_id) == {0, 1}
    # the day appears later in the cache: the gap is now stale and the build says how to recover it
    write_days(tmp_path, [1])
    _, res2 = build(tmp_path, extra=["bars.on_unconfirmed_missing_day=gap"])
    stale = [w for w in res2["warnings"] if "now in the cache" in w]
    assert stale and "2024-03-02" in stale[0] and "rebuild" in stale[0]


def test_recent_404_is_not_a_confirmed_missing_day(tmp_path):
    from cryptods import timeutil as tu
    p = Pipeline(cfg_for(tmp_path), offline=True)
    today = tu.now_us() // DAY_US * DAY_US
    assert p._record_notfound("data/x/recent.zip", today - DAY_US) is False      # within the publication lag
    assert not p._notfound_confirmed("data/x/recent.zip", today - DAY_US)
    assert p._record_notfound("data/x/old.zip", today - 10 * DAY_US) is True
    assert p._notfound_confirmed("data/x/old.zip", today - 10 * DAY_US)


IB_SET = ["bars.types=[volume,tick_runs,tick_imbalance]", "bars.information.warmup_ticks=30",
          "bars.information.ewma_span_ticks=500", "bars.information.ewma_span_bars=10"]
# declared non-AFML stabilisers: with i.i.d. signs (these synthetic days) literal imbalance bars collapse
STAB = ["bars.information.expected_ticks_bounds=[20,40]", "bars.information.min_expected_imbalance_ratio=0.3"]


def test_information_bars_in_the_pipeline(tmp_path):
    """IB series are built by the same day loop (no reset at midnight; incremental == one-shot), carry their
    diagnostics columns and get shape QA in the manifest."""
    write_days(tmp_path / "a", [0, 1, 2])
    p = Pipeline(cfg_for(tmp_path / "a", extra=IB_SET + STAB), offline=True)
    res = p.build(allow_fetch=False)
    out = Path(res["out_dir"])
    assert res["status"] == "complete" and res["halted_series"] == {}
    for bt in ("tick_runs", "tick_imbalance"):
        ds = pd.read_parquet(out / f"{bt}.parquet")
        assert {"ib_theta", "ib_expected_ticks", "ib_expected_per_tick", "ib_warmup", "threshold_frozen"} <= set(ds.columns)
        assert len(ds) > 20 and ds.ib_warmup.iloc[0] and not ds.ib_warmup.iloc[1:].any()
        assert ds.threshold_frozen.iloc[1:].notna().all() and (ds.decision_time_us >= ds.bar_end_us).all()
        qa = res["series"][bt]["information_bar_qa"]
        assert qa["bars"] >= len(ds) and qa["ticks_per_bar"]["median"] > 2 and qa["state"]["initialised"]
        assert "ib_" not in " ".join(pd.read_parquet(out / "volume.parquet").columns)
    # incremental day-by-day build == one-shot build (state incl. EWMAs persisted between runs)
    write_days(tmp_path / "b", [0])
    Pipeline(cfg_for(tmp_path / "b", extra=IB_SET + STAB), offline=True).build(allow_fetch=False)
    write_days(tmp_path / "b", [1, 2])
    res_b = Pipeline(cfg_for(tmp_path / "b", extra=IB_SET + STAB), offline=True).build(allow_fetch=False)
    for bt in ("tick_runs", "tick_imbalance"):
        pd.testing.assert_frame_equal(pd.read_parquet(Path(res_b["out_dir"]) / f"{bt}.parquet"),
                                      pd.read_parquet(out / f"{bt}.parquet"), check_exact=True)


def test_collapse_guard_halts_only_the_runaway_series(tmp_path):
    """Literal AFML information bars on i.i.d. signs (these synthetic days) collapse: E0[|2P-1|] ~ 0.02 puts
    the tick-imbalance threshold below one tick, and the runs expectation E0[T] shrinks towards 2 ticks.
    The guard drops the offending day and halts those series only; with the guard disabled the collapse
    is reported by the shape QA."""
    write_days(tmp_path / "g", [0, 1, 2])
    p = Pipeline(cfg_for(tmp_path / "g", extra=IB_SET + ["bars.information.guard_max_bars_per_day=400"]), offline=True)
    res = p.build(allow_fetch=False)
    assert res["status"] == "incomplete"
    assert set(res["halted_series"]) == {"tick_imbalance", "tick_runs"}
    msg = res["halted_series"]["tick_imbalance"]
    assert "guard 400/day" in msg and "Bars before 2024-03-01 are kept" in msg
    assert res["series"]["tick_imbalance"]["rows"] == 0 and res["series"]["volume"]["rows"] > 0   # volume unaffected
    from cryptods import cli
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()) as err:
        code = cli._report(res)
    assert code == 3 and "HALTED tick_imbalance" in err.getvalue()
    write_days(tmp_path / "n", [0, 1, 2])
    res = Pipeline(cfg_for(tmp_path / "n", extra=IB_SET + ["bars.information.guard_max_bars_per_day=null"]),
                   offline=True).build(allow_fetch=False)
    qa = res["series"]["tick_imbalance"]["information_bar_qa"]
    assert res["status"] == "complete" and qa["ticks_per_bar"]["median"] == 1 and "collapsed" in qa["warning"]


def test_cli_exit_codes(tmp_path):
    import contextlib
    import io as _io
    from cryptods import cli
    err = _io.StringIO()
    with contextlib.redirect_stderr(err):
        assert cli.main(["build", "--offline", "--set", "instrument.symbol=ETHUSDT",
                         "--set", f"storage.root={tmp_path}"]) == 2
    assert "CONFIG ERROR" in err.getvalue()
    write_days(tmp_path, [0, 2])
    args = ["build", "--offline", "--set", f"storage.root={tmp_path}", "--set", "period.start=2024-03-01T00:00:00Z",
            "--set", "period.end=2024-03-04T00:00:00Z", "--set", "period.bar_origin=2024-03-01T00:00:00Z",
            "--set", "warmup.intraday_days=0", "--set", "warmup.daily_days=0", "--set", "bars.types=[volume]"]
    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()) as e2:
        assert cli.main(args) == 3
    assert "INCOMPLETE: trades for 2024-03-02" in e2.getvalue()
    confirm_404(tmp_path, 1)
    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
        assert cli.main(args) == 0


def test_days_inside_the_publication_lag_are_not_an_error(tmp_path, monkeypatch):
    """`update` near the present: the last day is not yet in the archive.  The build stops before it (no
    permanent gap) but reports complete_until_publication_lag instead of failing."""
    from cryptods import timeutil as tu
    write_days(tmp_path, [0, 1])
    monkeypatch.setattr(tu, "now_us", lambda: int(pd.Timestamp("2024-03-03T12:00", tz="UTC").value // 1000))
    _, res = build(tmp_path)
    assert res["stopped_before_day"]["day"] == "2024-03-03"
    assert res["status"] == "complete_until_publication_lag"


def test_rms_imbalance_and_per_type_parameters_in_the_pipeline(tmp_path):
    """One build, each information type with its own parameters; the RMS imbalance variant does not collapse
    on the i.i.d. synthetic days where the literal rule does (test_collapse_guard...)."""
    write_days(tmp_path / "a", [0, 1, 2])
    sets = ["bars.types=[tick_runs,tick_imbalance,volume_imbalance]", "bars.information.sign=aggressor",
            "bars.information.imbalance_threshold=rms", "bars.information.ewma_span_ticks=2000",
            "bars.information.per_type={tick_runs: {warmup_ticks: 150, ewma_span_bars: 20},"
            " tick_imbalance: {rms_horizon_ticks: 200, warmup_ticks: 2000},"
            " volume_imbalance: {rms_horizon_ticks: 100, warmup_ticks: 2000}}",
            "bars.information.guard_max_bars_per_day=400"]
    res = Pipeline(cfg_for(tmp_path / "a", extra=sets), offline=True).build(allow_fetch=False)
    assert res["status"] == "complete" and res["halted_series"] == {}
    out = Path(res["out_dir"])
    ds = {bt: pd.read_parquet(out / f"{bt}.parquet") for bt in ("tick_runs", "tick_imbalance", "volume_imbalance")}
    assert ds["tick_runs"].n_trades.iloc[0] == 150 and ds["tick_imbalance"].n_trades.iloc[0] == 2000
    for bt, h in (("tick_imbalance", 200), ("volume_imbalance", 100)):
        nw = ds[bt][~ds[bt].ib_warmup]
        assert (nw.ib_expected_ticks == h).all() and (nw.threshold_frozen > 0).all()
        qa = res["series"][bt]["information_bar_qa"]
        assert qa["threshold_rule"] == "rms" and "warning" not in qa and qa["share_bars_le_2_ticks"] < 0.05
        assert 0.7 * h < qa["mean_ticks_per_bar_excl_warmup"] < 2.5 * h
    assert res["series"]["tick_runs"]["information_bar_qa"]["threshold_rule"] == "afml"
    assert res["series"]["tick_runs"]["identity"]["information"]["warmup_ticks"] == 150

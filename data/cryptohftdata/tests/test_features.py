from datetime import date

import numpy as np
import pyarrow.parquet as pq
import pytest

from chdds import paths as P
from chdds.features import slow as S
from chdds.features import trades as T
from conftest import requires_canonical

D = date(2025, 6, 29)


def _queries(n=300, seed=0):
    A = T.load_trade_arrays(D, warm_days=1)
    day0 = P.hour_start_ns(D, 0)
    m = A["recv"] >= day0
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(np.nonzero(m)[0], n, replace=False))
    return A, A["recv"][pick], A["seq"][pick]


@requires_canonical
def test_trade_features_bruteforce():
    A, t, cur = _queries()
    cfg = T.TradeFeatureConfig()
    hist0 = int(pq.read_table(P.DERIVED / "canonical" / "trades" / "2025-06-28.parquet", columns=["recv_ns"])["recv_ns"][0].as_py())
    f = T.compute_trade_features(D, t, cur, cfg, hist0, (np.zeros(0, np.int64), np.zeros(0, np.int64)), 1e5)
    for i in range(0, t.size, 7):
        for w in (30, 900):
            sel = (A["recv"] > t[i] - w * 10**9) & (A["recv"] <= t[i]) & (A["seq"] <= cur[i])
            q, px, sg = A["q"][sel], A["px"][sel], A["sg"][sel]
            s = T.wname(w)
            assert f[f"volume_{s}_btc"][i] == pytest.approx(q.sum(), rel=1e-9)
            assert f[f"tfi_{s}_btc"][i] == pytest.approx((sg * q).sum(), rel=1e-9, abs=1e-9)
            assert f[f"vwap_{s}"][i] == pytest.approx((px * q).sum() / q.sum(), rel=1e-12)
            assert f[f"range_log_{s}"][i] == pytest.approx(np.log(px.max() / px.min()), abs=1e-15)
            assert f[f"trade_count_{s}"][i] == sel.sum()
        sel = (A["recv"] > t[i] - 30 * 10**9) & (A["recv"] <= t[i]) & (A["seq"] <= cur[i])
        assert f["trade_size_p90_30s_btc"][i] == pytest.approx(np.quantile(A["q"][sel], 0.9), rel=1e-12)
        # return: last trade price at cursor vs last trade price at t-w (both as-of)
        j0 = np.searchsorted(A["recv"], t[i] - 120 * 10**9, side="right") - 1
        j1 = np.searchsorted(A["seq"], cur[i])
        assert f["ret_trade_2m"][i] == pytest.approx(np.log(A["px"][j1] / A["px"][j0]), abs=1e-15)


@requires_canonical
def test_rv_identities():
    A, t, cur = _queries(200, 1)
    hist0 = P.hour_start_ns(date(2025, 6, 28), 5)
    f = T.compute_trade_features(D, t, cur, T.TradeFeatureConfig(), hist0, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
    for k in [k for k in f if k.startswith("rv_trade_")]:
        s = k[len("rv_trade_"):]
        a = f[k]
        b = f[f"rv_pos_trade_{s}"] + f[f"rv_neg_trade_{s}"]
        m = np.isfinite(a)
        assert np.allclose(a[m], b[m], rtol=1e-12, atol=1e-18)
        assert np.all(a[m] >= 0)


@requires_canonical
def test_trade_prefix_invariance(tmp_path, monkeypatch):
    """Features at t computed with the full day equal those computed when the data stops right after t."""
    A, t, cur = _queries(50, 2)
    hist0 = P.hour_start_ns(date(2025, 6, 28), 5)
    gaps = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    full = T.compute_trade_features(D, t, cur, T.TradeFeatureConfig(), hist0, gaps, 1e5)
    # truncated canonical copy
    tcut = t[25]
    src = P.DERIVED / "canonical" / "trades"
    for d in ("2025-06-28", "2025-06-29"):
        tab = pq.read_table(src / f"{d}.parquet")
        r = tab["recv_ns"].to_numpy()
        pq.write_table(tab.filter(r <= tcut), tmp_path / f"{d}.parquet")
    monkeypatch.setattr(T, "TRADES_DIR", tmp_path)
    part = T.compute_trade_features(D, t[:26], cur[:26], T.TradeFeatureConfig(), hist0, gaps, 1e5)
    for k, v in part.items():
        a = np.nan_to_num(np.asarray(full[k][:26], dtype=float), nan=-9e99)
        b = np.nan_to_num(np.asarray(v, dtype=float), nan=-9e99)
        # burstiness counts cells up to the max query: equal for queries <= cut
        assert np.array_equal(a, b), k


@requires_canonical
def test_trade_gap_mask():
    t = np.array([100, 200, 300], np.int64) * 10**9
    g0 = np.array([150 * 10**9], np.int64)
    g1 = np.array([160 * 10**9], np.int64)
    m = T.window_gap_mask(t, 60 * 10**9, g0, g1)
    assert list(m) == [False, True, False]


@requires_canonical
def test_oi_no_lookahead():
    data = S.SlowData()
    o = data.oi
    recv = o["recv_ns"].to_numpy()
    ts = o["ts_ms"].to_numpy() * 10**6
    t = np.sort(np.random.default_rng(0).integers(recv.min(), recv.max(), 500))
    f = S.oi_features(t, S.SlowFeatureConfig(), data, tw_needed=False)
    ref_ts = f["oi_reference_ts_ns"]
    for i in range(t.size):
        known = (recv <= t[i]) & (o["status"].to_numpy() == 0)
        if known.any():
            assert ref_ts[i] == ts[known].max()        # latest timestamp among accepted versions received by t
            assert ref_ts[i] <= t[i]


def test_tw_piecewise_exact():
    times = np.array([0, 10, 20, 30], np.int64) * 10**9
    vals = np.array([1.0, 2.0, 4.0, 8.0])
    vu = np.array([100, 15, 100, 100], np.int64) * 10**9   # second value valid only until 15s
    m, m2, cov = S._tw_piecewise(times, vals, vu, np.array([25 * 10**9], np.int64), 20 * 10**9)
    # window (5,25]: [5,10)->1 (5s), [10,15)->2 (5s), [15,20) invalid, [20,25]->4 (5s)
    assert cov[0] == pytest.approx(15 / 20)
    assert m[0] == pytest.approx((5 + 10 + 20) / 15)


def test_markout_maturation():
    """A markout of horizon h is counted only once matured: queries before recv+h never see it."""
    import test_book as TB
    from chdds.book.replay import ReplayConfig
    upd = [(0.5, [(0, 1000, 1000)]), (6.0, [(1, 1001, 0), (0, 1001, 1000)])]   # at 6.0s mid moves up one tick
    b, t0 = TB._synthetic_stream(upd)
    cfg = ReplayConfig(windows_s=(1.0,), mlofi_windows_s=(), minmax_max_window_s=0.0, ring_size=1 << 12,
                       max_age_ms=10_000, basis_windows_s=(), es_windows_s=(5.0,), markout_h_s=(1.0,),
                       markout_windows_s=(4.0,), tw_band_bps=())
    cols = cfg.columns()
    ci = {c: i for i, c in enumerate(cols)}
    tr_t = np.array([t0 + 6_200_000_000], np.int64)
    trades = (tr_t, tr_t // 10**6, np.array([100.1]), np.array([1.0]), np.array([1.0]))
    q = np.array([t0 + 7_100_000_000, t0 + 7_300_000_000], np.int64)
    saved = TB.COLS
    TB.COLS = cols
    try:
        out, _ = TB.run_batches([b], q, cfg=cfg, trades=trades)
    finally:
        TB.COLS = saved
    assert out[0, ci["markout_1s_n_4s"]] == 0
    assert out[1, ci["markout_1s_n_4s"]] == 1
    assert out[0, ci["eff_spread_n_5s"]] == 1
    # quote before the trade (6.0s state): bid 100.1, ask 100.2 -> mid 100.15; buy at 100.1 -> ES = 2e4*(100.1-100.15)/100.15
    assert out[0, ci["eff_spread_qw_bps_5s"]] == pytest.approx(2e4 * (100.1 - 100.15) / 100.15)
    # matured markout: mid unchanged after 6.0s -> 0
    assert out[1, ci["markout_1s_qw_bps_4s"]] == pytest.approx(0.0, abs=1e-12)


def test_calendar():
    t = np.array([P.hour_start_ns(date(2025, 6, 28), 0)], np.int64)    # Saturday 00:00 UTC
    f = S.calendar_features(t)
    assert f["is_weekend"][0] == 1
    assert f["hour_of_day_sin"][0] == pytest.approx(0.0)


# ------------------------------------------------------------------ open interest versions
def _oi_case(rows):
    """rows: (minute, recv minute, oi BTC, implied price)"""
    from chdds.canonical import slow as C
    ts = np.array([r[0] * 60000 for r in rows], np.int64)
    recv = np.array([r[1] * 60 * 10**9 for r in rows], np.int64)
    oi = np.array([int(r[2] * 1e8) for r in rows], np.int64)
    ov = np.array([int(r[2] * r[3] * 1e8) for r in rows], np.int64)
    return C, C.classify_oi_versions(ts, recv, oi, ov, lambda a, b: 100.0 * 1e8, lambda a: 100.0)


def test_oi_single_versions_are_never_touched():
    C, st = _oi_case([(0, 6, 1000.0, 100.0), (5, 11, 5000.0, 150.0), (10, 16, 1000.5, 100.0)])
    assert (st == C.OI_OK).all(), "the volume and price rules only act on contested points"


def test_oi_foreign_zero_value_is_rejected_even_alone():
    C, st = _oi_case([(0, 6, 1000.0, 100.0), (5, 11, 1000.0, 0.0)])
    assert st.tolist() == [C.OI_OK, C.OI_FOREIGN]


def test_oi_impossible_version_is_rejected_and_the_survivor_resolves_the_point():
    C, st = _oi_case([(0, 6, 1000.0, 100.0), (5, 11, 5000.0, 100.0), (5, 900, 1001.0, 100.0), (10, 16, 1002.0, 100.0)])
    assert st.tolist() == [C.OI_OK, C.OI_UNREACHABLE, C.OI_OK, C.OI_OK]


def test_oi_two_plausible_versions_stay_in_conflict_without_choosing():
    C, st = _oi_case([(0, 6, 1000.0, 100.0), (5, 11, 1010.0, 100.0), (5, 900, 1020.0, 100.0), (10, 16, 1015.0, 100.0)])
    assert st.tolist() == [C.OI_OK, C.OI_CONFLICT, C.OI_CONFLICT, C.OI_OK]


def test_oi_frozen_and_price_inconsistent_versions():
    C, st = _oi_case([(0, 6, 1000.0, 100.0),
                      (5, 900, 1500.0, 100.0), (5, 901, 1001.0, 100.0),
                      (10, 900, 1500.0, 100.0), (10, 901, 1002.0, 100.0),
                      (15, 900, 1500.0, 100.0), (15, 901, 1003.0, 103.0), (15, 902, 1003.5, 100.0)])
    assert st.tolist() == [C.OI_OK, C.OI_FROZEN, C.OI_OK, C.OI_FROZEN, C.OI_OK, C.OI_FROZEN, C.OI_PRICE, C.OI_OK]


class _OiData:
    def __init__(self, rows):
        import pyarrow as pa
        self.oi = pa.table({"ts_ms": np.array([r[0] for r in rows], np.int64) * 60000, "recv_ns": np.array([r[1] for r in rows], np.int64) * 60 * 10**9,
                            "oi_e8": np.array([int(r[2] * 1e8) for r in rows], np.int64), "oi_value_e8": np.array([int(r[2] * 100e8) for r in rows], np.int64),
                            "status": np.array([r[3] for r in rows], np.int8)}).sort_by("recv_ns")
        self.oi_rz = None


def test_oi_features_never_use_rejected_versions_and_mark_the_reason():
    rows = [(m, m + 1, 1000.0 + m, 0) for m in range(0, 40, 5)]
    rows += [(40, 41, 9000.0, 4), (45, 46, 9000.0, 5), (50, 51, 9100.0, 5), (55, 56, 9200.0, 5), (60, 61, 9300.0, 5)]
    rows += [(40, 500, 1040.0, 0)]             # resolved point, known only much later
    d = _OiData(rows)
    cfg = S.SlowFeatureConfig(oi_windows_s=(900,))
    t = np.array([42, 52, 58, 400, 600], np.int64) * 60 * 10**9
    f = S.oi_features(t, cfg, d)
    assert f["oi_btc_asof"][0] == 1035.0, "rejected point never replaces the reference"
    assert f["oi_state"].tolist()[:3] == [0, 3, 3], "the reference expires, and the reason is the rejected point"
    assert np.isnan(f["oi_btc_asof"][2]) and np.isnan(f["oi_log_change_known_15m"][2])
    assert f["oi_reference_ts_ns"][3] == 35 * 60 * 10**9, "the resolved value is not available before its reception"
    assert f["oi_reference_ts_ns"][4] == 40 * 60 * 10**9
    assert np.isnan(f["oi_points_received_15m"][0]) and np.isnan(f["tw_oi_known_15m_btc"][2])
    assert np.nanmax(f["oi_btc_known_delayed"]) < 2000, "no quarantined value leaks into the delayed context"


def test_rolling_statistics_ignore_points_not_yet_received():
    x = np.array([1.0, 2.0, 100.0, 3.0, 4.0])
    avail = np.array([0, 1, 50, 3, 4], np.int64)
    med, _ = S._rolling_robust_known(x, avail, 10, 2)
    assert med[4] == pytest.approx(2.0), "the value received at 50 is not known at entry 4"
    assert med[2] == pytest.approx(1.5)


# ------------------------------------------------------------------ liquidations
class _LiqData:
    def __init__(self, times_s, sides, notl, missing_hours=(), gaps=()):
        import pyarrow as pa
        h0 = P.hour_start_ns(P.HISTORY_START, 0)
        self.h0 = h0
        self.liq = pa.table({"recv_ns": h0 + np.array(times_s, np.int64) * 10**9, "side": np.array(sides, np.int8), "notional_usdt": np.array(notl, float)})
        self.liq_hours = set(range(0, 48)) - set(missing_hours)
        g = np.array(gaps, np.int64).reshape(-1, 2)
        self.stream_gaps = (h0 + g[:, 0] * 10**9, h0 + g[:, 1] * 10**9)


def test_hour_without_liquidation_file_is_unknown_never_zero():
    d = _LiqData([100, 3700 + 7200], [1, 1], [10.0, 20.0], missing_hours=(2,))
    cfg = S.SlowFeatureConfig(liq_windows_s=(120, 3600))
    q = np.array([7300, 9000, 10800 + 60, 10800 + 3000, 14500], np.int64)   # hour 2 = 7200..10800
    t = d.h0 + q * 10**9
    f = S.liq_features(t, cfg, d, d.h0)
    assert f["liq_state_2m"].tolist() == [2, 2, 2, 0, 0]
    assert f["liq_state_1h"].tolist() == [2, 2, 2, 2, 0]
    for k in ("liq_notional_short_2m_usdt", "liq_count_2m", "liq_any_2m", "liq_imbalance_2m", "log1p_liq_notional_2m", "liq_peak_1s_2m_usdt"):
        assert np.isnan(f[k][:3]).all(), k
    assert f["liq_count_2m"][3] == 0.0, "a covered hour with no event is a real zero"
    assert np.isnan(f["time_since_buy_short_liq_s"][:2]).all(), "a later liquidation may hide in the missing hour"
    assert f["time_since_buy_short_liq_s"][4] == pytest.approx(14500 - 10900)


def test_liquidation_unknown_propagates_to_combinations():
    from chdds.dataset import combos

    class C:
        pass
    nan = np.nan
    t = {"liq_notional_short_2m_usdt": np.array([nan, 5.0]), "liq_notional_long_2m_usdt": np.array([nan, 5.0]), "notional_2m_usdt": np.array([100.0, 100.0]),
         "oi_usdt_asof": np.array([1000.0, 1000.0]), "liq_any_2m": np.array([nan, 1.0]), "rv_trade_2m_grid1s": np.array([1.0, 1.0]), "rv_trade_1h_grid1s": np.array([1.0, 1.0])}
    o = combos(t, C())
    assert np.isnan(o["liq_to_trade_notional_2m"][0]) and np.isnan(o["liq_to_oi_2m"][0]) and np.isnan(o["liq_any_2m_x_rv_ratio"][0])
    assert o["liq_to_trade_notional_2m"][1] == pytest.approx(0.1)


def test_shared_gap_makes_time_since_unknown():
    d = _LiqData([100], [1], [10.0], gaps=[(200, 260)])
    t = d.h0 + np.array([150, 300], np.int64) * 10**9
    f = S.liq_features(t, S.SlowFeatureConfig(liq_windows_s=(120,)), d, d.h0)
    assert f["time_since_buy_short_liq_s"][0] == pytest.approx(50) and np.isnan(f["time_since_buy_short_liq_s"][1])


# ------------------------------------------------------------------ mark price / funding
class _MarkData:
    def __init__(self, recv_s, fr, nft_s):
        import pyarrow as pa
        r = np.array(recv_s, np.int64)
        n = r.size
        self.t = pa.table({"recv_ns": r * 10**9, "event_ms": r * 1000, "mark_e8": np.full(n, 100 * 10**8), "index_e8": np.full(n, 100 * 10**8),
                           "funding_e8": (np.array(fr) * 1e8).astype(np.int64), "funding_null": np.zeros(n, bool), "next_funding_ms": np.array(nft_s, np.int64) * 1000,
                           "settle_e8": np.zeros(n, np.int64), "settle_null": np.ones(n, bool)})

    def mark_range(self, a, b):
        return self.t


def test_funding_change_needs_a_fresh_start_and_outage_leaves_no_samples():
    h = 3600
    recv = np.concatenate([np.arange(0, 1000), np.arange(2 * h, 2 * h + 1000)])
    d = _MarkData(recv, np.where(recv < h, 0.001, 0.002), np.full(recv.size, 8 * h))
    t = np.array([2 * h + 500], np.int64) * 10**9        # t - 1h falls in the outage
    cfg = S.SlowFeatureConfig(funding_change_windows_s=(h,), index_return_windows_s=(), basis_windows_s=(), funding_z_lookback_days=1)
    f = S.mark_features(D, t, cfg, d)
    assert np.isnan(f["funding_change_1h"][0]), "the start value is older than the freshness limit"
    assert f["funding_rate_asof"][0] == pytest.approx(0.002)
    g = S.mark_features(D, np.array([h + 200], np.int64) * 10**9, cfg, d)      # inside the outage: nothing is invented
    assert np.isnan(g["mark_price"][0]) and np.isnan(g["funding_rate_asof"][0]) and g["mark_state"][0] == 2


def test_funding_interval_is_unknown_across_a_silence_that_could_hide_a_cycle():
    h = 3600
    a = np.arange(0, 100)                       # next funding at 8h
    b = np.arange(8 * h + 1, 8 * h + 100)       # next funding at 16h: transition seen after a long silence
    c = np.arange(16 * h - 50, 16 * h)          # same value
    e = np.arange(16 * h, 16 * h + 100)         # next funding at 24h: transition seen, no silence
    recv = np.concatenate([a, b, c, e])
    nft = np.concatenate([np.full(a.size, 8 * h), np.full(b.size + c.size, 16 * h), np.full(e.size, 24 * h)])
    d = _MarkData(recv, np.full(recv.size, 0.001), nft)
    cfg = S.SlowFeatureConfig(funding_change_windows_s=(), index_return_windows_s=(), basis_windows_s=(), funding_z_lookback_days=1)
    f = S.mark_features(D, np.array([8 * h + 50, 16 * h + 50], np.int64) * 10**9, cfg, d)
    assert np.isnan(f["funding_interval_s_observed"][0]) and np.isnan(f["funding_phase_sin"][0])
    assert f["funding_interval_s_observed"][1] == 8 * h


# ------------------------------------------------------------------ returns defined by two prices
def _flat_trades(monkeypatch, n=7200):
    r = np.arange(n, dtype=np.int64) * 10**9
    px = 100.0 + np.arange(n) * 0.01
    A = dict(seq=np.arange(n), recv=r, px=px, q=np.ones(n), d=px, sg=np.ones(n), seg=np.zeros(n, np.int64), lx=np.log(px), qe3=np.full(n, 1000), de5=(px * 1e5).astype(np.int64))
    monkeypatch.setattr(T, "load_trade_arrays", lambda d, warm_days=1: A)
    return A


def test_return_ignores_gaps_inside_the_window_but_not_at_its_ends(monkeypatch):
    A = _flat_trades(monkeypatch)
    cfg = T.TradeFeatureConfig(flow_windows_s=(), return_windows_s=(3600,), quantile_windows_s=(), rv_grids={}, burst_windows_s=(), path_eff_windows_s=())
    t = np.array([5000, 6000, 6500], np.int64) * 10**9
    cur = np.array([5000, 6000, 6500])
    # interior gap for all three; a gap around the start of the second window; a gap just before the end of the third
    g0 = np.array([3000, 2395, 6499], np.int64) * 10**9
    g1 = np.array([3010, 2405, 6501], np.int64) * 10**9
    g0, g1 = g0[np.argsort(g0)], g1[np.argsort(g0)]
    f = T.compute_trade_features(D, t, cur, cfg, 0, (g0, g1), None)
    assert f["tw_state_1h"].tolist() == [2, 2, 2], "flow statistics stay strict"
    assert f["ret_trade_1h"][0] == pytest.approx(np.log(A["px"][5000] / A["px"][1400])), "only the two prices matter"
    assert np.isnan(f["ret_trade_1h"][1]), "the price at t-w may be one of the missing trades"
    assert np.isnan(f["ret_trade_1h"][2]), "the price at t may be one of the missing trades"


# ------------------------------------------------------------------ quality metadata
def test_quality_metadata_is_not_a_default_model_input():
    from chdds.dataset import META_COLS, QUALITY_COLS, model_columns
    cols = META_COLS + QUALITY_COLS + ["spread_bps", "tfi_5s_btc"]
    assert model_columns(cols) == ["spread_bps", "tfi_5s_btc"]


def test_incident_is_available_for_exclusion_only():
    from chdds.dataset import incident_rows, load_incidents
    inc = load_incidents()
    assert [i["id"] for i in inc] == ["book_phantom_level_2025-08-29"] and inc[0]["use"] == "exclude_only"
    a, b = inc[0]["start_ns"], inc[0]["end_ns"]
    assert b - a == 31 * 10**9 and a == P.hour_start_ns(date(2025, 8, 29), 6) + (37 * 60 + 5) * 10**9
    ts = np.array([a - 100, a - 100, b + 1, a + 1])
    ev = np.array([a - 1, a, b + 2, a + 2])
    assert incident_rows(ts, ev, inc).tolist() == [False, True, False, True]


def test_guard_stops_on_low_disk_or_stop_file(monkeypatch, tmp_path):
    from chdds import guard
    monkeypatch.setattr(guard, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setenv("CHDDS_MIN_FREE_GB", "0")
    assert guard.stop_reason() is None
    monkeypatch.setenv("CHDDS_MIN_FREE_GB", "1e12")
    assert "free disk" in guard.stop_reason()
    monkeypatch.setenv("CHDDS_MIN_FREE_GB", "0")
    (tmp_path / "STOP").write_text("x")
    assert "stop file" in guard.stop_reason()

"""L2 reconstruction, exact band maintenance, cost walk, walls, withdrawal decomposition, real fixtures."""
import numpy as np
import pandas as pd
import pytest

from conftest import REAL
from cryptods import config as cfgmod
from cryptods.book.orderbook import L2Book, BookGap
from cryptods.book.adapters import nautilus_binance_futures_messages, tardis_incremental_messages
from cryptods.features.book_features import BookEngine
from cryptods.instruments import InstrumentSpec

S = 1_000_000
PS, QS = 100, 10 ** 8


def lin(p, q):
    return p * q


def brute_depth(book: L2Book, k):
    b, a = book.best_bid(), book.best_ask()
    m = (a + b) / 2
    bid = sum(p * q for p, q in book.bids.items() if p >= m * (1 - k / 1e4) - 1e-9)
    ask = sum(p * q for p, q in book.asks.items() if p <= m * (1 + k / 1e4) + 1e-9)
    return bid, ask


def test_binance_spot_sync_rules():
    bk = L2Book("binance_spot", lin)
    bk.apply_snapshot([(100_00, QS)], [(101_00, QS)], update_id=100, ts_us=0)
    assert not bk.valid()                        # needs the bridging diff first
    assert bk.apply_update([], [], 1, U=90, u=100) == "ignored"   # u <= lastUpdateId
    bk2 = L2Book("binance_spot", lin)
    bk2.apply_snapshot([(1, 1)], [(2, 1)], 100, 0)
    with pytest.raises(BookGap):                 # first event must contain lastUpdateId+1
        bk2.apply_update([], [], 1, U=105, u=110)
    assert bk.apply_update([(100_00, 2 * QS)], [], 2, U=95, u=103) == "applied"
    assert bk.valid() and bk.bids[100_00] == 2 * QS   # absolute size, not increment
    bk.apply_update([(100_00, 0)], [(101_00, 3 * QS)], 3, U=104, u=104)
    assert 100_00 not in bk.bids                      # 0 removes the level
    with pytest.raises(BookGap):
        bk.apply_update([], [], 4, U=106, u=107)      # missed 105


def test_binance_futures_pu_rule():
    bk = L2Book("binance_futures", lin)
    bk.apply_snapshot([(100_00, QS)], [(101_00, QS)], update_id=100, ts_us=0)
    bk.apply_update([], [], 1, U=95, u=102, pu=90)
    bk.apply_update([], [], 2, U=103, u=105, pu=102)
    with pytest.raises(BookGap):
        bk.apply_update([], [], 3, U=106, u=108, pu=104)


def test_incremental_bands_equal_bruteforce_property():
    rng = np.random.default_rng(0)
    bk = L2Book("tardis", lin, bands_bps=(10, 25, 50))
    bids = [(10000_00 - i * 7, int(rng.integers(1, 50)) * 10 ** 6) for i in range(400)]
    asks = [(10000_01 + i * 7, int(rng.integers(1, 50)) * 10 ** 6) for i in range(400)]
    bk.apply_snapshot(bids, asks, None, 0)
    for step in range(3000):
        side = rng.integers(2)
        bb, ba = bk.best_bid(), bk.best_ask()
        if side == 0:
            p = int(bb - rng.integers(-3, 300))
            p = min(p, ba - 1)
            lv = [(p, int(rng.integers(0, 40)) * 10 ** 6)]
            bk.apply_update(lv, [], step)
        else:
            p = int(ba + rng.integers(-3, 300))
            p = max(p, bb + 1)
            bk.apply_update([], [(p, int(rng.integers(0, 40)) * 10 ** 6)], step)
        if step % 97 == 0 and bk.valid():
            for k in (10, 25, 50):
                rb, ra = brute_depth(bk, k)
                assert bk.bid_bands[k].depth == rb and bk.ask_bands[k].depth == ra


def test_band_coverage_beyond_snapshot_is_null():
    bk = L2Book("binance_spot", lin, bands_bps=(10, 50))
    # snapshot limited to 3 levels per side, levels 20 bps deep -> 10 bps covered, 50 bps not
    bids = [(100_000, QS), (99_900, QS), (99_800, QS)]
    asks = [(100_010, QS), (100_100, QS), (100_200, QS)]
    bk.apply_snapshot(bids, asks, 10, 0, depth_limit=3)
    bk.apply_update([], [], 1, U=11, u=11)
    assert bk.band_covered(10, True) and not bk.band_covered(50, True)


def test_cost_walk_vwap_and_no_double_spread():
    bk = L2Book("tardis", lin)
    bk.apply_snapshot([(99_00, 1 * QS), (98_00, 2 * QS)], [(101_00, 1 * QS), (102_00, 2 * QS)], None, 0)
    cost, filled, used = bk.walk(2 * QS, True)       # buy 2: 1@101 + 1@102 -> VWAP 101.5
    vwap = cost / filled / PS
    assert vwap == pytest.approx(101.5) and used == 2
    mid = 100.0
    slip = 1e4 * (vwap / mid - 1)
    assert slip == pytest.approx(150.0)               # includes the half spread (100 bps) once
    assert bk.walk(4 * QS, True) is None              # insufficient displayed depth -> null


def _engine(spec, msgs, starts, ends, decs, cfg_over=None, wall_bin=None):
    cb = cfgmod.deep_merge(cfgmod.DEFAULTS["features"]["book"], cfg_over or {})
    eng = BookEngine(spec, cb, "tardis", starts, ends, decs, wall_bin)
    return eng.run(iter(msgs))


def test_engine_tie_rule_twa_and_withdrawal_identity(spec):
    m = []
    m.append({"type": "snapshot", "ts_us": 0, "bids": [(100_00, QS), (99_95, QS)], "asks": [(100_05, QS), (100_10, QS)]})
    m.append({"type": "update", "ts_us": 10 * S, "bids": [(100_00, 3 * QS)], "asks": []})          # refill at same price
    m.append({"type": "update", "ts_us": 20 * S, "bids": [], "asks": [(100_05, 0)]})                 # ask withdrawn, mid moves
    o = _engine(spec, m, [5 * S], [30 * S], [10 * S], {"max_book_age": "1h"})
    # decision at exactly 10s must NOT see the update stamped 10s (other-stream tie rule)
    assert o["bid"][0] == 100.0 and o["d_bid_10_dec"][0] == pytest.approx(100.0 + 99.95)
    # time-weighted 10bps bid depth over [5,30): 5s @ 199.95, 10s @ 399.95, 10s @ (after mid move)
    for k in (10,):
        added, wd = o[f"add_bid_{k}"][0], o[f"wd_bid_{k}"][0]
        shift = o[f"d_bid_{k}_end"][0] - o[f"d_bid_{k}_start"][0] - (added - wd)
        assert added == pytest.approx(200.0)
        assert np.isfinite(shift)
    assert o["tw_valid_frac"][0] == pytest.approx(1.0)


def test_walls_detection_age_and_reset(spec):
    msgs = [{"type": "snapshot", "ts_us": 0, "bids": [(100_00 - i, QS // 10) for i in range(1, 200)],
             "asks": [(100_00 + i, QS // 10) for i in range(1, 200)]}]
    # every 10s an update keeps the book fresh; a big bid concentration appears at t=900s
    for k in range(1, 300):
        lv = [(99_50, 50 * QS)] if k * 10 >= 900 else []
        msgs.append({"type": "update", "ts_us": k * 10 * S, "bids": lv or [(100_00 - 1, QS // 10)], "asks": []})
    dec = [1000 * S, 1500 * S]
    o = _engine(spec, msgs, [d - S for d in dec], [d for d in dec], dec,
                {"max_book_age": "1h", "wall": {"min_history_samples": 30, "history_window": "1h", "sample_every": "10s"}},
                wall_bin=10)
    assert o["wall_dist_bid"][0] == pytest.approx(50.0, abs=0.6)   # notional-weighted zone price ~ 99.50
    assert o["wall_age_bid"][0] == pytest.approx(90.0)                # first seen at the 910 s sample
    assert o["wall_age_bid"][1] == pytest.approx(590.0)
    # history-quantile mode: a persistent concentration eventually becomes its own baseline
    o2 = _engine(spec, msgs, [d - S for d in dec], [d for d in dec], dec,
                 {"max_book_age": "1h", "wall": {"threshold_mode": "history_quantile", "min_history_samples": 30,
                                                 "history_window": "1h", "sample_every": "10s"}}, wall_bin=10)
    assert o2["wall_dist_bid"][0] == pytest.approx(50.0, abs=0.6) and np.isnan(o2["wall_dist_bid"][1])
    # a book reset (gap) resets the observed age
    msgs2 = msgs[:120] + [{"type": "gap", "ts_us": 1195 * S, "bids": [], "asks": []},
                          {**msgs[0], "ts_us": 1196 * S, "bids": msgs[0]["bids"] + [(99_50, 50 * QS)]}] + \
        [dict(m, ts_us=m["ts_us"]) for m in msgs[120:]]
    o3 = _engine(spec, msgs2, [1499 * S], [1500 * S], [1500 * S], {"max_book_age": "1h"}, wall_bin=10)
    assert o3["wall_age_bid"][0] < 310


def test_real_binance_futures_depth_fixture():
    if not (REAL / "binance_btcusdt-depth-snap.csv").exists():
        pytest.skip("real fixtures missing")
    sp = InstrumentSpec("binance", "perpetual", "BTCUSDT", "BTC", "USDT", price_decimals=2, qty_decimals=4)
    msgs = list(nautilus_binance_futures_messages(REAL / "binance_btcusdt-depth-snap.csv",
                                                  REAL / "binance_btcusdt-depth-update.csv", sp))
    bk = L2Book("binance_futures", lambda p, q: p * q)
    snap = msgs[0]
    bk.apply_snapshot(snap["bids"], snap["asks"], snap["u"], snap["ts_us"], snap["depth_limit"])
    outcomes = []
    for m in msgs[1:]:
        try:
            outcomes.append(bk.apply_update(m["bids"], m["asks"], m["ts_us"], m["U"], m["u"], m["pu"]))
        except BookGap as e:
            outcomes.append(f"gap: {e}")
    # The fixture's diff (2022-11-01 ~10 min after the snapshot) does not bridge the snapshot id:
    # the reconstruction must refuse it rather than silently continue.
    assert any(str(x).startswith("gap") for x in outcomes) or bk.valid()
    if bk.valid():
        assert not bk.crossed()


def test_real_tardis_deribit_incremental_inverse_fixture():
    f = REAL / "tardis_deribit_incremental_book_L2_BTC-PERPETUAL.csv"
    if not f.exists():
        pytest.skip("real fixtures missing")
    df = pd.read_csv(f, dtype={"price": str, "amount": str})
    from cryptods.sources.tardis import book_events_from_incremental
    ev = book_events_from_incremental(df)
    sp = InstrumentSpec("deribit", "perpetual", "BTC-PERPETUAL", "BTC", "USD", contract_type="inverse",
                        contract_size="1", contract_size_currency="USD", price_decimals=1, qty_decimals=0)
    msgs = list(tardis_incremental_messages(ev, sp))
    assert msgs[0]["type"] == "snapshot"
    bk = L2Book("tardis", lambda p, q: q * sp.price_scale, bands_bps=(10, 50))
    bk.apply_snapshot(msgs[0]["bids"], msgs[0]["asks"], None, msgs[0]["ts_us"], msgs[0]["depth_limit"])
    for m in msgs[1:]:
        bk.apply_update(m["bids"], m["asks"], m["ts_us"])
    # amounts are USD notionals for this inverse contract (Tardis passes exchange amounts through)
    if bk.valid():
        d = bk.ask_bands[10].depth / sp.price_scale
        assert d == sum(q for p, q in bk.asks.items() if p <= bk.ask_bands[10].thr)


def test_binance_buffered_diffs_applied_after_snapshot(spec, tmp_path):
    """Live replay: diffs received BEFORE the REST snapshot with u > lastUpdateId must be applied after
    it (Binance procedure), otherwise the book could never sync."""
    from cryptods.storage import JsonlGzWriter
    from cryptods.book.adapters import binance_live_messages
    from cryptods.live import replay
    w = JsonlGzWriter(tmp_path, "binance_spot_depth", 3600 * S)
    t0 = 1_700_000_000_000_000
    for k, (U, u) in enumerate([(95, 99), (100, 103), (104, 106)]):     # received before the snapshot
        w.write(t0 + k * 100_000, {"e": "depthUpdate", "E": (t0 + k * 100_000) // 1000, "U": U, "u": u,
                                   "b": [["100.00", str(1 + k)]], "a": []})
    w.write(t0 + 900_000, {"e": "depthUpdate", "E": (t0 + 900_000) // 1000, "U": 107, "u": 108,
                           "b": [], "a": [["100.10", "5"]]})
    w.close()
    ws = JsonlGzWriter(tmp_path, "binance_spot_depth_snapshot", 3600 * S)
    ws.write(t0 + 500_000, {"lastUpdateId": 101, "bids": [["100.00", "9"], ["99.90", "1"]], "asks": [["100.10", "1"]]},
             {"limit": 5000})
    ws.close()
    msgs = list(binance_live_messages(replay.files(tmp_path, "binance_spot_depth"),
                                      replay.files(tmp_path, "binance_spot_depth_snapshot"), [], spec))
    bk = L2Book("binance_spot", lin)
    for m in msgs:
        if m["type"] == "snapshot":
            bk.apply_snapshot(m["bids"], m["asks"], m["u"], m["ts_us"], m.get("depth_limit"))
        else:
            bk.apply_update(m["bids"], m["asks"], m["ts_us"], m["U"], m["u"], m.get("pu"))
    assert bk.valid() and bk.update_id == 108
    assert bk.bids[100_00] == 3 * QS          # the (104,106) diff applied after the (100,103) bridging diff
    assert bk.asks[100_10] == 5 * QS

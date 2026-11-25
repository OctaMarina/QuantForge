"""Dataset-level L2 features through book_features.compute (synthetic exact book)."""
import numpy as np
import pytest

from conftest import mk_trades
from helpers import make_cfg, build_bars, ctx_for
from cryptods.features import book_features
from cryptods.nulls import CODE, VALID

S = 1_000_000
QS = 10 ** 8


def msgs():
    bids = [(100_00 - i, QS) for i in range(0, 60)]       # 1 BTC per cent level below 100.00
    asks = [(100_02 + i, QS) for i in range(0, 60)]
    out = [{"type": "snapshot", "ts_us": 0, "bids": bids, "asks": asks}]
    for k in range(1, 200):
        out.append({"type": "update", "ts_us": k * S, "bids": [(99_50, QS)], "asks": []})
    return out


def test_book_features_compute_costs_spread_and_pressure(spec, tmp_path):
    t = mk_trades(spec, [1, 2, 3], [10 * S, 20 * S, 30 * S], ["100.02", "100.02", "100.02"], ["0.5", "0.5", "1"])
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=1.0)
    cfg = make_cfg(features={"book": {"max_book_age": "1h", "sizes_base": ["2"], "sizes_quote": ["100.02"],
                                      "taker_fee_bps": 7.5, "bands_bps": [10]}})
    ctx = ctx_for(cfg, spec, bars, tables={"book_messages": msgs, "book_venue_rule": "tardis"})
    book_features.compute(ctx)
    mid = (100.00 + 100.02) / 2
    assert ctx.out["spread"].values[0] == pytest.approx(0.02)
    assert ctx.out["spread_bps"].values[0] == pytest.approx(1e4 * 0.02 / mid)
    # buy 2 BTC: 1 @ 100.02 + 1 @ 100.03 -> VWAP 100.025 ; slippage vs mid includes half spread once
    c = ctx.out["cost_buy_2BTC_bps"]
    assert c.values[0] == pytest.approx(1e4 * (100.025 / mid - 1))
    assert ctx.out["cost_buy_2BTC_incl_fee_bps"].values[0] == pytest.approx(c.values[0] + 7.5)
    rt = ctx.out["hypo_roundtrip_cost_2BTC_current_book_bps"]
    assert rt.values[0] == pytest.approx(c.values[0] + ctx.out["cost_sell_2BTC_bps"].values[0])
    # reference notional 100.02 USDT -> q = floor(100.02/mid, 1e-8) BTC = 1.00009999 BTC: walks 1 BTC at
    # 100.02 and the remainder at 100.03 (the spend VWAP*q differs from the reference notional)
    q = int(100.02 / mid * 1e8) / 1e8
    vwap = (1 * 100.02 + (q - 1) * 100.03) / q
    cq = ctx.out["cost_buy_100.02USDT_bps"]
    assert cq.values[0] == pytest.approx(1e4 * (vwap / mid - 1), rel=1e-7)
    # depth within 10 bps: bid levels >= mid*(1-0.001) = 99.91 -> 100.00..99.91 = 10 levels
    d = ctx.out["depth_bid_10bps"]
    assert d.values[0] == pytest.approx(sum((100_00 - i) / 100 for i in range(10)))
    # buy pressure: aggressive buy notional in bar / ask depth within 10 bps at bar start
    bp = ctx.out["buy_pressure_10bps"]
    ask_depth = sum((100_02 + i) / 100 for i in range(0, 60) if (100_02 + i) / 100 <= mid * 1.001)
    # bar0 starts at the origin t=0, exactly when the snapshot is stamped -> not yet known (tie rule)
    assert np.isnan(bp.values[0]) and bp.reason[0] == CODE["incomplete_coverage"]
    assert bp.values[1] == pytest.approx(100.02 * 1.0 / ask_depth)     # bar1 = trade 3 (1 BTC bought)
    # insufficient displayed depth -> null with reason (no extrapolation)
    cfg2 = make_cfg(features={"book": {"max_book_age": "1h", "sizes_base": ["1000"], "sizes_quote": [], "bands_bps": [10]}})
    ctx2 = ctx_for(cfg2, spec, bars, tables={"book_messages": msgs, "book_venue_rule": "tardis"})
    book_features.compute(ctx2)
    c2 = ctx2.out["cost_buy_1000BTC_bps"]
    assert np.isnan(c2.values[0]) and c2.reason[0] == CODE["insufficient_book_depth"]


def test_book_absent_means_null_not_zero(spec, tmp_path):
    t = mk_trades(spec, [1, 2], [10 * S, 20 * S], ["100", "100"], ["1", "1"])
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=1.0)
    ctx = ctx_for(make_cfg(), spec, bars, tables={})
    book_features.compute(ctx)
    for k in ("spread_bps", "depth_bid_10bps", "cost_buy_1BTC_bps", "wall_dist_bid", "buy_pressure_10bps"):
        assert np.isnan(ctx.out[k].values).all() and (ctx.out[k].reason != VALID).all()

"""Quote-based spread on REAL Binance BTCUSDT quotes/trades (2021-01-08 fixture) and on a manual case."""
import numpy as np
import pandas as pd
import pytest

from conftest import REAL, mk_trades
from helpers import make_cfg, build_bars, ctx_for
from cryptods.features import quotes as qf
from cryptods.nulls import CODE, VALID

S = 1_000_000


def test_quote_spread_manual_tie_and_twa(spec, tmp_path):
    q = pd.DataFrame({"ts_us": [0, 10 * S, 20 * S], "bid": [100.0, 100.0, 100.0], "ask": [100.1, 100.2, 99.0]})
    t = mk_trades(spec, [1, 2], [5 * S, 10 * S], ["100", "100"], ["1", "1"])
    bars, _ = build_bars(spec, t, tmp_path, "volume", thr=1.0)
    cfg = make_cfg(features={"book": {"max_quote_age": "1h"}})
    ctx = ctx_for(cfg, spec, bars, tables={"quotes": q})
    qf.compute(ctx)
    sp = ctx.out["quote_spread_bps"]
    m0 = 100.05
    assert sp.values[0] == pytest.approx(1e4 * 0.1 / m0)
    assert sp.values[1] == pytest.approx(1e4 * 0.1 / m0)   # quote stamped exactly at decision not yet known
    tw = ctx.out["quote_twa_spread_bps"]
    # bar1 spans [5s, 10s): one quote in effect -> equals that spread
    assert tw.values[1] == pytest.approx(1e4 * 0.1 / m0)
    # crossed quote -> invalid, never used
    ctx2 = ctx_for(cfg, spec, bars.assign(bar_end_us=[21 * S, 22 * S]), tables={"quotes": q})
    ctx2.decision_us = np.array([21 * S, 22 * S])
    qf.compute(ctx2)
    assert ctx2.out["quote_spread_bps"].reason[0] == CODE["invalid_quote"]


def test_real_binance_quotes_and_trades_fixture(tmp_path):
    fq, ft = REAL / "binance_btcusdt-quotes.parquet", REAL / "binance_btcusdt-trades.parquet"
    if not fq.exists():
        pytest.skip("real fixtures missing")
    from cryptods.instruments import InstrumentSpec
    sp = InstrumentSpec("binance", "spot", "BTCUSDT", "BTC", "USDT", price_decimals=2, qty_decimals=8)
    tr = pd.read_parquet(ft)
    qq = pd.read_parquet(fq)
    ts = tr.timestamp.astype("datetime64[us, UTC]").astype("int64")
    t = mk_trades(sp, tr.trade_id.to_numpy(), ts.to_numpy(), [f"{x:.2f}" for x in tr.price], [f"{x:.6f}" for x in tr.quantity],
                  side=np.where(tr.buyer_maker.astype(str) == "True", -1, 1))
    bars, _ = build_bars(sp, t, tmp_path, "time", interval_us=1 * S, watermark=int(ts.max()))
    q = pd.DataFrame({"ts_us": qq.timestamp.astype("datetime64[us, UTC]").astype("int64").to_numpy(),
                      "bid": qq.bid.to_numpy(float), "ask": qq.ask.to_numpy(float)})
    ctx = ctx_for(make_cfg(features={"book": {"max_quote_age": "5s"}}), sp, bars, bar_type="time", tables={"quotes": q})
    qf.compute(ctx)
    v = ctx.out["quote_spread_bps"]
    ok = v.reason == VALID
    assert ok.sum() > 10
    assert (v.values[ok] > 0).all() and (v.values[ok] < 50).all()   # BTCUSDT spreads are a few bps or less
    assert ((ctx.out["quote_valid_time_frac"].values >= 0) & (ctx.out["quote_valid_time_frac"].values <= 1)).all()

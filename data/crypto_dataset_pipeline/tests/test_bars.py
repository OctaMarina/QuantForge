"""Bar construction: exactness, thresholds, ordering, restart/batch invariance, gaps, precision."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from conftest import mk_trades, random_trades
from cryptods.bars import BarBuilder, bars_to_float, STATUS_GAP_INVALIDATED, STATUS_INCOMPLETE, STATUS_EMPTY
from cryptods.numeric import scaled_int, PrecisionError
from cryptods.sources.binance import dedupe_and_check, DuplicateConflict, normalize_trades_table

S = 1_000_000


def vb(spec, tmp_path, thr, name="v", **kw):
    return BarBuilder(name, "volume", spec, tmp_path, threshold_i=int(thr * spec.qty_scale), **kw)


def test_time_bar_ohlc_half_open(spec, tmp_path):
    # interval 60s; a trade at exactly 60s belongs to the SECOND bar ([t, t+D))
    t = mk_trades(spec, [1, 2, 3, 4, 5], [0, 10 * S, 59 * S + 999_999, 60 * S, 61 * S],
                  ["100", "105", "99", "101", "102"], ["1", "2", "3", "4", "5"])
    b = BarBuilder("t", "time", spec, tmp_path, interval_us=60 * S)
    b.start_series_at(0)
    out = bars_to_float(b.process(t, watermark_us=120 * S), spec)
    r0, r1 = out.iloc[0], out.iloc[1]
    assert (r0.bar_start_us, r0.bar_end_us) == (0, 60 * S)
    assert (r0.open, r0.high, r0.low, r0.close, r0.volume, r0.n_trades) == (100, 105, 99, 99, 6, 3)
    assert (r1.open, r1.close, r1.volume, r1.n_trades) == (101, 102, 9, 2)
    assert r0.notional == pytest.approx(100 * 1 + 105 * 2 + 99 * 3)


def test_empty_interval_is_zero_volume_null_ohlc(spec, tmp_path):
    t = mk_trades(spec, [1, 2], [5 * S, 185 * S], ["100", "101"], ["1", "1"])
    b = BarBuilder("t", "time", spec, tmp_path, interval_us=60 * S)
    b.start_series_at(0)
    out = bars_to_float(b.process(t, watermark_us=240 * S), spec)
    assert list(out.bar_start_us) == [0, 60 * S, 120 * S, 180 * S]
    mid = out.iloc[1]
    assert mid.status == STATUS_EMPTY and mid.complete == 1
    assert mid.volume == 0 and mid.n_trades == 0 and np.isnan(mid.open) and np.isnan(mid.close)


def test_volume_threshold_crossing_trade_included(spec, tmp_path):
    t = mk_trades(spec, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], ["10"] * 5, ["0.4", "0.5", "0.2", "2.5", "0.1"])
    b = vb(spec, tmp_path, 1.0)
    out = bars_to_float(b.process(t), spec)
    # bar0: 0.4+0.5+0.2 = 1.1 >= 1 (closed by trade 3, overshoot kept); bar1: 2.5 (single huge trade)
    assert list(out.last_trade_id) == [3, 4]
    assert list(out.volume) == [pytest.approx(1.1), pytest.approx(2.5)]
    part = b.partial()
    assert list(part.first_trade_id) == [5] and part.partial.all()


def test_exact_threshold_without_float_error(spec, tmp_path):
    # 0.7 + 0.1 == 0.8 exactly in scaled ints; in float 0.7+0.1 = 0.7999999999999999 < 0.8
    assert 0.7 + 0.1 < 0.8
    t = mk_trades(spec, [1, 2, 3], [1, 2, 3], ["1", "1", "1"], ["0.7", "0.1", "0.3"])
    out = vb(spec, tmp_path, 0.8).process(t)
    assert list(out.last_trade_id) == [2]


def test_notional_bars_and_conservation(spec, tmp_path):
    t = random_trades(spec, 20000, seed=3)
    thr = 50_000 * spec.notional_scale
    b = BarBuilder("n", "notional", spec, tmp_path, threshold_i=thr)
    out = b.process(t)
    part = b.partial()
    assert (out.notional_i >= thr).all()
    # every closed bar minus its last trade is below threshold (closes at FIRST crossing trade)
    last_notional = t.set_index("trade_id").loc[out.last_trade_id, "notional_i"].to_numpy()
    assert (out.notional_i.to_numpy() - last_notional < thr).all()
    tot = int(out.notional_i.sum()) + int(part.notional_i.sum())
    assert tot == int(t.notional_i.sum())
    assert int(out.volume_i.sum()) + int(part.volume_i.sum()) == int(t.qty_i.sum())
    assert int(out.n_trades.sum()) + int(part.n_trades.sum()) == len(t)
    # contiguous, non-overlapping trade coverage
    assert (out.first_trade_id.to_numpy()[1:] == out.last_trade_id.to_numpy()[:-1] + 1).all()


def test_same_timestamp_ordering_and_unique_ids(spec, tmp_path):
    t = mk_trades(spec, [10, 11, 12, 13], [5, 5, 5, 5], ["1", "2", "3", "4"], ["1", "1", "1", "1"])
    out = vb(spec, tmp_path, 1.0).process(t.sample(frac=1, random_state=1))  # input order shuffled
    assert list(out.last_trade_id) == [10, 11, 12, 13]
    assert out.bar_seq.is_unique and (out.bar_end_us == 5).all()
    assert list(out.close_i) == [100, 200, 300, 400]


@pytest.mark.parametrize("bar_type", ["time", "volume", "notional"])
def test_batch_and_restart_invariance(spec, tmp_path, bar_type):
    t = random_trades(spec, 30000, seed=7)
    kw = dict(interval_us=60 * S) if bar_type == "time" else dict(
        threshold_i=(2 * spec.qty_scale if bar_type == "volume" else 40_000 * spec.notional_scale))
    wm = int(t.ts_us.max())
    a = BarBuilder("a", bar_type, spec, tmp_path / "a", **kw)
    full = a.process(t, watermark_us=wm)
    # random batches with a restart (new builder object loading the persisted state) in between
    rng = np.random.default_rng(1)
    cuts = np.sort(rng.choice(np.arange(1, len(t)), 12, replace=False))
    bounds = [0] + list(cuts) + [len(t)]
    pieces = [t.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    outs = []
    b = BarBuilder("b", bar_type, spec, tmp_path / "b", **kw)
    for i, p in enumerate(pieces):
        if i == 6:
            b.save()
            b = BarBuilder("b", bar_type, spec, tmp_path / "b", **kw)  # restart from disk
        outs.append(b.process(p, watermark_us=wm if i == len(pieces) - 1 else -1))
    inc = pd.concat(outs, ignore_index=True)
    cols = [c for c in full.columns if c not in ("series_id",)]
    pd.testing.assert_frame_equal(full[cols].reset_index(drop=True), inc[cols].reset_index(drop=True), check_exact=True)
    # overlapping re-feed (already processed trades are skipped)
    again = b.process(t.iloc[:5000])
    assert len(again) == 0


def test_gap_invalidates_event_segment_and_time_bars(spec, tmp_path):
    t = mk_trades(spec, [1, 2, 3, 10, 11, 12], [1 * S, 2 * S, 3 * S, 200 * S, 201 * S, 202 * S],
                  ["10"] * 6, ["0.6", "0.6", "0.6", "0.6", "0.6", "0.6"])
    b = vb(spec, tmp_path / "v", 1.0)
    out = b.process(t)
    st = out.status.to_numpy()
    # bar0 = trades 1,2 closed; bar1 had trade 3 when the gap hit -> invalidated; then a new segment
    assert list(out.last_trade_id) == [2, 3, 11]
    assert st[1] == STATUS_GAP_INVALIDATED and out.complete.iloc[1] == 0
    assert out.segment_id.iloc[2] == 1 and out.bar_start_us.iloc[2] == 200 * S
    assert b.meta["segments"][-1]["origin_us"] == 200 * S
    tb = BarBuilder("t", "time", spec, tmp_path / "t", interval_us=60 * S)
    tb.start_series_at(0)
    o = tb.process(t, watermark_us=240 * S)
    inc = o[o.status == STATUS_INCOMPLETE].bar_start_us.tolist()
    assert inc == [0, 60 * S, 120 * S, 180 * S]  # every bucket touching [3s, 200s] is incomplete
    assert (o[o.status == STATUS_INCOMPLETE].complete == 0).all()


def test_gap_recovery_equals_uninterrupted(spec, tmp_path):
    t = random_trades(spec, 3000, seed=5)
    missing = t[(t.trade_id >= 1000) & (t.trade_id < 1100)]
    holed = t[~t.trade_id.isin(missing.trade_id)]
    a = vb(spec, tmp_path / "a", 0.5).process(t)
    rec = vb(spec, tmp_path / "b", 0.5, gap_resolver=lambda lo, hi: missing[(missing.trade_id >= lo) & (missing.trade_id <= hi)])
    b = rec.process(holed)
    pd.testing.assert_frame_equal(a.drop(columns="series_id"), b.drop(columns="series_id"))
    assert rec.meta["recovered_gaps"] == [[999, 1100]]


def test_dedupe_identity(spec):
    df = mk_trades(spec, [1, 2, 2, 3], [1, 2, 2, 2], ["1", "1", "1", "1"], ["1", "2", "2", "3"])
    out, q = dedupe_and_check(df, "trade_id")
    assert list(out.trade_id) == [1, 2, 3] and q["duplicates_dropped"] == 1
    # distinct trades with identical ts/price/qty but different ids are kept
    df2 = mk_trades(spec, [1, 2], [5, 5], ["1", "1"], ["1", "1"])
    assert len(dedupe_and_check(df2, "trade_id")[0]) == 2
    bad = mk_trades(spec, [1, 1], [1, 1], ["1", "2"], ["1", "1"])
    with pytest.raises(DuplicateConflict):
        dedupe_and_check(bad, "trade_id")


def test_precision_errors():
    assert list(scaled_int(pa.array(["1.25", "0.00000001"]), 8)) == [125000000, 1]
    with pytest.raises(PrecisionError):
        scaled_int(pa.array(["1.234"]), 2)


def test_is_buyer_maker_mapping_and_units(spec):
    t = pa.table({"id": pa.array([1, 2, 3], pa.int64()), "price": pa.array(["100.00", "100.01", "99.99"]),
                  "qty": pa.array(["1", "2", "3"]), "quote_qty": pa.array(["100", "200.02", "299.97"]),
                  "time": pa.array([1_700_000_000_000, 1_700_000_000_001, 1_700_000_000_002], pa.int64()),
                  "is_buyer_maker": pa.array(["True", "False", "true"])})
    from datetime import date
    df, q = normalize_trades_table(t, spec, "spot", date(2023, 11, 14), "x")
    assert list(df.aggressor) == [-1, 1, -1]      # buyer maker -> seller aggressed
    assert q["quote_qty_mismatch"] == 0           # our exact p*q equals the venue's quoteQty
    assert df.ts_us.iloc[0] == 1_700_000_000_000_000
    # spot archive from 2025-01-01 is documented in microseconds: ms values must be rejected
    with pytest.raises(ValueError):
        normalize_trades_table(t, spec, "spot", date(2025, 2, 1), "x")


def test_timestamp_regression_rejected(spec, tmp_path):
    t = mk_trades(spec, [1, 2, 3], [10, 5, 20], ["1", "1", "1"], ["1", "1", "1"])
    with pytest.raises(ValueError):
        vb(spec, tmp_path, 10).process(t)


def test_adaptive_threshold_uses_only_prior_complete_days(spec, tmp_path):
    D = 86_400 * S
    rng = np.random.default_rng(0)
    rows = []
    tid = 1
    vols = {0: 10, 1: 20, 2: 30, 3: 1000}  # day 3 is huge: must not affect bars starting on day 2
    for d, v in vols.items():
        for k in range(100):
            rows.append((tid, d * D + k * 800 * S, "100", str(v / 100)))
            tid += 1
    t = mk_trades(spec, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])
    ad = {"enabled": True, "lookback_days": 2, "target_bars_per_day": 10}
    b = vb(spec, tmp_path / "a", 1.0, adaptive=ad)
    out = b.process(t)
    thr_by_day = out.groupby(out.bar_start_us // D).threshold_i.first()
    # day 2 bars: mean(day0, day1) / 10 = (10+20)/2/10 = 1.5 ; day 3: (20+30)/2/10 = 2.5
    assert thr_by_day.loc[2] == int(1.5 * spec.qty_scale)
    assert thr_by_day.loc[3] == int(2.5 * spec.qty_scale)
    # perturbing the future day leaves earlier bars identical
    t2 = t.copy()
    t2.loc[t2.ts_us >= 3 * D, "qty_i"] *= 7
    t2["notional_i"] = t2.price_i * t2.qty_i
    out2 = vb(spec, tmp_path / "b", 1.0, adaptive=ad).process(t2)
    early = out[out.bar_end_us < 3 * D]
    pd.testing.assert_frame_equal(early.drop(columns="series_id"), out2.iloc[:len(early)].drop(columns="series_id"))

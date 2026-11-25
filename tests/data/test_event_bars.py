import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import quant_forge.strategies as qs
from quant_forge.data import (BarConfig, EventBarSpec, daily_thresholds, event_bars_from_trades, event_bars_to_frame,
                              sampling_summary)
from quant_forge.data import archive as arch
from quant_forge.data import event_bars as eb

DAY = 86_400_000
T0 = int(pd.Timestamp("2021-01-01", tz="UTC").value // 10 ** 6)


def make_trades(days=50, per_day=1500, seed=0, gap_day=None, gap_hours=6.0):
    rng = np.random.default_rng(seed)
    n = days * per_day
    t = np.sort(T0 + rng.integers(0, days * DAY, n))
    t[100:110] = t[100]                                  # a burst of trades with the same timestamp
    if gap_day is not None:
        lo = T0 + gap_day * DAY
        t = t[(t < lo) | (t >= lo + int(gap_hours * 3_600_000))]
    n = len(t)
    first = np.arange(n) * 3
    return pd.DataFrame({"transact_time": t, "price": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))),
                         "quantity": rng.lognormal(0, 1, n), "first_trade_id": first,
                         "last_trade_id": first + rng.integers(0, 3, n), "is_buyer_maker": rng.random(n) < 0.5})


SPECS = [EventBarSpec("volume", 6), EventBarSpec("dollar", 24), EventBarSpec("volume", 1, window_days=10)]


def test_spec_validation_and_name():
    assert EventBarSpec("dollar", 6).name == "dollar_6pd_w30_g4h"
    with pytest.raises(ValueError):
        EventBarSpec("tick", 6)
    with pytest.raises(ValueError):
        EventBarSpec("volume", 0)


def test_thresholds_follow_the_trailing_rule_and_frequency_tracks_target():
    tr = make_trades()
    bars = event_bars_from_trades(tr, SPECS)
    dv = tr.groupby(tr["transact_time"] // DAY)["quantity"].sum()
    full = np.zeros(int(dv.index.max()) + 1)
    full[dv.index] = dv.to_numpy()
    for spec in (SPECS[0], SPECS[2]):
        x = bars[spec.name]
        thr = daily_thresholds(full, spec.bars_per_day, spec.window_days, first_day=int(dv.index.min()))
        np.testing.assert_allclose(x["threshold"].to_numpy(), thr[x["start_ms"] // DAY])
        # no bar opens during the warm-up
        assert x["start_ms"].min() >= (int(dv.index.min()) + spec.window_days) * DAY
        per_day = x.groupby(x["end_ms"] // DAY).size().iloc[1:-1].mean()
        assert abs(per_day / spec.bars_per_day - 1) < 0.2
    # every threshold-closed bar reached its threshold
    x = bars[SPECS[1].name]
    ok = x["closed_by"] == 0
    assert (x.loc[ok, "quote_volume"] >= x.loc[ok, "threshold"] * (1 - 1e-12)).all()


def test_batch_invariance_and_python_kernel():
    tr = make_trades(days=40, gap_day=35)
    a = event_bars_from_trades(tr, SPECS)
    b = event_bars_from_trades(tr, SPECS, batch_size=3_333)
    c = event_bars_from_trades(tr, SPECS, batch_size=20_000, use_numba=False)
    for s in SPECS:
        pd.testing.assert_frame_equal(a[s.name], b[s.name])
        pd.testing.assert_frame_equal(a[s.name], c[s.name])


def test_prefix_property_no_look_ahead():
    tr = make_trades()
    full = event_bars_from_trades(tr, SPECS)
    cut = T0 + 41 * DAY + 12_345
    part = event_bars_from_trades(tr.loc[tr["transact_time"] < cut], SPECS)
    for s in SPECS:
        k = len(part[s.name])
        assert k > 0
        pd.testing.assert_frame_equal(full[s.name].iloc[:k].reset_index(drop=True), part[s.name])


def test_ties_never_split_and_gaps_start_segments():
    tr = make_trades(days=45, gap_day=40)
    x = event_bars_from_trades(tr, [EventBarSpec("dollar", 48)])["dollar_48pd_w30_g4h"]
    assert (np.diff(x["start_ms"]) > 0).all()
    assert (x["start_ms"].to_numpy()[1:] > x["end_ms"].to_numpy()[:-1]).all()      # ties stay in one bar
    assert x["segment_id"].nunique() == 2
    jump = np.flatnonzero(np.diff(x["segment_id"]) == 1)[0]
    assert x["start_ms"].iloc[jump + 1] - x["end_ms"].iloc[jump] > 4 * 3_600_000
    assert x["n_trades"].sum() == pytest.approx((tr["last_trade_id"] - tr["first_trade_id"] + 1)
                                                [tr["transact_time"] >= x["start_ms"].iloc[0]].sum(), rel=0.01)


def test_invalid_records_are_dropped():
    tr = make_trades(days=45)
    bad = tr.copy()
    idx = bad.index[[5000, 20000, 40000, 40001]]
    bad.loc[idx[:3], "price"] = 0.0
    bad.loc[idx[3], "quantity"] = -1.0
    b = eb.EventBarBuilder(SPECS)
    b.feed(eb.normalize_epoch_ms(bad["transact_time"].to_numpy()), bad["price"].to_numpy(), bad["quantity"].to_numpy())
    got = b.take_bars()
    ref = event_bars_from_trades(tr.drop(index=idx)[["transact_time", "price", "quantity"]], SPECS)
    assert b.n_dropped == 4
    for s in SPECS:
        assert (got[s.name]["low"] > 0).all()
        pd.testing.assert_frame_equal(got[s.name], ref[s.name])


def test_repeated_blocks_are_dropped_by_aggtrade_id():
    tr = make_trades(days=45).reset_index(drop=True)
    tr["agg_trade_id"] = np.arange(len(tr)) + 1000
    # the file repeats a block of earlier trades (as ETHUSDT 2018-01 does) and later restarts its id sequence
    dup = pd.concat([tr.iloc[:30000], tr.iloc[10000:20000], tr.iloc[30000:]], ignore_index=True)
    restart = dup.copy()
    late = restart["transact_time"] > restart["transact_time"].iloc[-1] - 3 * DAY
    restart.loc[late, "agg_trade_id"] = np.arange(late.sum())          # newer timestamps, smaller ids: kept
    b = eb.EventBarBuilder(SPECS)
    for chunk in np.array_split(np.arange(len(restart)), 7):
        x = restart.iloc[chunk]
        b.feed(x["transact_time"].to_numpy(), x["price"].to_numpy(), x["quantity"].to_numpy(),
               (x["last_trade_id"] - x["first_trade_id"] + 1).to_numpy(), x["is_buyer_maker"].to_numpy(),
               x["agg_trade_id"].to_numpy())
    got = b.take_bars()
    ref = event_bars_from_trades(tr.drop(columns="agg_trade_id"), SPECS)
    assert b.n_duplicates == 10000 and b.n_reordered == 0
    for s in SPECS:
        pd.testing.assert_frame_equal(got[s.name], ref[s.name])


def test_frame_conversion_runs_every_strategy():
    tr = make_trades(days=80, per_day=800)
    raw = event_bars_from_trades(tr, [EventBarSpec("volume", 24)])["volume_24pd_w30_g4h"]
    bars = event_bars_to_frame(raw, bar_type="volume_24")
    for c in ["bar_start", "bar_end", "open", "high", "low", "close", "volume", "quote_volume", "n_trades",
              "taker_buy_base", "bar_start_us", "bar_end_us", "first_trade_ts_us", "segment_id", "complete"]:
        assert c in bars.columns
    assert bars.index.is_monotonic_increasing and (bars["bar_end_us"] >= bars["bar_start_us"]).all()
    for s in qs.Strategies:
        res = qs.run_strategy(bars, s, tick_size=0.01, fee_bps=5)
        assert res.diagnostics["n_valid_bars"] == len(bars)
    summ = sampling_summary({"X": bars}, 24)
    assert 0.7 < summ.loc["X", "ratio to target"] < 1.4
    trimmed = event_bars_to_frame(raw, end=bars.index[100])
    assert len(trimmed) == 101


def test_bar_config():
    assert BarConfig("time", 6).interval == "4h" and BarConfig("time", 48).key == "time_30m"
    assert BarConfig("dollar", 6).event_spec == EventBarSpec("dollar", 6) and BarConfig("volume", 1).key == "volume_1"
    with pytest.raises(ValueError):
        BarConfig("time", 5)


def test_plan_archive_files():
    plan = arch.plan_archive_files("2026-08-15", "2026-11-19", today="2026-11-23")
    assert plan[0] == ("daily", "2026-08-15") and ("monthly", "2026-09") in plan and ("monthly", "2026-10") in plan
    assert plan[-1] == ("daily", "2026-11-18") and ("monthly", "2026-11") not in plan
    assert len(plan) == 17 + 2 + 18
    assert arch.plan_archive_files("2026-11-01", "2026-12-31", today="2026-11-05")[-1] == ("daily", "2026-11-04")


def _zip_bytes(csv_text: str, name="X.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, csv_text)
    return buf.getvalue()


@pytest.mark.parametrize("header", [False, True])
def test_streaming_zip_reader_matches_csv(header, monkeypatch):
    import hashlib

    import pyarrow as pa

    rng = np.random.default_rng(1)
    n = 50_000
    df = pd.DataFrame({"agg_trade_id": np.arange(n), "price": rng.uniform(1, 2, n).round(6),
                       "quantity": rng.uniform(0, 5, n).round(4), "first_trade_id": np.arange(n),
                       "last_trade_id": np.arange(n), "transact_time": T0 + np.arange(n),
                       "is_buyer_maker": np.where(rng.random(n) < 0.5, "True", "False"), "is_best_match": "True"})
    text = df.to_csv(index=False, header=header)
    blob = _zip_bytes(text)
    files = {"x.zip": blob, "x.zip.CHECKSUM": f"{hashlib.sha256(blob).hexdigest()}  x.zip".encode()}

    monkeypatch.setattr(arch, "archive_url", lambda *a, **k: "x.zip")
    monkeypatch.setattr(arch, "_download", lambda url, timeout: files.get(url))

    class Resp(io.BytesIO):
        pass

    monkeypatch.setattr(arch.urllib.request, "urlopen", lambda url, timeout=None: Resp(files[url]))
    types = {"price": pa.float64(), "transact_time": pa.int64(), "is_buyer_maker": pa.bool_()}
    got = pa.Table.from_batches(list(arch.stream_archive_batches(
        "aggTrades", "X", "2021-01", columns=arch.AGGTRADE_COLUMNS, types=types, block_size=1 << 16))).to_pandas()
    assert len(got) == n
    np.testing.assert_allclose(got["price"], df["price"])
    assert (got["transact_time"].to_numpy() == df["transact_time"].to_numpy()).all()
    assert (got["is_buyer_maker"].to_numpy() == (df["is_buyer_maker"] == "True").to_numpy()).all()
    files["x.zip.CHECKSUM"] = b"0" * 64
    with pytest.raises(arch.ChecksumError):
        list(arch.stream_archive_batches("aggTrades", "X", "2021-01", columns=arch.AGGTRADE_COLUMNS, types=types))


def test_build_is_resumable_incremental_and_falls_back_to_daily(tmp_path, monkeypatch):
    import pyarrow as pa

    tr = make_trades(days=75, per_day=600, seed=3)
    tr["agg_trade_id"] = np.arange(len(tr))
    tr["day"] = pd.to_datetime(tr["transact_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    tr["month"] = tr["day"].str[:7]
    missing_monthly = {"2021-02"}                       # monthly file absent -> daily files are used

    def fake_stream(kind, symbol, period, interval=None, *, columns, types, timeout=120, stats=None, **kw):
        if len(period) == 7 and period in missing_monthly:
            raise arch.ArchiveFileMissing(period)
        part = tr.loc[(tr["month"] if len(period) == 7 else tr["day"]) == period]
        if stats is not None:
            stats.update(raw_bytes=len(part) * 40, seconds=0.0)
        for a in range(0, len(part), 5_000):
            yield pa.RecordBatch.from_pandas(part.iloc[a:a + 5_000][list(types)], preserve_index=False)

    monkeypatch.setattr(eb, "stream_archive_batches", fake_stream)
    specs = [EventBarSpec("volume", 6), EventBarSpec("dollar", 1)]
    kw = dict(start="2021-01-01", today="2021-06-01", verbose=False)
    eb.build_event_bars("XUSDT", specs, end="2021-02-10", cache_dir=tmp_path / "a", **kw)
    st = eb.build_event_bars("XUSDT", specs, end="2021-03-17", cache_dir=tmp_path / "a", **kw)
    one = eb.build_event_bars("XUSDT", specs, end="2021-03-17", cache_dir=tmp_path / "b", **kw)
    assert st["next_day"] == one["next_day"] == pd.Timestamp("2021-03-17", tz="UTC") and not st["missing"]
    for s in specs:
        a = pd.read_parquet(tmp_path / "a" / "XUSDT" / f"{s.name}.parquet")
        b = pd.read_parquet(tmp_path / "b" / "XUSDT" / f"{s.name}.parquet")
        pd.testing.assert_frame_equal(a, b)
        direct = event_bars_from_trades(tr.loc[tr["day"] < "2021-03-17"], [s])[s.name]
        pd.testing.assert_frame_equal(a, direct)
    files_b = eb._load_state(tmp_path / "b" / "XUSDT")["files"]
    assert "2021-01" in files_b and "2021-02-15" in files_b and "2021-02" not in files_b

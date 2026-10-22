"""Archive path end-to-end with daily files in the documented Binance layout (written to the raw cache):
no reset at day boundaries, incremental update == one-shot build, missing day -> gap handling."""
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


def cfg_for(root, end_day=3):
    return cfgmod.load(None, [f"storage.root={root}", "period.start=2024-03-01T00:00:00Z",
                              f"period.end=2024-03-0{1 + end_day}T00:00:00Z", "period.bar_origin=2024-03-01T00:00:00Z",
                              "warmup.intraday_days=0", "warmup.daily_days=0", "bars.volume.threshold=5",
                              "bars.notional.threshold=300000", "bars.time.interval=1h",
                              "features.structural.window_bars=40", "features.structural.min_length=15",
                              "features.structural.csw_window=40", "features.structural.bde_window=40"])


def build(root, end_day=3):
    p = Pipeline(cfg_for(root, end_day), offline=True)
    res = p.build(allow_fetch=False)
    out = Path(res["out_dir"])
    return {bt: pd.read_parquet(out / f"{bt}.parquet") for bt in ("time", "volume", "notional")}, res


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


def test_missing_day_gap_segment_and_incomplete_time_bars(tmp_path):
    write_days(tmp_path, [0, 2])                    # day 1 missing in the archive
    out, res = build(tmp_path)
    assert res["missing_trade_days"] == ["2024-03-02"]
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

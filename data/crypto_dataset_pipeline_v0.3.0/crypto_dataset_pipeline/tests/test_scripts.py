"""End-to-end check of the run scripts on a synthetic archive (Binance layout, persistent order flow):
classic build -> calibration on a pilot month -> build with the calibrated per-type parameters -> report."""
import io
import json
import subprocess
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import ROOT
from cryptods import cli
from cryptods.sources import binance as bn
from cryptods.storage import RawStore


def write_archive(root, d0, ndays, per_day, seed=0):
    rng = np.random.default_rng(seed)
    raw = RawStore(root)
    tid, px, side = 1, 6_000_000, 1
    for k in range(ndays):
        d = d0 + timedelta(days=k)
        n = int(per_day * rng.uniform(0.6, 1.4))
        flips = rng.random(n) > 0.9
        s = np.empty(n, np.int64)
        for i in range(n):
            side = -side if flips[i] else side
            s[i] = side
        p = px + np.cumsum(np.where(rng.random(n) < 0.3, s, 0) + rng.choice([-1, 0, 0, 0, 1], n))
        px = int(p[-1])
        q = np.round(rng.lognormal(-4, 1.2, n), 5) + 1e-5
        t0 = int(pd.Timestamp(d, tz="UTC").value // 1_000_000)
        ts = t0 + np.sort(rng.integers(0, 86_400_000, n))
        lines = [f"{tid + i},{p[i] / 100:.2f},{q[i]:.5f},{p[i] / 100 * q[i]:.8f},{ts[i]},{'True' if s[i] < 0 else 'False'},True"
                 for i in range(n)]
        tid += n
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr(f"BTCUSDT-trades-{d}.csv", "\n".join(lines) + "\n")
        raw.put("binance", bn.archive_rel("spot", "daily", "trades", "BTCUSDT", d), b.getvalue(), {"note": "synthetic"})


def test_calibrate_build_report(tmp_path):
    root = tmp_path / "data"
    write_archive(root, date(2024, 3, 1), 61, 1500)
    common = ["--set", f"storage.root={root}", "--set", "period.start=2024-03-08T00:00:00Z",
              "--set", "period.end=2024-05-01T00:00:00Z", "--set", "period.bar_origin=2024-03-01T00:00:00Z",
              "--set", "warmup.intraday_days=0", "--set", "warmup.daily_days=0"]
    assert cli.main(["build", "--offline"] + common + ["--set", "bars.types=[volume]"]) == 0
    out = tmp_path / "cal"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "calibrate_information_bars.py")] + common +
                       ["--targets", "8", "--pilot-months", "2024-04", "--types", "tick_runs,volume_imbalance",
                        "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    cal = json.loads((out / "calibration.json").read_text())
    t = cal["targets"]["8"]
    assert t["parameters"]["tick_runs"]["fixed_expected_ticks"] >= 2
    assert t["parameters"]["volume_imbalance"]["rms_horizon_ticks"] % 4 == 0
    assert t["parameters"]["volume_imbalance"]["warmup_ticks"] == cal["ewma_span_ticks"]
    args = [x for x in (out / "target_8.args").read_text().splitlines() if x]
    res_code = cli.main(["build", "--offline"] + common + args)
    assert res_code == 0
    # the calibrated series land close to the target over the whole period
    rep = subprocess.run([sys.executable, str(ROOT / "scripts" / "report_datasets.py")] + common,
                         capture_output=True, text=True, cwd=ROOT)
    assert rep.returncode == 0, rep.stderr[-2000:]
    s = pd.read_csv(root / "reports" / "datasets_summary.csv")
    for bt in ("tick_runs", "volume_imbalance"):
        bpd = s[s.bar_type == bt].bars_per_day.iloc[0]
        assert 8 * 0.75 < bpd < 8 * 1.25, (bt, bpd)
    assert (s.status == "complete").all() and s.halted.isna().all()

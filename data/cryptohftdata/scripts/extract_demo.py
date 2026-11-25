"""Extract a small demonstration slice of an assembled dataset (all bar sets) into derived/datasets/<out>/.

Usage: .venv/bin/python scripts/extract_demo.py btcusdt_initial demo_2026-10-01_03 2026-10-01 2026-10-03
The slice keeps the original values (no recomputation), so it is consistent with the full dataset; a manifest
records the source and the period. Then run scripts/verify_dataset.py on it.
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chdds import paths as P  # noqa: E402

src, out, a, b = sys.argv[1], sys.argv[2], date.fromisoformat(sys.argv[3]), date.fromisoformat(sys.argv[4])
sroot = P.DERIVED / "datasets" / src
oroot = P.DERIVED / "datasets" / out
man = json.loads((sroot / "manifest.json").read_text())
t0 = P.hour_start_ns(a, 0)
t1 = P.hour_start_ns(b + timedelta(days=1), 0)
newman = dict(name=out, source_dataset=src, period=[a.isoformat(), b.isoformat()], code=man["code"],
              config_hash=man["config_hash"], bar_sets={})
for spec_id, info in man["bar_sets"].items():
    months = sorted({a.isoformat()[:7], b.isoformat()[:7]})
    files = [sroot / spec_id / f"{m}.parquet" for m in months if (sroot / spec_id / f"{m}.parquet").exists()]
    if not files:
        continue
    df = pl.concat([pl.read_parquet(f) for f in files]).filter(
        (pl.col("decision_time_ns") >= t0) & (pl.col("decision_time_ns") < t1))
    (oroot / spec_id).mkdir(parents=True, exist_ok=True)
    df.write_parquet(oroot / spec_id / f"{a.isoformat()[:7]}.parquet", compression="zstd")
    newman["bar_sets"][spec_id] = dict(name=info["name"], rows=df.height, columns=df.columns, n_columns=df.width)
    print(spec_id, df.shape)
(oroot / "manifest.json").write_text(json.dumps(newman, indent=1, default=str))

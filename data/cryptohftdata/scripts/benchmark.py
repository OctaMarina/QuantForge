"""End-to-end benchmark on representative intervals (separate derived root; raw and staged data are shared read-only).

Usage: .venv/bin/python scripts/benchmark.py <bench_root> <start> <end> [workers]
Measures wall time, max RSS (children) and output size for each stage; prints a JSON line per stage.
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def du(p):
    tot = 0
    for dp, dn, fn in os.walk(p):
        for f in fn:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def stage(name, code, env):
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-v", PY, "-c", code], env=env, capture_output=True, text=True)
    wall = time.time() - t0
    rss = None
    for line in p.stderr.splitlines():
        if "Maximum resident set size" in line:
            rss = int(line.split(":")[1]) / 1e6
    if p.returncode != 0:
        print(p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"stage {name} failed")
    return dict(stage=name, wall_s=round(wall, 1), max_rss_gb=round(rss, 2) if rss else None)


def main():
    bench, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    bench = Path(bench)
    bench.mkdir(parents=True, exist_ok=True)
    for sub in ("staged", "audit"):
        link = bench / sub
        if not link.exists():
            link.symlink_to(ROOT / "derived" / sub)
    env = dict(os.environ, CHDDS_DERIVED=str(bench))
    cfg = bench / "bench.yaml"
    cfg.write_text(f"""dataset: {{name: bench, start: {start}, end: {end}}}
bars:
  - {{name: time_1m, kind: time, threshold: 60}}
  - {{name: volume_50btc, kind: volume, threshold: 50}}
  - {{name: dollar_5musdt, kind: dollar, threshold: 5000000}}
  - {{name: cusum_5bps, kind: cusum, threshold: 5}}
features: {{families: [bar, trades, mark, oi, liq, ticker, calendar, book, combos]}}
workers: {workers}
""")
    res = []
    pre = f"from datetime import date; s=date.fromisoformat('{start}'); e=date.fromisoformat('{end}');"
    res.append(stage("canonical", pre + "from chdds.canonical import trades as T, slow as S;"
                     "T.build(s,e,log=lambda *a:None);S.build_mark(s,e,log=lambda *a:None);S.build_ticker(s,e,log=lambda *a:None);"
                     "S.build_oi(s,e,log=lambda *a:None);S.build_liquidations(s,e,log=lambda *a:None)", env))
    res.append(stage("bars", f"from chdds.config import load_config; from chdds import dataset as D; c=load_config('{cfg}');"
                     "D.build_all_bars(c, workers=1, log=lambda *a:None)", env))
    res.append(stage("features", f"from chdds.config import load_config; from chdds import dataset as D; c=load_config('{cfg}');"
                     f"D.build_features(c, workers={workers}, log=lambda *a:None)", env))
    res.append(stage("book", f"from chdds.config import load_config; from chdds import dataset as D; c=load_config('{cfg}');"
                     f"print(D.build_book(c, workers={workers}, log=lambda *a:None))", env))
    res.append(stage("assemble", f"from chdds.config import load_config; from chdds import dataset as D; c=load_config('{cfg}');"
                     "D.assemble(c, log=lambda *a:None)", env))
    sizes = {d: round(du(bench / d) / 1e6, 1) for d in ("canonical", "bars", "features", "book_union", "book_features",
                                                         "book_grid", "book_state", "datasets")}
    out = dict(interval=f"{start}..{end}", workers=workers, stages=res, output_mb=sizes)
    print(json.dumps(out))
    (bench / "benchmark.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

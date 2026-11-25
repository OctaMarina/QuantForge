"""Command line interface.

  chdds audit        [--workers 9] [--streams trades,orderbook,...]   exhaustive, resumable per-file audit
  chdds consolidate                                                    per-file JSON -> audit tables
  chdds audit-report                                                   docs/AUDIT_REPORT.md + validity tables
  chdds canonical    [--start D] [--end D]                             canonical trades + slow streams
  chdds bars         --config C [--start D --end D]                    bar sets (incremental / resumable)
  chdds features     --config C [--workers N]                          per-day feature parts
  chdds book         --config C [--workers N] [--resume-from DAY]      single causal L2 replay for all bar sets
  chdds assemble     --config C                                        join + combinations + selection + manifest
  chdds build        --config C                                        bars -> features -> book -> assemble
  chdds registry                                                       docs/FEATURE_REGISTRY.md / .csv
  chdds show-config  --config C
Overrides: --set key.sub=value (YAML scalars/lists), e.g. --set dataset.end=2026-10-31
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

from . import paths as P


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _cfg(a):
    from .config import load_config
    ov = dict(x.split("=", 1) for x in (a.set or []))
    cfg = load_config(a.config, ov)
    if getattr(a, "start", None):
        cfg.start = date.fromisoformat(a.start)
    if getattr(a, "end", None):
        cfg.end = date.fromisoformat(a.end)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chdds", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("--config")
    ap.add_argument("--set", action="append")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--streams")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--resume-from")
    ap.add_argument("--drop-book-intermediates", action="store_true")
    a = ap.parse_args(argv)
    c = a.command
    if c == "audit":
        from .audit.run import run
        streams = a.streams.split(",") if a.streams else P.STREAMS
        run(streams, a.workers or 9, log=_log)
    elif c == "consolidate":
        from .audit.consolidate import consolidate
        consolidate(log=_log)
    elif c == "audit-report":
        from .audit.report import write_report
        write_report(log=_log)
    elif c == "canonical":
        from .canonical import slow as CS
        from .canonical import trades as CT
        s = date.fromisoformat(a.start) if a.start else P.HISTORY_START
        e = date.fromisoformat(a.end) if a.end else P.HISTORY_END
        CT.build(s, e, log=_log)
        CS.build_mark(s, e, log=_log)
        CS.build_ticker(s, e, log=_log)
        CS.build_oi(s, e, log=_log)
        CS.build_liquidations(s, e, log=_log)
    elif c in ("bars", "features", "book", "assemble", "build"):
        from . import dataset as DS
        cfg = _cfg(a)
        t0 = time.time()
        if c in ("bars", "build"):
            DS.build_all_bars(cfg, log=_log, workers=min(len(cfg.bars), a.workers or 4))
        if c in ("features", "build"):
            DS.build_features(cfg, workers=a.workers or cfg.workers, log=_log)
        if c in ("book", "build") and "book" in cfg.families:
            _log(json.dumps(DS.build_book(cfg, workers=a.workers or 4, log=_log, resume_from=a.resume_from)))
        if c in ("assemble", "build"):
            DS.assemble(cfg, log=_log, drop_book_intermediates=a.drop_book_intermediates)
        _log(f"{c} done in {time.time() - t0:.0f}s")
    elif c == "registry":
        from .registry import write_docs
        write_docs(P.PROJECT_ROOT / "docs" / "FEATURE_REGISTRY.md", P.PROJECT_ROOT / "docs" / "feature_registry.csv")
    elif c == "show-config":
        cfg = _cfg(a)
        print(json.dumps(cfg.raw, indent=1, default=str))
        for b in cfg.bars:
            print(b.spec_id, b.normalised())
    else:
        ap.error(f"unknown command {c}")


def run(argv=None):
    from .guard import EXIT_STOPPED, RunStopped
    try:
        return main(argv)
    except RunStopped as e:
        print(f"STOPPED: {e}", flush=True)
        return EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(run())

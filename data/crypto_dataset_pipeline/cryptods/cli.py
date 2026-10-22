"""Command line interface.

    python crypto_dataset.py dry-run      [--config CFG] [--set key.path=value ...]
    python crypto_dataset.py download     [...]           # fetch raw inputs into the cache (resumable)
    python crypto_dataset.py build        [...] [--fetch] # build datasets from the cache (no network unless --fetch)
    python crypto_dataset.py update       [...]           # download new days + incremental build
    python crypto_dataset.py live         [...] [--seconds N]   # collect raw live data (no orders)
    python crypto_dataset.py replay-live  [...]           # build datasets from live logs (same transformations)
    python crypto_dataset.py sample-real  [...]           # build a small dataset from real fixture data (see README)
    python crypto_dataset.py show-config  [...]
    python crypto_dataset.py registry     [...] [--fixtures OUT_DIR]   # feature registry for a config (no data)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config as cfgmod


def main(argv=None):
    ap = argparse.ArgumentParser(prog="crypto_dataset.py", description="Causal crypto bar datasets (AFML-style)")
    ap.add_argument("command", choices=["dry-run", "download", "build", "update", "live", "replay-live",
                                        "sample-real", "show-config", "registry"])
    ap.add_argument("--config", default=None, help="YAML config (default: built-in BTC demo)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override, e.g. --set period.start=2024-01-01T00:00:00Z --set bars.types=[volume]")
    ap.add_argument("--fetch", action="store_true", help="build: allow fetching missing inputs")
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    ap.add_argument("--seconds", type=float, default=None, help="live: stop after N seconds")
    ap.add_argument("--fixtures", default=None, help="sample-real: directory with the real fixture files")
    a = ap.parse_args(argv)
    cfg = cfgmod.load(a.config, a.set)
    if a.command == "registry":
        from .registry_dump import dump
        print(json.dumps(dump(cfg, a.fixtures or "registry_out"), indent=2))
        return 0
    if a.command == "show-config":
        print(json.dumps(cfg, indent=2, default=str))
        return 0
    if a.command == "sample-real":
        from .sample import build_real_sample
        res = build_real_sample(cfg, a.fixtures)
        print(json.dumps(res, indent=2, default=str))
        return 0
    from .pipeline import Pipeline
    p = Pipeline(cfg, offline=a.offline)
    if a.command == "dry-run":
        rep = p.dry_run()
        print(json.dumps(rep, indent=2, default=str))
    elif a.command == "download":
        print(json.dumps(p.download(), indent=2, default=str))
    elif a.command == "build":
        print(json.dumps(p.build(allow_fetch=a.fetch), indent=2, default=str))
    elif a.command == "update":
        p.download()
        print(json.dumps(p.build(allow_fetch=True), indent=2, default=str))
    elif a.command == "live":
        from .live.collector import run
        run(cfg, p.root, p.spec, a.seconds)
    elif a.command == "replay-live":
        from .live import replay as lr
        trades = lr.trades(p.root, p.spec, futures=p.spec.market_type != "spot")
        print(json.dumps(p.build(allow_fetch=False, live_trades=trades), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

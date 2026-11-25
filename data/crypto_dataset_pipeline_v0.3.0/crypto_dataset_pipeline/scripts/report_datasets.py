#!/usr/bin/env python3
"""Summary of every dataset built under storage.root: one line per (build, bar type).

Columns: build (config hash), bar type, series id, main parameters, rows, bars/day over the export period,
mean / median trades per bar, build status, number of warnings, halted series.  Written to
<root>/reports/datasets_summary.csv and printed.

  .venv/bin/python scripts/report_datasets.py --config configs/eth_spot.yaml [--set storage.root=...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cryptods import config as cfgmod  # noqa: E402


def params(ident: dict) -> str:
    if "threshold" in ident:
        return f"threshold={ident['threshold']}"
    if "interval_us" in ident:
        return f"interval={ident['interval_us'] // 1_000_000}s"
    ib = ident.get("information") or {}
    keep = ("sign", "fixed_expected_ticks", "rms_horizon_ticks", "warmup_ticks", "ewma_span_bars",
            "ewma_span_ticks", "imbalance_threshold", "expected_ticks_bounds", "min_expected_imbalance_ratio")
    return " ".join(f"{k}={ib[k]}" for k in keep if ib.get(k) is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = cfgmod.load(a.config, a.set)
    root = Path(cfg["storage"]["root"])
    rows = []
    for m in sorted((root / "datasets").glob("*/manifest.json")):
        man = json.loads(m.read_text())
        p = man["config"]["period"]
        days = (pd.Timestamp(p["end"]) - pd.Timestamp(p["start"])) / pd.Timedelta(days=1)
        for bt, d in man.get("datasets", {}).items():
            f = Path(d["file"])
            if not f.exists():
                f = m.parent / f"{bt}.parquet"
            n = pd.read_parquet(f, columns=["n_trades"])["n_trades"] if f.exists() else pd.Series(dtype=float)
            qa = d.get("information_bar_qa") or {}
            rows.append({"build": m.parent.name, "generated_at": man.get("generated_at"), "bar_type": bt,
                         "series_id": d["series_id"], "parameters": params(d.get("identity", {})),
                         "period": f"{p['start'][:10]}..{p['end'][:10]}", "rows": d["rows"],
                         "bars_per_day": round(d["rows"] / days, 1) if days else None,
                         "mean_trades_per_bar": round(float(n.mean()), 1) if len(n) else None,
                         "median_trades_per_bar": float(n.median()) if len(n) else None,
                         "share_bars_le_2_trades": qa.get("share_bars_le_2_ticks"),
                         "status": man.get("status"), "warnings": len(man.get("warnings") or []),
                         "halted": ",".join(man.get("halted_series") or {}) or "",
                         "qa_warning": qa.get("warning", "")})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"no datasets under {root / 'datasets'}")
        return
    df = df.sort_values(["period", "bar_type", "bars_per_day"])
    out = root / "reports" / "datasets_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.max_rows", 500, "display.max_colwidth", 90, "display.width", 250):
        print(df[["build", "bar_type", "period", "rows", "bars_per_day", "mean_trades_per_bar",
                  "status", "halted", "qa_warning", "parameters"]].to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

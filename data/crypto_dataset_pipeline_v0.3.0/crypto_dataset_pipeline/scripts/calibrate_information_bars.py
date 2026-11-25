#!/usr/bin/env python3
"""Calibrate the information-driven bars to a target number of bars per day.

What is calibrated (one value per bar type and target, everything else is fixed by declared rules):

* runs bars (tick/volume/notional): ``fixed_expected_ticks`` H, i.e. E0[T] := H in the AFML threshold
  E0[T] * max(P E[v|b=1], (1-P) E[v|b=-1]); P and E[v|b] stay EWMAs of prior bars (span = target bars/day,
  ~1 day of memory);
* imbalance bars (tick/volume/notional): ``rms_horizon_ticks`` H of the RMS threshold
  sqrt(mu^2 H^2 + s2 H); mu, s2 = EWMA mean / variance of b_t v_t over ~1 day of prior ticks.

Why a calibration is needed: the mean bar length is c * H with c depending on the order-flow persistence
(on real ETHUSDT trades c ~ 0.3 for tick imbalance, ~0.8-1 for the others).  c is measured on pilot
months (separate series, separate storage under <root>/_calibration), H is rescaled, and the result is
verified on the same pilots.  The mean number of trades per day comes from the per-day quality records
written by any earlier build of the full period (run the volume/notional builds first).

Outputs (``--out``, default calibration/):
  calibration.json        all inputs, measurements per iteration, final parameters, expected bars/day per year
  target_<N>.args         one CLI token per line, for:  mapfile -t A < calibration/target_25.args
                          python crypto_dataset.py build <common args> "${A[@]}"

Usage (from the project directory):
  .venv/bin/python scripts/calibrate_information_bars.py --config configs/eth_spot.yaml \\
      --set period.start=2021-09-01T00:00:00Z --set period.end=2026-09-01T00:00:00Z --targets 25,50,100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cryptods import config as cfgmod          # noqa: E402
from cryptods import timeutil as tu            # noqa: E402
from cryptods.pipeline import Pipeline         # noqa: E402

RUNS = ("tick_runs", "volume_runs", "notional_runs")
IMB = ("tick_imbalance", "volume_imbalance", "notional_imbalance")


def iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def trades_per_day(cfg) -> pd.Series:
    """Rows (deduplicated executions) per UTC day of the full period, from the build quality records."""
    p = Pipeline(cfg, offline=True)
    spec = p.spec
    f = (Path(cfg["storage"]["root"]) / "normalized" / "trades_quality" /
         f"{spec.exchange}_{spec.market_type}_{spec.symbol}_{cfg['bars']['trade_source']}.jsonl")
    if not f.exists():
        raise SystemExit(f"{f} not found: run at least one build of the full period first (e.g. the volume bars)")
    recs = {}
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            if "rows" in r:
                recs[r["day"]] = int(r["rows"])          # the last record of a day wins
    s = pd.Series(recs, dtype="int64")
    s.index = pd.to_datetime(s.index, utc=True)
    start = pd.Timestamp(tu.to_iso(p.start)); end = pd.Timestamp(tu.to_iso(p.end))
    s = s[(s.index >= start) & (s.index < end)].sort_index()
    n_days = int((end - start) / pd.Timedelta(days=1))
    if len(s) < 0.95 * n_days:
        raise SystemExit(f"per-day trade counts cover only {len(s)} of {n_days} days of the period: run a full-period "
                         f"build first (missing days are reported by that build)")
    return s


def fit(bt: str, h: float) -> int:
    """Runs: any integer >= 2.  Imbalance (RMS rule): a multiple of 4 (windows of 4 sub-blocks)."""
    if bt in IMB:
        return max(4, int(round(h / 4.0)) * 4)
    return max(2, int(round(h)))


def per_type_params(bt: str, h: int, target: int, span_ticks: int) -> dict:
    if bt in RUNS:
        return {"fixed_expected_ticks": int(h), "warmup_ticks": int(h), "ewma_span_bars": int(target)}
    # warm-up = one full memory span of the window EWMA, so the first threshold rests on ~a day of windows
    return {"imbalance_threshold": "rms", "rms_horizon_ticks": int(h), "warmup_ticks": int(span_ticks),
            "ewma_span_ticks": int(span_ticks)}


def build_args(types, params: dict, sign: str, guard_day: int, guard_open: int) -> list[str]:
    per = json.dumps({bt: params[bt] for bt in types}, separators=(", ", ": "))
    return ["--set", f"bars.types=[{','.join(types)}]",
            "--set", f"bars.information.sign={sign}",
            "--set", f"bars.information.per_type={per}",
            "--set", f"bars.information.guard_max_bars_per_day={guard_day}",
            "--set", f"bars.information.guard_max_open_days={guard_open}"]


def run_pilot(config, sets, calib_root: Path, month: str, settle_days: int, args_tokens: list[str], types):
    m0 = pd.Timestamp(f"{month}-01", tz="UTC")
    m1 = m0 + pd.offsets.MonthBegin(1)
    start = m0 + pd.Timedelta(days=settle_days)
    over = list(sets) + [f"storage.root={calib_root}", f"period.bar_origin={iso(m0)}", f"period.start={iso(start)}",
                         f"period.end={iso(m1)}", "warmup.intraday_days=0", "warmup.daily_days=0",
                         "storage.qa_kline_crosscheck=false"]
    over += [args_tokens[i + 1] for i in range(0, len(args_tokens), 2)]       # drop the "--set" tokens
    cfg = cfgmod.load(config, over)
    res = Pipeline(cfg, offline=True).build(allow_fetch=False)
    if res["status"] == "incomplete":
        raise SystemExit(f"pilot {month} incomplete: {res.get('stopped_before_day') or res.get('halted_series')}")
    out = Path(res["out_dir"])
    days = (m1 - start) / pd.Timedelta(days=1)
    meas = {}
    for bt in types:
        f = out / f"{bt}.parquet"
        if not f.exists():
            raise SystemExit(f"pilot {month}: no {bt} bars produced ({res['series'].get(bt)})")
        d = pd.read_parquet(f, columns=["n_trades", "ib_warmup", "bar_status"])
        d = d[(~d["ib_warmup"].astype(bool)) & (d["bar_status"] == "closed")]
        meas[bt] = {"bars": int(len(d)), "ticks": int(d["n_trades"].sum()), "days": float(days),
                    "bars_per_day": float(len(d) / days)}
    return meas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="same overrides as the builds")
    ap.add_argument("--targets", default="25,50,100", help="bars per day (mean over the period)")
    ap.add_argument("--types", default=",".join(RUNS + IMB))
    ap.add_argument("--pilot-months", default="2022-06,2024-03", help="YYYY-MM inside the downloaded period")
    ap.add_argument("--settle-days", type=int, default=7, help="days between the pilot origin and its measurement")
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--tolerance", type=float, default=0.05, help="accepted relative error of the mean bar length")
    ap.add_argument("--sign", default="aggressor", choices=["aggressor", "tick_rule"])
    ap.add_argument("--guard-bars-per-day", type=int, default=5000)
    ap.add_argument("--guard-open-days", type=int, default=2)
    ap.add_argument("--out", default="calibration")
    a = ap.parse_args()

    cfg = cfgmod.load(a.config, a.set)
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    for t in types:
        if t not in RUNS + IMB:
            raise SystemExit(f"unknown information bar type {t}")
    targets = [int(x) for x in a.targets.split(",")]
    months = [m.strip() for m in a.pilot_months.split(",") if m.strip()]
    tpd = trades_per_day(cfg)
    mean_tpd = float(tpd.mean())
    span_ticks = int(round(mean_tpd))                         # ~1 day of ticks for mu / s2
    root = Path(cfg["storage"]["root"]).resolve()
    calib_root = root / "_calibration"
    calib_root.mkdir(parents=True, exist_ok=True)
    link = calib_root / "raw"
    if not link.exists():
        os.symlink(root / "raw", link, target_is_directory=True)      # pilots read the same raw cache
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": a.config, "overrides": a.set, "sign": a.sign, "pilot_months": months,
              "settle_days": a.settle_days, "trades_per_day": {"mean": mean_tpd, "median": float(tpd.median()),
                                                               "days": int(len(tpd)),
                                                               "by_year_mean": {str(k): float(v) for k, v in
                                                                                tpd.groupby(tpd.index.year).mean().items()}},
              "ewma_span_ticks": span_ticks, "targets": {}}
    for n in targets:
        L = mean_tpd / n                                      # wanted mean ticks per bar
        h = {bt: fit(bt, L) for bt in types}
        hist, final, verified = [], None, False
        for it in range(a.max_iterations):
            params = {bt: per_type_params(bt, h[bt], n, span_ticks) for bt in types}
            tokens = build_args(types, params, a.sign, a.guard_bars_per_day, a.guard_open_days)
            tot = {bt: {"bars": 0, "ticks": 0, "days": 0.0} for bt in types}
            per_month = {}
            for m in months:
                print(f"[target {n}/day] iteration {it + 1}: pilot {m} ...", flush=True)
                meas = run_pilot(a.config, a.set, calib_root, m, a.settle_days, tokens, types)
                per_month[m] = meas
                for bt in types:
                    for k in ("bars", "ticks", "days"):
                        tot[bt][k] += meas[bt][k]
            mean_len = {bt: tot[bt]["ticks"] / max(tot[bt]["bars"], 1) for bt in types}
            err = {bt: mean_len[bt] / L - 1.0 for bt in types}
            hist.append({"iteration": it + 1, "H": dict(h), "pilot_mean_ticks_per_bar": mean_len,
                         "relative_error_vs_target_length": err, "pilots": per_month})
            print(f"[target {n}/day] wanted {L:,.0f} ticks/bar; " +
                  ", ".join(f"{bt}: H={h[bt]:,} -> {mean_len[bt]:,.0f} ({err[bt]:+.1%})" for bt in types), flush=True)
            if all(abs(e) <= a.tolerance for e in err.values()):
                final, verified = dict(h), True
                break
            # c = mean length / H is approximately constant: rescale H (only the types still outside tolerance)
            h = {bt: (h[bt] if abs(err[bt]) <= a.tolerance else fit(bt, h[bt] * L / mean_len[bt])) for bt in types}
            final = dict(h)
        params = {bt: per_type_params(bt, final[bt], n, span_ticks) for bt in types}
        tokens = build_args(types, params, a.sign, a.guard_bars_per_day, a.guard_open_days)
        (out_dir / f"target_{n}.args").write_text("\n".join(tokens) + "\n")
        report["targets"][str(n)] = {
            "wanted_mean_ticks_per_bar": L, "parameters": params, "verified_on_pilots": verified,
            "note": None if verified else ("final H rescaled after the last pilot but not re-measured: check "
                                           "information_bar_qa of the full build"),
            # bars/day follow the trading activity of each year, like volume bars with a fixed threshold
            "expected_bars_per_day_by_year": {y: round(v / L, 1)
                                              for y, v in report["trades_per_day"]["by_year_mean"].items()},
            "iterations": hist}
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"written: {out_dir / 'calibration.json'} and target_<N>.args for {targets}")
    if not all(report["targets"][str(n)]["verified_on_pilots"] for n in targets):
        print("WARNING: some targets were not verified within tolerance (see calibration.json)", file=sys.stderr)


if __name__ == "__main__":
    main()

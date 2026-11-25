"""Audit report: docs/AUDIT_REPORT.md + derived/audit/*.csv tables (monthly summaries, gap lists, validity map).

Every number labelled EXHAUSTIVE comes from reading all rows of all files; SAMPLE marks anything computed on a subset.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import polars as pl

from .. import paths as P
from .consolidate import OUT, load, load_events

DOCS = P.PROJECT_ROOT / "docs"


def _md(df: pl.DataFrame, max_rows=60, floatfmt=3):
    if df is None or df.height == 0:
        return "_(none)_\n"
    df = df.head(max_rows)
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if v != v else (f"{v:,.{floatfmt}f}" if abs(v) < 1e6 else f"{v:,.0f}"))
            elif isinstance(v, int) and not isinstance(v, bool):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _iso(ns):
    if ns is None:
        return ""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _month(df):
    return df.with_columns(pl.col("day").str.slice(0, 7).alias("month"))


def section_inventory():
    from .inventory import build_inventory, unavailable_intervals
    inv, checks = build_inventory()
    inv.write_parquet(OUT / "inventory.parquet")
    iv = unavailable_intervals(inv)
    iv.write_csv(OUT / "unavailable_intervals.csv")
    s = ["## 1. Inventory and manifests (EXHAUSTIVE over the 11,784-hour grid × 6 streams)\n"]
    s.append("Manifest/file-system cross-checks:\n\n```\n" + json.dumps(checks, indent=1, default=str) + "\n```\n")
    by = inv.group_by("stream").agg(pl.len().alias("hours"), pl.col("fs_exists").sum().alias("files"),
                                    (pl.col("fs_size").sum() / 1e9).alias("GB")).sort("stream")
    s.append(_md(by))
    multi = iv.filter(pl.col("hours") > 1)
    s.append("\nUnavailable intervals longer than one hour (end exclusive; full list incl. single hours in "
             "`derived/audit/unavailable_intervals.csv`):\n\n" + _md(multi))
    s.append(f"\nSingle-hour liquidation files missing: {iv.filter((pl.col('stream') == 'liquidations') & (pl.col('hours') == 1)).height}. "
             "A missing liquidation file is NOT treated as zero liquidations (default policy `unknown`).\n")
    both = inv.group_by(["day", "hour"]).agg(pl.col("fs_exists").all().alias("all6"))
    s.append(f"\nHours with all six files present: {both['all6'].sum():,} of {both.height:,} "
             f"({100 * both['all6'].mean():.3f}%). File presence is not completeness — see sections below.\n")
    return "\n".join(s)


def section_schemas():
    s = ["## 2. Schemas, versions, writers and repair markers (EXHAUSTIVE)\n"]
    for st in P.STREAMS:
        try:
            f = load(st)
        except Exception:
            continue
        g = (f.group_by(["schema_fp", "created_by", "kv_metadata"])
             .agg(pl.col("day").min().alias("first_day"), pl.col("day").max().alias("last_day"), pl.len().alias("files"),
                  pl.col("columns").first().alias("columns")).sort("first_day"))
        s.append(f"### {st}\n")
        s.append(_md(g.with_columns(pl.col("columns").str.replace_all('"', "").str.slice(0, 300))))
        if "s_n_rows" in f.columns:
            mm = f.filter(pl.col("s_n_rows") != pl.col("parquet_num_rows")).height
            s.append(f"Row count read vs parquet footer mismatches: {mm}.\n")
    return "\n".join(s)


def _nulls_summary(f):
    tot = {}
    for js in f["s_nulls"].to_list():
        for k, v in json.loads(js).items():
            tot[k] = tot.get(k, 0) + v
    return {k: v for k, v in tot.items() if v}


def section_trades():
    f = _month(load("trades"))
    s = ["## 3. Trades (EXHAUSTIVE: every row, identity, NA rows, ids, clocks)\n"]
    m = (f.group_by("month").agg(
        pl.len().alias("files"), pl.col("s_n_rows").sum().alias("rows"), pl.col("s_n_economic").sum().alias("economic"),
        pl.col("s_n_na").sum().alias("NA"), pl.col("s_n_na_nonzero").sum().alias("NA_nonzero"),
        pl.col("s_n_nonna_nonpos").sum().alias("nonNA_nonpos"), pl.col("s_n_dup_id_rows").sum().alias("dup_id_rows"),
        pl.col("s_n_dup_conflict").sum().alias("id_conflicts"), pl.col("s_n_missing_ids").sum().alias("missing_ids_in_file"),
        pl.col("s_n_id_gaps").sum().alias("id_gaps_in_file"), pl.col("s_n_id_inversion_in_recv_order").sum().alias("id_inversions"),
        pl.col("s_n_recv_regress_fileorder").sum().alias("recv_regress"), pl.col("s_delay_ms_p50").median().alias("delay_p50_ms"),
        pl.col("s_delay_ms_p99").median().alias("delay_p99_ms"), pl.col("s_delay_ms_max").max().alias("delay_max_ms"),
        pl.col("s_n_neg_delay").sum().alias("neg_delay"), pl.col("s_n_synthetic_recv").sum().alias("synthetic_recv"),
        pl.col("s_n_offtick").sum().alias("offtick"), pl.col("s_recv_n_gap_gt_5s").sum().alias("recv_gaps_gt5s"),
        pl.col("s_n_recv_outside_hour").sum().alias("recv_outside_hour"),
        pl.col("s_price_inexact").sum().alias("price_inexact"), pl.col("s_qty_inexact").sum().alias("qty_inexact"))
         .sort("month"))
    m.write_csv(OUT / "trades_monthly.csv")
    s.append(_md(m))
    s.append(f"\nNull counts over all files: `{_nulls_summary(f)}`. Order types: "
             f"`{_merge_vc(f['s_order_types'])}`.\n")
    cb = P.DERIVED / "canonical" / "trades_build.json"
    if cb.exists():
        s.append("\nCanonical stream build (global dedup across files, availability order, continuity):\n\n```\n"
                 + cb.read_text() + "\n```\n")
    gp = P.DERIVED / "canonical" / "trades_gaps.parquet"
    if gp.exists():
        g = pl.read_parquet(gp)
        if g.height and "kind" in g.columns:
            gg = g.filter(pl.col("kind") == "id_gap").with_columns(
                ((pl.col("t1") - pl.col("t0")) / 1e9).alias("recv_span_s"), pl.col("hour").str.slice(0, 7).alias("month"))
            gm = gg.group_by("month").agg(pl.len().alias("id_gaps"), pl.col("missing").sum().alias("missing_ids"),
                                          pl.col("recv_span_s").sum().alias("gap_span_s"),
                                          pl.col("recv_span_s").max().alias("max_span_s")).sort("month")
            gm.write_csv(OUT / "trades_gaps_monthly.csv")
            s.append("\nTrade-id continuity gaps (missing ids after counting NA ids) by month — each breaks the bar "
                     "segment and invalidates windows that intersect it:\n\n" + _md(gm))
            top = gg.sort("missing", descending=True).head(15).with_columns(
                pl.col("t0").map_elements(_iso, return_dtype=pl.Utf8).alias("from"),
                pl.col("t1").map_elements(_iso, return_dtype=pl.Utf8).alias("to")).select(
                ["from", "to", "missing", "recv_span_s"])
            s.append("\nLargest trade gaps:\n\n" + _md(top))
    return "\n".join(s)


def _merge_vc(col):
    tot = {}
    for js in col.to_list():
        if js is None:
            continue
        for k, v in json.loads(js).items():
            tot[k] = tot.get(k, 0) + v
    return tot


def section_book():
    f = _month(load("orderbook"))
    s = ["## 4. Order book (EXHAUSTIVE per-file pass: message reassembly, copies, sequence, snapshots, clocks)\n"]
    s.append("Per-file chain statistics below follow reception order inside each file only (no buffering); the "
             "causal resolver used for reconstruction (section 9) re-orders by sequence ids with buffering and "
             "is the reference for continuity.\n")
    m = (f.group_by("month").agg(
        pl.len().alias("files"), pl.col("s_n_rows").sum().alias("rows"), pl.col("s_n_msgs").sum().alias("messages"),
        pl.col("s_copies_mean").mean().alias("copies_per_msg"), pl.col("s_n_msgs_copy_conflict").sum().alias("copy_conflicts"),
        pl.col("s_n_msgs_snapshot").sum().alias("snapshots"), pl.col("s_n_snap_synthetic_only").sum().alias("snap_synthetic_recv"),
        pl.col("s_n_snap_checkpoint").sum().alias("hour_checkpoints"), pl.col("s_msg_interval_ms_p50").median().alias("msg_interval_p50_ms"),
        pl.col("s_levels_per_update_p50").median().alias("levels_p50"), pl.col("s_delay_ms_p50").median().alias("delay_p50_ms"),
        pl.col("s_delay_ms_p999").median().alias("delay_p999_ms"), pl.col("s_delay_ms_max").max().alias("delay_max_ms"),
        pl.col("s_n_neg_delay").sum().alias("neg_delay_msgs"), pl.col("s_chain_break").sum().alias("breaks_recvorder"),
        pl.col("s_chain_stale").sum().alias("stale_recvorder"), pl.col("s_recv_n_gap_gt_5s").sum().alias("recv_gaps_gt5s"),
        pl.col("s_recv_n_gap_gt_60s").sum().alias("recv_gaps_gt60s"), pl.col("s_recv_gap_time_gt_5s_s").sum().alias("gap_time_s"),
        pl.col("s_n_msgs_inconsistent").sum().alias("inconsistent_msgs"), pl.col("s_n_msgs_duplevel").sum().alias("dup_level_msgs"),
        pl.col("s_n_offtick").sum().alias("offtick_rows"), pl.col("s_n_zero_qty_rows").sum().alias("zero_qty_rows"),
        pl.col("s_n_msgs_noop").sum().alias("noop_msgs"), pl.col("s_n_update_txn_null").sum().alias("txn_null_updates"))
         .sort("month"))
    m.write_csv(OUT / "orderbook_monthly.csv")
    s.append(_md(m))
    s.append(f"\nNull counts: `{_nulls_summary(f)}`; event types `{_merge_vc(f['s_event_types'])}`; sides `{_merge_vc(f['s_sides'])}`.\n")
    e = load_events("orderbook")
    if e is not None:
        sn = e.filter(pl.col("kind") == "book_snapshot")
        if sn.height:
            sm = (sn.with_columns(pl.col("day").str.slice(0, 7).alias("month")).group_by("month").agg(
                pl.len().alias("snapshots"), pl.col("nbid").median().alias("bid_levels_p50"),
                pl.col("nbid").min().alias("bid_levels_min"), pl.col("synthetic").sum().alias("synthetic_recv"),
                pl.col("checkpoint").sum().alias("hour_checkpoint"), pl.col("connects_next").sum().alias("connects_next_in_file"),
                pl.col("crossed").sum().alias("crossed"), pl.col("day").n_unique().alias("days_with_snapshot")).sort("month"))
            sm.write_csv(OUT / "orderbook_snapshots_monthly.csv")
            s.append("\nSnapshots (anchors) by month:\n\n" + _md(sm))
        gaps = e.filter(pl.col("kind") == "recv_gap").sort("dur_s", descending=True).head(20)
        if gaps.height:
            s.append("\nLongest intra-file reception gaps of book messages:\n\n" + _md(gaps.with_columns(
                pl.col("t0").map_elements(_iso, return_dtype=pl.Utf8).alias("from")).select(["from", "dur_s"])))
    return "\n".join(s)


def section_simple(stream, title, extra_cols):
    f = _month(load(stream))
    aggs = [pl.len().alias("files"), pl.col("s_n_rows").sum().alias("rows")]
    for c, how, name in extra_cols:
        if c in f.columns:
            e = getattr(pl.col(c), how)()
            aggs.append(e.alias(name))
    m = f.group_by("month").agg(aggs).sort("month")
    m.write_csv(OUT / f"{stream}_monthly.csv")
    s = [f"## {title}\n", _md(m), f"\nNull counts: `{_nulls_summary(f)}`.\n"]
    return "\n".join(s), f


def section_slow():
    out = []
    t, f = section_simple("mark_price", "5. Mark price / index / funding (EXHAUSTIVE)", [
        ("s_interval_ms_p50", "median", "interval_p50_ms"), ("s_n_dup_evt", "sum", "dup_event_rows"),
        ("s_n_evt_with_conflicting_payload", "sum", "evt_conflicts"), ("s_delay_ms_p50", "median", "delay_p50_ms"),
        ("s_delay_ms_max", "max", "delay_max_ms"), ("s_n_delay_gt_3s", "sum", "delay_gt3s"),
        ("s_n_neg_delay", "sum", "neg_delay"), ("s_evt_n_gap_gt_3s", "sum", "evt_gaps_gt3s"),
        ("s_evt_n_gap_gt_60s", "sum", "evt_gaps_gt60s"), ("s_evt_gap_time_gt_5s_s", "sum", "evt_gap_time_s"),
        ("s_basis_mark_index_bps_min", "min", "basis_min_bps"), ("s_basis_mark_index_bps_max", "max", "basis_max_bps"),
        ("s_funding_min", "min", "funding_min"), ("s_funding_max", "max", "funding_max")])
    out.append(t)
    e = load_events("mark_price")
    if e is not None:
        g = e.filter(pl.col("kind") == "evt_gap").with_columns(
            ((pl.col("t0") // 1_000_000_000) % 3600 // 60).alias("minute_of_hour"), pl.col("day").str.slice(0, 7).alias("month"))
        if g.height:
            mh = g.filter(pl.col("dur_s") >= 30).group_by("month").agg(
                pl.len().alias("gaps_ge30s"), (pl.col("minute_of_hour").is_between(1, 5)).sum().alias("starting_hh01_05"),
                pl.col("dur_s").median().alias("dur_p50_s")).sort("month")
            out.append("\nMark-price market-clock gaps ≥30 s by month and how many start at minute 01–05 of the hour "
                       "(vendor-declared hh:02–05 outages since mid-Aug 2026):\n\n" + _md(mh))
    t, f = section_simple("ticker", "6. Ticker (EXHAUSTIVE)", [
        ("s_interval_ms_p50", "median", "interval_p50_ms"), ("s_window_s_p0", "min", "window_min_s"),
        ("s_window_s_max", "max", "window_max_s"), ("s_n_pc_mismatch", "sum", "price_change_mismatch"),
        ("s_n_pct_mismatch_gt_0p001", "sum", "pct_mismatch"), ("s_n_vwap_outside_hl", "sum", "vwap_outside_hl"),
        ("s_n_last_outside_hl", "sum", "last_outside_hl"), ("s_n_dup_evt", "sum", "dup_event_rows"),
        ("s_n_evt_with_conflicting_payload", "sum", "evt_conflicts"), ("s_delay_ms_p50", "median", "delay_p50_ms"),
        ("s_recv_n_gap_gt_10s", "sum", "recv_gaps_gt10s")])
    out.append(t)
    t, f = section_simple("open_interest", "7. Open interest (EXHAUSTIVE)", [
        ("s_n_unique_ts", "sum", "unique_ts_in_files"), ("s_n_ts_not_5m", "sum", "ts_not_5min"),
        ("s_n_batches", "sum", "poll_batches"), ("s_batch_interval_s_p50", "median", "batch_interval_p50_s"),
        ("s_delay_s_p0", "min", "delay_min_s"), ("s_delay_s_p50", "median", "delay_p50_s"), ("s_delay_s_max", "max", "delay_max_s"),
        ("s_n_ts_before_hour", "sum", "points_before_file_hour")])
    out.append(t)
    rep = f.group_by("s_repairs").agg(pl.len().alias("files"), pl.col("day").min().alias("first"), pl.col("day").max().alias("last"))
    out.append("\nRepair markers in OI parquet metadata:\n\n" + _md(rep))
    ob = P.DERIVED / "canonical" / "oi_build.json"
    if ob.exists():
        out.append("\nCanonical OI (versions by first receipt, cross-file repeats removed):\n\n```\n" + ob.read_text() + "\n```\n")
    t, f = section_simple("liquidations", "8. Liquidations (EXHAUSTIVE)", [
        ("s_n_filled_eq_qty", "sum", "filled_eq_qty"), ("s_n_filled_lt_qty", "sum", "filled_lt_qty"),
        ("s_n_lastfilled_lt_filled", "sum", "lastfilled_lt_filled"), ("s_n_dup_exact", "sum", "dup_exact"),
        ("s_n_dup_except_recv", "sum", "dup_except_recv"), ("s_n_same_order_key_rows", "sum", "same_order_key"),
        ("s_notional_usdt", "sum", "notional_usdt"), ("s_delay_ms_p50", "median", "delay_p50_ms"),
        ("s_delay_ms_max", "max", "delay_max_ms")])
    out.append(t)
    out.append(f"\nValues: status `{_merge_vc(f['s_order_status_values'])}`, type `{_merge_vc(f['s_order_type_values'])}`, "
               f"tif `{_merge_vc(f['s_time_in_force_values'])}`, side `{_merge_vc(f['s_side_values'])}`.\n")
    lb = P.DERIVED / "canonical" / "liq_build.json"
    if lb.exists():
        out.append("\nCanonical liquidations:\n\n```\n" + lb.read_text() + "\n```\n")
    return "\n".join(out)


def write_report(log=print, extra_sections=()):
    parts = ["# Audit of the CryptoHFTData BTCUSDT (Binance USD-M) history 2025-06-28 → 2026-10-31\n",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `chdds audit-report`. Raw files were read "
             "only; no raw file was modified. Tables are also written as CSV under `derived/audit/`.\n"]
    for fn in (section_inventory, section_schemas, section_trades, section_book, section_slow):
        try:
            parts.append(fn())
            log(f"  section {fn.__name__} ok")
        except Exception as ex:  # keep the report going; record the failure explicitly
            parts.append(f"\n> Section {fn.__name__} failed: {type(ex).__name__}: {ex}\n")
            log(f"  section {fn.__name__} FAILED {ex}")
    parts.extend(extra_sections)
    (DOCS / "AUDIT_TABLES.md").write_text("\n\n".join(parts))
    return DOCS / "AUDIT_TABLES.md"


def write_validity_map(cfg_name="btcusdt_initial", log=print):
    """docs/VALIDITY_MAP.md: monthly valid-time fractions per family + long invalid intervals (CSV with all)."""
    from .validity import book_hourly, invalid_intervals
    v = pl.read_parquet(OUT / "validity_hourly.parquet")
    bk = book_hourly(P.DERIVED / "book_grid" / cfg_name / "grid100ms")
    if bk is not None:
        v = v.join(bk, on=["day", "hour"], how="left").with_columns(
            [pl.col(c).fill_null(0.0) for c in ("book_strict", "book_unanchored", "book_stale", "book_top_uncertified")])
    v.write_parquet(OUT / "validity_hourly_all.parquet")
    fams = [c for c in v.columns if c not in ("day", "hour")]
    m = (v.with_columns(pl.col("day").str.slice(0, 7).alias("month")).group_by("month")
         .agg([(pl.col(c).mean() * 100).alias(c) for c in fams]).sort("month"))
    m.write_csv(OUT / "validity_monthly.csv")
    tot = v.select([(pl.col(c).mean() * 100).alias(c) for c in fams])
    ivs = []
    for c in ("trades", "mark_fast", "ticker", "oi", "liquidations") + (("book_strict",) if bk is not None else ()):
        ivs += invalid_intervals(v, c, 0.5, 1)
    ivd = pl.DataFrame(ivs) if ivs else pl.DataFrame({"family": []})
    ivd.write_csv(OUT / "invalid_intervals.csv")
    lines = ["# Harta de validitate pe familii (EXHAUSTIV pe tot istoricul)", "",
             "Procent din timp (grilă de 1 s; book: grilă de 100 ms a replay-ului strict) în care condiția familiei "
             "este satisfăcută. Condiții: `chdds/audit/validity.py`. Tabele: `derived/audit/validity_hourly_all.parquet`, "
             "`validity_monthly.csv`, `invalid_intervals.csv` (toate intervalele orare cu < 50% timp valid).", "",
             "Notă: ferestrele unui feature cer în plus coverage complet al ferestrei (strict), deci disponibilitatea "
             "unei coloane cu fereastră lungă este mai mică decât fracția de mai jos.", "",
             "## Total istoric", "", _md(tot, floatfmt=2), "## Pe luni", "", _md(m, max_rows=40, floatfmt=2)]
    long = ivd.filter(pl.col("hours") >= 2) if ivd.height else ivd
    lines += ["## Intervale invalide de cel puțin 2 ore (start inclus, end exclus)", "", _md(long, max_rows=200)]
    (DOCS / "VALIDITY_MAP.md").write_text("\n".join(lines) + "\n")
    log(f"  validity map written ({ivd.height} invalid hour-runs)")
    return m

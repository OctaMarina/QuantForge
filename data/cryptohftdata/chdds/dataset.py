"""Dataset construction: bars -> feature parts (per day, parallel) -> book replay (single pass for all bar sets)
-> assembly (join, combinations, column selection) -> parquet per month + manifest.

Directory layout (all under derived/):
  bars/<spec_id>/...                      (bars.build)
  features/<cfg>/<spec_id>/<day>.parquet  trade/slow/calendar/bar-intrinsic feature parts
  book_features/<cfg>/<spec_id>/<day>.parquet
  book_grid/<cfg>/grid100ms/<day>.parquet
  datasets/<cfg>/<spec_id>/<YYYY-MM>.parquet + manifest.json
Targets/labels are intentionally NOT produced here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import __version__
from . import paths as P
from .bars import engines as E
from .bars.build import BARS_DIR, build_bars, read_bars
from .config import DatasetConfig
from .features import slow as S
from .features import trades as T
from .guard import RunStopped, stop_reason

META_COLS = ["bar_id", "decision_time_ns", "eval_time_ns", "t_start_ns", "t_end_ns", "t_open_recv_ns",
             "t_close_recv_ns", "m_open_ms", "m_close_ms", "first_seq", "last_seq", "first_trade_id",
             "last_trade_id", "seg", "close_reason", "complete", "threshold_int", "feed_complete"]


# Quality metadata: kept in the files, excluded by default from model inputs (model_columns)
QUALITY_COLS = ["book_state", "cert_bid_floor_dist_bps", "cert_ask_ceiling_dist_bps"]


def model_columns(columns):
    """Default model inputs: everything except bar identity and quality metadata."""
    skip = set(META_COLS) | set(QUALITY_COLS)
    return [c for c in columns if c not in skip]


def load_incidents(path=None):
    """Quality incidents. Retrospective knowledge: for excluding data from experiments only, never written as a dataset column."""
    import yaml
    p = Path(path) if path else P.PROJECT_ROOT / "configs" / "quality_incidents.yaml"
    out = []
    for r in yaml.safe_load(p.read_text())["incidents"]:
        q = dict(r)
        for k in ("start", "end"):
            q[k + "_ns"] = int(np.datetime64(str(r[k]).replace("Z", "").replace("+00:00", ""), "ns").astype(np.int64))
            q[k] = str(r[k])
        out.append(q)
    return out


def incident_rows(t_start_ns, eval_time_ns, incidents, scope=None):
    """True for bars whose span [t_start, eval_time] touches an incident."""
    m = np.zeros(len(eval_time_ns), bool)
    for i in incidents:
        if scope is None or i["scope"] == scope:
            m |= (np.asarray(t_start_ns) <= i["end_ns"]) & (np.asarray(eval_time_ns) >= i["start_ns"])
    return m


def _days(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def history_start_ns():
    files = sorted((P.DERIVED / "canonical" / "trades").glob("*.parquet"))
    t = pq.read_table(files[0], columns=["recv_ns"])
    return int(t["recv_ns"][0].as_py())


def code_fingerprint():
    h = hashlib.sha1()
    root = Path(__file__).parent
    for f in sorted(root.rglob("*.py")):
        h.update(f.read_bytes())
    try:
        git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git = ""
    return dict(package_version=__version__, source_sha1=h.hexdigest(), git_head=git)


# ------------------------------------------------------------------------------- bar-intrinsic features
def bar_features(bars: pa.Table, prev_close_e2: int | None, kind: str):
    c = {k: bars[k].to_numpy() for k in bars.column_names}
    n = bars.num_rows
    out = {}
    has = c["n_trades"] > 0
    O = np.where(has, c["open_e2"] / 100.0, np.nan)
    H = np.where(has, c["high_e2"] / 100.0, np.nan)
    L = np.where(has, c["low_e2"] / 100.0, np.nan)
    C = np.where(has, c["close_e2"] / 100.0, np.nan)
    V = c["vol_e3"] / 1e3
    D = c["notional_e5"] / 1e5
    Vb = c["vol_buy_e3"] / 1e3
    Db = c["notional_buy_e5"] / 1e5
    N = c["n_trades"].astype(np.float64)
    prevC = np.concatenate([[prev_close_e2 / 100.0 if prev_close_e2 else np.nan], C[:-1]])
    prev_seg = np.concatenate([[c["seg"][0]], c["seg"][:-1]])
    out["bar_open"], out["bar_high"], out["bar_low"], out["bar_close"] = O, H, L, C
    out["bar_n_trades"] = N
    out["bar_volume_btc"] = V
    out["bar_notional_usdt"] = D
    with np.errstate(invalid="ignore", divide="ignore"):
        out["bar_vwap"] = np.where(V > 0, D / V, np.nan)
        out["bar_return_log"] = np.where(has & (prev_seg == c["seg"]), np.log(C / prevC), np.nan)
        out["bar_oc_log"] = np.log(C / O)
        out["bar_range_log"] = np.log(H / L)
        out["bar_close_location"] = np.where(H > L, (C - L) / (H - L), np.nan)
        out["bar_tfi_btc"] = 2 * Vb - V
        out["bar_tfi_usdt"] = 2 * Db - D
        out["bar_tfi_ratio"] = np.where(V > 0, (2 * Vb - V) / V, np.nan)
        out["bar_count_imbalance"] = np.where(N > 0, (2 * c["n_buy"] - N) / N, np.nan)
        out["bar_mean_trade_size_btc"] = np.where(N > 0, V / N, np.nan)
    if kind == "time":
        dur = (c["t_end"] - c["t_start"]) / 1e9
    else:
        dur = (c["t_close"] - c["t_open"]) / 1e9
    out["bar_duration_s"] = dur
    with np.errstate(invalid="ignore", divide="ignore"):
        out["bar_volume_rate_btc_s"] = np.where(dur > 0, V / dur, np.nan)
        out["bar_notional_rate_usdt_s"] = np.where(dur > 0, D / dur, np.nan)
    if kind == "volume":
        out["bar_overshoot_btc"] = np.where(c["complete"] == 1, c["overshoot_int"] / 1e3, np.nan)
        out["bar_threshold_btc"] = c["threshold_int"] / 1e3
    if kind == "dollar":
        out["bar_overshoot_usdt"] = np.where(c["complete"] == 1, c["overshoot_int"] / 1e5, np.nan)
        out["bar_threshold_usdt"] = c["threshold_int"] / 1e5
    if kind == "cusum":
        out["cusum_trigger_side"] = c["trigger"].astype(np.float64)
        out["cusum_pre_reset"] = c["cusum_pre"]
        out["cusum_opposite_state"] = c["cusum_opp"]
        out["cusum_filter_overshoot"] = c["cusum_overshoot"]
        out["cusum_threshold_log"] = c["threshold_log"]
        out["bar_threshold_bps"] = c["threshold_int"] / 100.0
    return out


def meta_columns(bars: pa.Table, kind: str, hist0: int, gaps):
    c = {k: bars[k].to_numpy() for k in bars.column_names}
    g0, g1 = gaps
    if kind == "time":
        span_w = c["t_end"] - c["t_start"]
        gapped = T.window_gap_mask(c["t_end"] - 1, span_w, g0, g1)
        feed_ok = (~gapped) & (c["t_start"] >= hist0)
    else:
        feed_ok = c["complete"] == 1
    return {
        "bar_id": c["bar_id"], "decision_time_ns": c["decision_time"], "eval_time_ns": c["eval_time"],
        "t_start_ns": c["t_start"], "t_end_ns": c["t_end"], "t_open_recv_ns": c["t_open"],
        "t_close_recv_ns": c["t_close"], "m_open_ms": c["m_open"], "m_close_ms": c["m_close"],
        "first_seq": c["first_seq"], "last_seq": c["last_seq"], "first_trade_id": c["first_id"],
        "last_trade_id": c["last_id"], "seg": c["seg"], "close_reason": c["reason"].astype(np.int8),
        "complete": c["complete"].astype(np.int8), "threshold_int": c["threshold_int"],
        "feed_complete": feed_ok.astype(np.int8)}


# ------------------------------------------------------------------------------- per-day feature worker
def _feature_day(cfg_path_or_obj, d_iso: str, specs: list, cfg_name: str, slots_days: dict | None):
    cfg = cfg_path_or_obj
    d = date.fromisoformat(d_iso)
    hist0 = history_start_ns()
    gaps = T.load_gaps()
    per_spec = []
    allq_t = []
    allq_c = []
    for spec_id, kind in specs:
        p = BARS_DIR / spec_id / f"{d_iso}.parquet"
        if not p.exists():
            per_spec.append(None)
            continue
        bars = pq.read_table(p)
        if bars.num_rows == 0:
            per_spec.append(None)
            continue
        prevp = BARS_DIR / spec_id / f"{(d - timedelta(days=1)).isoformat()}.parquet"
        prev_close = None
        if prevp.exists():
            pb = pq.read_table(prevp, columns=["close_e2", "n_trades"])
            if pb.num_rows and pb["n_trades"][-1].as_py() > 0:
                prev_close = pb["close_e2"][-1].as_py()
        per_spec.append((spec_id, kind, bars, prev_close))
        allq_t.append(bars["eval_time"].to_numpy())
        allq_c.append(bars["last_seq"].to_numpy())
    if not allq_t:
        return d_iso, 0
    qt = np.concatenate(allq_t)
    qc = np.concatenate(allq_c)
    feats = {}
    fam = set(cfg.families)
    if "trades" in fam or "combos" in fam:
        thr = T.previous_day_large_threshold(d, cfg.trades.large_trade_quantile)
        feats.update(T.compute_trade_features(d, qt, qc, cfg.trades, hist0, gaps, thr))
    data = S.SlowData()
    if "mark" in fam:
        feats.update(S.mark_features(d, qt, cfg.slow, data, trade_ret=feats))
    if "oi" in fam:
        feats.update(S.oi_features(qt, cfg.slow, data))
    if "liq" in fam:
        feats.update(S.liq_features(qt, cfg.slow, data, hist0))
    if "ticker" in fam:
        feats.update(S.ticker_features(d, qt, cfg.slow, data, feats.get("last_trade_price")))
    if "calendar" in fam:
        feats.update(S.calendar_features(qt))
        if slots_days is not None and "notional_15m_usdt" in feats:
            feats["notional_15m_vs_slot_median_28d"] = S.seasonal_ratio(d, qt, feats["notional_15m_usdt"], slots_days,
                                                                      cfg.slow)
    off = 0
    n_total = 0
    for item in per_spec:
        if item is None:
            continue
        spec_id, kind, bars, prev_close = item
        n = bars.num_rows
        cols = meta_columns(bars, kind, hist0, gaps)
        if "bar" in fam:
            cols.update(bar_features(bars, prev_close, kind))
        for k, v in feats.items():
            cols[k] = v[off:off + n]
        off += n
        n_total += n
        outd = P.DERIVED / "features" / cfg_name / spec_id
        outd.mkdir(parents=True, exist_ok=True)
        tmp = outd / f".{d_iso}.tmp"
        pq.write_table(pa.table(cols), tmp, compression="zstd")
        os.replace(tmp, outd / f"{d_iso}.parquet")
    return d_iso, n_total


def build_features(cfg: DatasetConfig, days=None, workers=None, log=print, skip_existing=True):
    specs = [(b.spec_id, b.kind) for b in cfg.bars]
    days = list(days or _days(cfg.start, cfg.end))
    if skip_existing:
        days = [d for d in days if not all((P.DERIVED / "features" / cfg.name / s / f"{d.isoformat()}.parquet").exists()
                                           for s, _ in specs)]
    slots = None
    if "calendar" in cfg.families:
        slots = S.seasonal_slot_table(cfg.start - timedelta(days=cfg.slow.seasonal_lookback_days), cfg.end)
    workers = workers or cfg.workers
    t0 = time.time()
    done = 0
    rows = 0
    if workers <= 1:
        for d in days:
            _, n = _feature_day(cfg, d.isoformat(), specs, cfg.name, slots)
            rows += n
            done += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_feature_day, cfg, d.isoformat(), specs, cfg.name, slots) for d in days]
            for f in as_completed(futs):
                _, n = f.result()
                rows += n
                done += 1
                if done % 20 == 0 or done == len(days):
                    log(f"  features {done}/{len(days)} days, {rows:,} rows, {time.time() - t0:.0f}s")
                why = stop_reason()
                if why:
                    ex.shutdown(cancel_futures=True)
                    raise RunStopped(f"features stopped after {done} days: {why}")
    return rows


# ------------------------------------------------------------------------------- book replay for all specs
def build_book(cfg: DatasetConfig, workers=4, log=print, resume_from: str | None = None):
    """One causal L2 replay over [book_start, book_end] evaluating the union of all bar sets' eval times.
    Results are streamed per UTC day, then split per bar set (joined on eval_time)."""
    from .book.driver import run_replay
    b0 = cfg.book_start or cfg.start
    b1 = cfg.book_end or cfg.end
    allt = []
    for b in cfg.bars:
        t = read_bars(b.spec_id, b0, b1)
        if t is not None:
            allt.append(t["eval_time"].to_numpy())
    qt = np.unique(np.concatenate(allt))
    outdir = P.DERIVED / "book_grid" / cfg.name
    ck = P.DERIVED / "book_state" / cfg.name
    fdir = P.DERIVED / "book_union" / cfg.name
    resume = (ck / resume_from) if resume_from else None
    _, cols, stats, counters, _ = run_replay(b0, b1, qt, outdir, cfg.book, workers=workers, checkpoint_dir=ck,
                                             log=log, features_out_dir=fdir, resume_from=resume)
    # after a resume the days replayed earlier still have to be split: their union files are already on disk
    split_book(cfg, cfg.start if resume else b0, b1)
    statp = P.DERIVED / "book_state" / cfg.name
    statp.mkdir(parents=True, exist_ok=True)
    tag = f"{b0.isoformat()}_{b1.isoformat()}"
    sd = statp / "_stats"
    hrs = [pq.read_table(f) for f in sorted(sd.glob("hours_*.parquet"))] if sd.exists() else []
    rest = [s for s in stats if s.get("kind") != "verify"]
    if rest:
        hrs.append(pa.Table.from_pylist(rest))
    if hrs:
        pq.write_table(pa.concat_tables(hrs, promote_options="default"), statp / f"hour_stats_{tag}.parquet")
    ver = [pq.read_table(f) for f in sorted(sd.glob("verify_*.parquet"))] if sd.exists() else []
    vrest = [s for s in stats if s.get("kind") == "verify"]
    if vrest:
        ver.append(pa.Table.from_pylist(vrest))
    if ver:
        pq.write_table(pa.concat_tables(ver, promote_options="default"), statp / f"verify_log_{tag}.parquet")
    (statp / f"counters_{tag}.json").write_text(json.dumps(counters, indent=1))
    return counters


def split_book(cfg: DatasetConfig, b0: date, b1: date):
    fdir = P.DERIVED / "book_union" / cfg.name
    for d in _days(b0, b1):
        fp = fdir / f"{d.isoformat()}.parquet"
        if not fp.exists():
            continue
        # bars of day d may carry an eval time on day d-1 (partial bar closed by the first trade of day d after a
        # threshold change / gap): look the eval time up in the union files of d-1 and d
        prevp = fdir / f"{(d - timedelta(days=1)).isoformat()}.parquet"
        parts_u = [pq.read_table(prevp)] if prevp.exists() else []
        parts_u.append(pq.read_table(fp))
        u = pa.concat_tables(parts_u)
        ut = u["eval_time_ns"].to_numpy()
        if np.any(np.diff(ut) < 0):
            o = np.argsort(ut, kind="stable")
            u = u.take(pa.array(o))
            ut = ut[o]
        for b in cfg.bars:
            bp = BARS_DIR / b.spec_id / f"{d.isoformat()}.parquet"
            if not bp.exists():
                continue
            bars = pq.read_table(bp, columns=["bar_id", "eval_time"])
            et = bars["eval_time"].to_numpy()
            idx = np.searchsorted(ut, et)
            found = ut[np.minimum(idx, ut.size - 1)] == et
            # a subset run starts at b0: a bar of that day evaluated just before it was never replayed, so its book values are null
            assert np.all(found | (et < P.hour_start_ns(b0, 0))), "eval time missing from book union"
            tab = u.take(pa.array(idx, mask=~found)).drop(["eval_time_ns"])
            tab = tab.add_column(0, "bar_id", bars["bar_id"])
            od = P.DERIVED / "book_features" / cfg.name / b.spec_id
            od.mkdir(parents=True, exist_ok=True)
            pq.write_table(tab, od / f"{d.isoformat()}.parquet", compression="zstd")


# ------------------------------------------------------------------------------- combinations
def combos(t: dict, cfg: DatasetConfig):
    out = {}
    g = t.get
    nan = np.nan

    def safe_div(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.isfinite(a) & np.isfinite(b) & (b > 0), a / b, nan)

    for w in ("5s", "30s", "2m", "15m"):
        vb = g(f"tfi_{w}_btc")
        vol = g(f"volume_{w}_btc")
        if vb is not None and vol is not None:
            vbuy = (vol + vb) / 2
            vsell = (vol - vb) / 2
            da = g(f"tw_depth_ask_5bps_{w}_btc")
            db = g(f"tw_depth_bid_5bps_{w}_btc")
            if da is not None:
                pb = safe_div(vbuy, da)
                ps = safe_div(vsell, db)
                out[f"pressure_buy_5bps_{w}"] = pb
                out[f"pressure_sell_5bps_{w}"] = ps
                out[f"pressure_net_5bps_{w}"] = pb - ps
            d1 = g(f"tw_depth1_{w}_btc")
            if d1 is not None:
                out[f"tfi_over_tw_depth1_{w}"] = safe_div(vb, d1)
            on = g(f"ofi_norm_{w}")
            tr = g(f"tfi_ratio_{w}")
            if on is not None and tr is not None:
                out[f"tfi_ratio_x_ofi_norm_{w}"] = tr * on
        tr = g(f"tfi_ratio_{w}")
        rt = g(f"ret_trade_{w}")
        if tr is not None and rt is not None:
            out[f"tfi_ratio_x_ret_trade_{w}"] = tr * rt
    f = g("funding_rate_asof")
    if f is not None:
        if g("oi_level_rz_7d") is not None:
            out["funding_x_oi_level_rz"] = f * g("oi_level_rz_7d")
        if g("oi_log_change_known_1h") is not None:
            out["funding_x_oi_log_change_1h"] = f * g("oi_log_change_known_1h")
    for w in ("15m", "1h"):
        if g(f"volume_{w}_btc") is not None and g(f"tw_oi_known_{w}_btc") is not None:
            out[f"turnover_to_oi_{w}"] = safe_div(g(f"volume_{w}_btc"), g(f"tw_oi_known_{w}_btc"))
    for w in ("30s", "2m", "15m", "1h"):
        ls = g(f"liq_notional_short_{w}_usdt")
        ll = g(f"liq_notional_long_{w}_usdt")
        if ls is not None:
            tot = ls + ll
            if g(f"notional_{w}_usdt") is not None:
                out[f"liq_to_trade_notional_{w}"] = safe_div(tot, g(f"notional_{w}_usdt"))
            if g("oi_usdt_asof") is not None:
                out[f"liq_to_oi_{w}"] = safe_div(tot, g("oi_usdt_asof"))
            na = g("notional_ask_10bps_usdt")
            nb = g("notional_bid_10bps_usdt")
            if na is not None:
                out[f"liq_short_to_ask_depth_10bps_{w}"] = safe_div(ls, na)
                out[f"liq_long_to_bid_depth_10bps_{w}"] = safe_div(ll, nb)
    for w in ("2m", "15m", "1h"):
        nr = g(f"notional_rate_{w}_usdt_s")
        tq = g("ticker_quote_rate_usdt_s")
        if nr is not None and tq is not None:
            out[f"activity_vs_ticker24h_{w}"] = safe_div(nr, tq)
    r2 = g("rv_trade_2m_grid1s")
    r60 = g("rv_trade_1h_grid1s")
    if r2 is not None and r60 is not None:
        out["rv_ratio_2m_1h_grid1s"] = safe_div(r2 / 120.0, r60 / 3600.0)
        la = g("liq_any_2m")
        if la is not None:
            out["liq_any_2m_x_rv_ratio"] = la * out["rv_ratio_2m_1h_grid1s"]
    return out


# ------------------------------------------------------------------------------- assembly
def assemble(cfg: DatasetConfig, log=print, drop_book_intermediates=False):
    """drop_book_intermediates removes the per-day book files of this dataset once their month is written, to bound the disk peak.
    Finished months are recorded, so a stopped run continues where it was and never rebuilds a month from missing files."""
    out_root = P.DERIVED / "datasets" / cfg.name
    out_root.mkdir(parents=True, exist_ok=True)
    state_p = out_root / "_assemble_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    book_lo = (cfg.book_start or cfg.start).isoformat()
    book_hi = (cfg.book_end or cfg.end).isoformat()
    manifest = dict(name=cfg.name, config_hash=cfg.config_hash, config=cfg.raw, code=code_fingerprint(),
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), bar_sets={})
    for b in cfg.bars:
        fdir = P.DERIVED / "features" / cfg.name / b.spec_id
        bdir = P.DERIVED / "book_features" / cfg.name / b.spec_id
        files = sorted(fdir.glob("*.parquet"))
        months = {}
        for f in files:
            months.setdefault(f.stem[:7], []).append(f)
        done_m = state.setdefault(b.spec_id, {})
        sdir = out_root / b.spec_id
        sdir.mkdir(parents=True, exist_ok=True)
        nrows = 0
        cols_final = None
        book_states = {}
        for m, fl in sorted(months.items()):
            if m in done_m:
                nrows += done_m[m]["rows"]
                cols_final = done_m[m]["columns"]
                book_states[m] = done_m[m]["book_states"]
                continue
            why = stop_reason()
            if why:
                raise RunStopped(f"assemble stopped before {b.spec_id} {m}: {why}")
            parts = []
            for f in fl:
                t = pq.read_table(f)
                bp = bdir / f.name
                if "book" in cfg.families and book_lo <= f.stem <= book_hi and not bp.exists():
                    raise FileNotFoundError(f"book features missing for {b.spec_id} {f.stem}")
                if "book" in cfg.families and bp.exists():
                    bt = pq.read_table(bp)
                    # align on bar_id (same order by construction; verified)
                    if not np.array_equal(bt["bar_id"].to_numpy(), t["bar_id"].to_numpy()):
                        bt = bt.join(t.select(["bar_id"]), "bar_id", join_type="right outer").sort_by("bar_id")
                    for c in bt.column_names:
                        if c != "bar_id":
                            t = t.append_column(c, bt[c])
                parts.append(t)
            tab = pa.concat_tables(parts, promote_options="default")
            d = {c: tab[c].to_numpy(zero_copy_only=False) for c in tab.column_names}
            if "combos" in cfg.families:
                for k, v in combos(d, cfg).items():
                    tab = tab.append_column(k, pa.array(v))
            keep = META_COLS + [c for c in cfg.selected([c for c in tab.column_names if c not in META_COLS])]
            tab = tab.select([c for c in keep if c in tab.column_names])
            tmp = sdir / f".{m}.tmp"
            pq.write_table(tab, tmp, compression="zstd")
            tmp.replace(sdir / f"{m}.parquet")
            nrows += tab.num_rows
            if "book_state" in tab.column_names:
                sv, sc = np.unique(np.nan_to_num(tab["book_state"].to_numpy(zero_copy_only=False), nan=-1), return_counts=True)
                book_states[m] = {str(int(a)): int(b) for a, b in zip(sv, sc)}
            done_m[m] = dict(rows=tab.num_rows, columns=tab.column_names, book_states=book_states.get(m, {}))
            state_p.write_text(json.dumps(state))
            if drop_book_intermediates:
                for f in fl:
                    (bdir / f.name).unlink(missing_ok=True)
            cols_final = tab.column_names
        spec = json.loads((BARS_DIR / b.spec_id / "spec.json").read_text())
        manifest["bar_sets"][b.spec_id] = dict(name=b.name, spec=spec, rows=nrows, columns=cols_final,
                                               n_columns=len(cols_final or []), book_state_rows_by_month=book_states,
                                               model_feature_columns=model_columns(cols_final or []))
        log(f"  assembled {b.spec_id}: {nrows:,} rows, {len(cols_final or [])} columns")
    manifest["book_provenance"] = dict(
        book_mode=cfg.book.book_mode, legacy_chain=cfg.book.legacy_chain, checkpoint_cert_levels=cfg.book.checkpoint_cert_levels,
        rebuilt_cross_quarantine_s=cfg.book.rebuilt_cross_quarantine_s,
        tiers={"0": "anchored: snapshot + continuous updates", "9": "rebuilt without anchor, top exact by protocol certificate",
               "10": "rebuilt without anchor, top not certified, book features null"},
        per_row="book_state; certified reach in cert_bid_floor_dist_bps and cert_ask_ceiling_dist_bps")
    manifest["quality_columns"] = QUALITY_COLS
    manifest["quality_incidents"] = load_incidents()
    (out_root / "quality_incidents.json").write_text(json.dumps(manifest["quality_incidents"], indent=1))
    for k in ("trades_build.json", "mark_build.json", "ticker_build.json", "oi_build.json", "liq_build.json"):
        p = P.DERIVED / "canonical" / k
        if p.exists():
            manifest.setdefault("canonical_builds", {})[k] = json.loads(p.read_text())
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def build_all_bars(cfg: DatasetConfig, log=print, workers=4):
    if workers <= 1:
        for b in cfg.bars:
            build_bars(b, cfg.start, cfg.end, log=log)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_bars, b, cfg.start, cfg.end): b for b in cfg.bars}
        for f in as_completed(futs):
            log(f"  bars {futs[f].spec_id}: {f.result():,} bars")

"""Orchestration of the causal L2 replay over hourly files.

* book files are decoded in worker processes (prefetch), resolved + replayed sequentially;
* mark/index (staged) and canonical trades of the same hour are merged in the replay kernel;
* query times (bar eval times of any number of bar sets) are evaluated exactly;
* a 100 ms grid of top-of-book primitives is written per day;
* state is checkpointed daily; a run can resume from any saved checkpoint (equivalence tested).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import paths as P
from ..audit.run import staged_path
from ..guard import RunStopped, stop_reason
from . import chain as CH
from .core import ReplayState, rebase, replay_kernel
from .replay import GRID_COLS, ReplayConfig

MSG_FIELDS = ("recv", "evt", "txn", "et", "U", "u", "pu", "L", "hash", "synth")


def load_book_hour(day_iso: str, hour: int):
    """Decode one hourly book file into message metadata + flattened canonical rows (availability order)."""
    from ..book.messages import build_messages, gather_rows
    from ..rawio import read_raw
    d = date.fromisoformat(day_iso)
    path = P.raw_path(d, hour, "orderbook")
    if not path.exists():
        return None
    table, info = read_raw(path, with_hash=False)
    bm = build_messages(table)
    del table
    row_start, side, idx, qty, off = gather_rows(bm)
    return dict(recv=bm.m_recv, evt=bm.m_evt, txn=bm.m_txn, et=bm.m_et, U=bm.m_U, u=bm.m_u, pu=bm.m_pu, L=bm.m_L,
                hash=bm.m_hash, synth=bm.m_synth_only, row_start=row_start, side=side, idx=idx, qty=qty, off=off,
                n_rows_raw=bm.n_rows, ncopies=bm.m_ncopies)


def _concat_batch(carry, cur):
    """Prepend carried pending messages (with their rows) to the current file's messages."""
    if carry is None or carry["recv"].size == 0:
        return cur, 0
    if cur is None:
        return carry, carry["recv"].size
    out = {}
    for k in MSG_FIELDS:
        out[k] = np.concatenate([carry[k], cur[k]])
    nrc = carry["row_start"][-1]
    out["row_start"] = np.concatenate([carry["row_start"][:-1], cur["row_start"] + nrc])
    for k in ("side", "idx", "qty", "off"):
        out[k] = np.concatenate([carry[k], cur[k]])
    return out, carry["recv"].size


def _take_pending(batch, pend_idx):
    if pend_idx.size == 0:
        return None
    pend_idx = np.sort(pend_idx)
    out = {k: batch[k][pend_idx] for k in MSG_FIELDS}
    rs = batch["row_start"]
    sizes = rs[pend_idx + 1] - rs[pend_idx]
    out["row_start"] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    rows = np.concatenate([np.arange(rs[i], rs[i + 1]) for i in pend_idx])
    for k in ("side", "idx", "qty", "off"):
        out[k] = batch[k][rows]
    return out


def load_mark_hour(d: date, h: int):
    sp = staged_path("mark_price", d, h)
    if not sp.exists():
        return None
    t = pq.read_table(sp)
    recv = t["recv_ns"].to_numpy()
    evt = t["event_ms"].to_numpy()
    mk = t["mark_e8"].to_numpy()
    ix = t["index_e8"].to_numpy()
    synth = recv == evt * P.MS_NS
    # identical copies (same event_time and payload): keep earliest authentic receipt
    o = np.lexsort((recv, synth, ix, mk, evt))
    keep = np.ones(o.size, bool)
    keep[1:] = ~((evt[o][1:] == evt[o][:-1]) & (mk[o][1:] == mk[o][:-1]) & (ix[o][1:] == ix[o][:-1]))
    sel = o[keep]
    sel = sel[np.argsort(recv[sel], kind="stable")]
    ok = (mk[sel] > 0) & (ix[sel] > 0)
    sel = sel[ok]
    return recv[sel], evt[sel], mk[sel] / 1e8, ix[sel] / 1e8


class TradeSource:
    def __init__(self):
        self.day = None
        self.tab = None

    def hour(self, d: date, h: int):
        if self.day != d:
            p = P.DERIVED / "canonical" / "trades" / f"{d.isoformat()}.parquet"
            self.tab = pq.read_table(p, columns=["recv_ns", "trade_time_ms", "price_e2", "qty_e3", "sign"]) \
                if p.exists() else None
            self.day = d
            if self.tab is not None:
                self.recv = self.tab["recv_ns"].to_numpy()
        if self.tab is None:
            z = np.zeros(0, np.int64)
            return z, z, np.zeros(0), np.zeros(0), np.zeros(0)
        h0 = P.hour_start_ns(d, h)
        a = np.searchsorted(self.recv, h0)
        b = np.searchsorted(self.recv, h0 + P.HOUR_NS)
        s = self.tab.slice(a, b - a)
        return (s["recv_ns"].to_numpy(), s["trade_time_ms"].to_numpy(), s["price_e2"].to_numpy() / 100.0,
                s["qty_e3"].to_numpy() / 1000.0, s["sign"].to_numpy().astype(np.float64))


def snapshot_hours():
    """Hour keys (day_iso, hour) whose book file contains at least one snapshot message (from the exhaustive audit)."""
    p = P.DERIVED / "audit" / "files_orderbook.parquet"
    if not p.exists():
        return None
    t = pq.read_table(p, columns=["day", "hour", "s_n_msgs_snapshot"]).to_pydict()
    return {(d, h) for d, h, n in zip(t["day"], t["hour"], t["s_n_msgs_snapshot"]) if n and n > 0}


def run_replay(start: date, end: date, query_times: np.ndarray, out_dir: Path, cfg: ReplayConfig = None,
               workers: int = 4, resume_from: Path | None = None, checkpoint_dir: Path | None = None,
               write_grid: bool = True, chain_timeout_s: float = 60.0, log=print, hours=None,
               features_out_dir: Path | None = None, heal: bool = False, skip_unanchored: bool = True,
               legacy_chain: bool = False, chain_probe: int | None = None, start_after: int = 0,
               top_levels: int = 0, loader=None, trades: "TradeSource | None" = None, extras: dict | None = None,
               chain_heal: int | None = None):
    """Replay [start, end] (inclusive days). query_times: sorted int64 eval times (ns, inclusive as-of).
    Returns (features matrix aligned with query_times, column names, per-hour stats list).

    Experiment hooks (defaults leave the production path unchanged): chain_probe overrides the resolver's snapshot policy
    (3 = this book never sees a snapshot), start_after = earliest empty start in rebuilt mode, top_levels > 0 stores the best
    levels of each side per query in extras["top"], loader/trades share decoded hours between runs."""
    cfg = cfg or ReplayConfig()
    cols = cfg.columns()
    pa_ = cfg.param_arrays()
    state = ReplayState(cfg)
    cst, ring_u, ring_h = CH.new_state()
    carry = None
    if resume_from is not None:
        state.load_into(resume_from / "replay_state.npz")
        z = np.load(resume_from / "chain_state.npz", allow_pickle=True)
        cst[:] = z["st"]
        ring_u[:] = z["ring_u"]
        ring_h[:] = z["ring_h"]
        carry = z["carry"].item() if z["carry"].size and z["carry"].item() is not None else None
    # in streaming mode (features_out_dir) results are written per UTC day and not kept in memory
    out = None if features_out_dir is not None else np.full((query_times.size, len(cols)), np.nan)
    day_q, day_o = [], []
    feat_day = None
    hour_list = hours if hours is not None else list(P.iter_hours(start, end))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    verify_log = np.zeros((100000, 9), np.int64)
    trades = trades or TradeSource()
    load_hour = loader or load_book_hour
    xo_all = np.full((query_times.size, 4 * top_levels), -1, np.int64)
    grid_rows = []
    grid_day = None
    t_start = time.time()
    ex = ProcessPoolExecutor(max_workers=workers) if workers > 0 else None
    futs = {}
    ahead = max(2 * workers, 1)

    rebuilt = cfg.book_mode == "rebuilt"
    snaps = snapshot_hours() if (skip_unanchored and not heal and not rebuilt) else None

    def needs_decode(i):
        # hour i is needed if it has a snapshot, or if the next hour has one (client-side buffer of the tail of hour i)
        d, h = hour_list[i]
        if (d.isoformat(), h) in snaps:
            return True
        if i + 1 < len(hour_list):
            d2, h2 = hour_list[i + 1]
            return (d2.isoformat(), h2) in snaps
        return False

    def skippable(i):
        # exact in strict mode: an unanchored resolver can only leave M_NONE through a snapshot
        if snaps is None or cst[CH.ST_MODE] != CH.M_NONE:
            return False
        return not needs_decode(i)

    def submit(i):
        if ex is not None and i < len(hour_list) and i not in futs:
            d, h = hour_list[i]
            if snaps is not None and cst[CH.ST_MODE] == CH.M_NONE and not needs_decode(i):
                return   # likely skippable: decode lazily only if needed
            futs[i] = ex.submit(load_book_hour, d.isoformat(), h)

    for i in range(min(ahead, len(hour_list))):
        submit(i)
    prev_hour_present = None
    try:
        for i, (d, h) in enumerate(hour_list):
            t0 = time.time()
            submit(i + ahead)
            skipped = False
            if skippable(i):
                fut = futs.pop(i, None)
                if fut is not None:
                    fut.cancel()
                cur = None if not P.raw_path(d, h, "orderbook").exists() else "skip"
                skipped = cur == "skip"
                cur = None
                carry = None   # the rolling client buffer is not needed: no snapshot in this hour or the next
            elif ex is not None and i in futs:
                cur = futs.pop(i).result()
            else:
                cur = load_hour(d.isoformat(), h)
            h0 = P.hour_start_ns(d, h)
            forced_inval = False
            if cur is None and not skipped and cst[CH.ST_MODE] != CH.M_NONE:
                # a missing hourly file is a gap: never assume automatic recovery
                cst[CH.ST_MODE] = CH.M_NONE
                cst[CH.ST_GAP_SINCE] = -1
                carry = None
                forced_inval = True
            batch, ncar = _concat_batch(carry, cur)
            if batch is not None and batch["recv"].size:
                (seq_msg, seq_time, seq_kind, seq_aux, action, ev_kind, ev_t0, ev_t1, ev_u0, ev_u1, pend) = CH.resolve(
                    batch["recv"], batch["et"], batch["U"], batch["u"], batch["pu"], batch["L"], batch["hash"],
                    cst, ring_u, ring_h, ncar, int(chain_timeout_s * 1e9),
                    (2 if rebuilt else (1 if heal else 0)) if chain_heal is None else chain_heal,
                    int(cfg.snapshot_probe_only) if chain_probe is None else chain_probe,
                    1 if (legacy_chain or cfg.legacy_chain) else 0,
                    start_after)
                carry = _take_pending(batch, pend)
            else:
                seq_msg = np.zeros(0, np.int64)
                seq_time = np.zeros(0, np.int64)
                seq_kind = np.zeros(0, np.int8)
                seq_aux = np.zeros(0, np.int64)
                action = np.zeros(0, np.int8)
                ev_kind = np.zeros(0, np.int8)
                ev_t0 = ev_t1 = ev_u0 = ev_u1 = np.zeros(0, np.int64)
                batch = dict(evt=np.zeros(1, np.int64), txn=np.zeros(1, np.int64), synth=np.zeros(1, bool),
                             row_start=np.zeros(2, np.int64), side=np.zeros(0, np.int8), idx=np.zeros(0, np.int64),
                             qty=np.zeros(0, np.int64), off=np.zeros(0, bool))
            if forced_inval:
                seq_msg = np.concatenate([[-1], seq_msg]).astype(np.int64)
                seq_time = np.concatenate([[h0], seq_time]).astype(np.int64)
                seq_kind = np.concatenate([[CH.Q_INVALIDATE], seq_kind]).astype(np.int8)
                seq_aux = np.concatenate([[-1], seq_aux]).astype(np.int64)
            mk = load_mark_hour(d, h)
            if mk is None:
                z = np.zeros(0, np.int64)
                mk = (z, z, np.zeros(0), np.zeros(0))
            tr = trades.hour(d, h)
            qa_ = np.searchsorted(query_times, h0)
            qb_ = np.searchsorted(query_times, h0 + P.HOUR_NS)
            qsl = query_times[qa_:qb_]
            outh = np.full((qsl.size, len(cols)), np.nan)
            gstep = pa_["ip"][18]
            g_t = np.arange(h0 + gstep, h0 + P.HOUR_NS + 1, gstep, dtype=np.int64) if write_grid else np.zeros(0, np.int64)
            g_out = np.zeros((g_t.size, len(GRID_COLS)), np.int64)
            nver = replay_kernel(seq_msg, seq_time, seq_kind, seq_aux,
                                 batch["evt"], batch["txn"], batch["synth"], batch["row_start"], batch["side"],
                                 batch["idx"], batch["qty"], batch["off"],
                                 mk[0], mk[1], mk[2], mk[3],
                                 tr[0], tr[1], tr[2], tr[3], tr[4],
                                 qsl, outh, xo_all[qa_:qb_], g_t, g_out,
                                 *state.kernel_args(),
                                 pa_["ip"], pa_["bands_c"], pa_["ks"], pa_["win_ns"], pa_["win_mlofi"],
                                 pa_["win_minmax"], pa_["tw_bands"], pa_["cost_q"], pa_["cost_d"], pa_["bwin_ns"],
                                 pa_["eswin_ns"], pa_["mo_h_ns"], pa_["mowin_ns"], pa_["min_cov"], verify_log)
            if out is not None:
                out[qa_:qb_] = outh
            else:
                if feat_day is not None and feat_day != d:
                    _write_feat_day(features_out_dir, feat_day, day_q, day_o, cols)
                    day_q, day_o = [], []
                feat_day = d
                day_q.append(qsl)
                day_o.append(outh)
            rebase(state.cum, state.ring_c, state.si, state.es_cum, state.es_c, state.mo_cum, state.mo_c)
            # ---- per-hour statistics
            ac = np.bincount(action, minlength=len(CH.ACTION_NAMES)) if action.size else np.zeros(len(CH.ACTION_NAMES), int)
            evc = np.bincount(ev_kind, minlength=len(CH.EVENT_NAMES)) if ev_kind.size else np.zeros(len(CH.EVENT_NAMES), int)
            st_codes = g_out[:, 1] if g_t.size else np.zeros(0, np.int64)
            rec = dict(day=d.isoformat(), hour=h, file=(cur is not None) or skipped, skipped_unanchored=skipped,
                       n_msgs=int(cur["recv"].size) if cur else 0,
                       n_carried=int(ncar), forced_invalidation=forced_inval,
                       **{f"act_{n}": int(ac[k]) for k, n in enumerate(CH.ACTION_NAMES)},
                       **{f"ev_{n}": int(evc[k]) for k, n in enumerate(CH.EVENT_NAMES)},
                       buffered_delay_ms_max=float(((seq_time - batch["recv"][np.maximum(seq_msg, 0)])[seq_msg >= 0]).max() / 1e6)
                       if (seq_msg >= 0).any() else 0.0,
                       verify=int(nver), n_queries=int(qsl.size), secs=time.time() - t0)
            for code in range(11):
                rec[f"grid_state_{code}"] = int(np.count_nonzero(st_codes == code))
            for k in range(nver):
                stats.append(dict(kind="verify", t=int(verify_log[k, 0]), mismatches=int(verify_log[k, 1]),
                                  snapshot_levels=int(verify_log[k, 2]), overlay=int(verify_log[k, 4]),
                                  healed=int(verify_log[k, 5]), ms_since_heal=int(verify_log[k, 6]),
                                  mismatches_within_10bps=int(verify_log[k, 7]), top_equal=int(verify_log[k, 8])))
            events = [dict(kind=CH.EVENT_NAMES[int(ev_kind[j])], t0=int(ev_t0[j]), t1=int(ev_t1[j]), u0=int(ev_u0[j]),
                           u1=int(ev_u1[j])) for j in range(ev_kind.size)]
            rec["events"] = json.dumps(events)
            stats.append(rec)
            # ---- grid output (per day)
            if write_grid:
                if grid_day is not None and grid_day != d:
                    _write_grid(out_dir, grid_day, grid_rows)
                    grid_rows = []
                grid_day = d
                grid_rows.append(g_out)
            if checkpoint_dir is not None and h == 23:
                # the outputs of the day reach disk before its checkpoint, so resuming from it never loses a day
                if out is None:
                    _write_feat_day(features_out_dir, d, day_q, day_o, cols)
                    day_q, day_o, feat_day = [], [], None
                if write_grid:
                    _write_grid(out_dir, d, grid_rows)
                    grid_rows, grid_day = [], None
                # per-day statistics are persisted incrementally (survive interruptions; memory stays bounded)
                sd = checkpoint_dir / "_stats"
                sd.mkdir(parents=True, exist_ok=True)
                hrs = [x for x in stats if x.get("kind") != "verify"]
                ver = [x for x in stats if x.get("kind") == "verify"]
                if hrs:
                    pq.write_table(pa.Table.from_pylist(hrs), sd / f"hours_{d.isoformat()}.parquet")
                if ver:
                    pq.write_table(pa.Table.from_pylist(ver), sd / f"verify_{d.isoformat()}.parquet")
                stats = []
                ck = checkpoint_dir / d.isoformat()
                ck.mkdir(parents=True, exist_ok=True)
                state.save(ck / "replay_state.npz")
                np.savez(ck / "chain_state.npz", st=cst, ring_u=ring_u, ring_h=ring_h,
                         carry=np.array(carry, dtype=object))
                # retention: keep month-end checkpoints and the latest one
                prev = d - timedelta(days=1)
                pk = checkpoint_dir / prev.isoformat()
                if pk.exists() and (prev + timedelta(days=1)).month == prev.month:
                    for f in pk.iterdir():
                        f.unlink()
                    pk.rmdir()
                why = stop_reason()
                if why:
                    raise RunStopped(f"book stopped after the checkpoint of {d}: {why}")
            if h == 0 or i == len(hour_list) - 1:
                el = time.time() - t_start
                log(f"  replay {d} h{h:02d} msgs={rec['n_msgs']} state0={rec.get('grid_state_0')} "
                    f"applied={rec['act_apply'] + rec['act_apply_buffered']} events={evc.tolist()} {el:.0f}s")
    finally:
        if ex is not None:
            ex.shutdown(cancel_futures=True)
    if write_grid and grid_rows:
        _write_grid(out_dir, grid_day, grid_rows)
    if out is None and feat_day is not None:
        _write_feat_day(features_out_dir, feat_day, day_q, day_o, cols)
    if extras is not None:
        extras["top"] = xo_all
        extras["verify_log"] = verify_log
    from . import replay as RP
    counters = {k: int(state.si[getattr(RP, x)]) for k, x in [
        ("nmsg_applied", "I_NAPPLY"), ("nlevels", "I_NLEV"), ("nverify", "I_NVERIFY"), ("nverify_fail", "I_NVERIFY_FAIL"),
        ("ncross", "I_NCROSS"), ("ncross_inval", "I_NCROSS_INVAL"), ("noff_pos", "I_NOFFTICK_POS"),
        ("nanchor", "I_NANCHOR"), ("ninval", "I_NINVAL"), ("nes", "I_NES"), ("nmo", "I_NMO"),
        ("verify_mismatch_levels", "I_VERIFY_MISMATCH_LEVELS"), ("nempty", "I_NEMPTY")]}
    return out, cols, stats, counters, state


def _write_feat_day(fdir, d, qs, os_, cols):
    fdir.mkdir(parents=True, exist_ok=True)
    q = np.concatenate(qs) if qs else np.zeros(0, np.int64)
    o = np.concatenate(os_) if os_ else np.zeros((0, len(cols)))
    tab = {"eval_time_ns": q}
    for j, c in enumerate(cols):
        tab[c] = o[:, j]
    tmp = fdir / f".{d.isoformat()}.tmp"
    pq.write_table(pa.table(tab), tmp, compression="zstd")
    tmp.replace(fdir / f"{d.isoformat()}.parquet")


def _write_grid(out_dir, d, rows):
    g = np.concatenate(rows)
    tab = pa.table({c: g[:, j] for j, c in enumerate(GRID_COLS)})
    p = out_dir / "grid100ms"
    p.mkdir(parents=True, exist_ok=True)
    pq.write_table(tab, p / f"{d.isoformat()}.parquet", compression="zstd")

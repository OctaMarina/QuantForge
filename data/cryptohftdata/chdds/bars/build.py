"""Bar-set specifications, causal threshold schedules, resumable day-by-day construction.

Output (append-only, one file per UTC day of the availability clock):
  derived/bars/<spec_id>/YYYY-MM-DD.parquet   complete bars + partial bars closed by gaps/resets
  derived/bars/<spec_id>/_state/YYYY-MM-DD.npz engine state after that day (resume point)
  derived/bars/<spec_id>/_partial_end_<end>.parquet  the still-open bar at the end of the built range (never
                                               merged into day files; regenerated on extension)
  derived/bars/<spec_id>/spec.json             normalised spec + version + threshold schedule
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import paths as P
from . import engines as E

BARS_DIR = P.DERIVED / "bars"
TRADES_DIR = P.DERIVED / "canonical" / "trades"


@dataclass
class BarSpec:
    kind: str                       # time | volume | dollar | cusum
    threshold: float | None = None  # seconds | BTC | USDT | bps (fixed schedule)
    adaptive: dict | None = None    # see threshold_schedule()
    cusum_reset: str = "triggered"  # triggered | both
    daily_reset: bool = False
    origin_ns: int = 0              # time-bar grid origin (UTC epoch)
    name: str = ""

    def normalised(self):
        d = asdict(self)
        d.pop("name")
        d["version"] = E.BAR_SPEC_VERSION
        return d

    @property
    def spec_id(self):
        h = hashlib.sha1(json.dumps(self.normalised(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.name or self.kind}-{h}"


def day_index(d: date) -> int:
    return (datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)).days


def load_trades(d: date):
    p = TRADES_DIR / f"{d.isoformat()}.parquet"
    if not p.exists():
        return None
    return pq.read_table(p)


# ------------------------------------------------------------------------- causal threshold schedules
def _day_stats_for_schedule(d: date, target_s: float):
    """Per-day statistics used by adaptive schedules (computed from that day's trades only):
    volume/notional in consecutive target_s windows (availability clock) and 1-min log returns."""
    t = load_trades(d)
    if t is None or t.num_rows == 0:
        return None
    recv = t["recv_ns"].to_numpy()
    q = t["qty_e3"].to_numpy()
    px = t["price_e2"].to_numpy()
    d0 = day_index(d) * E.DAY_NS
    w = int(target_s * 1e9)
    cell = (recv - d0) // w
    ncell = int(E.DAY_NS // w)
    vol = np.bincount(cell, weights=q, minlength=ncell)[:ncell]
    notl = np.bincount(cell, weights=q * px.astype(np.float64), minlength=ncell)[:ncell]
    # 1-minute last-price grid (as-of) returns
    m = (recv - d0) // 60_000_000_000
    last = np.full(1440, np.nan)
    idx = np.searchsorted(m, np.arange(1440), side="right") - 1
    ok = idx >= 0
    last[ok] = px[idx[ok]]
    r = np.diff(np.log(last))
    return dict(vol=vol, notional=notl, r1m=r[np.isfinite(r)])


def threshold_schedule(spec: BarSpec, start: date, end: date, log=print):
    """Integer threshold per day for days [start, end], using only strictly previous days.
    volume -> e3 BTC, dollar -> e5 USDT (price_e2*qty_e3 units), cusum -> 1e-2 bp, time -> ns.
    adaptive: {"target_s": 60, "lookback_days": 7, "stat": "median"} for volume/dollar;
              {"c": 1.0, "lookback_days": 7, "scale_s": 60} for cusum (h = c * std of 1-min log returns)."""
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    if spec.adaptive is None:
        if spec.kind == "time":
            v = int(round(spec.threshold * 1e9))
        elif spec.kind == "volume":
            v = int(round(spec.threshold * 1e3))
        elif spec.kind == "dollar":
            v = int(round(spec.threshold * 1e5))
        else:
            v = int(round(spec.threshold * 100))
        return {dd: v for dd in days}, {}
    a = spec.adaptive
    lb = int(a.get("lookback_days", 7))
    sched = {}
    info = {}
    cache = {}
    for dd in days:
        hist = []
        for k in range(1, lb + 1):
            pd_ = dd - timedelta(days=k)
            if pd_ not in cache:
                cache[pd_] = _day_stats_for_schedule(pd_, a.get("target_s", 60))
            if cache[pd_] is not None:
                hist.append(cache[pd_])
        if len(hist) < int(a.get("min_days", lb)):
            sched[dd] = None          # warm-up: no threshold yet -> no bars emitted that day
            info[dd.isoformat()] = "warmup"
            continue
        if spec.kind in ("volume", "dollar"):
            key = "vol" if spec.kind == "volume" else "notional"
            x = np.concatenate([h[key] for h in hist])
            x = x[x > 0]
            v = float(np.median(x))
            sched[dd] = int(round(v)) if spec.kind == "volume" else int(round(v))  # already e3 / e5 units
            info[dd.isoformat()] = v / (1e3 if spec.kind == "volume" else 1e5)
        else:
            r = np.concatenate([h["r1m"] for h in hist])
            sd = float(np.std(r))
            bps = a.get("c", 1.0) * sd * 1e4
            sched[dd] = max(1, int(round(bps * 100)))
            info[dd.isoformat()] = sched[dd] / 100
    return sched, info


# ------------------------------------------------------------------------- construction
def _to_table(out_i, out_f, spec_id, bar_id0):
    n = out_i.shape[0]
    cols = {c: out_i[:, j] for j, c in enumerate(E.I_COLS)}
    cols.update({c: out_f[:, j] for j, c in enumerate(E.F_COLS)})
    cols["bar_id"] = np.arange(bar_id0, bar_id0 + n, dtype=np.int64)
    return pa.table(cols)


def build_bars(spec: BarSpec, start: date, end: date, resume: bool = True, finalize_partial: bool = True,
               log=print):
    out_dir = BARS_DIR / spec.spec_id
    st_dir = out_dir / "_state"
    st_dir.mkdir(parents=True, exist_ok=True)
    sched, info = threshold_schedule(spec, start, end, log)
    (out_dir / "spec.json").write_text(json.dumps(dict(spec=spec.normalised(), name=spec.name, spec_id=spec.spec_id,
                                                       schedule_info=info), indent=1, default=str))
    state = E.new_event_state()
    bar_id = 0
    d = start
    if resume:
        # incremental: continue after the latest day already built inside [start, end] (its state is the
        # resume point); otherwise resume from the state saved for `start - 1` if any.
        built = sorted(p.stem for p in st_dir.glob("*.npz") if start.isoformat() <= p.stem <= end.isoformat())
        if built and (out_dir / f"{built[-1]}.parquet").exists():
            last = date.fromisoformat(built[-1])
            z = np.load(st_dir / f"{built[-1]}.npz")
            state = z["state"].copy()
            bar_id = int(z["bar_id"])
            d = last + timedelta(days=1)
        else:
            prev = start - timedelta(days=1)
            sp = st_dir / f"{prev.isoformat()}.npz"
            if sp.exists():
                z = np.load(sp)
                state = z["state"].copy()
                bar_id = int(z["bar_id"])
    # a run started at a later date without saved state: the first event bar does not contain the trades before
    # `start` that the full-history construction would have aggregated -> declared partial (report §7.5 pt. 8)
    first_day = min((p.stem for p in TRADES_DIR.glob("*.parquet")), default=start.isoformat())
    fresh_partial = (state[E.S_SEG] == -1) and start.isoformat() > first_day and spec.kind != "time"
    total = 0
    while d <= end:
        t = load_trades(d)
        di = day_index(d)
        if t is None or t.num_rows == 0:
            d += timedelta(days=1)
            continue
        seq = t["seq"].to_numpy()
        tid = t["trade_id"].to_numpy()
        recv = t["recv_ns"].to_numpy()
        tt = t["trade_time_ms"].to_numpy()
        px = t["price_e2"].to_numpy()
        qx = t["qty_e3"].to_numpy()
        sg = t["sign"].to_numpy().astype(np.int64)
        seg = t["seg"].to_numpy().astype(np.int64)
        CH = 200_000          # bounded memory: the engine state carries across chunks (exactly equivalent)
        outs_i, outs_f = [], []
        n = seq.size
        if spec.kind == "time":
            dt = sched[d]
            t_from = di * E.DAY_NS
            day_end = (di + 1) * E.DAY_NS
            for a in range(0, n, CH):
                b = min(a + CH, n)
                t_until = int(recv[b]) if b < n else day_end
                oi_, of_, k_ = E.time_bars(seq[a:b], tid[a:b], recv[a:b], tt[a:b], px[a:b], qx[a:b], sg[a:b], seg[a:b],
                                           dt, spec.origin_ns, state, t_from, t_until)
                outs_i.append(oi_)
                outs_f.append(of_)
        else:
            thr = sched.get(d)
            if thr is None:
                d += timedelta(days=1)
                continue
            mode = {"volume": 0, "dollar": 1, "cusum": 2}[spec.kind]
            day_thr = np.array([thr], np.int64)
            for a in range(0, n, CH):
                b = min(a + CH, n)
                oi_, of_, k_ = E.event_bars(mode, seq[a:b], tid[a:b], recv[a:b], tt[a:b], px[a:b], qx[a:b], sg[a:b],
                                            seg[a:b], day_thr, di, state, spec.cusum_reset == "both", spec.daily_reset)
                outs_i.append(oi_)
                outs_f.append(of_)
        oi = np.concatenate(outs_i) if outs_i else np.zeros((0, len(E.I_COLS)), np.int64)
        of = np.concatenate(outs_f) if outs_f else np.zeros((0, len(E.F_COLS)))
        k = oi.shape[0]
        if fresh_partial and k > 0:
            oi = oi.copy()
            oi[0, E.IC["complete"]] = 0
            oi[0, E.IC["reason"]] = E.R_FRESH_START
            fresh_partial = False
        tab = _to_table(oi, of, spec.spec_id, bar_id)
        bar_id += k
        total += k
        tmp = out_dir / f".{d.isoformat()}.parquet.tmp"
        pq.write_table(tab, tmp, compression="zstd")
        tmp.replace(out_dir / f"{d.isoformat()}.parquet")
        np.savez(st_dir / f"{d.isoformat()}.npz", state=state, bar_id=bar_id)
        if d.day == 1:
            log(f"  bars {spec.spec_id} {d} total={total:,}")
        d += timedelta(days=1)
    # still-open event bar at the end of the built range: written separately as partial
    if finalize_partial and spec.kind != "time" and state[E.S_OPEN] == 1:
        oi = np.zeros((1, len(E.I_COLS)), np.int64)
        of = np.full((1, len(E.F_COLS)), np.nan)
        E._emit(oi, of, 0, state, E.R_END, 0, 0, state[E.S_THR], 0, np.nan, np.nan, np.nan, np.nan)
        pq.write_table(_to_table(oi, of, spec.spec_id, bar_id), out_dir / f"_partial_end_{end.isoformat()}.parquet")
    return total


def read_bars(spec_id: str, start: date | None = None, end: date | None = None):
    out_dir = BARS_DIR / spec_id
    files = sorted(p for p in out_dir.glob("*.parquet") if not p.name.startswith("_"))
    if start:
        files = [f for f in files if f.stem >= start.isoformat()]
    if end:
        files = [f for f in files if f.stem <= end.isoformat()]
    if not files:
        return None
    return pa.concat_tables([pq.read_table(f) for f in files])

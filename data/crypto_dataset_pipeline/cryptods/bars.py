"""Exact, restartable bar construction from individual trades.

Rules implemented (see docs/BARS.md):

* Trades are processed in exchange sequence order (trade id); identical timestamps are ordered by id.
* Time bars: UTC half-open buckets [k*D, (k+1)*D) anchored at the Unix epoch (stable for any export
  range).  With a complete feed an interval without trades is a bar with volume 0, n_trades 0 and
  null OHLC.  An interval overlapping a feed gap is *incomplete* (its OHLC is never presented as
  complete).
* Volume / notional bars: a bar closes at the first trade whose cumulative volume (resp. quote
  notional) reaches or exceeds the threshold; the crossing trade is included entirely (overshoot is
  kept, no redistribution); the next bar starts with the next trade.  Accumulation is never reset at
  file/day/batch boundaries; it starts at ``bar_origin`` and only restarts at an unrecoverable gap,
  which opens a new *segment* (explicit policy ``invalidate_and_restart_segment``).
* All accumulations of prices/volumes/notional are exact int64 at instrument scale.  Float sums
  used by the per-bar regressions are accumulated sequentially in trade order, so results are
  bit-identical for one-shot, batched and restarted processing.
* The open (unfinished) bar is part of the persisted state and is exported only as *partial*.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import timeutil as tu
from .storage import atomic_write_json, read_json, write_parquet

try:  # numba is optional; the pure-Python path computes identical results (slowly)
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

MODE_TIME, MODE_VOLUME, MODE_NOTIONAL = 0, 1, 2
STATUS_OK, STATUS_GAP_INVALIDATED, STATUS_INCOMPLETE, STATUS_EMPTY = 0, 1, 2, 3
STATUS_NAMES = {0: "closed", 1: "invalidated_by_gap", 2: "incomplete_interval", 3: "empty_interval"}

# ---- int state slots
(S_HAS_OPEN, S_SEQ_NEXT, S_SEGMENT, S_OPEN_START, S_FIRST_TS, S_LAST_TS, S_FIRST_TID, S_LAST_TID,
 S_OPEN, S_HIGH, S_LOW, S_CLOSE, S_NMSG, S_NEXEC, S_VOL, S_NOT, S_BUYV, S_SELLV, S_UNKV, S_BUYN,
 S_SELLN, S_UNKN, S_NBUY, S_NSELL, S_TBUYV, S_TSELLV, S_TUNKV, S_TBUYN, S_TSELLN, S_TUNKN, S_ACC,
 S_THR_OPEN, S_LAST_PX, S_LAST_SIGN, S_OPEN_INCOMPLETE, S_NEXT_BUCKET, S_LAST_CLOSE, S_PROCESSED,
 S_NREG, S_LAST_TS_SEEN, S_KT_N, S_KA_N, S_HT_N, S_HA_N, S_T_UNK_N, S_A_UNK_N, S_INCOMPLETE_UNTIL,
 S_OPEN_BUCKET) = range(48)
NI = 48
# ---- float state slots
(F_KT_XX, F_KT_XY, F_KT_YY, F_KA_XX, F_KA_XY, F_KA_YY, F_HT_XX, F_HT_XY, F_HT_YY, F_HA_XX, F_HA_XY,
 F_HA_YY, F_XB_T, F_XB_A) = range(14)
NF = 14

OUT_INT = ["bar_seq", "segment_id", "bar_start_us", "bar_end_us", "first_trade_ts_us", "last_trade_ts_us",
           "first_trade_id", "last_trade_id", "open_i", "high_i", "low_i", "close_i", "n_messages", "n_trades",
           "volume_i", "notional_i", "buy_volume_i", "sell_volume_i", "unknown_side_volume_i",
           "buy_notional_i", "sell_notional_i", "unknown_side_notional_i", "n_buy", "n_sell",
           "tick_buy_volume_i", "tick_sell_volume_i", "tick_unknown_volume_i", "tick_buy_notional_i",
           "tick_sell_notional_i", "tick_unknown_notional_i", "threshold_i", "complete", "status",
           "kyle_tick_n", "kyle_aggr_n", "hasb_tick_n", "hasb_aggr_n", "tick_unknown_n", "aggr_unknown_n"]
OUT_FLT = ["kyle_tick_sxx", "kyle_tick_sxy", "kyle_tick_syy", "kyle_aggr_sxx", "kyle_aggr_sxy", "kyle_aggr_syy",
           "hasb_tick_sxx", "hasb_tick_sxy", "hasb_tick_syy", "hasb_aggr_sxx", "hasb_aggr_sxy", "hasb_aggr_syy",
           "hasb_tick_xb", "hasb_aggr_xb"]
NOI, NOF = len(OUT_INT), len(OUT_FLT)


@njit(cache=True)
def _reset_bar(st, sf):
    st[S_HAS_OPEN] = 0
    for k in (S_FIRST_TS, S_LAST_TS, S_FIRST_TID, S_LAST_TID, S_OPEN, S_HIGH, S_LOW, S_CLOSE, S_NMSG,
              S_NEXEC, S_VOL, S_NOT, S_BUYV, S_SELLV, S_UNKV, S_BUYN, S_SELLN, S_UNKN, S_NBUY, S_NSELL,
              S_TBUYV, S_TSELLV, S_TUNKV, S_TBUYN, S_TSELLN, S_TUNKN, S_ACC, S_OPEN_INCOMPLETE,
              S_KT_N, S_KA_N, S_HT_N, S_HA_N, S_T_UNK_N, S_A_UNK_N):
        st[k] = 0
    for k in range(NF):
        sf[k] = 0.0


@njit(cache=True)
def _emit(st, sf, oi, of, row, end_us, status, complete):
    oi[row, 0] = st[S_SEQ_NEXT]
    oi[row, 1] = st[S_SEGMENT]
    oi[row, 2] = st[S_OPEN_START]
    oi[row, 3] = end_us
    oi[row, 4] = st[S_FIRST_TS]
    oi[row, 5] = st[S_LAST_TS]
    oi[row, 6] = st[S_FIRST_TID]
    oi[row, 7] = st[S_LAST_TID]
    oi[row, 8] = st[S_OPEN]
    oi[row, 9] = st[S_HIGH]
    oi[row, 10] = st[S_LOW]
    oi[row, 11] = st[S_CLOSE]
    oi[row, 12] = st[S_NMSG]
    oi[row, 13] = st[S_NEXEC]
    oi[row, 14] = st[S_VOL]
    oi[row, 15] = st[S_NOT]
    oi[row, 16] = st[S_BUYV]
    oi[row, 17] = st[S_SELLV]
    oi[row, 18] = st[S_UNKV]
    oi[row, 19] = st[S_BUYN]
    oi[row, 20] = st[S_SELLN]
    oi[row, 21] = st[S_UNKN]
    oi[row, 22] = st[S_NBUY]
    oi[row, 23] = st[S_NSELL]
    oi[row, 24] = st[S_TBUYV]
    oi[row, 25] = st[S_TSELLV]
    oi[row, 26] = st[S_TUNKV]
    oi[row, 27] = st[S_TBUYN]
    oi[row, 28] = st[S_TSELLN]
    oi[row, 29] = st[S_TUNKN]
    oi[row, 30] = st[S_THR_OPEN]
    oi[row, 31] = complete
    oi[row, 32] = status
    oi[row, 33] = st[S_KT_N]
    oi[row, 34] = st[S_KA_N]
    oi[row, 35] = st[S_HT_N]
    oi[row, 36] = st[S_HA_N]
    oi[row, 37] = st[S_T_UNK_N]
    oi[row, 38] = st[S_A_UNK_N]
    for k in range(NF):
        of[row, k] = sf[k]
    st[S_SEQ_NEXT] += 1


@njit(cache=True)
def _emit_empty_until(st, sf, oi, of, row, bucket_stop, interval):
    """Emit empty time bars for buckets [S_NEXT_BUCKET, bucket_stop)."""
    b = st[S_NEXT_BUCKET]
    while b < bucket_stop:
        _reset_bar(st, sf)
        st[S_OPEN_START] = b * interval
        incomplete = 1 if b <= st[S_INCOMPLETE_UNTIL] else 0
        if incomplete == 1:
            _emit(st, sf, oi, of, row, (b + 1) * interval, STATUS_INCOMPLETE, 0)
        else:
            _emit(st, sf, oi, of, row, (b + 1) * interval, STATUS_EMPTY, 1)
        row += 1
        b += 1
    if bucket_stop > st[S_NEXT_BUCKET]:
        st[S_NEXT_BUCKET] = bucket_stop
    return row


@njit(cache=True)
def _close_time_bar(st, sf, oi, of, row, interval):
    b = st[S_OPEN_BUCKET]
    incomplete = 1 if (st[S_OPEN_INCOMPLETE] == 1 or b <= st[S_INCOMPLETE_UNTIL]) else 0
    status = STATUS_INCOMPLETE if incomplete == 1 else STATUS_OK
    _emit(st, sf, oi, of, row, (b + 1) * interval, status, 1 - incomplete)
    st[S_NEXT_BUCKET] = b + 1
    _reset_bar(st, sf)
    return row + 1


@njit(cache=True)
def process_kernel(ts, tid, seq_first, seq_last, px, qty, notl, aggr, nexec,
                   mode, interval, thr_new, price_scale, qty_scale, notional_scale,
                   watermark, st, sf, oi, of):
    """Sequentially process trades; returns number of emitted rows.

    ``watermark``: (time mode) all trades with ts < watermark are known to have been delivered, so
    buckets ending at or before it can be closed.  Use -1 for "no watermark".
    """
    row = 0
    n = ts.shape[0]
    for i in range(n):
        if st[S_PROCESSED] >= 0 and seq_last[i] <= st[S_PROCESSED]:
            continue  # already processed (overlapping batch / restart)
        t = ts[i]
        if t < st[S_LAST_TS_SEEN]:
            st[S_NREG] += 1
        else:
            st[S_LAST_TS_SEEN] = t
        if mode == MODE_TIME:
            b = t // interval
            if st[S_HAS_OPEN] == 1 and b > st[S_OPEN_BUCKET]:
                row = _close_time_bar(st, sf, oi, of, row, interval)
            if st[S_HAS_OPEN] == 0:
                if b < st[S_NEXT_BUCKET]:
                    raise ValueError("trade belongs to an already closed time bucket (watermark violated)")
                row = _emit_empty_until(st, sf, oi, of, row, b, interval)
                _reset_bar(st, sf)
                st[S_HAS_OPEN] = 1
                st[S_OPEN_BUCKET] = b
                st[S_OPEN_START] = b * interval
                st[S_THR_OPEN] = 0
        else:
            if st[S_HAS_OPEN] == 0:
                _reset_bar(st, sf)
                st[S_HAS_OPEN] = 1
                st[S_OPEN_START] = st[S_LAST_CLOSE]
                st[S_THR_OPEN] = thr_new
        p = px[i]
        q = qty[i]
        nv = notl[i]
        if st[S_NMSG] == 0:
            st[S_FIRST_TS] = t
            st[S_FIRST_TID] = seq_first[i]
            st[S_OPEN] = p
            st[S_HIGH] = p
            st[S_LOW] = p
        else:
            if p > st[S_HIGH]:
                st[S_HIGH] = p
            if p < st[S_LOW]:
                st[S_LOW] = p
        st[S_CLOSE] = p
        st[S_LAST_TS] = t
        st[S_LAST_TID] = seq_last[i]
        st[S_NMSG] += 1
        st[S_NEXEC] += nexec[i]
        st[S_VOL] += q
        st[S_NOT] += nv
        a = aggr[i]
        if a > 0:
            st[S_BUYV] += q
            st[S_BUYN] += nv
            st[S_NBUY] += 1
        elif a < 0:
            st[S_SELLV] += q
            st[S_SELLN] += nv
            st[S_NSELL] += 1
        else:
            st[S_UNKV] += q
            st[S_UNKN] += nv
            st[S_A_UNK_N] += 1
        # tick rule (state carried across bars/files; reset only at segment start)
        lp = st[S_LAST_PX]
        if lp < 0:
            s = 0
        elif p > lp:
            s = 1
        elif p < lp:
            s = -1
        else:
            s = st[S_LAST_SIGN]
        if s > 0:
            st[S_TBUYV] += q
            st[S_TBUYN] += nv
        elif s < 0:
            st[S_TSELLV] += q
            st[S_TSELLN] += nv
        else:
            st[S_TUNKV] += q
            st[S_TUNKN] += nv
            st[S_T_UNK_N] += 1
        qf = q / qty_scale
        nf = nv / notional_scale
        if lp >= 0:
            dp = (p - lp) / price_scale
            r = np.log(p / lp)
            sq = np.sqrt(nf)
            if s != 0:
                x = s * qf
                sf[F_KT_XX] += x * x
                sf[F_KT_XY] += x * dp
                sf[F_KT_YY] += dp * dp
                st[S_KT_N] += 1
                xh = s * sq
                sf[F_HT_XX] += xh * xh
                sf[F_HT_XY] += xh * r
                sf[F_HT_YY] += r * r
                st[S_HT_N] += 1
            if a != 0:
                x = a * qf
                sf[F_KA_XX] += x * x
                sf[F_KA_XY] += x * dp
                sf[F_KA_YY] += dp * dp
                st[S_KA_N] += 1
                xh = a * sq
                sf[F_HA_XX] += xh * xh
                sf[F_HA_XY] += xh * r
                sf[F_HA_YY] += r * r
                st[S_HA_N] += 1
        # bar-level Hasbrouck regressor sum_i b_i sqrt(p_i q_i) (all trades with a known sign)
        if s != 0:
            sf[F_XB_T] += s * np.sqrt(nf)
        if a != 0:
            sf[F_XB_A] += a * np.sqrt(nf)
        st[S_LAST_PX] = p
        if s != 0:
            st[S_LAST_SIGN] = s
        st[S_PROCESSED] = seq_last[i]
        if mode != MODE_TIME:
            st[S_ACC] += q if mode == MODE_VOLUME else nv
            if st[S_ACC] >= st[S_THR_OPEN]:
                _emit(st, sf, oi, of, row, t, STATUS_OK, 1)
                row += 1
                st[S_LAST_CLOSE] = t
                _reset_bar(st, sf)
    if mode == MODE_TIME and watermark >= 0:
        wb = watermark // interval  # buckets with index < wb end at or before the watermark
        if st[S_HAS_OPEN] == 1 and st[S_OPEN_BUCKET] < wb:
            row = _close_time_bar(st, sf, oi, of, row, interval)
        if st[S_HAS_OPEN] == 0:
            row = _emit_empty_until(st, sf, oi, of, row, wb, interval)
    return row


class GapPolicyHalt(RuntimeError):
    pass


@dataclass
class Gap:
    after_seq: int        # last sequence id seen before the gap
    before_seq: int       # first sequence id after the gap
    start_ts_us: int      # timestamp of the last trade before the gap
    end_ts_us: int        # timestamp of the first trade after the gap
    reason: str


class BarBuilder:
    """Restartable builder for one bar series (time, volume or notional)."""

    def __init__(self, series_id: str, bar_type: str, spec, state_dir: Path, *, interval_us: int = 0,
                 threshold_i: int = 0, origin_us: int = 0, gap_policy: str = "invalidate_and_restart_segment",
                 adaptive: dict | None = None, gap_resolver=None, max_ts_regressions: int = 0):
        self.series_id = series_id
        self.bar_type = bar_type
        self.mode = {"time": MODE_TIME, "volume": MODE_VOLUME, "notional": MODE_NOTIONAL}[bar_type]
        self.spec = spec
        self.interval = int(interval_us)
        self.threshold_i = int(threshold_i)
        self.origin_us = int(origin_us)
        self.gap_policy = gap_policy
        self.adaptive = adaptive if adaptive and adaptive.get("enabled") else None
        self.gap_resolver = gap_resolver
        self.max_ts_regressions = max_ts_regressions
        self.state_dir = Path(state_dir) / series_id
        self.st = np.zeros(NI, dtype=np.int64)
        self.sf = np.zeros(NF, dtype=np.float64)
        self.meta = {"gaps": [], "daily_totals": {}, "parts": [], "segments": []}
        self._init_state()
        self.load()
        if not self.state_path().exists() and self.mode != MODE_TIME and self.origin_us % tu.DAY != 0:
            self.mark_day_incomplete((self.origin_us // tu.DAY) * tu.DAY)  # origin day only partly covered

    # ------------------------------------------------------------------ state
    def _init_state(self):
        self.st[:] = 0
        self.sf[:] = 0
        self.st[S_LAST_PX] = -1
        self.st[S_PROCESSED] = -1
        self.st[S_INCOMPLETE_UNTIL] = -1
        self.st[S_LAST_CLOSE] = self.origin_us
        self.st[S_NEXT_BUCKET] = -1
        self.st[S_LAST_TS_SEEN] = -(2 ** 62)
        self.meta["segments"] = [{"segment_id": 0, "origin_us": self.origin_us, "reason": "bar_origin"}]

    def state_path(self) -> Path:
        return self.state_dir / "state.json"

    def load(self):
        s = read_json(self.state_path())
        if not s:
            return
        if s.get("series_id") != self.series_id:
            raise ValueError("state belongs to another series")
        self.st[:] = np.array(s["st"], dtype=np.int64)
        self.sf[:] = np.array([float.fromhex(x) for x in s["sf_hex"]], dtype=np.float64)
        self.meta = s["meta"]

    def save(self):
        atomic_write_json(self.state_path(), {
            "series_id": self.series_id, "bar_type": self.bar_type, "st": [int(x) for x in self.st],
            # floats are stored as exact hex to make restart bit-identical
            "sf_hex": [float(x).hex() for x in self.sf], "meta": self.meta,
            "spec": self.spec.to_dict(), "interval_us": self.interval, "threshold_i": self.threshold_i,
            "origin_us": self.origin_us,
        })

    @property
    def last_seq(self) -> int:
        return int(self.st[S_PROCESSED])

    # ------------------------------------------------------------------ helpers
    def start_series_at(self, start_us: int):
        """Time bars: first bucket to emit (the start of the first covered day)."""
        if self.mode == MODE_TIME and self.st[S_NEXT_BUCKET] < 0:
            self.st[S_NEXT_BUCKET] = int(start_us) // self.interval

    def _thr_for_day(self, day_us: int) -> int:
        if self.mode == MODE_TIME:
            return 0
        if not self.adaptive:
            return self.threshold_i
        lb = int(self.adaptive["lookback_days"])
        days = sorted(int(d) for d, v in self.meta["daily_totals"].items() if v["complete"] and int(d) < day_us)
        days = [d for d in days if d >= day_us - lb * tu.DAY]
        if len(days) < lb:
            return self.threshold_i  # declared initial threshold until enough complete history
        tot = sum(int(self.meta["daily_totals"][str(d)]["total"]) for d in days)
        thr = tot // (lb * int(self.adaptive["target_bars_per_day"]))
        return max(int(thr), 1)

    def _empty_out(self, n):
        return np.zeros((n, NOI), dtype=np.int64), np.zeros((n, NOF), dtype=np.float64)

    def _run(self, arrs, thr_new, watermark):
        n = arrs["ts"].shape[0]
        if self.mode == MODE_TIME and self.st[S_NEXT_BUCKET] < 0:
            first = int(arrs["ts"][0]) if n else int(watermark)
            if first < 0:
                return self._empty_out(0)
            self.st[S_NEXT_BUCKET] = first // self.interval
        extra = 2
        if self.mode == MODE_TIME:
            hi = max(int(arrs["ts"].max()) if n else -1, int(watermark))
            nb = self.st[S_NEXT_BUCKET] if self.st[S_NEXT_BUCKET] >= 0 else (hi // self.interval)
            extra += max(0, hi // self.interval - nb + 2)
        oi, of = self._empty_out(n + extra)
        k = process_kernel(arrs["ts"], arrs["tid"], arrs["seq_first"], arrs["seq_last"], arrs["px"], arrs["qty"],
                           arrs["notl"], arrs["aggr"], arrs["nexec"], self.mode, max(self.interval, 1),
                           int(thr_new), float(self.spec.price_scale), float(self.spec.qty_scale),
                           float(self.spec.notional_scale), int(watermark), self.st, self.sf, oi, of)
        if self.st[S_NREG] > self.max_ts_regressions:
            raise ValueError(f"{self.st[S_NREG]} exchange-timestamp regressions in id order exceed tolerance")
        return oi[:k], of[:k]

    @staticmethod
    def _arrays(df: pd.DataFrame) -> dict:
        tid = df["trade_id"].to_numpy(np.int64)
        sf = df["first_trade_id"].to_numpy(np.int64) if "first_trade_id" in df else tid
        sl = df["last_trade_id"].to_numpy(np.int64) if "last_trade_id" in df else tid
        nexec = (sl - sf + 1) if "first_trade_id" in df else np.ones(len(df), dtype=np.int64)
        return {"ts": df["ts_us"].to_numpy(np.int64), "tid": tid, "seq_first": sf, "seq_last": sl,
                "px": df["price_i"].to_numpy(np.int64), "qty": df["qty_i"].to_numpy(np.int64),
                "notl": df["notional_i"].to_numpy(np.int64), "aggr": df["aggressor"].to_numpy(np.int8),
                "nexec": nexec.astype(np.int64)}

    # ------------------------------------------------------------------ public API
    def apply_gap(self, gap: Gap) -> list[np.ndarray]:
        """Register an unrecoverable gap.  Returns emitted rows (invalidated / incomplete bars)."""
        self.meta["gaps"].append(dict(gap.__dict__))
        d = (gap.start_ts_us // tu.DAY) * tu.DAY
        while d <= gap.end_ts_us:
            self.mark_day_incomplete(d)
            d += tu.DAY
        if self.gap_policy == "halt":
            self.save()
            raise GapPolicyHalt(f"unrecoverable gap after seq {gap.after_seq} ({gap.reason}); policy=halt")
        oi, of = self._empty_out(2)
        k = 0
        if self.mode == MODE_TIME:
            self.st[S_INCOMPLETE_UNTIL] = max(int(self.st[S_INCOMPLETE_UNTIL]), gap.end_ts_us // self.interval)
            if self.st[S_HAS_OPEN] == 1:
                self.st[S_OPEN_INCOMPLETE] = 1
        else:
            if self.st[S_HAS_OPEN] == 1:
                _emit(self.st, self.sf, oi, of, 0, gap.start_ts_us, STATUS_GAP_INVALIDATED, 0)
                k = 1
            _reset_bar(self.st, self.sf)
            self.st[S_SEGMENT] += 1
            self.st[S_LAST_CLOSE] = gap.end_ts_us
            self.meta["segments"].append({"segment_id": int(self.st[S_SEGMENT]), "origin_us": gap.end_ts_us,
                                          "reason": gap.reason, "after_seq": gap.after_seq})
        # price-change based quantities must not bridge a gap
        self.st[S_LAST_PX] = -1
        self.st[S_LAST_SIGN] = 0
        self.st[S_PROCESSED] = max(int(self.st[S_PROCESSED]), gap.before_seq - 1)
        return [oi[:k], of[:k]]

    def process(self, df: pd.DataFrame, watermark_us: int = -1, complete_days: set | None = None) -> pd.DataFrame:
        """Process a batch of normalised trades (any size, any split).  ``watermark_us`` (time bars):
        the feed is known complete up to this time.  Returns the closed bars emitted."""
        if df is None or len(df) == 0:
            if self.mode == MODE_TIME and watermark_us >= 0:
                oi, of = self._run(self._arrays(_EMPTY), 0, watermark_us)
                return self._frame(oi, of)
            return self._frame(*self._empty_out(0))
        df = df.sort_values("trade_id", kind="stable")
        arrs_all = self._arrays(df)
        keep = arrs_all["seq_last"] > self.st[S_PROCESSED]
        if not keep.all():
            df = df[keep]
            arrs_all = {k: v[keep] for k, v in arrs_all.items()}
        if len(df) == 0:
            return self._frame(*self._empty_out(0))
        outs_i, outs_f = [], []
        # --- split on sequence gaps (between state and batch, and inside the batch)
        sfirst, slast = arrs_all["seq_first"], arrs_all["seq_last"]
        prev_last = np.concatenate([[self.st[S_PROCESSED]], slast[:-1]])
        brk = np.flatnonzero((prev_last >= 0) & (sfirst != prev_last + 1))
        if np.any(sfirst <= prev_last):
            raise ValueError("overlapping / non-increasing sequence ids inside a batch after dedup")
        brk_set = set(int(x) for x in brk)
        cuts = sorted(set([0, len(df)] + list(brk_set)))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if a in brk_set:
                after, before = int(prev_last[a]), int(sfirst[a])
                ts_a = int(self.st[S_LAST_TS_SEEN]) if a == 0 else int(arrs_all["ts"][a - 1])
                recovered = False
                if self.gap_resolver is not None:
                    rec = self.gap_resolver(after + 1, before - 1)
                    if rec is not None and len(rec):
                        ra = self._arrays(rec.sort_values("trade_id"))
                        if ra["seq_first"][0] == after + 1 and ra["seq_last"][-1] == before - 1 and \
                                np.all(ra["seq_first"][1:] == ra["seq_last"][:-1] + 1):
                            oi, of = self._run_split(ra, -1)
                            outs_i.append(oi), outs_f.append(of)
                            self.meta.setdefault("recovered_gaps", []).append([after, before])
                            recovered = True
                if not recovered:
                    gi, gf = self.apply_gap(Gap(after, before, ts_a, int(arrs_all["ts"][a]), "trade_id_gap"))
                    outs_i.append(gi), outs_f.append(gf)
            piece = {k: v[a:b] for k, v in arrs_all.items()}
            wm = watermark_us if b == len(df) else -1
            oi, of = self._run_split(piece, wm)
            outs_i.append(oi), outs_f.append(of)
        oi = np.concatenate(outs_i) if outs_i else self._empty_out(0)[0]
        of = np.concatenate(outs_f) if outs_f else self._empty_out(0)[1]
        return self._frame(oi, of)

    def _run_split(self, arrs, watermark):
        """Split at UTC day boundaries to (a) update adaptive thresholds and (b) track daily totals."""
        ts = arrs["ts"]
        if ts.size == 0:
            return self._run(arrs, 0, watermark)
        day = ts // tu.DAY
        cut = np.flatnonzero(np.diff(day)) + 1
        starts = np.concatenate([[0], cut])
        ends = np.concatenate([cut, [ts.size]])
        ois, ofs = [], []
        for a, b in zip(starts, ends):
            d_us = int(day[a]) * tu.DAY
            sub = {k: v[a:b] for k, v in arrs.items()}
            tot_key = str(d_us)
            rec = self.meta["daily_totals"].setdefault(tot_key, {"total": 0, "complete": True, "n": 0})
            add = int(sub["qty"].sum()) if self.mode == MODE_VOLUME else int(sub["notl"].sum())
            rec["total"] = int(rec["total"]) + add
            rec["n"] = int(rec["n"]) + int(b - a)
            thr = self._thr_for_day(d_us)
            wm = watermark if b == ts.size else -1
            oi, of = self._run(sub, thr, wm)
            ois.append(oi), ofs.append(of)
        return np.concatenate(ois), np.concatenate(ofs)

    def mark_day_incomplete(self, day_us: int):
        rec = self.meta["daily_totals"].setdefault(str(int(day_us)), {"total": 0, "complete": False, "n": 0})
        rec["complete"] = False

    def flush(self, watermark_us: int) -> pd.DataFrame:
        return self.process(None, watermark_us=watermark_us)

    def partial(self) -> pd.DataFrame:
        """Current open bar as a *partial* record (never labelled final)."""
        if self.st[S_HAS_OPEN] == 0:
            return self._frame(*self._empty_out(0)).assign(partial=True)
        oi, of = self._empty_out(1)
        st, sf = self.st.copy(), self.sf.copy()
        end = (st[S_OPEN_BUCKET] + 1) * self.interval if self.mode == MODE_TIME else st[S_LAST_TS]
        _emit(st, sf, oi, of, 0, end, STATUS_OK, 0)
        return self._frame(oi, of).assign(partial=True)

    def _frame(self, oi, of) -> pd.DataFrame:
        df = pd.DataFrame(oi, columns=OUT_INT)
        for j, c in enumerate(OUT_FLT):
            df[c] = of[:, j] if len(of) else np.zeros(0)
        df["series_id"] = self.series_id
        df["bar_type"] = self.bar_type
        return df


_EMPTY = pd.DataFrame({"trade_id": np.zeros(0, np.int64), "ts_us": np.zeros(0, np.int64),
                       "price_i": np.zeros(0, np.int64), "qty_i": np.zeros(0, np.int64),
                       "notional_i": np.zeros(0, np.int64), "aggressor": np.zeros(0, np.int8)})


def bars_to_float(bars: pd.DataFrame, spec) -> pd.DataFrame:
    """Human-unit view of the integer bar table (kept alongside exact integer columns)."""
    ps, qs, ns = spec.price_scale, spec.qty_scale, spec.notional_scale
    out = bars.copy()
    has_trades = out["n_messages"].to_numpy() > 0
    for c in ("open", "high", "low", "close"):
        v = out[f"{c}_i"].to_numpy(np.float64) / ps
        v[~has_trades] = np.nan
        out[c] = v
    for c in ("volume", "buy_volume", "sell_volume", "unknown_side_volume", "tick_buy_volume",
              "tick_sell_volume", "tick_unknown_volume"):
        out[c] = out[f"{c}_i"].to_numpy(np.float64) / qs
    for c in ("notional", "buy_notional", "sell_notional", "unknown_side_notional", "tick_buy_notional",
              "tick_sell_notional", "tick_unknown_notional"):
        out[c] = out[f"{c}_i"].to_numpy(np.float64) / ns
    btype = out["bar_type"].iloc[0] if len(out) else "time"
    if btype == "volume":
        out["threshold"] = out["threshold_i"].to_numpy(np.float64) / qs
    elif btype == "notional":
        out["threshold"] = out["threshold_i"].to_numpy(np.float64) / ns
    else:
        out["threshold"] = np.nan
    out["status_name"] = out["status"].map(STATUS_NAMES)
    return out

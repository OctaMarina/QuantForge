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
# information-driven bars (AFML 2018, section 2.3.2)
MODE_TIB, MODE_VIB, MODE_DIB, MODE_TRB, MODE_VRB, MODE_DRB = 3, 4, 5, 6, 7, 8
INFO_MODES = {"tick_imbalance": MODE_TIB, "volume_imbalance": MODE_VIB, "notional_imbalance": MODE_DIB,
              "tick_runs": MODE_TRB, "volume_runs": MODE_VRB, "notional_runs": MODE_DRB}
INFO_BAR_TYPES = tuple(INFO_MODES)
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
I64_MAX = 9223372036854775807
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
# information-bar diagnostics (written only for information-driven modes; dropped otherwise)
OUT_IB = ["ib_threshold_raw", "ib_theta_raw", "ib_expected_ticks", "ib_expected_per_tick_raw", "ib_warmup"]
OUT_FLT = OUT_FLT + OUT_IB
NOI, NOF = len(OUT_INT), len(OUT_FLT)

# ---- information-bar persistent state (NOT reset per bar)
# EWMAs are bias-corrected ("adjusted", identical to pandas ewm(adjust=True)): value = num / den with
# num <- x + (1-a) num, den <- 1 + (1-a) den.  A plain recursion seeded with the first observation would
# let a single initial tick dominate a long-span EWMA for thousands of ticks.
# RMS imbalance threshold (declared NON-AFML variant): the signed flow x_t = b_t v_t is cut into sub-blocks of
# m ticks; every completed sub-block closes a window of RMS_K consecutive sub-blocks (H = RMS_K * m ticks,
# windows overlap by H - m) and the EWMA of window_sum^2 is updated (IBF_BLK_N / IBF_BLK_D).
# Older state files (15 slots) are padded with zeros on load.
(IBF_ET_N, IBF_ET_D, IBF_EB_N, IBF_EB_D, IBF_EP_N, IBF_EP_D, IBF_EVB_N, IBF_EVB_D, IBF_EVS_N, IBF_EVS_D,
 IBF_THR, IBF_ET_FROZEN, IBF_EXP_FROZEN, IBF_EV_N, IBF_EV_D, IBF_BLK_N, IBF_BLK_D, IBF_SUB_CUR,
 IBF_SUB0, IBF_SUB1, IBF_SUB2, IBF_SUB3) = range(22)
NIBF = 22
RMS_K = 4
THR_AFML, THR_RMS = 0, 1
(IBI_INIT, IBI_EB_INIT, IBI_T, IBI_NBUY, IBI_NSELL, IBI_BUYV, IBI_SELLV, IBI_THETA, IBI_WARMUP_OPEN,
 IBI_SUB_CNT, IBI_SUB_DONE) = range(11)
NIBI = 11


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
def _ib_reset_bar(ibi):
    ibi[IBI_T] = 0
    ibi[IBI_NBUY] = 0
    ibi[IBI_NSELL] = 0
    ibi[IBI_BUYV] = 0
    ibi[IBI_SELLV] = 0
    ibi[IBI_THETA] = 0


@njit(cache=True)
def _ew(num, den):
    return num / den if den > 0 else np.nan


@njit(cache=True)
def _ib_open(mode, ibf, ibi, t_min, t_max, imb_floor, thr_kind, sub_len):
    """Freeze the threshold at the beginning of the bar from expectations of PRIOR bars only."""
    _ib_reset_bar(ibi)
    if ibi[IBI_INIT] == 0:
        ibi[IBI_WARMUP_OPEN] = 1
        ibf[IBF_THR] = np.nan
        ibf[IBF_ET_FROZEN] = np.nan
        ibf[IBF_EXP_FROZEN] = np.nan
        return
    ibi[IBI_WARMUP_OPEN] = 0
    if mode <= MODE_DIB and thr_kind == THR_RMS:
        # NOT AFML (declared variant): threshold = empirical RMS of the signed flow summed over windows of
        # H ticks of the recent past, sqrt(EWMA(window_sum^2)).  It replaces |E0[theta_T]| = E0[T] |E[b v]|,
        # which is ~0 for balanced flow and is fed back through E0[T] (collapse / explosion).  Measured on
        # fixed tick windows (not on bars), so the estimate does not depend on where bars closed.
        h = float(RMS_K * sub_len)
        thr = np.sqrt(ibf[IBF_BLK_N] / ibf[IBF_BLK_D])
        ibf[IBF_ET_FROZEN] = h
        ibf[IBF_EXP_FROZEN] = thr / h
        ibf[IBF_THR] = thr
        return
    et = _ew(ibf[IBF_ET_N], ibf[IBF_ET_D])
    if t_min > 0 and et < t_min:
        et = t_min
    if t_max > 0 and et > t_max:
        et = t_max
    if mode <= MODE_DIB:
        # imbalance bars: |2P[b=1]-1| (ticks) or |2v+ - E0[v]| (volume/notional), both estimated by the
        # EWMA of b_t v_t over the ticks of prior bars
        e = abs(_ew(ibf[IBF_EB_N], ibf[IBF_EB_D]))
        if imb_floor > 0:
            # NOT AFML: floor the expected per-tick imbalance at imb_floor * E[v_t] (declared variant)
            fl = imb_floor * _ew(ibf[IBF_EV_N], ibf[IBF_EV_D])
            if fl > e:
                e = fl
    else:
        p = _ew(ibf[IBF_EP_N], ibf[IBF_EP_D])
        evb = _ew(ibf[IBF_EVB_N], ibf[IBF_EVB_D])
        evs = _ew(ibf[IBF_EVS_N], ibf[IBF_EVS_D])
        a1 = p * evb if evb == evb else 0.0     # no buy observed yet -> buy side contributes 0
        a2 = (1.0 - p) * evs if evs == evs else 0.0
        e = a1 if a1 > a2 else a2
    ibf[IBF_ET_FROZEN] = et
    ibf[IBF_EXP_FROZEN] = e
    ibf[IBF_THR] = et * e


@njit(cache=True)
def _ewu(ibf, kn, kd, x, a):
    ibf[kn] = x + (1.0 - a) * ibf[kn]
    ibf[kd] = 1.0 + (1.0 - a) * ibf[kd]


@njit(cache=True)
def _ib_after_close(mode, ibf, ibi, of, row, a_bars, thr_kind):
    """Write diagnostics of the closed bar and update bar-level EWMAs (they only affect LATER bars)."""
    tn = ibi[IBI_T]
    nb = ibi[IBI_NBUY]
    ns = ibi[IBI_NSELL]
    if mode <= MODE_DIB:
        theta = float(ibi[IBI_THETA])
    else:
        theta = float(ibi[IBI_BUYV] if ibi[IBI_BUYV] > ibi[IBI_SELLV] else ibi[IBI_SELLV])
    of[row, NF + 0] = ibf[IBF_THR]
    of[row, NF + 1] = theta
    of[row, NF + 2] = ibf[IBF_ET_FROZEN]
    of[row, NF + 3] = ibf[IBF_EXP_FROZEN]
    of[row, NF + 4] = float(ibi[IBI_WARMUP_OPEN])
    ncls = nb + ns
    ready = ncls > 0 and (mode > MODE_DIB or ibi[IBI_EB_INIT] == 1)
    if mode <= MODE_DIB and thr_kind == THR_RMS:
        ready = ready and ibf[IBF_BLK_D] > 0.0      # at least one complete H-tick window
    if ibi[IBI_WARMUP_OPEN] == 1 and not ready:
        _ib_reset_bar(ibi)   # no classified tick yet: the next bar is a warm-up bar again
        return
    _ewu(ibf, IBF_ET_N, IBF_ET_D, float(tn), a_bars)
    if ncls > 0:
        _ewu(ibf, IBF_EP_N, IBF_EP_D, nb / ncls, a_bars)
    if nb > 0:
        _ewu(ibf, IBF_EVB_N, IBF_EVB_D, ibi[IBI_BUYV] / nb, a_bars)
    if ns > 0:
        _ewu(ibf, IBF_EVS_N, IBF_EVS_D, ibi[IBI_SELLV] / ns, a_bars)
    ibi[IBI_INIT] = 1
    _ib_reset_bar(ibi)


@njit(cache=True)
def process_kernel(ts, tid, seq_first, seq_last, px, qty, notl, aggr, nexec,
                   mode, interval, thr_new, price_scale, qty_scale, notional_scale,
                   watermark, st, sf, oi, of, ibf, ibi, ib_sign_aggr, ib_warmup_ticks,
                   a_bars, a_ticks, t_min, t_max, imb_floor, thr_kind, sub_len, a_blk):
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
                if mode >= MODE_TIB:
                    _ib_open(mode, ibf, ibi, t_min, t_max, imb_floor, thr_kind, sub_len)
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
        # exact int64 sums: refuse to wrap around silently (every other per-bar quantity is a partial sum
        # of these two, including the information-bar theta / run sums)
        if st[S_NOT] > I64_MAX - nv or st[S_VOL] > I64_MAX - q:
            raise OverflowError("per-bar volume/notional exceeds the exact int64 range at instrument scale "
                                "(spot 2+8 decimals: about 9.2e8 quote units per bar): use a shorter time "
                                "interval / lower threshold, or fewer instrument.qty_decimals")
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
        if mode == MODE_VOLUME or mode == MODE_NOTIONAL:
            st[S_ACC] += q if mode == MODE_VOLUME else nv
            if st[S_ACC] >= st[S_THR_OPEN]:
                _emit(st, sf, oi, of, row, t, STATUS_OK, 1)
                row += 1
                st[S_LAST_CLOSE] = t
                _reset_bar(st, sf)
        elif mode >= MODE_TIB:
            # b_t: tick rule (AFML; boundary b0 = terminal b of the previous bar, carried in state) or the
            # observed aggressor (separate, explicitly named variant).  Unknown sign -> counted in T only.
            bsg = a if ib_sign_aggr == 1 else s
            if mode == MODE_TIB or mode == MODE_TRB:
                v = 1
            elif mode == MODE_VIB or mode == MODE_VRB:
                v = q
            else:
                v = nv
            ibi[IBI_T] += 1
            if bsg > 0:
                ibi[IBI_NBUY] += 1
                ibi[IBI_BUYV] += v
                ibi[IBI_THETA] += v
            elif bsg < 0:
                ibi[IBI_NSELL] += 1
                ibi[IBI_SELLV] += v
                ibi[IBI_THETA] -= v
            if bsg != 0:
                _ewu(ibf, IBF_EB_N, IBF_EB_D, float(bsg) * float(v), a_ticks)
                _ewu(ibf, IBF_EV_N, IBF_EV_D, float(v), a_ticks)
                ibi[IBI_EB_INIT] = 1
            if mode <= MODE_DIB and thr_kind == THR_RMS:
                # every tick (unknown sign: x = 0) advances the fixed tick windows of the RMS rule
                ibf[IBF_SUB_CUR] += float(bsg) * float(v)
                ibi[IBI_SUB_CNT] += 1
                if ibi[IBI_SUB_CNT] >= sub_len:
                    ibf[IBF_SUB0 + ibi[IBI_SUB_DONE] % RMS_K] = ibf[IBF_SUB_CUR]
                    ibf[IBF_SUB_CUR] = 0.0
                    ibi[IBI_SUB_CNT] = 0
                    ibi[IBI_SUB_DONE] += 1
                    if ibi[IBI_SUB_DONE] >= RMS_K:
                        w = ibf[IBF_SUB0] + ibf[IBF_SUB1] + ibf[IBF_SUB2] + ibf[IBF_SUB3]   # fixed slot order
                        _ewu(ibf, IBF_BLK_N, IBF_BLK_D, w * w, a_blk)
            if ibi[IBI_WARMUP_OPEN] == 1:
                close = ibi[IBI_T] >= ib_warmup_ticks
            elif mode <= MODE_DIB:
                close = abs(float(ibi[IBI_THETA])) >= ibf[IBF_THR]
            else:
                mx = ibi[IBI_BUYV] if ibi[IBI_BUYV] > ibi[IBI_SELLV] else ibi[IBI_SELLV]
                close = float(mx) >= ibf[IBF_THR]
            if close:
                _emit(st, sf, oi, of, row, t, STATUS_OK, 1)
                _ib_after_close(mode, ibf, ibi, of, row, a_bars, thr_kind)
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
    """Restartable builder for one bar series: time, volume, notional, or one of the information-driven
    types (tick/volume/notional imbalance, tick/volume/notional runs)."""

    def __init__(self, series_id: str, bar_type: str, spec, state_dir: Path, *, interval_us: int = 0,
                 threshold_i: int = 0, origin_us: int = 0, gap_policy: str = "invalidate_and_restart_segment",
                 adaptive: dict | None = None, gap_resolver=None, max_ts_regressions: int = 0,
                 info: dict | None = None):
        self.series_id = series_id
        self.bar_type = bar_type
        modes = {"time": MODE_TIME, "volume": MODE_VOLUME, "notional": MODE_NOTIONAL, **INFO_MODES}
        if bar_type not in modes:
            raise ValueError(f"unknown bar type {bar_type}")
        self.mode = modes[bar_type]
        info = info or {}
        if self.mode >= MODE_TIB:
            for k in ("ewma_span_bars", "ewma_span_ticks", "warmup_ticks", "sign"):
                if k not in info:
                    raise ValueError(f"information bars need bars.information.{k}")
        self.ib_sign_aggr = 1 if info.get("sign") == "aggressor" else 0
        self.ib_warmup_ticks = int(info.get("warmup_ticks", 0))
        self.ib_a_bars = 2.0 / (float(info.get("ewma_span_bars", 1)) + 1.0)
        self.ib_a_ticks = 2.0 / (float(info.get("ewma_span_ticks", 1)) + 1.0)
        bounds = info.get("expected_ticks_bounds") or [0, 0]
        self.ib_t_min, self.ib_t_max = float(bounds[0] or 0), float(bounds[1] or 0)
        self.ib_fixed_t = float(info.get("fixed_expected_ticks") or 0.0)
        if self.ib_fixed_t > 0:
            # E0[T] := H (declared NON-AFML variant): implemented as the degenerate clamp [H, H]
            self.ib_t_min = self.ib_t_max = self.ib_fixed_t
        self.ib_imb_floor = float(info.get("min_expected_imbalance_ratio") or 0.0)
        kind = info.get("imbalance_threshold", "afml") or "afml"
        if kind not in ("afml", "rms"):
            raise ValueError("bars.information.imbalance_threshold must be afml or rms")
        self.ib_thr_kind = THR_RMS if kind == "rms" else THR_AFML
        h = float(info.get("rms_horizon_ticks") or 0.0)
        if MODE_TIB <= self.mode <= MODE_DIB and self.ib_thr_kind == THR_RMS and h < RMS_K:
            raise ValueError(f"imbalance_threshold=rms needs bars.information.rms_horizon_ticks >= {RMS_K}")
        # windows of RMS_K sub-blocks of m ticks: effective horizon RMS_K * m (= H when H is a multiple of 4)
        self.ib_sub_len = max(1, int(round(h / RMS_K))) if h > 0 else 1
        self.ib_horizon = float(RMS_K * self.ib_sub_len) if h > 0 else 0.0
        # EWMA over windows with the same memory in ticks as ewma_span_ticks (one window every m ticks)
        self.ib_a_blk = 2.0 / (float(info.get("ewma_span_ticks", 1)) / float(self.ib_sub_len) + 1.0)
        self.ibf = np.zeros(NIBF, dtype=np.float64)
        self.ibi = np.zeros(NIBI, dtype=np.int64)
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
        self.ibf[:] = 0.0
        self.ibi[:] = 0
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
        if "ibf_hex" in s:
            vals = [float.fromhex(x) for x in s["ibf_hex"]]
            self.ibf[:] = 0.0
            self.ibf[:len(vals)] = np.array(vals, dtype=np.float64)
        if "ibi" in s:
            iv = [int(x) for x in s["ibi"]]
            self.ibi[:] = 0
            self.ibi[:len(iv)] = np.array(iv, dtype=np.int64)            # older states had fewer slots
        self.meta = s["meta"]

    def save(self):
        atomic_write_json(self.state_path(), {
            "series_id": self.series_id, "bar_type": self.bar_type, "st": [int(x) for x in self.st],
            # floats are stored as exact hex to make restart bit-identical
            "sf_hex": [float(x).hex() for x in self.sf], "meta": self.meta,
            "ibf_hex": [float(x).hex() for x in self.ibf], "ibi": [int(x) for x in self.ibi],
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
                           float(self.spec.notional_scale), int(watermark), self.st, self.sf, oi, of,
                           self.ibf, self.ibi, int(self.ib_sign_aggr), int(self.ib_warmup_ticks),
                           float(self.ib_a_bars), float(self.ib_a_ticks), float(self.ib_t_min), float(self.ib_t_max),
                           float(self.ib_imb_floor), int(self.ib_thr_kind), int(self.ib_sub_len), float(self.ib_a_blk))
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
            if self.mode >= MODE_TIB:
                # a new segment is self-contained: expectations restart with a warm-up bar
                self.ibf[:] = 0.0
                self.ibi[:] = 0
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

    def reset_to_saved(self):
        """Discard in-memory progress since the last save (used to drop a rejected day)."""
        self.meta = {"gaps": [], "daily_totals": {}, "parts": [], "segments": []}
        self._init_state()
        self.load()

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
        if self.mode >= MODE_TIB:
            ibi = self.ibi
            theta = float(ibi[IBI_THETA]) if self.mode <= MODE_DIB else float(max(ibi[IBI_BUYV], ibi[IBI_SELLV]))
            of[0, NF:NF + 5] = [self.ibf[IBF_THR], theta, self.ibf[IBF_ET_FROZEN], self.ibf[IBF_EXP_FROZEN],
                                float(ibi[IBI_WARMUP_OPEN])]
        return self._frame(oi, of).assign(partial=True)

    def info_state(self) -> dict:
        """Current information-bar expectations (for manifests / debugging)."""
        f = self.ibf
        ew = lambda n, d: float(f[n] / f[d]) if f[d] > 0 else None
        return {"initialised": bool(self.ibi[IBI_INIT]), "E0_T": ew(IBF_ET_N, IBF_ET_D),
                "ewma_b_v_raw": ew(IBF_EB_N, IBF_EB_D), "P_buy": ew(IBF_EP_N, IBF_EP_D),
                "E_v_buy_raw": ew(IBF_EVB_N, IBF_EVB_D), "E_v_sell_raw": ew(IBF_EVS_N, IBF_EVS_D),
                "rms_window_ticks": self.ib_horizon or None, "rms_threshold_raw": (
                    float(np.sqrt(f[IBF_BLK_N] / f[IBF_BLK_D])) if f[IBF_BLK_D] > 0 else None),
                "threshold_rule": "rms" if self.ib_thr_kind == THR_RMS and self.mode <= MODE_DIB else "afml",
                "open_bar_ticks": int(self.ibi[IBI_T]), "open_bar_threshold_raw": float(f[IBF_THR])}

    def _frame(self, oi, of) -> pd.DataFrame:
        df = pd.DataFrame(oi, columns=OUT_INT)
        for j, c in enumerate(OUT_FLT):
            if c in OUT_IB and self.mode < MODE_TIB:
                continue
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
    elif btype in INFO_MODES:
        # v_t units: ticks (count), base quantity, or quote notional
        unit = 1.0 if btype.startswith("tick") else (qs if btype.startswith("volume") else ns)
        out["threshold"] = out["ib_threshold_raw"].to_numpy(np.float64) / unit
        out["ib_theta"] = out["ib_theta_raw"].to_numpy(np.float64) / unit
        out["ib_expected_per_tick"] = out["ib_expected_per_tick_raw"].to_numpy(np.float64) / unit
        out["ib_warmup"] = out["ib_warmup"].to_numpy(np.float64) == 1.0
    else:
        out["threshold"] = np.nan
    out["status_name"] = out["status"].map(STATUS_NAMES)
    return out

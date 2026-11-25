"""Bar engines over the canonical economic trade stream (availability order).

Conventions (report §7.1–7.5), all versioned in BAR_SPEC_VERSION:
  * trades are never split; the trade crossing a threshold belongs entirely to the bar it closes;
    at most one bar closes per trade; excess is stored as overshoot, not carried over.
  * state (open bar, CUSUM accumulators) carries over file/day boundaries; never reset per file.
  * a trade-continuity break (new `seg`) closes the open event bar as PARTIAL (reason=gap),
    resets CUSUM accumulators and restarts the bar.
  * time bars: [T_k, T_k+Δ) on the availability clock (received_time); an event exactly at
    T_{k+1} belongs to the next bar; decision_time = T_{k+1}; eval_time = T_{k+1}-1ns (state left
    of the boundary). Empty bars are emitted with n=0 and null OHLC.
  * event bars: decision_time = eval_time = received_time of the closing trade; trade cursor = seq.
  * CUSUM (symmetric, zero drift, threshold h=log(1+b/1e4)): S+_i = max(0,S+_{i-1}+r_i),
    S-_i = min(0,S-_{i-1}+r_i), r_i = log p_i - log p_{i-1}. Equivalent exact form used here:
    S+_i = log(p_i / min p since S+ reset), S-_i = log(p_i / max p since S- reset). Trigger on
    S+ >= h  <=>  p_i*1e6 >= pmin*(1e6+b_c)  (b_c = threshold in 1e-2 bp), equality triggers.
    Only the triggered accumulator resets (variant 'triggered'); 'both' is a distinct variant.
    The bar aggregates ALL trades since the previous boundary, including the trigger trade.
"""
from __future__ import annotations

import numpy as np
from numba import njit

BAR_SPEC_VERSION = "bars/v1"

# integer output columns
I_COLS = ["first_seq", "last_seq", "first_id", "last_id", "t_open", "t_close", "m_open", "m_close",
          "open_e2", "high_e2", "low_e2", "close_e2", "n_trades", "n_buy", "vol_e3", "vol_buy_e3",
          "notional_e5", "notional_buy_e5", "seg", "reason", "complete", "decision_time", "eval_time",
          "t_start", "t_end", "trigger", "threshold_int", "overshoot_int", "day"]
F_COLS = ["cusum_pre", "cusum_opp", "cusum_overshoot", "threshold_log"]
IC = {c: i for i, c in enumerate(I_COLS)}
FC = {c: i for i, c in enumerate(F_COLS)}
NI, NF = len(I_COLS), len(F_COLS)

# close reasons
R_THRESHOLD, R_GAP, R_END, R_THRESHOLD_CHANGE, R_DAILY_RESET, R_TIME, R_FRESH_START = 0, 1, 2, 3, 4, 5, 6
REASONS = {R_THRESHOLD: "threshold", R_GAP: "gap_partial", R_END: "end_partial",
           R_THRESHOLD_CHANGE: "threshold_change_partial", R_DAILY_RESET: "daily_reset_partial", R_TIME: "time",
           R_FRESH_START: "fresh_start_partial"}

# event-bar state layout (int64)
S_OPEN, S_FSEQ, S_FID, S_TOPEN, S_MOPEN, S_O, S_H, S_L, S_N, S_NBUY, S_VOL, S_VOLB, S_NOT, S_NOTB, S_SEG, \
    S_LSEQ, S_LID, S_TLAST, S_MLAST, S_C, S_PMIN, S_PMAX, S_CINIT, S_DAY, S_THR = range(25)
NS_STATE = 25
DAY_NS = 86_400_000_000_000


def new_event_state():
    st = np.zeros(NS_STATE, np.int64)
    st[S_SEG] = -1
    st[S_DAY] = -1
    st[S_LSEQ] = -1      # no trade seen yet: time-bar cursor -1 means "no trade available"
    return st


@njit(cache=True)
def _emit(out_i, out_f, k, st, reason, complete, trig, thr_int, over_int, pre, opp, fover, thr_log):
    out_i[k, 0] = st[S_FSEQ]
    out_i[k, 1] = st[S_LSEQ]
    out_i[k, 2] = st[S_FID]
    out_i[k, 3] = st[S_LID]
    out_i[k, 4] = st[S_TOPEN]
    out_i[k, 5] = st[S_TLAST]
    out_i[k, 6] = st[S_MOPEN]
    out_i[k, 7] = st[S_MLAST]
    out_i[k, 8] = st[S_O]
    out_i[k, 9] = st[S_H]
    out_i[k, 10] = st[S_L]
    out_i[k, 11] = st[S_C]
    out_i[k, 12] = st[S_N]
    out_i[k, 13] = st[S_NBUY]
    out_i[k, 14] = st[S_VOL]
    out_i[k, 15] = st[S_VOLB]
    out_i[k, 16] = st[S_NOT]
    out_i[k, 17] = st[S_NOTB]
    out_i[k, 18] = st[S_SEG]
    out_i[k, 19] = reason
    out_i[k, 20] = complete
    out_i[k, 21] = st[S_TLAST]
    out_i[k, 22] = st[S_TLAST]
    out_i[k, 23] = st[S_TOPEN]
    out_i[k, 24] = st[S_TLAST]
    out_i[k, 25] = trig
    out_i[k, 26] = thr_int
    out_i[k, 27] = over_int
    out_i[k, 28] = st[S_DAY]
    out_f[k, 0] = pre
    out_f[k, 1] = opp
    out_f[k, 2] = fover
    out_f[k, 3] = thr_log


@njit(cache=True)
def _reset_bar(st):
    st[S_OPEN] = 0
    st[S_N] = 0
    st[S_NBUY] = 0
    st[S_VOL] = 0
    st[S_VOLB] = 0
    st[S_NOT] = 0
    st[S_NOTB] = 0


@njit(cache=True)
def _add_trade(st, seq, tid, recv, tt, px, qx, sg):
    if st[S_OPEN] == 0:
        st[S_OPEN] = 1
        st[S_FSEQ] = seq
        st[S_FID] = tid
        st[S_TOPEN] = recv
        st[S_MOPEN] = tt
        st[S_O] = px
        st[S_H] = px
        st[S_L] = px
    if px > st[S_H]:
        st[S_H] = px
    if px < st[S_L]:
        st[S_L] = px
    st[S_C] = px
    st[S_LSEQ] = seq
    st[S_LID] = tid
    st[S_TLAST] = recv
    st[S_MLAST] = tt
    st[S_N] += 1
    st[S_VOL] += qx
    nt = px * qx
    st[S_NOT] += nt
    if sg > 0:
        st[S_NBUY] += 1
        st[S_VOLB] += qx
        st[S_NOTB] += nt


@njit(cache=True)
def event_bars(mode, seq, tid, recv, tt, px, qx, sign, seg, day_thr, day0, st, cusum_both, daily_reset):
    """mode: 0 volume (threshold e3 BTC), 1 dollar (threshold e5 USDT), 2 CUSUM (threshold 1e-2 bp).
    day_thr: per-day threshold array indexed by (recv//DAY - day0); constant schedule = same value.
    Returns (out_i, out_f, n_out). State `st` is updated in place."""
    n = seq.shape[0]
    out_i = np.zeros((n + 2, 29), np.int64)
    out_f = np.full((n + 2, 4), np.nan)
    k = 0
    for i in range(n):
        dabs = recv[i] // DAY_NS          # absolute UTC day (state is compared across calls/chunks)
        d = dabs - day0
        thr = day_thr[d]
        # --- segment break: close open bar as partial, reset CUSUM
        if seg[i] != st[S_SEG]:
            if st[S_OPEN] == 1:
                _emit(out_i, out_f, k, st, R_GAP, 0, 0, st[S_THR], 0, np.nan, np.nan, np.nan, np.nan)
                k += 1
                _reset_bar(st)
            st[S_SEG] = seg[i]
            st[S_CINIT] = 0
        # --- day boundary: threshold change / optional daily reset
        if dabs != st[S_DAY]:
            if st[S_DAY] >= 0 and st[S_OPEN] == 1:
                if daily_reset:
                    _emit(out_i, out_f, k, st, R_DAILY_RESET, 0, 0, st[S_THR], 0, np.nan, np.nan, np.nan, np.nan)
                    k += 1
                    _reset_bar(st)
                    st[S_CINIT] = 0
                elif mode == 2 and thr != st[S_THR]:
                    _emit(out_i, out_f, k, st, R_THRESHOLD_CHANGE, 0, 0, st[S_THR], 0, np.nan, np.nan, np.nan, np.nan)
                    k += 1
                    _reset_bar(st)
                    st[S_CINIT] = 0
            elif mode == 2 and thr != st[S_THR]:
                st[S_CINIT] = 0
            st[S_DAY] = dabs
            st[S_THR] = thr
        p = px[i]
        _add_trade(st, seq[i], tid[i], recv[i], tt[i], p, qx[i], sign[i])
        if mode == 0:
            if st[S_VOL] >= thr:
                _emit(out_i, out_f, k, st, R_THRESHOLD, 1, 0, thr, st[S_VOL] - thr, np.nan, np.nan, np.nan, np.nan)
                k += 1
                _reset_bar(st)
        elif mode == 1:
            if st[S_NOT] >= thr:
                _emit(out_i, out_f, k, st, R_THRESHOLD, 1, 0, thr, st[S_NOT] - thr, np.nan, np.nan, np.nan, np.nan)
                k += 1
                _reset_bar(st)
        else:
            if st[S_CINIT] == 0:
                # first trade of a segment / after threshold change: both accumulators at 0
                st[S_PMIN] = p
                st[S_PMAX] = p
                st[S_CINIT] = 1
                continue
            if p < st[S_PMIN]:
                st[S_PMIN] = p
            if p > st[S_PMAX]:
                st[S_PMAX] = p
            up = p * 1000000 >= st[S_PMIN] * (1000000 + thr)
            dn = p * (1000000 + thr) <= st[S_PMAX] * 1000000
            if up or dn:
                hlog = np.log1p(thr / 1e6)
                if up:
                    pre = np.log(p / st[S_PMIN])
                    opp = np.log(p / st[S_PMAX])
                    trig = 1
                else:
                    pre = np.log(p / st[S_PMAX])
                    opp = np.log(p / st[S_PMIN])
                    trig = -1
                fover = abs(pre) - hlog
                _emit(out_i, out_f, k, st, R_THRESHOLD, 1, trig, thr, 0, pre, opp, fover, hlog)
                k += 1
                _reset_bar(st)
                if cusum_both:
                    st[S_PMIN] = p
                    st[S_PMAX] = p
                elif up:
                    st[S_PMIN] = p
                else:
                    st[S_PMAX] = p
    return out_i[:k], out_f[:k], k


@njit(cache=True)
def time_bars(seq, tid, recv, tt, px, qx, sign, seg, dt_ns, origin_ns, st, t_from, t_until):
    """Time bars on the availability clock. Emits every grid interval [T, T+dt) with T >= t_from
    whose end T+dt <= t_until or that is followed by a trade (recv >= T+dt).
    st: event-state array reused; st[S_TOPEN] stores the start T of the current open interval."""
    n = seq.shape[0]
    span = (t_until - t_from) // dt_ns + 3
    out_i = np.zeros((n + span, 29), np.int64)
    out_f = np.full((n + span, 4), np.nan)
    k = 0
    if st[S_DAY] < 0:
        st[S_DAY] = 1
        st[S_THR] = t_from - ((t_from - origin_ns) % dt_ns)  # current interval start
    for i in range(n + 1):
        if i < n:
            r = recv[i]
        else:
            r = t_until
        # close intervals ending at or before r
        while st[S_THR] + dt_ns <= r:
            T = st[S_THR]
            out_i[k, 23] = T
            out_i[k, 24] = T + dt_ns
            out_i[k, 21] = T + dt_ns
            out_i[k, 22] = T + dt_ns - 1
            out_i[k, 19] = R_TIME
            out_i[k, 20] = 1
            out_i[k, 26] = dt_ns
            out_i[k, 28] = T // DAY_NS
            if st[S_OPEN] == 1:
                out_i[k, 0] = st[S_FSEQ]
                out_i[k, 1] = st[S_LSEQ]
                out_i[k, 2] = st[S_FID]
                out_i[k, 3] = st[S_LID]
                out_i[k, 4] = st[S_TOPEN]
                out_i[k, 5] = st[S_TLAST]
                out_i[k, 6] = st[S_MOPEN]
                out_i[k, 7] = st[S_MLAST]
                out_i[k, 8] = st[S_O]
                out_i[k, 9] = st[S_H]
                out_i[k, 10] = st[S_L]
                out_i[k, 11] = st[S_C]
                out_i[k, 12] = st[S_N]
                out_i[k, 13] = st[S_NBUY]
                out_i[k, 14] = st[S_VOL]
                out_i[k, 15] = st[S_VOLB]
                out_i[k, 16] = st[S_NOT]
                out_i[k, 17] = st[S_NOTB]
                out_i[k, 18] = st[S_SEG]
            else:
                out_i[k, 0] = -1
                out_i[k, 1] = st[S_LSEQ]  # cursor: last trade before the boundary
                out_i[k, 2] = -1
                out_i[k, 3] = -1
                out_i[k, 18] = st[S_SEG]
            k += 1
            _reset_bar(st)
            st[S_THR] = T + dt_ns
        if i == n:
            break
        if r < st[S_THR]:
            # trade older than the current interval (should not happen in availability order)
            continue
        _add_trade(st, seq[i], tid[i], recv[i], tt[i], px[i], qx[i], sign[i])
        st[S_SEG] = seg[i]
    return out_i[:k], out_f[:k], k

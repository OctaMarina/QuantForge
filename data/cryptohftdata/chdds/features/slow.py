"""Features from mark/index/funding (F7 without book), open interest (F8), liquidations (F9),
ticker (F10) and calendar (F10 P1). All as-of the availability clock (received_time <= eval_t).

Validity encodings (NaN value + shared state column per block):
  mark_state : 0 valid, 1 no observation yet, 2 stale (market age > threshold)
  liq_state_<w> : 0 valid, 1 warmup, 2 no liquidation file for an hour in the window,
                  3 shared market-stream reception gap in the window (feed state unknown)
  oi_state  : 0 valid, 1 no point known, 2 reference older than oi_max_ref_age,
              3 reference too old because a newer point is rejected or in quarantine (canonical status)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

from .. import paths as P
from .trades import wname

CAN = P.DERIVED / "canonical"
NS = 1_000_000_000


@dataclass
class SlowFeatureConfig:
    mark_fast_age_s: float = 3.0
    mark_slow_age_s: float = 10.0
    basis_windows_s: tuple = (30, 120, 900, 3600)
    index_return_windows_s: tuple = (30, 120, 900, 3600)
    funding_change_windows_s: tuple = (3600, 14400)
    funding_z_lookback_days: int = 7
    oi_windows_s: tuple = (900, 3600, 14400, 86400)
    oi_max_ref_age_s: float = 900.0
    oi_z_lookback_points: int = 2016          # 7 days of 5-minute points
    liq_windows_s: tuple = (30, 120, 900, 3600)
    liq_missing_file_policy: str = "unknown"  # "unknown" (default) | "inferred_zero" (opt-in, flagged)
    stream_gap_s: float = 5.0
    funding_min_interval_s: float = 3600.0    # shortest Binance funding interval: a shorter silence cannot hide a whole cycle
    ticker_max_age_s: float = 10.0
    seasonal_lookback_days: int = 28
    seasonal_min_obs: int = 8


# ------------------------------------------------------------------------------- loaders (cached)
class SlowData:
    def __init__(self):
        self._mark = {}
        self._ticker = {}
        self.oi = pq.read_table(CAN / "open_interest.parquet") if (CAN / "open_interest.parquet").exists() else None
        self.liq = pq.read_table(CAN / "liquidations.parquet") if (CAN / "liquidations.parquet").exists() else None
        lf = CAN / "liquidation_file_hours.parquet"
        self.liq_hours = set(pq.read_table(lf)["hour_index"].to_numpy().tolist()) if lf.exists() else set()
        self.stream_gaps = None
        self.oi_rz = None

    def mark_range(self, d0: date, d1: date):
        tabs = []
        d = d0
        while d <= d1:
            p = CAN / "mark" / f"{d.isoformat()}.parquet"
            if p.exists():
                tabs.append(pq.read_table(p))
            d += timedelta(days=1)
        return pa.concat_tables(tabs) if tabs else None

    def ticker_range(self, d0: date, d1: date):
        tabs = []
        d = d0
        while d <= d1:
            p = CAN / "ticker" / f"{d.isoformat()}.parquet"
            if p.exists():
                tabs.append(pq.read_table(p))
            d += timedelta(days=1)
        return pa.concat_tables(tabs) if tabs else None


def _asof(times, t):
    return np.searchsorted(times, t, side="right") - 1


def _tw_piecewise(times, values, valid_until, t, w_ns):
    """Exact time-weighted mean/second moment over (t-w, t] of a step function that takes `values[i]` from
    times[i] until min(times[i+1], valid_until[i]); time outside validity excluded. Returns mean, m2, coverage."""
    return _tw_kernel(times, values, valid_until, t, w_ns)


@njit(cache=True)
def _tw_kernel(times, values, vu, t, w_ns):
    n = t.shape[0]
    mean = np.full(n, np.nan)
    m2 = np.full(n, np.nan)
    cov = np.zeros(n)
    m = times.shape[0]
    for qi in range(n):
        s = t[qi] - w_ns
        e = t[qi]
        # first segment starting at or before s
        lo = np.searchsorted(times, s, side="right") - 1
        if lo < 0:
            lo = 0
        acc = 0.0
        acc2 = 0.0
        dur = 0.0
        k = lo
        while k < m and times[k] <= e:
            a = times[k] if times[k] > s else s
            b_ = times[k + 1] if k + 1 < m else e
            if b_ > e:
                b_ = e
            if vu[k] < b_:
                b_ = vu[k]
            if b_ > a and values[k] == values[k]:
                dt = (b_ - a) / 1e9
                acc += values[k] * dt
                acc2 += values[k] * values[k] * dt
                dur += dt
            k += 1
        cov[qi] = dur / (w_ns / 1e9)
        if dur > 0:
            mean[qi] = acc / dur
            m2[qi] = acc2 / dur
    return mean, m2, cov


@njit(cache=True)
def _rolling_robust_known(x, avail, lookback, min_obs):
    """Median and 1.4826*MAD of the previous `lookback` entries, using only those already available at entry i: a point
    that arrived late (for example a revised value) cannot enter the statistics of points published before it."""
    n = x.shape[0]
    med = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    for i in range(n):
        a = i - lookback
        if a < 0:
            a = 0
        w = x[a:i]
        w = w[np.isfinite(w) & (avail[a:i] <= avail[i])]
        if w.shape[0] < min_obs:
            continue
        mm = np.median(w)
        med[i] = mm
        mad[i] = 1.4826 * np.median(np.abs(w - mm))
    return med, mad


@njit(cache=True)
def _rolling_robust(x, lookback, min_obs):
    """Median and 1.4826*MAD of x over the previous `lookback` entries (strictly prior)."""
    n = x.shape[0]
    med = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    for i in range(n):
        a = i - lookback
        if a < 0:
            a = 0
        w = x[a:i]
        w = w[np.isfinite(w)]
        if w.shape[0] < min_obs:
            continue
        mm = np.median(w)
        med[i] = mm
        mad[i] = 1.4826 * np.median(np.abs(w - mm))
    return med, mad


# ------------------------------------------------------------------------------- mark / funding
def mark_features(d: date, t: np.ndarray, cfg: SlowFeatureConfig, data: SlowData, trade_ret=None):
    out = {}
    lb = max(cfg.funding_z_lookback_days, 1)
    tab = data.mark_range(d - timedelta(days=lb), d)
    n = t.size
    if tab is None:
        return out
    recv = tab["recv_ns"].to_numpy()
    evt = tab["event_ms"].to_numpy() * P.MS_NS
    mk = tab["mark_e8"].to_numpy() / 1e8
    ix = tab["index_e8"].to_numpy() / 1e8
    fr = np.where(tab["funding_null"].to_numpy(), np.nan, tab["funding_e8"].to_numpy() / 1e8)
    nft = tab["next_funding_ms"].to_numpy() * P.MS_NS
    st_ = np.where(tab["settle_null"].to_numpy(), np.nan, tab["settle_e8"].to_numpy() / 1e8)
    j = _asof(recv, t)
    has = j >= 0
    jj = np.maximum(j, 0)
    age_mkt = np.where(has, t - evt[jj], np.iinfo(np.int64).max)
    fast = has & (age_mkt <= cfg.mark_fast_age_s * NS)
    slow = has & (age_mkt <= cfg.mark_slow_age_s * NS)
    state = np.where(~has, 1, np.where(fast, 0, 2)).astype(np.int8)
    out["mark_state"] = state
    out["mark_recv_age_ms"] = np.where(has, (t - recv[jj]) / 1e6, np.nan)
    out["mark_mkt_age_ms"] = np.where(has, age_mkt / 1e6, np.nan)
    out["mark_price"] = np.where(slow, mk[jj], np.nan)
    out["index_price"] = np.where(slow, ix[jj], np.nan)
    bmi = 1e4 * (mk / ix - 1)
    out["basis_mark_index_bps"] = np.where(fast, bmi[jj], np.nan)
    vu = evt + int(cfg.mark_slow_age_s * NS)
    for w in cfg.basis_windows_s:
        m1, m2, cov = _tw_piecewise(recv, bmi, vu, t, int(w * NS))
        ok = cov >= 1 - 1e-7
        s = wname(w)
        out[f"basis_mark_index_coverage_{s}"] = cov
        out[f"tw_basis_mark_index_bps_{s}"] = np.where(ok, m1, np.nan)
        out[f"std_basis_mark_index_bps_{s}"] = np.where(ok, np.sqrt(np.maximum(m2 - m1 * m1, 0)), np.nan)
        j0 = _asof(recv, t - int(w * NS))
        ok0 = (j0 >= 0) & ((t - int(w * NS)) - evt[np.maximum(j0, 0)] <= cfg.mark_fast_age_s * NS)
        out[f"basis_mark_index_change_{s}_bps"] = np.where(fast & ok0, bmi[jj] - bmi[np.maximum(j0, 0)], np.nan)
    for w in cfg.index_return_windows_s:
        j0 = _asof(recv, t - int(w * NS))
        ok0 = (j0 >= 0) & ((t - int(w * NS)) - evt[np.maximum(j0, 0)] <= cfg.mark_fast_age_s * NS)
        r = np.where(fast & ok0, np.log(ix[jj] / ix[np.maximum(j0, 0)]), np.nan)
        out[f"ret_index_{wname(w)}"] = r
        out[f"ret_mark_{wname(w)}"] = np.where(fast & ok0, np.log(mk[jj] / mk[np.maximum(j0, 0)]), np.nan)
        if trade_ret is not None and f"ret_trade_{wname(w)}" in trade_ret:
            out[f"ret_trade_minus_index_{wname(w)}"] = trade_ret[f"ret_trade_{wname(w)}"] - r
    # funding
    out["funding_rate_asof"] = np.where(slow, fr[jj], np.nan)
    out["funding_rate_age_ms"] = out["mark_recv_age_ms"]
    for w in cfg.funding_change_windows_s:
        j0 = _asof(recv, t - int(w * NS))
        ok0 = (j0 >= 0) & ((t - int(w * NS)) - evt[np.maximum(j0, 0)] <= cfg.mark_slow_age_s * NS)
        s = wname(w)
        out[f"funding_change_{s}"] = np.where(slow & ok0, fr[jj] - fr[np.maximum(j0, 0)], np.nan)
        out[f"funding_cycle_changed_{s}"] = np.where(slow & ok0, (nft[jj] != nft[np.maximum(j0, 0)]).astype(float), np.nan)
    ttf = (nft[jj] - t) / 1e9
    out["time_to_funding_s"] = np.where(slow & (ttf > 0), ttf, np.nan)
    # funding interval: difference between the current and the previous distinct next_funding_time seen so far
    uniq_nft, first_idx = np.unique(nft, return_index=True)
    order = np.argsort(first_idx)
    nft_seq = uniq_nft[order]
    first_seen = recv[first_idx[order]]
    last_idx = nft.size - 1 - np.unique(nft[::-1], return_index=True)[1]
    last_seen = recv[last_idx[order]]
    interval = np.full(nft.size, np.nan)
    pos = np.searchsorted(nft_seq, nft)  # nft_seq sorted? (next funding times increase with time)
    if np.all(np.diff(nft_seq) > 0):
        prev = np.where(pos > 0, nft_seq[np.maximum(pos - 1, 0)], np.nan)
        # a silence long enough to hide a whole cycle leaves the interval unknown: the difference could span two cycles
        seen = np.zeros(nft_seq.size, bool)
        seen[1:] = first_seen[1:] - last_seen[:-1] < cfg.funding_min_interval_s * NS
        interval = np.where((pos > 0) & seen[pos], (nft - prev) / 1e9, np.nan)
    iv = interval[jj]
    out["funding_interval_s_observed"] = np.where(slow, iv, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        phase = 1.0 - ttf / iv
        okp = slow & (ttf > 0) & np.isfinite(iv) & (iv > 0) & (phase >= 0) & (phase <= 1)
    out["funding_phase_sin"] = np.where(okp, np.sin(2 * np.pi * phase), np.nan)
    out["funding_phase_cos"] = np.where(okp, np.cos(2 * np.pi * phase), np.nan)
    # funding surprise: robust z vs previous N days of 1-minute as-of samples (strictly before t)
    g = np.arange((recv[0] // (60 * NS) + 1) * 60 * NS, t.max() + 1, 60 * NS, dtype=np.int64)
    jg = _asof(recv, g)
    # a sample exists only where the value is still fresh: an outage must not repeat the last rate into the statistics
    fg = np.where((jg >= 0) & (g - evt[np.maximum(jg, 0)] <= cfg.mark_slow_age_s * NS), fr[np.maximum(jg, 0)], np.nan)
    lbn = cfg.funding_z_lookback_days * 1440
    med, mad = _rolling_robust(fg, lbn, lbn // 2)
    gi = np.searchsorted(g, t, side="right") - 1   # last grid point <= t; stats use points strictly before it
    okz = (gi >= 0) & slow
    gic = np.maximum(gi, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (fr[jj] - med[gic]) / mad[gic]
    out[f"funding_rz_{cfg.funding_z_lookback_days}d"] = np.where(okz & (mad[gic] > 0), z, np.nan)
    # estimated settle price (P2, applicability unconfirmed for perpetuals -> diagnostic only)
    out["settle_est_index_bps_p2"] = np.where(fast, 1e4 * (st_[jj] / ix[jj] - 1), np.nan)
    return out


# ------------------------------------------------------------------------------- open interest
def oi_features(t: np.ndarray, cfg: SlowFeatureConfig, data: SlowData, tw_needed=True):
    out = {}
    if data.oi is None:
        return out
    o = data.oi
    # only accepted versions are ever used; rejected or quarantined versions are never chosen between (canonical/slow.py)
    status = o["status"].to_numpy()
    rej = o.filter(pa.array(status != 0))
    q_ts = np.unique(rej["ts_ms"].to_numpy()) * P.MS_NS
    q_recv = np.sort(rej["recv_ns"].to_numpy())
    o = o.filter(pa.array(status == 0))
    recv = o["recv_ns"].to_numpy()
    ts = o["ts_ms"].to_numpy() * P.MS_NS
    O = o["oi_e8"].to_numpy() / 1e8
    OV = o["oi_value_e8"].to_numpy() / 1e8
    # running "best known": max ts (latest version on ties) over recv order
    best = np.zeros(recv.size, np.int64)
    b = 0
    for i in range(recv.size):
        if i == 0 or ts[i] >= ts[b]:
            b = i
        best[i] = b
    j = _asof(recv, t)
    has = j >= 0
    k = best[np.maximum(j, 0)]
    ref_age = t - ts[k]
    ok = has & (ref_age <= cfg.oi_max_ref_age_s * NS)
    state = np.where(~has, 1, np.where(ok, 0, 2)).astype(np.int8)
    if q_ts.size:
        # a rejected or quarantined point newer than the reference explains why the reference is too old
        newer = np.searchsorted(q_ts, t, side="right") > np.searchsorted(q_ts, ts[k], side="right")
        state[(state == 2) & newer] = 3
    out["oi_state"] = state
    out["oi_btc_asof"] = np.where(ok, O[k], np.nan)
    out["oi_usdt_asof"] = np.where(ok, OV[k], np.nan)
    out["oi_reference_age_s"] = np.where(has, ref_age / 1e9, np.nan)
    out["oi_recv_age_s"] = np.where(has, (t - recv[k]) / 1e9, np.nan)
    out["oi_reference_ts_ns"] = np.where(has, ts[k], -1)
    # delayed context (explicitly different definition, no age limit)
    out["oi_btc_known_delayed"] = np.where(has, O[k], np.nan)
    for w in cfg.oi_windows_s:
        w_ns = int(w * NS)
        j0 = _asof(recv, t - w_ns)
        h0 = j0 >= 0
        k0 = best[np.maximum(j0, 0)]
        s = wname(w)
        okw = ok & h0
        out[f"oi_log_change_known_{s}"] = np.where(okw, np.log(O[k] / O[k0]), np.nan)
        out[f"oi_ref_dtau_known_{s}_s"] = np.where(okw, (ts[k] - ts[k0]) / 1e9, np.nan)
        clean = np.searchsorted(q_recv, t, side="right") == np.searchsorted(q_recv, t - w_ns, side="right")
        out[f"oi_points_received_{s}"] = np.where((h0 | has) & clean, (j - j0).astype(float), np.nan)
    # reference change between the two most recent distinct points known at t
    tsu, iu = np.unique(ts, return_index=True)          # first version per ts (by ts order)
    first_recv = np.full(tsu.size, np.iinfo(np.int64).max)
    np.minimum.at(first_recv, np.searchsorted(tsu, ts), recv)
    pos = np.searchsorted(tsu, ts[k])
    prev = pos - 1
    found = np.zeros(t.size, bool)
    pidx = np.full(t.size, -1)
    for _ in range(24):
        cand = prev >= 0
        okc = cand & ~found
        okc[okc] = first_recv[prev[okc]] <= t[okc]
        pidx[okc] = prev[okc]
        found |= okc
        prev = np.where(found, prev, prev - 1)
    val0 = np.full(t.size, np.nan)
    tau0 = np.full(t.size, -1, np.int64)
    sel = found & has
    val0[sel] = O[iu[pidx[sel]]]
    tau0[sel] = tsu[pidx[sel]]
    dtau = (ts[k] - tau0) / 1e9
    okr = ok & found
    out["oi_reference_change_btc"] = np.where(okr, O[k] - val0, np.nan)
    out["oi_reference_dtau_s"] = np.where(okr, dtau, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["oi_change_rate_per_s"] = np.where(okr & (dtau > 0) & (dtau <= cfg.oi_max_ref_age_s),
                                               np.log(O[k] / val0) / dtau, np.nan)
    out["oi_value_price_implied"] = np.where(ok, OV[k] / O[k], np.nan)
    # surprises on the grid of distinct points (strictly prior points), attached to the reference point
    order_ts = np.argsort(ts, kind="stable")
    pts_ts = ts[order_ts]
    pts_O = O[order_ts]
    pts_recv = recv[order_ts]
    if data.oi_rz is None:
        lo_ = np.log(pts_O)
        dlo_ = np.concatenate([[np.nan], np.diff(lo_)])
        dlo_[np.concatenate([[False], np.diff(pts_ts) != 300 * NS])] = np.nan   # only true 5-minute steps
        av_d = np.concatenate([pts_recv[:1], np.maximum(pts_recv[1:], pts_recv[:-1])])
        data.oi_rz = (dlo_, *_rolling_robust_known(lo_, pts_recv, cfg.oi_z_lookback_points, cfg.oi_z_lookback_points // 2),
                      *_rolling_robust_known(dlo_, av_d, cfg.oi_z_lookback_points, cfg.oi_z_lookback_points // 2))
    dlo, med, mad, medd, madd = data.oi_rz
    pk = np.searchsorted(pts_ts, ts[k])
    pkc = np.clip(pk, 0, pts_ts.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["oi_level_rz_7d"] = np.where(ok & (mad[pkc] > 0), (np.log(O[k]) - med[pkc]) / mad[pkc], np.nan)
        out["oi_change_rz_7d"] = np.where(ok & (madd[pkc] > 0), (dlo[pkc] - medd[pkc]) / madd[pkc], np.nan)
    # time-weighted known OI (for turnover_to_oi): step function on availability changes of the best point
    if tw_needed:
        chg = np.concatenate([[True], best[1:] != best[:-1]])
        tt_ = recv[chg]
        vv = O[best[chg]]
        # known OI has no age limit by definition (references normally reach 10-17 minutes between polls), except when a rejected
        # or quarantined point should have replaced it: then it expires like a reference would
        bt = ts[best[chg]]
        vu = np.full(tt_.size, np.iinfo(np.int64).max)
        if q_ts.size:
            nxt = np.concatenate([bt[1:], [np.iinfo(np.int64).max]])
            iq = np.searchsorted(q_ts, bt, side="right")
            hit = (iq < q_ts.size) & (q_ts[np.minimum(iq, q_ts.size - 1)] < nxt)
            vu[hit] = bt[hit] + int(cfg.oi_max_ref_age_s * NS)
        for w in cfg.oi_windows_s[:2]:
            m1, _, cov = _tw_piecewise(tt_, vv, vu, t, int(w * NS))
            out[f"tw_oi_known_{wname(w)}_btc"] = np.where((cov >= 1 - 1e-7) & ok, m1, np.nan)
    return out


# ------------------------------------------------------------------------------- liquidations
def liq_features(t: np.ndarray, cfg: SlowFeatureConfig, data: SlowData, history_start_ns: int):
    out = {}
    if data.liq is None:
        return out
    L = data.liq
    recv = L["recv_ns"].to_numpy()
    side = L["side"].to_numpy()
    notl = L["notional_usdt"].to_numpy()
    o = np.argsort(recv, kind="stable")
    recv, side, notl = recv[o], side[o], notl[o]
    if data.stream_gaps is None:
        from ..canonical.slow import market_stream_gaps
        data.stream_gaps = market_stream_gaps(cfg.stream_gap_s)
    g0, g1 = data.stream_gaps
    from .trades import window_gap_mask
    cS = np.concatenate([[0], np.cumsum(notl * (side == 1))])    # BUY = short liquidations
    cL = np.concatenate([[0], np.cumsum(notl * (side == -1))])   # SELL = long liquidations
    cN = np.concatenate([[0], np.cumsum(np.ones(recv.size))])
    hi = np.searchsorted(recv, t, side="right")
    hidx_t = (t - P.hour_start_ns(P.HISTORY_START, 0)) // P.HOUR_NS
    for w in cfg.liq_windows_s:
        w_ns = int(w * NS)
        lo = np.searchsorted(recv, t - w_ns, side="right")
        st = np.zeros(t.size, np.int8)
        st[t - w_ns < history_start_ns] = 1
        # hours touched by the window must all have a liquidation file
        h_lo = (t - w_ns + 1 - P.hour_start_ns(P.HISTORY_START, 0)) // P.HOUR_NS
        missing = np.zeros(t.size, bool)
        hrs = np.unique(np.concatenate([h_lo, hidx_t]))
        miss_h = np.array([h for h in range(int(hrs.min()), int(hrs.max()) + 1) if h not in data.liq_hours], np.int64)
        if miss_h.size:
            a = np.searchsorted(miss_h, h_lo, side="left")
            b = np.searchsorted(miss_h, hidx_t, side="right")
            missing = b > a
        if cfg.liq_missing_file_policy == "unknown":
            st[(st == 0) & missing] = 2
        gm = window_gap_mask(t, w_ns, g0, g1)
        st[(st == 0) & gm] = 3
        ok = st == 0
        s = wname(w)
        Ls = cS[hi] - cS[lo]
        Ll = cL[hi] - cL[lo]
        Nn = cN[hi] - cN[lo]
        out[f"liq_state_{s}"] = st
        if cfg.liq_missing_file_policy != "unknown":
            out[f"liq_inferred_zero_{s}"] = (missing & ok).astype(np.int8)
        out[f"liq_notional_short_{s}_usdt"] = np.where(ok, Ls, np.nan)
        out[f"liq_notional_long_{s}_usdt"] = np.where(ok, Ll, np.nan)
        out[f"liq_count_{s}"] = np.where(ok, Nn, np.nan)
        out[f"liq_rate_{s}_per_s"] = np.where(ok, Nn / w, np.nan)
        out[f"liq_signed_notional_{s}_usdt"] = np.where(ok, Ls - Ll, np.nan)
        tot = Ls + Ll
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"liq_imbalance_{s}"] = np.where(ok, np.where(tot > 0, (Ls - Ll) / tot, 0.0), np.nan)
        out[f"liq_any_{s}"] = np.where(ok, (Nn > 0).astype(float), np.nan)
        out[f"log1p_liq_notional_{s}"] = np.where(ok, np.log1p(tot), np.nan)
        out[f"liq_notional_rate_{s}_usdt_s"] = np.where(ok, tot / w, np.nan)
        # peak 1 s cell within the window (cells by availability second)
        out[f"liq_peak_1s_{s}_usdt"] = np.where(ok, _peak_cell(recv, notl, t, w_ns), np.nan)
    if 120 in cfg.liq_windows_s and 900 in cfg.liq_windows_s:
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = out["liq_rate_2m_per_s"]
            r15 = out["liq_rate_15m_per_s"]
            out["liq_burst_ratio_2m_15m"] = np.where(np.isfinite(r1) & np.isfinite(r15) & (r15 > 0), r1 / r15, np.nan)
    h0 = P.hour_start_ns(P.HISTORY_START, 0)
    all_miss = np.array([h for h in range(int(hidx_t.max()) + 1) if h not in data.liq_hours], np.int64)
    for sd, nm in ((1, "buy_short_liq"), (-1, "sell_long_liq")):
        tt_ = recv[side == sd]
        j = np.searchsorted(tt_, t, side="right") - 1
        last = tt_[np.maximum(j, 0)] if tt_.size else np.zeros(t.size, np.int64)
        # unknown if the feed cannot be vouched for since that notification: a later one may exist in a missing hour or gap
        unknown = window_gap_mask(t, t - last, g0, g1)
        if cfg.liq_missing_file_policy == "unknown":
            unknown |= np.searchsorted(all_miss, hidx_t, side="right") > np.searchsorted(all_miss, (last - h0) // P.HOUR_NS, side="left")
        out[f"time_since_{nm}_s"] = np.where((j >= 0) & ~unknown, (t - last) / 1e9, np.nan)
        out[f"time_since_{nm}_censored"] = (j < 0).astype(np.int8)
    return out


@njit(cache=True)
def _peak_cell(recv, notl, t, w_ns):
    n = t.shape[0]
    out = np.zeros(n)
    for i in range(n):
        lo = np.searchsorted(recv, t[i] - w_ns, side="right")
        hi = np.searchsorted(recv, t[i], side="right")
        best = 0.0
        cell = -1
        acc = 0.0
        for k in range(lo, hi):
            c = recv[k] // 1000000000
            if c != cell:
                if acc > best:
                    best = acc
                acc = 0.0
                cell = c
            acc += notl[k]
        if acc > best:
            best = acc
        out[i] = best
    return out


# ------------------------------------------------------------------------------- ticker
def ticker_features(d: date, t: np.ndarray, cfg: SlowFeatureConfig, data: SlowData, last_trade_price=None):
    out = {}
    tab = data.ticker_range(d - timedelta(days=1), d)
    if tab is None:
        return out
    recv = tab["recv_ns"].to_numpy()
    evt = tab["event_ms"].to_numpy() * P.MS_NS
    j = _asof(recv, t)
    has = j >= 0
    jj = np.maximum(j, 0)
    fresh = has & (t - evt[jj] <= cfg.ticker_max_age_s * NS)
    out["ticker_state"] = np.where(~has, 1, np.where(fresh, 0, 2)).astype(np.int8)
    out["ticker_mkt_age_ms"] = np.where(has, (t - evt[jj]) / 1e6, np.nan)

    def col(c, scale):
        return tab[c].to_numpy()[jj] / scale

    last = col("last_price", 100)
    opn = col("open_price", 100)
    hi_ = col("high_price", 100)
    lo_ = col("low_price", 100)
    wav = col("weighted_average_price", 100)
    pct = col("price_change_percent", 1e4)
    base = col("base_asset_volume", 1e3)
    quote = col("quote_asset_volume", 1e4)
    tot = tab["total_trades"].to_numpy()[jj].astype(np.float64)
    win = (tab["stat_close_ms"].to_numpy()[jj] - tab["stat_open_ms"].to_numpy()[jj]) / 1e3
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ticker_return"] = np.where(fresh, pct / 100.0, np.nan)
        out["ticker_return_consistency"] = np.where(fresh, last / opn - 1 - pct / 100.0, np.nan)
        out["ticker_range_log"] = np.where(fresh & (lo_ > 0), np.log(hi_ / lo_), np.nan)
        P_ = last_trade_price if last_trade_price is not None else last
        out["ticker_range_position_trade"] = np.where(fresh & (hi_ > lo_), (P_ - lo_) / (hi_ - lo_), np.nan)
        out["ticker_vwap_distance_trade_bps"] = np.where(fresh & (wav > 0), 1e4 * (P_ / wav - 1), np.nan)
        out["ticker_window_s"] = np.where(fresh, win, np.nan)
        out["ticker_quote_rate_usdt_s"] = np.where(fresh & (win > 0), quote / win, np.nan)
        out["ticker_avg_fill_size_btc"] = np.where(fresh & (tot > 0), base / tot, np.nan)
        out["ticker_base_volume_24h_btc"] = np.where(fresh, base, np.nan)
    return out


# ------------------------------------------------------------------------------- calendar / seasonality
def calendar_features(t: np.ndarray):
    out = {}
    sec = (t // NS) % 86400
    dow = ((t // NS) // 86400 + 3) % 7          # 1970-01-01 was a Thursday -> Monday=0
    out["hour_of_day_sin"] = np.sin(2 * np.pi * sec / 86400)
    out["hour_of_day_cos"] = np.cos(2 * np.pi * sec / 86400)
    out["day_of_week_sin"] = np.sin(2 * np.pi * (dow + sec / 86400) / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * (dow + sec / 86400) / 7)
    out["is_weekend"] = (dow >= 5).astype(np.int8)
    return out


def seasonal_slot_table(start: date, end: date):
    """Per (day, 15-min UTC slot) notional totals from canonical trades (input of the causal seasonal ratio)."""
    rows = {}
    d = start
    while d <= end:
        p = CAN / "trades" / f"{d.isoformat()}.parquet"
        if p.exists():
            t = pq.read_table(p, columns=["recv_ns", "price_e2", "qty_e3"])
            recv = t["recv_ns"].to_numpy()
            dn = t["price_e2"].to_numpy() / 100.0 * t["qty_e3"].to_numpy() / 1000.0
            slot = ((recv // NS) % 86400) // 900
            rows[d] = np.bincount(slot, weights=dn, minlength=96)[:96]
        d += timedelta(days=1)
    return rows


def seasonal_ratio(d: date, t: np.ndarray, notional_15m: np.ndarray, slots: dict, cfg: SlowFeatureConfig):
    """notional over the trailing 15 min divided by the median, over the previous N days, of the notional in the
    same 15-min UTC slot (>= min_obs valid days); strictly prior days only."""
    hist = [slots.get(d - timedelta(days=k)) for k in range(1, cfg.seasonal_lookback_days + 1)]
    hist = [h for h in hist if h is not None]
    slot = ((t // NS) % 86400) // 900
    if len(hist) < cfg.seasonal_min_obs:
        return np.full(t.size, np.nan)
    H = np.vstack(hist)
    med = np.median(H, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(med[slot] > 0, notional_15m / med[slot], np.nan)

"""Streaming L2 replay with exact, causal book features evaluated at arbitrary query times.

State: dense quantity arrays on a 0.1 USDT tick grid (index = price*10), per-side certified
domain [cert_bid_lo, +inf) / (-inf, cert_ask_hi] given by the deepest level of the anchoring
snapshot (levels outside it are unknown until updated; an isolated far update certifies nothing).

Clock: availability (effective application time from chain.resolve, ns). Exchange times are kept
for staleness ("market age" = query time - event_time of the last applied message).

Time-weighted window statistics are exact integrals of piecewise-constant states between
applications, restricted to the validity domain; the left window edge uses the state in force
at t-w (ring buffer of cumulative integrals at every event, binary search).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from .chain import Q_ANCHOR, Q_INVALIDATE, Q_UPDATE, Q_VERIFY

NT = 4_000_000            # ticks: 0 .. 400,000 USDT at 0.1
TICK = 0.1
NK = 20                   # top-K levels tracked
NML = 5                   # MLOFI ranks

# book state codes
BS_VALID, BS_UNANCHORED, BS_STALE, BS_CROSSED, BS_AWAIT, BS_SETTLE, BS_EMPTY, BS_TOP_UNCERT, BS_HEALED, \
    BS_REBUILT, BS_REBUILT_UNC = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
BS_NAMES = ["valid", "unanchored_book", "stale", "crossed_book", "awaiting_anchor_connection",
            "anchor_settle_warmup", "empty_side", "top_outside_certified_domain", "healed_unverified(experimental)",
            "rebuilt_top_certified", "rebuilt_top_uncertified"]

# int scalar state (si)
(I_BB, I_BA, I_CERT_LO, I_CERT_HI, I_ANCH, I_MODE_OK, I_LAST_T, I_LAST_EVT, I_LAST_TXN, I_CROSS_SINCE,
 I_RHEAD, I_RCOUNT, I_ESHEAD, I_ESCOUNT, I_MOHEAD, I_MOCOUNT, I_QHEAD, I_SETTLE_UNTIL, I_PREV_VALID,
 I_NMSG, I_NLEV, I_NVERIFY, I_NVERIFY_FAIL, I_NCROSS, I_NOFFTICK_POS, I_KHAS, I_KT, I_KEVT, I_NAPPLY,
 I_NANCHOR, I_ANCHOR_SYNTH, I_TMIN_B, I_TMAX_B, I_TMIN_A, I_TMAX_A, I_NINVAL, I_NCROSS_INVAL,
 I_LAST_APPLY_T, I_NES, I_NMO, I_VERIFY_MISMATCH_LEVELS, I_SHADOW, I_HEALED, I_HEAL_T, I_NHEAL, I_EPOCH,
 I_HEXT, I_LEXT, I_UNANCH, I_EMPTY_T, I_NEMPTY, I_FORCED, I_QUAR) = range(53)
NI = 53
# float scalar state (sf)
F_MARK, F_INDEX = 0, 1
NF = 4


@dataclass
class ReplayConfig:
    bands_bps: tuple = (1.0, 5.0, 10.0, 25.0, 50.0)
    ks: tuple = (5, 10, 20)
    windows_s: tuple = (1.0, 5.0, 30.0, 120.0, 900.0)
    mlofi_windows_s: tuple = (5.0, 30.0)
    tw_band_bps: tuple = (5.0, 25.0)
    minmax_max_window_s: float = 120.0
    minmax_band_bps: float = 5.0
    ar_band_bps: float = 10.0
    conc_band_bps: float = 10.0
    cost_qty_btc: tuple = (0.1, 1.0, 5.0)
    cost_notional_usdt: tuple = (1e4, 1e5, 1e6)
    basis_windows_s: tuple = (30.0, 120.0, 900.0, 3600.0)
    es_windows_s: tuple = (30.0, 120.0, 900.0)
    es_max_quote_age_ms: int = 1000
    markout_h_s: tuple = (1.0, 5.0, 30.0)
    markout_windows_s: tuple = (120.0, 900.0)
    max_age_ms: int = 1000            # book freshness for fast features (market age)
    mark_max_age_ms: int = 3000       # mark/index freshness for basis
    crossed_tolerance_ms: int = 1000  # crossed longer than this -> anchor invalidated
    synthetic_anchor_settle_ms: int = 1000
    grid_ms: int = 100
    min_coverage: float = 1.0         # strict: full coverage required for window statistics
    ring_size: int = 1 << 18
    snapshot_probe_only: bool = False  # EXPERIMENT ONLY: verification mismatches do not rebuild the state
    experiment_emit_healed: bool = False  # EXPERIMENT ONLY: emit features for healed (unverified) states
    checkpoint_cert_levels: int = 0    # >0: certify only the best N levels/side of a snapshot (vendor checkpoints hold the
                                       # collector's whole book, whose deep part is not exchange-certified); 0 = all rows
    legacy_chain: bool = False         # reproduce the resolver as it was before the D1 corrections (same as the existing datasets)
    book_mode: str = "strict"          # strict | rebuilt (optional: empty restart when no anchor, tiers kept separate)
    rebuilt_cross_quarantine_s: float = 600.0  # rebuilt mode: a crossed stream contradicts the premise of the certificate. Nothing
                                               # is emitted until this long after the last crossed message, and then the book
                                               # restarts empty, so any certificate comes from reports made after that moment
    frontier_extension: bool = True    # exact certified-domain extension (no-crossing + reported levels); False =
                                       # snapshot range only (more conservative, top of book often uncertified)

    def param_arrays(self):
        bands_c = np.array([int(round(b * 100)) for b in self.bands_bps], np.int64)
        def bi(b):
            return int(np.argmin(np.abs(np.array(self.bands_bps) - b)))
        ip = np.array([len(self.bands_bps), len(self.ks), len(self.windows_s), len(self.tw_band_bps),
                       bi(self.minmax_band_bps), bi(self.ar_band_bps), bi(self.conc_band_bps),
                       len(self.cost_qty_btc), len(self.cost_notional_usdt), len(self.basis_windows_s),
                       len(self.es_windows_s), self.es_max_quote_age_ms, len(self.markout_h_s),
                       len(self.markout_windows_s), self.max_age_ms, self.mark_max_age_ms,
                       self.crossed_tolerance_ms, self.synthetic_anchor_settle_ms, self.grid_ms * 1_000_000,
                       self.ring_size, 1 if self.snapshot_probe_only else 0,
                       1 if self.frontier_extension else 0, 1 if self.experiment_emit_healed else 0,
                       self.checkpoint_cert_levels, 1 if self.book_mode == "rebuilt" else 0,
                       int(self.rebuilt_cross_quarantine_s * 1000)], np.int64)
        return dict(
            ip=ip, bands_c=bands_c, ks=np.array(self.ks, np.int64),
            win_ns=np.array([int(w * 1e9) for w in self.windows_s], np.int64),
            win_mlofi=np.array([1 if w in self.mlofi_windows_s else 0 for w in self.windows_s], np.int64),
            win_minmax=np.array([1 if w <= self.minmax_max_window_s else 0 for w in self.windows_s], np.int64),
            tw_bands=np.array([bi(b) for b in self.tw_band_bps], np.int64),
            cost_q=np.array([int(round(q * 1000)) for q in self.cost_qty_btc], np.int64),
            cost_d=np.array(self.cost_notional_usdt, np.float64),
            bwin_ns=np.array([int(w * 1e9) for w in self.basis_windows_s], np.int64),
            eswin_ns=np.array([int(w * 1e9) for w in self.es_windows_s], np.int64),
            mo_h_ns=np.array([int(h * 1e9) for h in self.markout_h_s], np.int64),
            mowin_ns=np.array([int(w * 1e9) for w in self.markout_windows_s], np.int64),
            min_cov=float(self.min_coverage))

    # ------------------------------------------------------------------ output layout
    def columns(self):
        c = ["book_state", "book_recv_age_ms", "book_mkt_age_ms", "best_bid", "best_ask", "mid", "spread_bps",
             "qty_bid1_btc", "qty_ask1_btc", "qi_1", "weighted_mid_offset_bps", "gap_ask12_bps", "gap_bid12_bps"]
        for k in self.ks:
            c += [f"depth_bid_top{k}_btc", f"depth_ask_top{k}_btc", f"qi_top{k}"]
        for b in self.bands_bps:
            t = _bname(b)
            c += [f"depth_bid_{t}_btc", f"depth_ask_{t}_btc", f"notional_bid_{t}_usdt", f"notional_ask_{t}_usdt",
                  f"qi_{t}"]
        t = _bname(self.conc_band_bps)
        c += [f"conc_bid_{t}", f"conc_ask_{t}"]
        for q in self.cost_qty_btc:
            c += [f"cost_buy_{_qname(q)}btc_bps", f"cost_sell_{_qname(q)}btc_bps"]
        for d in self.cost_notional_usdt:
            c += [f"cost_buy_{int(d)}usdt_atmid_bps", f"cost_sell_{int(d)}usdt_atmid_bps"]
        c += ["cert_bid_floor_dist_bps", "cert_ask_ceiling_dist_bps"]
        for w, ml, mm in zip(self.windows_s, *[list(x) for x in (self.param_arrays()["win_mlofi"],
                                                                  self.param_arrays()["win_minmax"])]):
            s = _wname(w)
            c += [f"book_coverage_{s}", f"ofi_{s}_btc", f"ofi_rate_{s}_btc_s", f"ofi_norm_{s}"]
            if ml:
                for r in range(1, NML + 1):
                    c += [f"mlofi_l{r}_{s}_btc", f"mlofi_norm_l{r}_{s}"]
            c += [f"tw_spread_bps_{s}", f"tw_qi_1_{s}", f"tw_depth1_{s}_btc"]
            for b in self.tw_band_bps:
                t = _bname(b)
                c += [f"tw_depth_bid_{t}_{s}_btc", f"tw_depth_ask_{t}_{s}_btc"]
            if mm:
                t = _bname(self.minmax_band_bps)
                c += [f"max_spread_bps_{s}", f"min_depth1_{s}_btc", f"min_depth_bid_{t}_{s}_btc",
                      f"min_depth_ask_{t}_{s}_btc"]
            t = _bname(self.ar_band_bps)
            c += [f"add_bid_{t}_{s}_btc", f"rem_bid_{t}_{s}_btc", f"add_ask_{t}_{s}_btc", f"rem_ask_{t}_{s}_btc",
                  f"book_msg_count_{s}", f"book_level_changes_{s}"]
        c += ["basis_mid_index_bps", "basis_mid_mark_bps", "basis_mark_recv_age_ms", "basis_mark_mkt_age_ms"]
        for w in self.basis_windows_s:
            s = _wname(w)
            c += [f"basis_coverage_{s}", f"tw_basis_mid_index_bps_{s}", f"std_basis_mid_index_bps_{s}",
                  f"tw_basis_mid_mark_bps_{s}", f"std_basis_mid_mark_bps_{s}"]
        for w in self.es_windows_s:
            s = _wname(w)
            c += [f"eff_spread_qw_bps_{s}", f"eff_spread_dw_bps_{s}", f"eff_spread_n_{s}"]
        for h in self.markout_h_s:
            for w in self.markout_windows_s:
                c += [f"markout_{_wname(h)}_qw_bps_{_wname(w)}", f"markout_{_wname(h)}_n_{_wname(w)}"]
        return c


def _bname(b):
    return f"{b:g}bps".replace(".", "p")


def _qname(q):
    return f"{q:g}".replace(".", "p")


def _wname(w):
    w = float(w)
    if w >= 3600 and w % 3600 == 0:
        return f"{int(w // 3600)}h"
    if w >= 60 and w % 60 == 0:
        return f"{int(w // 60)}m"
    if w < 1:
        return f"{int(w * 1000)}ms"
    return f"{w:g}s"




GRID_COLS = ["t_ns", "state", "mid2_ticks", "best_bid_ticks", "best_ask_ticks", "qty_bid1_e3", "qty_ask1_e3",
             "mkt_age_ms"]

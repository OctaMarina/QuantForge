"""Volume and dollar bars built from Binance spot aggTrades (event bars, AFML §2.3.1).

A bar closes as soon as the base volume (``kind="volume"``) or the quote volume (``kind="dollar"``) traded since
it opened reaches a threshold.  The threshold of a bar is fixed when the bar opens and is estimated **per asset
from activity already observable at that moment**:

    threshold(bar opening on UTC day D) = mean of the daily traded measure over the complete days D-W .. D-1
                                          / bars_per_day

so that a day as active as the trailing W days produces ``bars_per_day`` bars on average, while busier days
produce more bars and quieter days fewer: the realised frequency varies with market activity.  No bar opens during
the first W days of an asset (the warm-up, like ``warmup_days`` of the time bars).

Conventions (the bar frame is accepted by :mod:`quant_forge.strategies` and :mod:`quant_forge.features`):

* trades with the same timestamp belong to the same bar: a bar that reached its threshold closes when the next
  trade has a strictly later timestamp, so bar starts strictly increase and every bar starts after the previous
  one ended;
* ``bar_start`` = first trade, ``bar_end`` = last trade of the bar = the time a decision can be taken;
  ``next_open`` orders of the strategies fill at the first trade of the next bar;
* a pause of more than ``max_gap_hours`` between two trades (an outage or a trading halt) closes the open bar
  (``closed_by = "gap"``) and starts a new ``segment_id``: indicators restart and open positions are censored,
  exactly as at a missing kline of the time bars;
* the bar still open at the data end is not emitted (it is incomplete, like a kline that is still forming);
* ``n_trades`` counts the underlying trades (``last_trade_id - first_trade_id + 1`` of each aggTrade, as in
  klines), ``taker_buy_base`` the volume of trades whose taker bought (``is_buyer_maker`` false);
* records with a non-positive price or quantity are dropped (the archive holds zero-price aggTrades of BTC, ETH,
  BNB and LTC on 2017-12-04 and 2018-01-13 .. 31 that the klines do not contain), and repeated records are dropped by
  aggTrade id (the ETHUSDT 2018-01 file repeats parts of the month four times; BTC and XRP repeat some trades of
  2026-02-11).  After both rules the builder's daily totals equal the 1-day klines.

Building is a single chronological pass over the archive files of a symbol (:func:`build_event_bars`): each
file is streamed, inflated and parsed on the fly (no raw data is stored), all requested specs are built at once,
and after every file the bars and the builder state are written to the cache, so an interrupted build resumes
where it stopped and a later call only processes the new days (incremental, like the kline cache).
"""

from __future__ import annotations

import http.client
import os
import pickle
import socket
import tempfile
import time
import urllib.error
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.cache import DEFAULT_CACHE_ROOT
from .archive import (AGGTRADE_COLUMNS, ArchiveFileMissing, ChecksumError, normalize_epoch_ms, plan_archive_files,
                      stream_archive_batches)

__all__ = ["EventBarSpec", "DEFAULT_EVENT_SPECS", "EventBarBuilder", "event_bars_from_trades", "build_event_bars",
           "build_event_bars_many", "event_bars_to_frame", "load_event_bars", "default_event_bars_dir",
           "event_cache_status", "daily_thresholds", "RAW_BAR_COLUMNS", "HAVE_NUMBA"]

try:  # optional accelerator (the pure-Python kernel gives identical results, but is ~100x slower)
    from numba import njit as _njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    _njit = None
    HAVE_NUMBA = False

DAY_MS = 86_400_000
_N_DAYS = 40_000                       # day index since 1970-01-01 (enough until 2079)
_F_COLS = ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote", "threshold")
_I_COLS = ("start_ms", "end_ms", "n_trades", "n_agg", "segment_id", "closed_by")
RAW_BAR_COLUMNS = _I_COLS + _F_COLS
CLOSED_BY = ("threshold", "gap")
_STATE_VERSION = 1


@dataclass(frozen=True)
class EventBarSpec:
    """One event-bar series.

    kind           ``"volume"`` (threshold in base units) or ``"dollar"`` (threshold in quote units, here USDT);
    bars_per_day   target average number of bars per day (48, 24, 6 and 1 match 30m, 1h, 4h and 1d time bars);
    window_days    complete trailing UTC days whose mean daily volume sets the threshold (also the warm-up);
    max_gap_hours  a longer pause between two trades closes the bar and starts a new segment.
    """

    kind: str
    bars_per_day: float
    window_days: int = 30
    max_gap_hours: float = 4.0

    def __post_init__(self):
        if self.kind not in ("volume", "dollar"):
            raise ValueError("kind must be 'volume' or 'dollar'")
        if not (np.isfinite(self.bars_per_day) and self.bars_per_day > 0):
            raise ValueError("bars_per_day must be > 0")
        if int(self.window_days) != self.window_days or self.window_days < 1:
            raise ValueError("window_days must be a positive integer")
        if not self.max_gap_hours > 0:
            raise ValueError("max_gap_hours must be > 0")

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.bars_per_day:g}pd_w{self.window_days}_g{self.max_gap_hours:g}h"

    @property
    def max_gap_ms(self) -> int:
        return int(round(self.max_gap_hours * 3_600_000))


#: The eight series of the bar-type study: volume and dollar bars at 48, 24, 6 and 1 bars per day.
DEFAULT_EVENT_SPECS = tuple(EventBarSpec(k, n) for k in ("volume", "dollar") for n in (48, 24, 6, 1))


def default_event_bars_dir() -> Path:
    return DEFAULT_CACHE_ROOT / "binance_event_bars"


# ----------------------------------------------------------------------------------------------- thresholds
def daily_thresholds(daily_measure, bars_per_day: float, window_days: int = 30, first_day: int = 0) -> np.ndarray:
    """Threshold in force for bars opening on each day: mean of the ``window_days`` previous days / bars_per_day.

    ``daily_measure[d]`` is the volume (or quote volume) traded on day ``d``; ``first_day`` is the first day with
    trades.  NaN while fewer than ``window_days`` complete days exist or when they traded nothing.  This is the
    rule applied inside the bar kernel (exposed for documentation and tests).
    """
    x = np.asarray(daily_measure, dtype=float)
    csum = np.r_[0.0, np.cumsum(x)]
    d = np.arange(len(x))
    out = np.full(len(x), np.nan)
    ok = d - first_day >= window_days
    s = csum[d[ok]] - csum[d[ok] - window_days]
    out[ok] = np.where(s > 0, s / window_days / bars_per_day, np.nan)
    return out


# ----------------------------------------------------------------------------------------------- kernel
def _bar_kernel_py(t, p, q, ntr, buy, use_quote, day_meas, first_day, window, bpd, max_gap, sf, si, of, oi):
    """Process one batch of trades for one spec; completed bars go to ``of`` / ``oi``.  Returns their number.

    State ``si`` = [in_bar, start, last, n_trades, n_agg, segment, previous trade time],
    ``sf`` = [open, high, low, close, volume, quote, taker base, taker quote, threshold, accumulated measure].
    """
    k = 0
    in_bar, start, last, n_acc, n_agg, seg, prev = si[0], si[1], si[2], si[3], si[4], si[5], si[6]
    o, h, lo, c, vol, quo, tb, tq, thr, acc = sf[0], sf[1], sf[2], sf[3], sf[4], sf[5], sf[6], sf[7], sf[8], sf[9]
    for i in range(t.shape[0]):
        ti = t[i]
        # 1. a bar that reached its threshold closes once a strictly later timestamp arrives
        if in_bar == 1 and acc >= thr and ti > last:
            of[k, 0] = o; of[k, 1] = h; of[k, 2] = lo; of[k, 3] = c; of[k, 4] = vol
            of[k, 5] = quo; of[k, 6] = tb; of[k, 7] = tq; of[k, 8] = thr
            oi[k, 0] = start; oi[k, 1] = last; oi[k, 2] = n_acc; oi[k, 3] = n_agg; oi[k, 4] = seg; oi[k, 5] = 0
            k += 1
            in_bar = 0
        # 2. an outage / halt closes the open bar and starts a new segment
        if prev >= 0 and ti - prev > max_gap:
            if in_bar == 1:
                of[k, 0] = o; of[k, 1] = h; of[k, 2] = lo; of[k, 3] = c; of[k, 4] = vol
                of[k, 5] = quo; of[k, 6] = tb; of[k, 7] = tq; of[k, 8] = thr
                oi[k, 0] = start; oi[k, 1] = last; oi[k, 2] = n_acc; oi[k, 3] = n_agg; oi[k, 4] = seg; oi[k, 5] = 1
                k += 1
                in_bar = 0
            seg += 1
        prev = ti
        pi = p[i]
        qi = q[i]
        dq = pi * qi
        if in_bar == 0:
            d = ti // 86400000
            if d - first_day < window:
                continue                      # warm-up: no threshold yet
            s = 0.0
            for j in range(d - window, d):
                s += day_meas[j]
            if not s > 0.0:
                continue
            thr = s / window / bpd
            in_bar = 1
            start = ti
            o = pi; h = pi; lo = pi
            vol = 0.0; quo = 0.0; tb = 0.0; tq = 0.0; acc = 0.0
            n_acc = 0; n_agg = 0
        if pi > h:
            h = pi
        if pi < lo:
            lo = pi
        c = pi
        last = ti
        vol += qi
        quo += dq
        n_acc += ntr[i]
        n_agg += 1
        if use_quote:
            acc += dq
        else:
            acc += qi
        if buy[i]:
            tb += qi
            tq += dq
    si[0] = in_bar; si[1] = start; si[2] = last; si[3] = n_acc; si[4] = n_agg; si[5] = seg; si[6] = prev
    sf[0] = o; sf[1] = h; sf[2] = lo; sf[3] = c; sf[4] = vol; sf[5] = quo; sf[6] = tb; sf[7] = tq
    sf[8] = thr; sf[9] = acc
    return k


def _accumulate_days_py(t, p, q, day_base, day_quote):
    """Add each trade to its day's totals in stream order (bitwise independent of how the stream is cut)."""
    for i in range(t.shape[0]):
        d = t[i] // 86400000
        day_base[d] += q[i]
        day_quote[d] += p[i] * q[i]


def _dedupe_py(ids, t, state, keep):
    """Mark records to keep: a record is a duplicate when its aggTrade id is not above every id seen so far AND its
    timestamp is not newer than every timestamp seen (repeated blocks in an archive file).  A smaller id with a newer
    timestamp is a restart of the id sequence and is kept.  ``state`` = [last id, last timestamp]."""
    last_id, last_t = state[0], state[1]
    n_dup = 0
    for i in range(ids.shape[0]):
        if ids[i] > last_id:
            keep[i] = True
            last_id = ids[i]
            if t[i] > last_t:
                last_t = t[i]
        elif t[i] > last_t:
            keep[i] = True
            last_id = ids[i]
            last_t = t[i]
        else:
            keep[i] = False
            n_dup += 1
    state[0], state[1] = last_id, last_t
    return n_dup


_bar_kernel_nb = _njit(cache=True)(_bar_kernel_py) if HAVE_NUMBA else None
_dedupe_nb = _njit(cache=True)(_dedupe_py) if HAVE_NUMBA else None
_accumulate_days_nb = _njit(cache=True)(_accumulate_days_py) if HAVE_NUMBA else None


# ----------------------------------------------------------------------------------------------- builder
class EventBarBuilder:
    """Streaming construction of several event-bar series from chronologically ordered trades.

    ``feed`` accepts trades in any batch sizes (the result does not depend on how the stream is cut);
    ``take_bars`` returns the bars completed since the previous call.  Timestamps are milliseconds; a timestamp
    earlier than its predecessor (never seen in the Binance archive) is raised to the running maximum and counted
    in ``n_reordered``; trades with a non-positive or non-finite price or quantity are dropped and counted in
    ``n_dropped``; with ``agg_ids`` a record whose aggTrade id is not above every id seen (and whose timestamp is
    not newer than every timestamp seen) is a duplicate, dropped and counted in ``n_duplicates``.
    """

    def __init__(self, specs: Iterable[EventBarSpec], *, use_numba: bool | None = None):
        self.specs = tuple(dict.fromkeys(specs))
        if not self.specs:
            raise ValueError("at least one spec is needed")
        self.use_numba = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
        self.first_day = -1
        self.day_base = np.zeros(_N_DAYS)
        self.day_quote = np.zeros(_N_DAYS)
        self.last_ms = -1
        self.n_fed = 0
        self.n_reordered = 0
        self.n_dropped = 0
        self.n_duplicates = 0
        self._dedupe_state = np.array([-1, -1], dtype=np.int64)     # last aggTrade id, last timestamp
        self._sf = {s.name: np.array([np.nan] * 9 + [0.0]) for s in self.specs}
        self._si = {s.name: np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.int64) for s in self.specs}
        self._chunks: dict[str, list] = {s.name: [] for s in self.specs}

    # -- state for rollback / persistence ---------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"first_day": self.first_day, "day_base": self.day_base.copy(), "day_quote": self.day_quote.copy(),
                "last_ms": self.last_ms, "n_fed": self.n_fed, "n_reordered": self.n_reordered, "n_dropped": self.n_dropped,
                "n_duplicates": self.n_duplicates, "dedupe": self._dedupe_state.copy(),
                "sf": {k: v.copy() for k, v in self._sf.items()}, "si": {k: v.copy() for k, v in self._si.items()},
                "chunks": {k: len(v) for k, v in self._chunks.items()}}

    def restore(self, snap: dict) -> None:
        self.first_day, self.last_ms = snap["first_day"], snap["last_ms"]
        self.day_base, self.day_quote = snap["day_base"].copy(), snap["day_quote"].copy()
        self.n_fed, self.n_reordered, self.n_dropped = snap["n_fed"], snap["n_reordered"], snap["n_dropped"]
        self.n_duplicates, self._dedupe_state = snap["n_duplicates"], snap["dedupe"].copy()
        self._sf = {k: v.copy() for k, v in snap["sf"].items()}
        self._si = {k: v.copy() for k, v in snap["si"].items()}
        self._chunks = {k: v[:snap["chunks"][k]] for k, v in self._chunks.items()}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_chunks"] = {k: [] for k in self._chunks}          # completed bars are persisted separately
        return d

    def __setstate__(self, state):
        state.setdefault("n_dropped", 0)                       # states written before these fields existed
        state.setdefault("n_duplicates", 0)
        state.setdefault("_dedupe_state", np.array([-1, state.get("last_ms", -1)], dtype=np.int64))
        self.__dict__.update(state)

    # -- streaming ------------------------------------------------------------------------------------------
    def feed(self, t_ms, price, qty, n_trades=None, is_buyer_maker=None, agg_ids=None) -> None:
        t = np.ascontiguousarray(t_ms, dtype=np.int64)
        n = len(t)
        if n == 0:
            return
        p = np.ascontiguousarray(price, dtype=np.float64)
        q = np.ascontiguousarray(qty, dtype=np.float64)
        ntr = (np.ones(n, dtype=np.int64) if n_trades is None else np.ascontiguousarray(n_trades, dtype=np.int64))
        buy = (np.zeros(n, dtype=np.bool_) if is_buyer_maker is None
               else ~np.ascontiguousarray(is_buyer_maker, dtype=np.bool_))
        # the archive holds a few invalid records (e.g. zero-price aggTrades of BTC/ETH/BNB/LTC on 2017-12-04 and
        # 2018-01-13..15/31, absent from the klines): they are dropped and counted
        valid = np.isfinite(p) & np.isfinite(q) & (p > 0) & (q > 0)
        if not valid.all():
            self.n_dropped += int(n - valid.sum())
        if agg_ids is not None:
            # repeated blocks (the ETHUSDT 2018-01 file holds each trade of parts of the month four times; BTC and XRP
            # repeat some trades of 2026-02-11): keep each aggTrade id once
            ids = np.ascontiguousarray(agg_ids, dtype=np.int64)[valid]
            keep = np.empty(len(ids), dtype=np.bool_)
            self.n_duplicates += int((_dedupe_nb if self.use_numba else _dedupe_py)(ids, t[valid], self._dedupe_state, keep))
            idx = np.flatnonzero(valid)[keep]
        else:
            idx = np.flatnonzero(valid)
        if len(idx) < n:
            t, p, q, ntr, buy = t[idx], p[idx], q[idx], ntr[idx], buy[idx]
            n = len(t)
            if n == 0:
                return
        if (self.last_ms >= 0 and t[0] < self.last_ms) or (n > 1 and np.any(t[1:] < t[:-1])):
            fixed = np.maximum.accumulate(np.maximum(t, self.last_ms))
            self.n_reordered += int(np.count_nonzero(fixed != t))
            t = fixed
        if self.first_day < 0:
            self.first_day = int(t[0] // DAY_MS)
        # day totals first: bars opening on day D only read days < D, which are complete once a trade of D is seen
        (_accumulate_days_nb if self.use_numba else _accumulate_days_py)(t, p, q, self.day_base, self.day_quote)
        self.last_ms = int(t[-1])
        self.n_fed += n
        of = np.empty((n + 1, len(_F_COLS)))
        oi = np.empty((n + 1, len(_I_COLS)), dtype=np.int64)
        kernel = _bar_kernel_nb if self.use_numba else _bar_kernel_py
        for s in self.specs:
            dollar = s.kind == "dollar"
            k = kernel(t, p, q, ntr, buy, dollar, self.day_quote if dollar else self.day_base, self.first_day,
                       int(s.window_days), float(s.bars_per_day), s.max_gap_ms, self._sf[s.name], self._si[s.name],
                       of, oi)
            if k:
                self._chunks[s.name].append((of[:k].copy(), oi[:k].copy()))

    def take_bars(self) -> dict[str, pd.DataFrame]:
        """``{spec name: raw bars}`` completed since the previous call (columns :data:`RAW_BAR_COLUMNS`)."""
        out = {}
        for s in self.specs:
            chunks = self._chunks[s.name]
            fv = np.concatenate([c[0] for c in chunks]) if chunks else np.empty((0, len(_F_COLS)))
            iv = np.concatenate([c[1] for c in chunks]) if chunks else np.empty((0, len(_I_COLS)), dtype=np.int64)
            df = pd.DataFrame(iv, columns=list(_I_COLS))
            for j, c in enumerate(_F_COLS):
                df[c] = fv[:, j]
            df["closed_by"] = df["closed_by"].astype(np.int8)
            out[s.name] = df.loc[:, list(RAW_BAR_COLUMNS)]
            self._chunks[s.name] = []
        return out

    def open_bar(self, spec: EventBarSpec) -> dict | None:
        """The bar still forming for ``spec`` (start, accumulated measure, threshold) or ``None``."""
        si, sf = self._si[spec.name], self._sf[spec.name]
        if si[0] != 1:
            return None
        return {"start_ms": int(si[1]), "last_ms": int(si[2]), "accumulated": float(sf[9]), "threshold": float(sf[8])}


def event_bars_from_trades(trades: pd.DataFrame, specs: Iterable[EventBarSpec], *, time_col: str = "transact_time",
                           price_col: str = "price", qty_col: str = "quantity", batch_size: int | None = None,
                           use_numba: bool | None = None) -> dict[str, pd.DataFrame]:
    """Raw event bars of an in-memory trade table (``transact_time`` in ms or µs, ``price``, ``quantity`` and
    optionally ``first_trade_id`` / ``last_trade_id`` / ``is_buyer_maker``).  ``batch_size`` feeds the trades in
    pieces (the result is the same)."""
    b = EventBarBuilder(specs, use_numba=use_numba)
    t = normalize_epoch_ms(trades[time_col].to_numpy())
    ntr = ((trades["last_trade_id"] - trades["first_trade_id"] + 1).to_numpy()
           if {"first_trade_id", "last_trade_id"} <= set(trades.columns) else None)
    ibm = trades["is_buyer_maker"].to_numpy() if "is_buyer_maker" in trades.columns else None
    ids = trades["agg_trade_id"].to_numpy() if "agg_trade_id" in trades.columns else None
    step = batch_size or max(len(trades), 1)
    for a in range(0, len(trades), step):
        sl = slice(a, a + step)
        b.feed(t[sl], trades[price_col].to_numpy()[sl], trades[qty_col].to_numpy()[sl],
               None if ntr is None else ntr[sl], None if ibm is None else ibm[sl], None if ids is None else ids[sl])
    return b.take_bars()


# ----------------------------------------------------------------------------------------------- cache / build
_AGG_TYPES = None


def _agg_types():
    global _AGG_TYPES
    if _AGG_TYPES is None:
        import pyarrow as pa

        _AGG_TYPES = {"agg_trade_id": pa.int64(), "price": pa.float64(), "quantity": pa.float64(), "first_trade_id": pa.int64(),
                      "last_trade_id": pa.int64(), "transact_time": pa.int64(), "is_buyer_maker": pa.bool_()}
    return _AGG_TYPES


def _utc_day(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return ts.normalize()


def _atomic_write(path: Path, write) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_state(folder: Path) -> dict | None:
    p = folder / "state.pkl"
    if not p.exists():
        return None
    st = pickle.loads(p.read_bytes())
    return st if st.get("version") == _STATE_VERSION else None


def event_cache_status(symbol: str, *, cache_dir=None) -> dict | None:
    """Summary of the cached build of ``symbol`` (specs, next day to process, files, missing files) or None."""
    st = _load_state(Path(cache_dir or default_event_bars_dir()).expanduser() / symbol)
    if st is None:
        return None
    return {"specs": [s.name for s in st["builder"].specs], "next_day": st["next_day"], "first_day":
            (pd.Timestamp(st["builder"].first_day * DAY_MS, unit="ms", tz="UTC") if st["builder"].first_day >= 0
             else None), "files": len(st["files"]), "missing": list(st["missing"]), "trades": st["builder"].n_fed,
            "reordered": st["builder"].n_reordered, "dropped": getattr(st["builder"], "n_dropped", 0),
            "duplicates": getattr(st["builder"], "n_duplicates", 0),
            "seconds": st.get("seconds", 0.0), "bytes": st.get("bytes", 0)}


_RETRYABLE = (ChecksumError, IOError, urllib.error.URLError, http.client.HTTPException, ConnectionError,
              socket.timeout, TimeoutError)


def _stream_file(builder: EventBarBuilder, symbol: str, period: str, timeout: float) -> dict:
    stats: dict = {}
    for batch in stream_archive_batches("aggTrades", symbol, period, columns=AGGTRADE_COLUMNS, types=_agg_types(),
                                        timeout=timeout, stats=stats):
        t = normalize_epoch_ms(batch.column("transact_time").to_numpy())
        ntr = (batch.column("last_trade_id").to_numpy() - batch.column("first_trade_id").to_numpy() + 1)
        builder.feed(t, batch.column("price").to_numpy(), batch.column("quantity").to_numpy(), ntr,
                     batch.column("is_buyer_maker").to_numpy(zero_copy_only=False), batch.column("agg_trade_id").to_numpy())
    return stats


def build_event_bars(symbol: str, specs: Sequence[EventBarSpec] = DEFAULT_EVENT_SPECS, *, end, start="2017-08-01",
                     cache_dir=None, timeout: float = 180, retries: int = 4, today=None, verbose: bool = True,
                     log=print) -> dict:
    """Build or extend the cached event bars of ``symbol`` (e.g. ``"BTCUSDT"``) through the whole UTC days before
    ``end`` (a mid-day ``end`` is floored to its day).

    All ``specs`` are built in one pass; if the cache was built for other specs, the symbol is rebuilt from
    scratch for the union.  Files: ``<cache_dir>/<symbol>/<spec.name>.parquet`` (raw bars) and ``state.pkl``
    (builder state, processed files, files missing from the archive).  Returns :func:`event_cache_status`.
    """
    folder = Path(cache_dir or default_event_bars_dir()).expanduser() / symbol
    folder.mkdir(parents=True, exist_ok=True)
    end_day = _utc_day(end)
    st = _load_state(folder)
    wanted = list(dict.fromkeys(specs))
    if st is not None and not {s.name for s in wanted} <= {s.name for s in st["builder"].specs}:
        wanted = list(dict.fromkeys([*st["builder"].specs, *wanted]))
        for f in folder.glob("*.parquet"):
            f.unlink()
        st = None
    if st is None:
        st = {"version": _STATE_VERSION, "builder": EventBarBuilder(wanted), "next_day": _utc_day(start),
              "files": [], "missing": [], "seconds": 0.0, "bytes": 0}
    builder: EventBarBuilder = st["builder"]
    if st["next_day"] >= end_day:
        return event_cache_status(symbol, cache_dir=cache_dir)
    stored = {s.name: (pd.read_parquet(folder / f"{s.name}.parquet") if (folder / f"{s.name}.parquet").exists()
                       else pd.DataFrame(columns=list(RAW_BAR_COLUMNS))) for s in builder.specs}

    def commit(next_day: pd.Timestamp) -> None:
        new = builder.take_bars()
        for s in builder.specs:
            if len(new[s.name]):
                stored[s.name] = (pd.concat([stored[s.name], new[s.name]], ignore_index=True) if len(stored[s.name])
                                  else new[s.name])
                _atomic_write(folder / f"{s.name}.parquet", lambda tmp, df=stored[s.name]: df.to_parquet(tmp, index=False))
        st["next_day"] = next_day
        _atomic_write(folder / "state.pkl", lambda tmp: Path(tmp).write_bytes(pickle.dumps(st, protocol=5)))

    def process(kind: str, period: str) -> str:
        """``"ok"``, ``"missing"`` or raises after ``retries`` failed attempts."""
        snap = builder.snapshot()
        for attempt in range(retries):
            t0 = time.time()
            try:
                stats = _stream_file(builder, symbol, period, timeout)
                st["seconds"] += time.time() - t0
                st["bytes"] += stats.get("raw_bytes", 0)
                if verbose:
                    mb = stats.get("raw_bytes", 0) / 1e6
                    log(f"{symbol} {period}: {mb:8.1f} MB in {time.time() - t0:6.1f} s "
                        f"({mb / max(time.time() - t0, 1e-9):5.1f} MB/s), {builder.n_fed:,} trades so far")
                return "ok"
            except ArchiveFileMissing:
                builder.restore(snap)
                return "missing"
            except _RETRYABLE as exc:
                builder.restore(snap)
                if verbose:
                    log(f"{symbol} {period}: attempt {attempt + 1} failed ({exc!r}); retrying")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{symbol} {period}: giving up after {retries} attempts")

    for kind, period in plan_archive_files(st["next_day"], end_day, today=today):
        if kind == "monthly":
            p0 = pd.Timestamp(period + "-01", tz="UTC")
            p1 = p0 + pd.offsets.MonthBegin(1)
            res = process(kind, period)
            if res == "missing" and builder.first_day >= 0:
                # a month without a monthly file after the listing: fall back to its daily files
                for day in pd.date_range(p0, p1 - pd.Timedelta(days=1), freq="D"):
                    d = day.strftime("%Y-%m-%d")
                    if process("daily", d) == "missing":
                        st["missing"].append(d)
                    else:
                        st["files"].append(d)
                    commit(day + pd.Timedelta(days=1))
                continue
            if res == "ok":
                st["files"].append(period)
            commit(p1)
        else:
            day = pd.Timestamp(period, tz="UTC")
            res = process(kind, period)
            if res == "ok":
                st["files"].append(period)
            elif builder.first_day >= 0:
                st["missing"].append(period)
            commit(day + pd.Timedelta(days=1))
    return event_cache_status(symbol, cache_dir=cache_dir)


def _build_worker(symbol, specs, end, cache_dir, log_path):
    with open(log_path, "a", buffering=1) as fh:
        def log(msg):
            fh.write(f"{pd.Timestamp.now():%H:%M:%S} {msg}\n")
        t0 = time.time()
        try:
            status = build_event_bars(symbol, specs, end=end, cache_dir=cache_dir, log=log)
            log(f"{symbol} DONE in {(time.time() - t0) / 60:.1f} min: {status}")
            return symbol, status, None
        except Exception as exc:  # reported to the caller, the other symbols continue
            log(f"{symbol} FAILED: {exc!r}")
            return symbol, None, repr(exc)


def build_event_bars_many(symbols: Sequence[str], specs: Sequence[EventBarSpec] = DEFAULT_EVENT_SPECS, *, end,
                          workers: int = 4, cache_dir=None, log_path=None) -> dict:
    """:func:`build_event_bars` for several symbols in parallel processes (in the given order: put the largest
    first).  Progress lines go to ``log_path`` (default: ``<cache_dir>/build.log``).  Returns
    ``{symbol: status or error string}``."""
    cache_dir = Path(cache_dir or default_event_bars_dir()).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_path or cache_dir / "build.log")
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_build_worker, s, list(specs), end, str(cache_dir), log_path) for s in symbols]
        for f in as_completed(futs):
            sym, status, err = f.result()
            out[sym] = status if err is None else err
    return out


# ----------------------------------------------------------------------------------------------- bar frames
def event_bars_to_frame(raw: pd.DataFrame, *, end=None, bar_type: str | None = None) -> pd.DataFrame:
    """Raw cached bars -> the bar frame of :func:`quant_forge.data.klines_to_bars` (index ``t`` = bar end).

    Extra columns: ``first_trade_ts_us``, ``segment_id`` (a change = a new segment), ``taker_buy_quote``,
    ``threshold``, ``n_agg``, ``closed_by`` and ``duration_h`` (time since the previous bar's end, NaN at a new
    segment).  ``bar_seq`` is consecutive (breaks are carried by ``segment_id``); ``complete`` is 1 for every
    emitted bar.  Bars ending after ``end`` are dropped.
    """
    raw = raw.sort_values("start_ms", kind="stable")
    if end is not None:
        end_ms = int(pd.Timestamp(end).tz_convert("UTC").value // 10 ** 6) if pd.Timestamp(end).tzinfo else \
            int(pd.Timestamp(end).tz_localize("UTC").value // 10 ** 6)
        raw = raw.loc[raw["end_ms"] <= end_ms]
    start_ms = raw["start_ms"].to_numpy(np.int64)
    end_ms_ = raw["end_ms"].to_numpy(np.int64)
    bar_start = pd.to_datetime(start_ms, unit="ms", utc=True)
    bar_end = pd.to_datetime(end_ms_, unit="ms", utc=True)
    seg = raw["segment_id"].to_numpy(np.int64)
    df = pd.DataFrame({"bar_start": bar_start, "bar_end": bar_end,
                       "open": raw["open"].to_numpy(), "high": raw["high"].to_numpy(), "low": raw["low"].to_numpy(),
                       "close": raw["close"].to_numpy(), "volume": raw["volume"].to_numpy(),
                       "quote_volume": raw["quote_volume"].to_numpy(), "n_trades": raw["n_trades"].to_numpy(float),
                       "taker_buy_base": raw["taker_buy_base"].to_numpy()}, index=pd.DatetimeIndex(bar_end, name="t"))
    df["bar_start_us"] = start_ms * 1000
    df["bar_end_us"] = end_ms_ * 1000
    df["first_trade_ts_us"] = df["bar_start_us"]
    df["bar_seq"] = np.arange(len(df), dtype=np.int64)
    df["segment_id"] = seg
    df["complete"] = 1
    df["taker_buy_quote"] = raw["taker_buy_quote"].to_numpy()
    df["threshold"] = raw["threshold"].to_numpy()
    df["n_agg"] = raw["n_agg"].to_numpy(np.int64)
    df["closed_by"] = np.asarray(CLOSED_BY, dtype=object)[raw["closed_by"].to_numpy(np.int64)]
    dur = np.r_[np.nan, np.diff(end_ms_) / 3.6e6]
    dur[np.r_[True, seg[1:] != seg[:-1]]] = np.nan
    df["duration_h"] = dur
    if bar_type is not None:
        df["bar_type"] = bar_type
    return df


def load_event_bars(assets: Sequence[str], spec: EventBarSpec, *, end, quote: str = "USDT", cache_dir=None,
                    update: bool = True, build_specs: Sequence[EventBarSpec] | None = None) -> dict[str, pd.DataFrame]:
    """``{asset: bar frame}`` of event bars for Binance spot ``<asset><quote>``, trimmed at ``end``.

    With ``update`` a symbol whose cache does not reach ``end`` is built / extended first (this downloads the
    aggTrades archive and can take long for large symbols; :func:`build_event_bars_many` does it in parallel).
    ``build_specs`` are built together with ``spec`` when a build is needed (default: :data:`DEFAULT_EVENT_SPECS`).
    The order of the returned dict is the order of ``assets``.
    """
    folder = Path(cache_dir or default_event_bars_dir()).expanduser()
    specs = list(dict.fromkeys([spec, *(build_specs or DEFAULT_EVENT_SPECS)]))
    out = {}
    for a in assets:
        sym = f"{a}{quote}"
        status = event_cache_status(sym, cache_dir=folder)
        complete = (status is not None and spec.name in status["specs"] and status["next_day"] >= _utc_day(end))
        if not complete:
            if not update:
                raise FileNotFoundError(f"event bars {spec.name} of {sym} are not cached through {end}; "
                                        "call build_event_bars / build_event_bars_many first")
            build_event_bars(sym, specs, end=end, cache_dir=folder)
        raw = pd.read_parquet(folder / sym / f"{spec.name}.parquet")
        out[a] = event_bars_to_frame(raw, end=end, bar_type=spec.name)
    return out


def _spec_dict(spec: EventBarSpec) -> dict:   # pragma: no cover - convenience for logs
    return asdict(spec)

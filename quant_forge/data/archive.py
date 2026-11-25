"""Binance public data archive (``data.binance.vision``): zipped CSV files of spot klines and aggTrades.

The archive publishes one zip per symbol and month (``monthly/``) and one per day (``daily/``), each with a
``.CHECKSUM`` (SHA-256) companion.  It is the practical source for fine data: one month of 1-minute klines is
about 2 MB, whereas the REST endpoint pages 1 000 klines per request.

* :func:`fetch_klines_archive` - klines of an interval over ``[start, end)`` from monthly files plus daily files
  for the last, incomplete month; same columns as :func:`quant_forge.data.binance.fetch_klines`, so it can be
  passed to :func:`quant_forge.data.binance.load_klines` as ``fetcher``;
* :func:`fetch_aggtrades_archive` - aggTrades of a range of days (``transact_time`` in milliseconds);
* :func:`stream_archive_batches` - one archive file as Arrow record batches *while it downloads* (the zip is
  inflated on the fly, nothing is written to disk); this is how months of aggTrades (up to 16 GB of CSV for one
  BTC month) are turned into event bars (:mod:`quant_forge.data.event_bars`);
* :func:`plan_archive_files` - the monthly / daily files that cover a range of whole UTC days.

Timestamps: files from 2025-01-01 onwards carry microseconds instead of milliseconds; both are normalised to
**milliseconds** here (:func:`normalize_epoch_ms`).  A missing file (HTTP 404: the symbol was not listed yet, is
delisted, or the file is not published yet) is skipped.
"""

from __future__ import annotations

import hashlib
import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["ARCHIVE_URL", "AGGTRADE_COLUMNS", "archive_url", "fetch_archive_csv", "fetch_klines_archive",
           "fetch_aggtrades_archive", "normalize_epoch_ms", "archive_periods", "plan_archive_files",
           "stream_archive_batches", "ArchiveFileMissing", "ChecksumError"]


class ArchiveFileMissing(FileNotFoundError):
    """The archive has no such file (HTTP 404)."""


class ChecksumError(IOError):
    """The downloaded bytes do not match the published SHA-256."""

ARCHIVE_URL = "https://data.binance.vision/data/spot"
AGGTRADE_COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time",
                    "is_buyer_maker", "is_best_match"]
_KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades",
                  "taker_buy_base", "taker_buy_quote", "ignore"]
_US_THRESHOLD = 10 ** 14          # epoch values above this are microseconds (ms epochs are ~1.5e12 in 2017-2030)


def normalize_epoch_ms(values) -> np.ndarray:
    """Epoch integers in milliseconds or microseconds -> milliseconds (int64)."""
    v = np.asarray(values, dtype=np.int64)
    return np.where(v >= _US_THRESHOLD, v // 1000, v)


def archive_url(kind: str, symbol: str, period: str, interval: str | None = None, *, base: str = ARCHIVE_URL) -> str:
    """URL of one archive file.  ``period`` is ``"YYYY-MM"`` (monthly) or ``"YYYY-MM-DD"`` (daily)."""
    scope = "daily" if len(period) == 10 else "monthly"
    if kind == "klines":
        if not interval:
            raise ValueError("klines need an interval")
        return f"{base}/{scope}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip"
    if kind == "aggTrades":
        return f"{base}/{scope}/aggTrades/{symbol}/{symbol}-aggTrades-{period}.zip"
    raise ValueError(f"unknown archive kind {kind!r}")


def _download(url: str, timeout: float) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_archive_csv(kind: str, symbol: str, period: str, interval: str | None = None, *, columns: Sequence[str],
                      timeout: float = 120, verify_checksum: bool = True, base: str = ARCHIVE_URL) -> pd.DataFrame | None:
    """One archive file as a DataFrame with ``columns`` (``None`` when the file does not exist).

    Files have no header line in the older years and a header line in the newer ones; both are handled.
    """
    url = archive_url(kind, symbol, period, interval, base=base)
    blob = _download(url, timeout)
    if blob is None:
        return None
    if verify_checksum:
        chk = _download(url + ".CHECKSUM", timeout)
        if chk is not None:
            expected = chk.decode().split()[0].strip().lower()
            if hashlib.sha256(blob).hexdigest() != expected:
                raise IOError(f"checksum mismatch for {url}")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, names=list(columns), dtype=str)
    if len(df) and not df.iloc[0, 0].lstrip("-").isdigit():       # header line present
        df = df.iloc[1:]
    return df.reset_index(drop=True)


def archive_periods(start, end, *, today=None) -> tuple[list[str], list[str]]:
    """``(months, days)`` covering ``[start, end)``: whole months, then the days of the last incomplete month.

    Daily files exist up to the day before ``today`` (UTC); later days are left to the REST endpoint.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC")
    today = (today.tz_convert("UTC") if today.tzinfo else today.tz_localize("UTC")).normalize()
    last_full_month_end = min(end, today).normalize().replace(day=1)       # first day of the current/end month
    months = [p.strftime("%Y-%m") for p in pd.period_range(start.strftime("%Y-%m"), freq="M",
                                                             end=(last_full_month_end - pd.Timedelta(days=1)).strftime("%Y-%m"))
              if p.start_time.tz_localize("UTC") < last_full_month_end]
    day_start = max(last_full_month_end, start.normalize())
    day_stop = min(end.normalize() + (pd.Timedelta(days=1) if end != end.normalize() else pd.Timedelta(0)), today)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(day_start, day_stop - pd.Timedelta(days=1), freq="D")] \
        if day_stop > day_start else []
    return months, days


def _parallel(fn, periods, workers):
    if workers <= 1 or len(periods) <= 1:
        return [fn(p) for p in periods]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, periods))


def fetch_klines_archive(symbol: str, interval: str = "1m", start="2017-08-01", end=None, *, workers: int = 6,
                         timeout: float = 120, verify_checksum: bool = True, today=None,
                         base: str = ARCHIVE_URL) -> pd.DataFrame:
    """Klines with open time in ``[start, end)`` from the archive (monthly files, then daily files).

    Returns the columns of :data:`quant_forge.data.binance.KLINE_COLUMNS` without ``ignore``; ``open_time`` and
    ``close_time`` in milliseconds; empty frame when nothing exists.  Days after the last published daily file
    are not covered (use the REST endpoint for them).
    """
    end = pd.Timestamp.now(tz="UTC") if end is None else end
    months, days = archive_periods(start, end, today=today)

    def one(period):
        return fetch_archive_csv("klines", symbol, period, interval, columns=_KLINE_COLUMNS, timeout=timeout,
                                 verify_checksum=verify_checksum, base=base)

    parts = [p for p in _parallel(one, months + days, workers) if p is not None and len(p)]
    cols = _KLINE_COLUMNS[:-1]
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, ignore_index=True)[cols]
    df["open_time"] = normalize_epoch_ms(df["open_time"].astype(np.int64))
    df["close_time"] = normalize_epoch_ms(df["close_time"].astype(np.int64))
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(np.int64)
    s_ms = int(pd.Timestamp(start).tz_localize("UTC").timestamp() * 1000) if pd.Timestamp(start).tzinfo is None \
        else int(pd.Timestamp(start).timestamp() * 1000)
    e_ms = int(pd.Timestamp(end).tz_localize("UTC").timestamp() * 1000) if pd.Timestamp(end).tzinfo is None \
        else int(pd.Timestamp(end).timestamp() * 1000)
    df = df.loc[(df["open_time"] >= s_ms) & (df["open_time"] < e_ms)]
    return df.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)


def _utc_day(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return ts.normalize()


def plan_archive_files(start, end, *, today=None) -> list[tuple[str, str]]:
    """Archive files covering the whole UTC days ``[floor(start), floor(end))`` in chronological order.

    A month entirely inside the range whose last day is before ``today`` (UTC) is one monthly file
    (``("monthly", "YYYY-MM")``); the other days are daily files (``("daily", "YYYY-MM-DD")``), only those before
    ``today`` (later days are not published yet).
    """
    d0, d1 = _utc_day(start), _utc_day(end)
    today = _utc_day(today if today is not None else pd.Timestamp.now(tz="UTC"))
    d1 = min(d1, today)
    out: list[tuple[str, str]] = []
    d = d0
    while d < d1:
        m_start = d.replace(day=1)
        m_end = m_start + pd.offsets.MonthBegin(1)
        if d == m_start and m_end <= d1:
            out.append(("monthly", d.strftime("%Y-%m")))
            d = m_end
        else:
            out.append(("daily", d.strftime("%Y-%m-%d")))
            d = d + pd.Timedelta(days=1)
    return out


class _ZipEntryStream(io.RawIOBase):
    """Inflate the first entry of a zip file read sequentially from ``response`` (local header + deflate data).

    Every raw byte read is fed to ``hasher`` so that the SHA-256 of the whole file can be checked at the end
    (:meth:`drain` reads the rest of the body: data descriptor and central directory).
    """

    def __init__(self, response, hasher, chunk: int = 1 << 20, piece: int = 1 << 23):
        super().__init__()
        self._r, self._h, self._chunk, self._piece = response, hasher, chunk, piece
        self._z = zlib.decompressobj(-zlib.MAX_WBITS)
        self._pending = b""
        self._out, self._off = b"", 0
        self.raw_bytes = 0
        sig, _, _, method, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", self._raw(30))
        if sig != 0x04034B50:
            raise IOError("not a zip file")
        if method != 8:
            raise IOError(f"unsupported zip compression method {method}")
        self._raw(nlen + xlen)

    def _read_raw(self) -> bytes:
        chunk = self._r.read(self._chunk)
        if chunk:
            self._h.update(chunk)
            self.raw_bytes += len(chunk)
        return chunk

    def _raw(self, n: int) -> bytes:
        while len(self._pending) < n:
            c = self._read_raw()
            if not c:
                raise IOError("truncated zip header")
            self._pending += c
        out, self._pending = self._pending[:n], self._pending[n:]
        return out

    def _next_piece(self) -> bool:
        while not self._z.eof:
            src = self._z.unconsumed_tail or self._pending or self._read_raw()
            self._pending = b""
            if not src:
                raise IOError("download ended before the end of the compressed data")
            self._out, self._off = self._z.decompress(src, self._piece), 0
            if self._out:
                return True
        return False

    def peek(self, n: int) -> bytes:
        while self._off >= len(self._out):
            if not self._next_piece():
                return b""
        return self._out[self._off:self._off + n]

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        mv, filled = memoryview(b), 0
        while filled < len(mv):
            if self._off >= len(self._out) and not self._next_piece():
                break
            take = min(len(mv) - filled, len(self._out) - self._off)
            mv[filled:filled + take] = self._out[self._off:self._off + take]
            self._off += take
            filled += take
        return filled

    def drain(self) -> None:
        while self._read_raw():
            pass


def stream_archive_batches(kind: str, symbol: str, period: str, interval: str | None = None, *,
                           columns: Sequence[str], types: Mapping[str, object], block_size: int = 64 << 20,
                           timeout: float = 120, verify_checksum: bool = True, base: str = ARCHIVE_URL,
                           stats: dict | None = None) -> Iterator:
    """Yield :class:`pyarrow.RecordBatch` es of one archive CSV while the zip downloads.

    columns  the file's column names in order; ``types``  ``{column: pyarrow type}`` of the columns to keep
    (the others are not converted).  A header line (newer files) is detected and skipped.
    Raises :class:`ArchiveFileMissing` for HTTP 404 and :class:`ChecksumError` *after the last batch* when the
    SHA-256 does not match (the caller must then discard what it built from this file).  ``stats`` (optional
    dict) receives ``raw_bytes`` and ``seconds``.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    url = archive_url(kind, symbol, period, interval, base=base)
    expected = None
    if verify_checksum:
        chk = _download(url + ".CHECKSUM", timeout)
        if chk is None:
            raise ArchiveFileMissing(url)
        expected = chk.decode().split()[0].strip().lower()
    t0 = time.time()
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ArchiveFileMissing(url) from None
        raise
    hasher = hashlib.sha256()
    with response:
        stream = _ZipEntryStream(response, hasher)
        first = stream.peek(1)
        skip = 0 if (not first or first.isdigit()) else 1
        reader = pacsv.open_csv(
            pa.PythonFile(stream, mode="r"),
            read_options=pacsv.ReadOptions(column_names=list(columns), skip_rows=skip, block_size=block_size),
            convert_options=pacsv.ConvertOptions(include_columns=list(types), column_types=dict(types)))
        for batch in reader:
            yield batch
        stream.drain()
    if stats is not None:
        stats.update(raw_bytes=stream.raw_bytes, seconds=time.time() - t0)
    if expected is not None and hasher.hexdigest() != expected:
        raise ChecksumError(f"checksum mismatch for {url}")


def fetch_aggtrades_archive(symbol: str, days: Sequence[str], *, workers: int = 4, timeout: float = 300,
                            verify_checksum: bool = True, base: str = ARCHIVE_URL) -> pd.DataFrame:
    """aggTrades of the given ``"YYYY-MM-DD"`` days (daily files); ``transact_time`` in milliseconds.

    Columns: :data:`AGGTRADE_COLUMNS` with numeric types (``is_buyer_maker`` as bool).
    """
    def one(day):
        return fetch_archive_csv("aggTrades", symbol, day, columns=AGGTRADE_COLUMNS, timeout=timeout,
                                 verify_checksum=verify_checksum, base=base)

    parts = [p for p in _parallel(one, list(days), workers) if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=AGGTRADE_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    for c in ["agg_trade_id", "first_trade_id", "last_trade_id"]:
        df[c] = df[c].astype(np.int64)
    df["price"], df["quantity"] = df["price"].astype(float), df["quantity"].astype(float)
    df["transact_time"] = normalize_epoch_ms(df["transact_time"].astype(np.int64))
    for c in ["is_buyer_maker", "is_best_match"]:
        df[c] = df[c].str.strip().str.lower().isin(["true", "1"])
    return df.sort_values(["transact_time", "agg_trade_id"]).reset_index(drop=True)

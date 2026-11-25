"""Binance: public data archive (data.binance.vision) + public REST for gap recovery and specs.

Verified on 2026-10-20 against github.com/binance/binance-public-data (README, python/utility.py)
and github.com/binance/binance-spot-api-docs (rest-api.md, web-socket-streams.md, CHANGELOG.md):

* Archive paths: data/spot/{daily|monthly}/{trades|aggTrades|klines}/SYMBOL/[interval/]FILE.zip,
  futures: data/futures/{um|cm}/... ; each zip has a sibling ``.CHECKSUM`` (sha256).
* Spot trades columns: trade Id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch.
  USD-M trades: id, price, qty, quoteQty, time, isBuyerMaker.  COIN-M: id, price, qty, baseQty, ...
* "The timestamp for SPOT Data from January 1st 2025 onwards will be in microseconds."  Futures
  archives are in milliseconds.  The parser checks the magnitude against this rule and raises on
  disagreement instead of guessing.
* "Archived files may be updated at a later date" -> raw files are versioned, never overwritten.
* REST ``GET /api/v3/historicalTrades`` (fromId, limit<=1000) no longer needs an API key
  (CHANGELOG: security type changed MARKET_DATA -> NONE).  Used only to recover ID gaps.
* isBuyerMaker=true -> the buyer was the maker -> the aggressor (taker) sold -> sign -1.
* Block trades (introduced 2026-05) have their own endpoints/stream (``@blockTrade``,
  ``historicalBlockTrades``) and ID space; whether they appear in the regular trade archive was NOT
  verified.  They are not merged into bars.
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .. import timeutil as tu
from ..http import Http, NotFound, SourceUnreachable
from ..instruments import InstrumentSpec
from ..numeric import scaled_int, safe_mul_int64, PrecisionError
from ..storage import RawStore, sha256_bytes

ARCHIVE_BASE = "https://data.binance.vision/"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
REST = {"spot": "https://api.binance.com", "um": "https://fapi.binance.com", "cm": "https://dapi.binance.com"}
SPOT_US_FROM = date(2025, 1, 1)
PROVIDER = "binance"


class DuplicateConflict(ValueError):
    """Same trade ID with different content: the archive is inconsistent; never pick one silently."""


def archive_rel(market: str, freq: str, dtype: str, symbol: str, when, interval: str | None = None) -> str:
    base = "data/spot" if market == "spot" else f"data/futures/{market}"
    sym = symbol.upper()
    if freq == "daily":
        tag = when.isoformat() if isinstance(when, date) else str(when)
    else:
        tag = when if isinstance(when, str) else f"{when.year:04d}-{when.month:02d}"
    if interval:
        return f"{base}/{freq}/{dtype}/{sym}/{interval}/{sym}-{interval}-{tag}.zip"
    return f"{base}/{freq}/{dtype}/{sym}/{sym}-{dtype}-{tag}.zip"


def documented_ts_unit(market: str, day: date) -> str:
    if market == "spot" and day >= SPOT_US_FROM:
        return "us"
    return "ms"


def _read_zip_csv(data: bytes) -> tuple[bytes, str]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one csv in zip, got {names}")
        return z.read(names[0]), names[0]


def _has_header(csv_bytes: bytes) -> bool:
    first = csv_bytes.split(b"\n", 1)[0].strip().split(b",")[0]
    return not first.lstrip(b"-").isdigit()


_HEADER_ALIASES = {
    "aggtradeid": "aggid", "quantity": "qty", "transacttime": "time", "count": "ntrades",
    "takerbuyvolume": "takerbuybase", "takerbuyquotevolume": "takerbuyquote", "quoteqty": "quoteqty",
    "isbuyermaker": "isbuyermaker", "tradeid": "id",
}


def _norm_header(s: str) -> str:
    k = s.strip().lower().replace("_", "").replace(" ", "")
    return _HEADER_ALIASES.get(k, k)


def _read_csv(csv_bytes: bytes, names: list[str], str_cols: list[str]) -> pa.Table:
    """Read a Binance archive CSV with exact string columns.  Header-less files use the documented
    column order; files with a header must match it (after alias normalisation) or we raise."""
    header = _has_header(csv_bytes)
    if header:
        hdr = csv_bytes.split(b"\n", 1)[0].decode().strip().split(",")
        if len(hdr) != len(names) or any(_norm_header(a) != _norm_header(b) for a, b in zip(hdr, names)):
            raise ValueError(f"unexpected archive header {hdr}; expected layout {names}")
    ro = pacsv.ReadOptions(column_names=names, skip_rows=1 if header else 0)
    co = pacsv.ConvertOptions(column_types={c: pa.string() for c in str_cols}, strings_can_be_null=False)
    return pacsv.read_csv(io.BytesIO(csv_bytes), read_options=ro, convert_options=co)


def _bool_col(col) -> np.ndarray:
    """True -> 1, False -> 0, anything else -> -1 (unknown; never coerced)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if pa.types.is_boolean(col.type):
        vals = np.asarray(col.to_numpy(zero_copy_only=False), dtype=object)
        out = np.full(len(col), -1, dtype=np.int8)
        valid = np.asarray(col.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
        out[valid & (vals == True)] = 1  # noqa: E712
        out[valid & (vals == False)] = 0  # noqa: E712
        return out
    s = np.asarray(pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(col, pa.string())))
                   .to_numpy(zero_copy_only=False), dtype=object)
    out = np.full(len(s), -1, dtype=np.int8)
    out[s == "true"] = 1
    out[s == "false"] = 0
    return out


def _ts_to_us(raw: np.ndarray, market: str, day: date) -> tuple[np.ndarray, str]:
    unit = documented_ts_unit(market, day)
    inferred = tu.infer_epoch_unit(raw)
    if inferred != "unknown" and inferred != unit:
        raise ValueError(f"timestamp unit mismatch for {market} {day}: documented {unit}, data looks {inferred}")
    return tu.normalize_exchange_ts(raw, unit), unit


def normalize_trades_table(t: pa.Table, spec: InstrumentSpec, market: str, day: date,
                           source_file: str, aggregated: bool = False, ts_unit: str | None = None
                           ) -> tuple[pd.DataFrame, dict]:
    """Columns are already named; returns the normalised frame and a quality record.
    ``ts_unit`` overrides the archive rule (e.g. live streams subscribed with timeUnit=MICROSECOND)."""
    q = {"file": source_file, "rows_raw": t.num_rows}
    price_i = scaled_int(t.column("price"), spec.price_decimals)
    qty_i = scaled_int(t.column("qty"), spec.qty_decimals)
    raw_ts = np.asarray(t.column("time").to_numpy(), dtype=np.int64)
    if ts_unit is None:
        ts_us, unit = _ts_to_us(raw_ts, market, day)
    else:
        inferred = tu.infer_epoch_unit(raw_ts)
        if inferred not in ("unknown", ts_unit):
            raise ValueError(f"declared timestamp unit {ts_unit} but data looks {inferred}")
        ts_us, unit = tu.normalize_exchange_ts(raw_ts, ts_unit), ts_unit
    q["ts_unit_original"] = unit
    ibm = _bool_col(t.column("is_buyer_maker"))
    ids = np.asarray(t.column("id").to_numpy(), dtype=np.int64)
    if spec.contract_type == "linear":
        notional_i = safe_mul_int64(price_i, qty_i)
    else:
        # inverse: qty = contracts; notional (contract currency) = contracts * contract_size
        cs = int(spec.contract_size)
        notional_i = qty_i * cs * spec.price_scale  # scale price_scale*qty_scale
    df = pd.DataFrame({
        "trade_id": ids, "ts_us": ts_us, "price_i": price_i, "qty_i": qty_i, "notional_i": notional_i,
        "aggressor": np.where(ibm == 1, -1, np.where(ibm == 0, 1, 0)).astype(np.int8),
    })
    if aggregated:
        df["first_trade_id"] = np.asarray(t.column("first_trade_id").to_numpy(), dtype=np.int64)
        df["last_trade_id"] = np.asarray(t.column("last_trade_id").to_numpy(), dtype=np.int64)
    if "quote_qty" in t.column_names and spec.contract_type == "linear":
        # Cross-check our exact price*qty against the venue's reported quote quantity.
        try:
            rq = scaled_int(t.column("quote_qty"), spec.price_decimals + spec.qty_decimals)
            q["quote_qty_mismatch"] = int(np.count_nonzero(rq != notional_i))
        except PrecisionError:
            q["quote_qty_mismatch"] = "not_comparable_precision"
    df, dq = dedupe_and_check(df, "trade_id")
    q.update(dq)
    return df, q


def dedupe_and_check(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, dict]:
    """Deduplicate on the real event identity; identical duplicates are dropped and counted,
    conflicting duplicates raise.  Sorts by the exchange sequence (trade id)."""
    q = {}
    df = df.sort_values(key, kind="stable").reset_index(drop=True)
    dup_mask = df.duplicated(key, keep=False)
    if dup_mask.any():
        dups = df[dup_mask]
        full = dups.duplicated(keep=False)
        if not full.all():
            bad = dups[~full][key].unique()[:5]
            raise DuplicateConflict(f"conflicting rows for {key} {list(bad)}")
        n_before = len(df)
        df = df.drop_duplicates(key, keep="first").reset_index(drop=True)
        q["duplicates_dropped"] = int(n_before - len(df))
    else:
        q["duplicates_dropped"] = 0
    ids = df[key].to_numpy()
    d = np.diff(ids)
    q["internal_id_gaps"] = int(np.count_nonzero(d > 1)) if len(ids) > 1 else 0
    q["internal_gap_ranges"] = [(int(ids[i]), int(ids[i + 1])) for i in np.flatnonzero(d > 1)[:100]]
    ts = df["ts_us"].to_numpy()
    q["ts_regressions"] = int(np.count_nonzero(np.diff(ts) < 0)) if len(ts) > 1 else 0
    q["rows"] = int(len(df))
    if len(df):
        q["first_id"], q["last_id"] = int(ids[0]), int(ids[-1])
        q["first_ts_us"], q["last_ts_us"] = int(ts.min()), int(ts.max())
    return df, q


TRADE_COLS = {
    "spot": ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"],
    "um": ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
    "cm": ["id", "price", "qty", "base_qty", "time", "is_buyer_maker"],
}
AGG_COLS = {
    "spot": ["agg_id", "price", "qty", "first_trade_id", "last_trade_id", "time", "is_buyer_maker", "is_best_match"],
    "um": ["agg_id", "price", "qty", "first_trade_id", "last_trade_id", "time", "is_buyer_maker"],
    "cm": ["agg_id", "price", "qty", "first_trade_id", "last_trade_id", "time", "is_buyer_maker"],
}
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def parse_trades_zip(data: bytes, spec: InstrumentSpec, day: date, source_file: str):
    market = spec.archive_market()
    csv_bytes, _ = _read_zip_csv(data)
    t = _read_csv(csv_bytes, TRADE_COLS[market], ["price", "qty", TRADE_COLS[market][3]])
    return normalize_trades_table(t, spec, market, day, source_file)


def parse_aggtrades_zip(data: bytes, spec: InstrumentSpec, day: date, source_file: str):
    market = spec.archive_market()
    csv_bytes, _ = _read_zip_csv(data)
    t = _read_csv(csv_bytes, AGG_COLS[market], ["price", "qty"])
    t = t.rename_columns(["id" if c == "agg_id" else c for c in t.column_names])
    return normalize_trades_table(t, spec, market, day, source_file, aggregated=True)


def parse_klines_zip(data: bytes, market: str, day: date, interval_us: int) -> pd.DataFrame:
    csv_bytes, _ = _read_zip_csv(data)
    t = _read_csv(csv_bytes, KLINE_COLS, ["open", "high", "low", "close", "volume", "quote_volume",
                                          "taker_buy_base", "taker_buy_quote"])
    return normalize_klines(t.to_pandas(), market, day, interval_us)


def normalize_klines(df: pd.DataFrame, market: str, day: date | None, interval_us: int) -> pd.DataFrame:
    """Klines are labelled by OPEN time.  We store period_start/period_end; the values describe the
    closed interval and become available at period_end (market-time replay)."""
    ot = np.asarray(df["open_time"], dtype=np.int64)
    if day is not None:
        ot_us, unit = _ts_to_us(ot, market, day)
    else:
        unit = tu.infer_epoch_unit(ot)
        ot_us = tu.normalize_exchange_ts(ot, unit)
    out = pd.DataFrame({
        "period_start_us": ot_us,
        "period_end_us": ot_us + interval_us,
        "open": pd.to_numeric(df["open"]).astype(float), "high": pd.to_numeric(df["high"]).astype(float),
        "low": pd.to_numeric(df["low"]).astype(float), "close": pd.to_numeric(df["close"]).astype(float),
        "volume": pd.to_numeric(df["volume"]).astype(float),
        "quote_volume": pd.to_numeric(df["quote_volume"]).astype(float),
        "n_trades": pd.to_numeric(df["n_trades"]).astype("Int64"),
        "taker_buy_base": pd.to_numeric(df["taker_buy_base"]).astype(float),
    })
    ct = np.asarray(df["close_time"], dtype=np.int64)
    ct_us = tu.normalize_exchange_ts(ct, unit)
    tick = 1000 if unit == "ms" else 1
    bad = ct_us + tick != out["period_end_us"].to_numpy()
    out["close_time_consistent"] = ~bad
    return out.sort_values("period_start_us").drop_duplicates("period_start_us").reset_index(drop=True)


class BinanceArchive:
    def __init__(self, http: Http, raw: RawStore):
        self.http = http
        self.raw = raw

    def fetch(self, rel: str) -> tuple[bytes, dict]:
        """Download (or reuse) an archive file, verifying the published sha256 CHECKSUM."""
        latest = self.raw.latest(PROVIDER, rel)
        if latest is not None:
            meta = self.raw.meta(latest)
            if meta.get("checksum_verified"):
                return latest.read_bytes(), meta
        url = ARCHIVE_BASE + rel
        chk = None
        try:
            r = self.http.get(url + ".CHECKSUM", PROVIDER)
            chk = r.content.decode().split()[0].strip().lower()
        except NotFound:
            chk = None
        r = self.http.get(url, PROVIDER)
        digest = sha256_bytes(r.content)
        verified = (chk == digest) if chk else None
        if chk and not verified:
            raise ValueError(f"CHECKSUM mismatch for {rel}: published {chk}, got {digest}")
        meta = {"url": url, "checksum_published": chk, "checksum_verified": bool(verified),
                "http_last_modified": r.headers.get("Last-Modified"), "http_etag": r.headers.get("ETag")}
        path, status = self.raw.put(PROVIDER, rel, r.content, meta)
        meta["store_status"] = status
        return r.content, meta

    def list_prefix(self, prefix: str) -> list[dict]:
        """List archive objects via the public S3 bucket index (same listing data.binance.vision uses)."""
        out, marker = [], None
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        while True:
            params = {"prefix": prefix, "delimiter": "/"}
            if marker:
                params["marker"] = marker
            r = self.http.get(S3_LIST, PROVIDER, params=params)
            root = ET.fromstring(r.content)
            for c in root.findall("s3:Contents", ns):
                out.append({"key": c.find("s3:Key", ns).text, "size": int(c.find("s3:Size", ns).text),
                            "last_modified": c.find("s3:LastModified", ns).text})
            trunc = root.find("s3:IsTruncated", ns)
            if trunc is None or trunc.text != "true" or not out:
                break
            marker = out[-1]["key"]
        return out


class BinanceRest:
    def __init__(self, http: Http, raw: RawStore):
        self.http = http
        self.raw = raw

    def historical_trades(self, spec: InstrumentSpec, from_id: int, to_id: int) -> pd.DataFrame:
        """Recover trades [from_id, to_id] (inclusive) for gap repair; raw pages are stored."""
        market = spec.archive_market()
        path = {"spot": "/api/v3/historicalTrades", "um": "/fapi/v1/historicalTrades",
                "cm": "/dapi/v1/historicalTrades"}[market]
        rows, cur = [], from_id
        while cur <= to_id:
            r = self.http.get(REST[market] + path, PROVIDER, params={"symbol": spec.symbol, "fromId": cur, "limit": 1000})
            self.raw.put(PROVIDER, f"rest/historicalTrades/{spec.symbol}/{cur}.json", r.content,
                         {"url": r.url, "received_at_us": tu.now_us()})
            page = __import__("json").loads(r.content)
            if not page:
                break
            rows.extend(page)
            cur = page[-1]["id"] + 1
        rows = [x for x in rows if from_id <= x["id"] <= to_id]
        if not rows:
            return pd.DataFrame()
        t = pa.table({
            "id": pa.array([x["id"] for x in rows], pa.int64()),
            "price": pa.array([x["price"] for x in rows]), "qty": pa.array([x["qty"] for x in rows]),
            "quote_qty": pa.array([x.get("quoteQty", "0") for x in rows]),
            "time": pa.array([x["time"] for x in rows], pa.int64()),
            "is_buyer_maker": pa.array([x["isBuyerMaker"] for x in rows]),
        })
        day = tu.us_to_datetime_index([t.column("time")[0].as_py() * 1000])[0].date()
        df, _ = normalize_trades_table(t, spec, market, day, "rest:historicalTrades")
        return df

    def depth_snapshot(self, spec: InstrumentSpec, limit: int = 5000) -> tuple[dict, int]:
        market = spec.archive_market()
        path = {"spot": "/api/v3/depth", "um": "/fapi/v1/depth", "cm": "/dapi/v1/depth"}[market]
        if market != "spot":
            limit = min(limit, 1000)
        r = self.http.get(REST[market] + path, PROVIDER, params={"symbol": spec.symbol, "limit": limit})
        return __import__("json").loads(r.content), tu.now_us()

    def funding_rate_history(self, symbol: str, start_us: int, end_us: int) -> pd.DataFrame:
        """GET /fapi/v1/fundingRate (ascending; max 1000 per call)."""
        rows, cur = [], start_us // 1000
        while cur < end_us // 1000:
            r = self.http.get(REST["um"] + "/fapi/v1/fundingRate", PROVIDER,
                              params={"symbol": symbol, "startTime": cur, "endTime": end_us // 1000 - 1, "limit": 1000})
            self.raw.put(PROVIDER, f"rest/fundingRate/{symbol}/{cur}.json", r.content, {"url": r.url})
            page = __import__("json").loads(r.content)
            if not page:
                break
            rows.extend(page)
            nxt = int(page[-1]["fundingTime"]) + 1
            if nxt <= cur:
                break
            cur = nxt
        return normalize_funding(rows)

    def klines(self, market: str, symbol: str, interval: str, start_us: int, end_us: int,
               kind: str = "klines") -> pd.DataFrame:
        """kind in klines | premiumIndexKlines | markPriceKlines | indexPriceKlines (futures)."""
        path = {"spot": "/api/v3/klines", "um": f"/fapi/v1/{kind}", "cm": f"/dapi/v1/{kind}"}[market]
        key = "pair" if kind == "indexPriceKlines" else "symbol"
        iv = tu.parse_duration(interval.replace("m", "min") if interval.endswith("m") else interval)
        rows, cur = [], start_us // 1000
        while cur < end_us // 1000:
            r = self.http.get(REST[market] + path, PROVIDER, params={key: symbol, "interval": interval, "startTime": cur,
                                                                     "endTime": end_us // 1000 - 1, "limit": 1000})
            page = __import__("json").loads(r.content)
            if not page:
                break
            rows.extend(page)
            cur = int(page[-1][0]) + iv // 1000
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=KLINE_COLS[: len(rows[0])])
        return normalize_klines(df, market, None, iv)


def normalize_funding(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["funding_time_us", "funding_rate", "mark_price"])
    df = pd.DataFrame({
        "funding_time_us": [int(r["fundingTime"]) * 1000 for r in rows],
        "funding_rate": [float(r["fundingRate"]) for r in rows],
        "mark_price": [float(r["markPrice"]) if r.get("markPrice") not in (None, "") else np.nan for r in rows],
    })
    return df.drop_duplicates("funding_time_us").sort_values("funding_time_us").reset_index(drop=True)


def parse_funding_archive(data: bytes) -> pd.DataFrame:
    """Monthly USD-M ``fundingRate`` archive.  Column names are read from the file header (the
    archive README does not document this file); unknown layouts raise."""
    csv_bytes, _ = _read_zip_csv(data)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("calc_time") or cols.get("funding_time") or cols.get("fundingtime")
    rcol = cols.get("last_funding_rate") or cols.get("funding_rate") or cols.get("fundingrate")
    if not tcol or not rcol:
        raise ValueError(f"unrecognised fundingRate archive header {list(df.columns)}")
    t = np.asarray(df[tcol], dtype=np.int64)
    out = pd.DataFrame({"funding_time_us": tu.normalize_exchange_ts(t, tu.infer_epoch_unit(t)),
                        "funding_rate": df[rcol].astype(float), "mark_price": np.nan})
    ih = cols.get("funding_interval_hours")
    if ih:
        out["funding_interval_hours_reported"] = df[ih].astype(float)
    return out.sort_values("funding_time_us").reset_index(drop=True)


def parse_book_ticker_archive(data: bytes) -> pd.DataFrame:
    """USD-M ``bookTicker`` archive (best bid/ask updates).  Layout read from the header; the README
    does not document it, so unknown headers raise rather than being guessed."""
    csv_bytes, _ = _read_zip_csv(data)
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str)
    need = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time", "event_time"]
    if not all(c in df.columns for c in need):
        raise ValueError(f"unrecognised bookTicker header {list(df.columns)}")
    tt = df["transaction_time"].astype(np.int64).to_numpy()
    et = df["event_time"].astype(np.int64).to_numpy()
    return pd.DataFrame({
        "update_id": df["update_id"].astype(np.int64), "ts_us": tu.normalize_exchange_ts(tt, tu.infer_epoch_unit(tt)),
        "event_ts_us": tu.normalize_exchange_ts(et, tu.infer_epoch_unit(et)),
        "bid": df["best_bid_price"].astype(float), "bid_qty": df["best_bid_qty"].astype(float),
        "ask": df["best_ask_price"].astype(float), "ask_qty": df["best_ask_qty"].astype(float),
    }).sort_values(["ts_us", "update_id"]).reset_index(drop=True)

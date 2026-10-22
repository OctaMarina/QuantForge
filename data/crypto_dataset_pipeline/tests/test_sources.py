"""Provider adapters against documented payload layouts (simulated HTTP; the real endpoints are not
reachable from the build environment - see docs/STATUS.md)."""
import hashlib
import io
import json
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from cryptods.http import Response, NotFound
from cryptods.sources import binance as bn
from cryptods.sources.coinbase import Coinbase
from cryptods.sources.deribit import Deribit
from cryptods.storage import RawStore


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, provider, params=None, headers=None, **kw):
        self.calls.append((url, params))
        for key, fn in self.routes.items():
            if key in url:
                body = fn(url, params) if callable(fn) else fn
                if body is None:
                    raise NotFound(url)
                return Response(url, 200, body, {"Last-Modified": "x"})
        raise NotFound(url)

    def get_json(self, url, provider, params=None):
        return json.loads(self.get(url, provider, params).content)


def zipped(name, text):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, text)
    return b.getvalue()


SPOT_CSV = "51175358,17.80180000,5.69000000,101.29224200,1735689600010866,True,True\n" \
           "51175359,17.80190000,1.00000000,17.80190000,1735689600020000,False,True\n"


def test_archive_fetch_checksum_cache_and_revision(tmp_path, spec):
    from cryptods.instruments import InstrumentSpec
    sp = InstrumentSpec("binance", "spot", "XUSDT", "X", "USDT", price_decimals=8, qty_decimals=8)
    data = zipped("XUSDT-trades-2025-01-01.csv", SPOT_CSV)
    good = hashlib.sha256(data).hexdigest()
    rel = bn.archive_rel("spot", "daily", "trades", "XUSDT", date(2025, 1, 1))
    assert rel == "data/spot/daily/trades/XUSDT/XUSDT-trades-2025-01-01.zip"
    http = FakeHttp({".CHECKSUM": f"{good}  XUSDT-trades-2025-01-01.zip\n".encode(), ".zip": data})
    raw = RawStore(tmp_path)
    a = bn.BinanceArchive(http, raw)
    content, meta = a.fetch(rel)
    assert meta["checksum_verified"] is True
    df, q = bn.parse_trades_zip(content, sp, date(2025, 1, 1), rel)
    assert q["ts_unit_original"] == "us" and df.ts_us.iloc[0] == 1735689600010866   # microseconds from 2025
    assert list(df.aggressor) == [-1, 1] and q["quote_qty_mismatch"] == 0
    n_calls = len(http.calls)
    a.fetch(rel)
    assert len(http.calls) == n_calls          # cached, verified -> no network
    # a revised archive is stored as a new version, never overwriting the original
    p, status = raw.put("binance", rel, data + b"x", {"url": "u"})
    assert status == "revision" and p.name.endswith(".v2") and (tmp_path / "raw" / "binance" / "_revisions.jsonl").exists()
    bad = FakeHttp({".CHECKSUM": b"0" * 64 + b"  f\n", ".zip": data})
    with pytest.raises(ValueError):
        bn.BinanceArchive(bad, RawStore(tmp_path / "b")).fetch(rel)


def test_futures_header_and_klines(tmp_path):
    from cryptods.instruments import InstrumentSpec
    sp = InstrumentSpec("binance", "perpetual", "BTCUSDT", "BTC", "USDT", price_decimals=2, qty_decimals=4)
    csv = "id,price,qty,quote_qty,time,is_buyer_maker\n28457,4.00,12.0000,48.00,1499865549590,true\n"
    df, q = bn.parse_trades_zip(zipped("a.csv", csv), sp, date(2017, 7, 12), "f")
    assert df.ts_us.iloc[0] == 1499865549590000 and df.aggressor.iloc[0] == -1
    with pytest.raises(ValueError):   # unexpected header layout is rejected, not guessed
        bn.parse_trades_zip(zipped("a.csv", "x,price,qty,quote_qty,time,is_buyer_maker\n1,1,1,1,1499865549590,true\n"),
                            sp, date(2017, 7, 12), "f")
    kl = ("1499040000000,0.01634790,0.80000000,0.01575800,0.01577100,148976.11427815,1499040059999,2434.19055334,308,"
          "1756.87402397,28.46694368,17928899.62484339\n")
    k = bn.parse_klines_zip(zipped("k.csv", kl), "um", date(2017, 7, 3), 60_000_000)
    assert k.period_end_us.iloc[0] - k.period_start_us.iloc[0] == 60_000_000 and k.close_time_consistent.all()


def test_s3_listing_pagination(tmp_path):
    ns = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'
    page1 = (f'<ListBucketResult {ns}><IsTruncated>true</IsTruncated>'
             '<Contents><Key>p/BTCUSDT-trades-2024-05-01.zip</Key><Size>10</Size><LastModified>t</LastModified></Contents>'
             '</ListBucketResult>').encode()
    page2 = (f'<ListBucketResult {ns}><IsTruncated>false</IsTruncated>'
             '<Contents><Key>p/BTCUSDT-trades-2024-05-02.zip</Key><Size>20</Size><LastModified>t</LastModified></Contents>'
             '</ListBucketResult>').encode()
    http = FakeHttp({"s3-ap-northeast-1": lambda u, p: page2 if p.get("marker") else page1})
    objs = bn.BinanceArchive(http, RawStore(tmp_path)).list_prefix("p/")
    assert [o["size"] for o in objs] == [10, 20]


def test_coinbase_pagination_and_missing_minutes(tmp_path):
    def route(url, params):
        s = pd.Timestamp(params["start"]).value // 1000
        e = pd.Timestamp(params["end"]).value // 1000
        rows = []
        t = s
        while t <= e:                                  # Coinbase 'end' is inclusive
            if (t // 60_000_000) % 7 != 3:             # some minutes have no trades -> no candle
                rows.append([t // 1_000_000, 1, 2, 1.5, 1.6, 10])
            t += 60_000_000
        return json.dumps(rows[::-1]).encode()         # newest first, as the API returns
    cb = Coinbase(FakeHttp({"/candles": route}), RawStore(tmp_path))
    s = 1_700_000_040_000_000 - (1_700_000_040_000_000 % 60_000_000)
    df = cb.candles("BTC-USD", s, s + 700 * 60_000_000)
    assert df.period_start_us.is_monotonic_increasing and df.period_start_us.is_unique
    assert len(df) == sum(1 for k in range(700) if ((s // 60_000_000) + k) % 7 != 3)
    assert ((df.period_end_us - df.period_start_us) == 60_000_000).all()


def test_deribit_dvol_continuation(tmp_path):
    t0 = 1_700_000_000_000

    def route(url, params):
        end = params["end_timestamp"]
        if end > t0 + 120_000:
            return json.dumps({"result": {"data": [[t0 + 120_000, 1, 2, 0.5, 55.0], [t0 + 180_000, 1, 2, 0.5, 56.0]],
                                          "continuation": t0 + 119_999}}).encode()
        return json.dumps({"result": {"data": [[t0, 1, 2, 0.5, 53.0], [t0 + 60_000, 1, 2, 0.5, 54.0]],
                                      "continuation": None}}).encode()
    d = Deribit(FakeHttp({"get_volatility_index_data": route}), RawStore(tmp_path))
    df = d.dvol("BTC", t0 * 1000, (t0 + 240_000) * 1000, "60")
    assert list(df.close) == [53.0, 54.0, 55.0, 56.0]


def test_wayback_captures_as_vintages(tmp_path):
    """Two archived captures of the Farside page: the second revises 05 Oct.  Captures become vintages
    with available_at = capture time (public no later than then)."""
    from test_external import FARSIDE_HTML
    from cryptods.sources.publications import Publications, farside_daily, vintages_from_snapshots
    rev = FARSIDE_HTML.replace("<td>980.4</td></tr>\n<tr><td>Total</td>", "<td>1000.0</td></tr>\n<tr><td>Total</td>")
    cdx = [["timestamp", "original", "digest"], ["20261006020000", "https://farside.co.uk/btc/", "A"],
           ["20261007010000", "https://farside.co.uk/btc/", "B"]]

    def route_web(url, params):
        if "cdx/search" in url:
            return json.dumps(cdx).encode()
        return (FARSIDE_HTML if "20261006020000" in url else rev).encode()
    pubs = Publications(FakeHttp({"web.archive.org": route_web}), RawStore(tmp_path))
    wv = pubs.wayback_vintages("farside", "BTC", 1_700_000_000_000_000, 1_800_000_000_000_000)
    frames = [(farside_daily(g), int(c)) for c, g in wv.groupby("available_at_us")]
    v = vintages_from_snapshots(frames, ["total_musd", "sum_funds_musd", "n_funds_reported"], ["economic_date"])
    oct5 = v[v.economic_date == "2026-10-05"].sort_values("version")
    assert list(oct5.total_musd) == [980.4, 1000.0] and list(oct5.version) == [1, 2]
    t1 = pd.Timestamp("2026-10-06 02:00", tz="UTC").value // 1000
    assert oct5.available_at_us.iloc[0] == t1

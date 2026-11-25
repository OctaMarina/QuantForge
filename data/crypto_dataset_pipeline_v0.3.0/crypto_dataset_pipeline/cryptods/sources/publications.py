"""External daily publications: Alternative.me Fear & Greed, Farside spot-ETF flows, and Wayback
Machine captures used as *evidence of first public availability* (vintages).

Point-in-time rules (strict mode):
* A historical table downloaded today proves nothing about when each value was first public, nor
  about revisions.  Such rows get ``available_at = null`` with reason ``unknown_publication_time``.
* Evidence that *does* establish availability: (a) our own live poll logs (received_at), and
  (b) third-party archive captures (Wayback ``timestamp``): a value visible in a capture at time T was
  public no later than T.  Each capture is a vintage; differences between captures are revisions.

Verified 2026-10-20:
* Alternative.me: GET https://api.alternative.me/fng/?limit=0 -> data[{value, value_classification,
  timestamp (unix s, daily), time_until_update (latest only)}].  The page gives no publication hour
  and no revision policy.  Index is Bitcoin-centric (volatility, momentum/volume, social, dominance,
  trends; surveys paused) - market context, not coin-specific sentiment.
* Farside (farside.co.uk/btc/, /eth/): title "Bitcoin ETF Flow (US$m)"; one column per fund + Total;
  negatives in parentheses; rows "DD Mon YYYY" for US trading sessions; summary rows Total/Average/
  Maximum/Minimum at the bottom and a Fee row at the top must be excluded.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lxml import html as lxml_html

from .. import timeutil as tu
from ..http import Http, NotFound
from ..storage import RawStore

FNG_URL = "https://api.alternative.me/fng/"
FARSIDE = {"BTC": "https://farside.co.uk/btc/", "ETH": "https://farside.co.uk/eth/"}
CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK = "https://web.archive.org/web/{ts}id_/{url}"
SUMMARY_ROWS = {"total", "average", "maximum", "minimum", "fee", "fees", ""}


# ------------------------------------------------------------------ Fear & Greed
def parse_fng(data: bytes) -> pd.DataFrame:
    js = json.loads(data)
    if js.get("metadata", {}).get("error"):
        raise RuntimeError(f"alternative.me error: {js['metadata']['error']}")
    rows = []
    for r in js.get("data", []):
        ts = int(r["timestamp"]) * tu.SEC
        rows.append({"economic_date": tu.to_iso(ts)[:10], "value": float(r["value"]),
                     "classification": r.get("value_classification"), "api_timestamp_us": ts,
                     "time_until_update_s": float(r["time_until_update"]) if r.get("time_until_update") else np.nan})
    return pd.DataFrame(rows).drop_duplicates("economic_date").sort_values("economic_date").reset_index(drop=True)


# ------------------------------------------------------------------ Farside
_NUM = re.compile(r"^\(?-?[\d,]*\.?\d+\)?$")


def _cell(text: str):
    t = (text or "").strip().replace("\xa0", "")
    if t in ("", "-", "–", "—", "n/a", "N/A"):
        return np.nan, t
    if not _NUM.match(t):
        return np.nan, t
    neg = t.startswith("(") and t.endswith(")")
    v = float(t.strip("()").replace(",", ""))
    return (-v if neg else v), t


def parse_farside(page: bytes, asset: str) -> pd.DataFrame:
    """Return long table: economic_date, column (fund ticker or 'Total'), value_musd, raw_cell."""
    doc = lxml_html.fromstring(page)
    best = None
    for tbl in doc.xpath("//table"):
        rows = tbl.xpath(".//tr")
        dated = sum(1 for r in rows if r.xpath("./td") and _parse_date(r.xpath("./td")[0].text_content()))
        if best is None or dated > best[0]:
            best = (dated, tbl)
    if best is None or best[0] == 0:
        raise ValueError("no dated flow table found on Farside page")
    tbl = best[1]
    header = None
    out = []
    for r in tbl.xpath(".//tr"):
        cells = [c.text_content().strip() for c in r.xpath("./th|./td")]
        if not cells:
            continue
        d = _parse_date(cells[0])
        if d is None:
            if header is None and any(c.upper() == "TOTAL" for c in cells):
                header = cells
            continue
        if header is None:
            raise ValueError("Farside table header with 'Total' not found before data rows")
        if len(cells) != len(header):
            raise ValueError(f"row width {len(cells)} != header width {len(header)} for {cells[0]}")
        for name, txt in zip(header[1:], cells[1:]):
            v, raw = _cell(txt)
            out.append({"asset": asset, "economic_date": d, "column": name.strip() or "?", "value_musd": v, "raw_cell": raw})
    df = pd.DataFrame(out)
    return df


def _parse_date(text: str):
    t = (text or "").strip()
    if t.lower() in SUMMARY_ROWS:
        return None
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def farside_daily(long: pd.DataFrame) -> pd.DataFrame:
    """Per economic date: farside_total (as published), sum of reported fund cells, counts."""
    if long is None or len(long) == 0:
        return pd.DataFrame(columns=["economic_date", "total_musd", "sum_funds_musd", "n_funds_reported", "n_fund_columns"])
    g = []
    for d, grp in long.groupby("economic_date"):
        tot = grp[grp["column"].str.upper() == "TOTAL"]["value_musd"]
        funds = grp[grp["column"].str.upper() != "TOTAL"]
        g.append({"economic_date": d, "total_musd": float(tot.iloc[0]) if len(tot) else np.nan,
                  "sum_funds_musd": float(funds["value_musd"].sum(min_count=1)),
                  "n_funds_reported": int(funds["value_musd"].notna().sum()),
                  "n_fund_columns": int(len(funds))})
    return pd.DataFrame(g).sort_values("economic_date").reset_index(drop=True)


# ------------------------------------------------------------------ Wayback evidence
def wayback_captures(http: Http, url: str, start_us: int, end_us: int) -> list[dict]:
    params = {"url": url, "output": "json", "from": tu.to_iso(start_us)[:10].replace("-", ""),
              "to": tu.to_iso(end_us)[:10].replace("-", ""), "filter": "statuscode:200",
              "fl": "timestamp,original,digest"}
    js = http.get_json(CDX, "wayback", params=params)
    if not js:
        return []
    head, rows = js[0], js[1:]
    out = []
    for r in rows:
        rec = dict(zip(head, r))
        ts = rec["timestamp"]
        dt = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        rec["capture_us"] = tu.parse_time(dt)
        out.append(rec)
    return out


class Publications:
    def __init__(self, http: Http, raw: RawStore):
        self.http = http
        self.raw = raw

    def fetch_fng(self) -> tuple[pd.DataFrame, int]:
        r = self.http.get(FNG_URL, "alternative_me", params={"limit": 0, "format": "json"})
        rec = tu.now_us()
        self.raw.put("alternative_me", f"fng/snapshot_{rec}.json", r.content, {"url": r.url, "received_at_us": rec})
        return parse_fng(r.content), rec

    def fetch_farside(self, asset: str) -> tuple[pd.DataFrame, int]:
        r = self.http.get(FARSIDE[asset], "farside")
        rec = tu.now_us()
        self.raw.put("farside", f"{asset.lower()}/snapshot_{rec}.html", r.content, {"url": r.url, "received_at_us": rec})
        return parse_farside(r.content, asset), rec

    def wayback_vintages(self, kind: str, asset: str, start_us: int, end_us: int, max_captures: int = 400) -> pd.DataFrame:
        """Fetch archive captures and parse each as a vintage (available_at = capture time)."""
        url = FARSIDE[asset] if kind == "farside" else FNG_URL + "?limit=0"
        caps = wayback_captures(self.http, url, start_us, end_us)
        seen_digest = set()
        frames = []
        for c in caps[:max_captures]:
            if c["digest"] in seen_digest:
                continue
            seen_digest.add(c["digest"])
            try:
                r = self.http.get(WAYBACK.format(ts=c["timestamp"], url=c["original"]), "wayback")
            except NotFound:
                continue
            self.raw.put("wayback", f"{kind}/{asset}/{c['timestamp']}", r.content,
                         {"url": r.url, "capture_us": c["capture_us"], "digest": c["digest"]})
            try:
                df = parse_farside(r.content, asset) if kind == "farside" else parse_fng(r.content)
            except Exception:
                continue
            df["available_at_us"] = c["capture_us"]
            df["evidence"] = "wayback_capture"
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def vintages_from_snapshots(frames: list[tuple[pd.DataFrame, int]], value_cols: list[str], key_cols: list[str]) -> pd.DataFrame:
    """Turn successive snapshots (df, received_at) into first-seen vintages: a row (key, value) gets
    available_at = received_at of the first snapshot that shows exactly that value.  Later changes of
    the same key become new versions (revisions)."""
    seen: dict = {}
    out = []
    for df, rec in sorted(frames, key=lambda x: x[1]):
        for r in df.itertuples(index=False):
            key = tuple(getattr(r, k) for k in key_cols)
            val = tuple(getattr(r, v) for v in value_cols)
            prev = seen.get(key)
            if prev is None or not _same(prev, val):
                seen[key] = val
                row = dict(zip(key_cols, key))
                row.update(dict(zip(value_cols, val)))
                row["available_at_us"] = rec
                row["version"] = 1 if prev is None else None
                out.append(row)
    res = pd.DataFrame(out)
    if len(res):
        res["version"] = res.groupby(key_cols).cumcount() + 1
    return res


def _same(a, b):
    for x, y in zip(a, b):
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False
    return True

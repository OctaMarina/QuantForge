"""Orchestration of the CLI modes: dry-run, download, build (from cache), update, live, replay-live."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, config as cfgmod, instruments, timeutil as tu
from .bars import BarBuilder, bars_to_float, GapPolicyHalt, S_NEXT_BUCKET
from .coverage import feature_month_coverage, summarize, source_coverage
from . import status as status_mod
from .features import basic, flow, structural, crossmarket, external, options, liquidations, book_features, quotes
from .features.context import Ctx
from .http import Http, NotFound, SourceUnreachable
from .nulls import compress_reasons, VALID, CODE, Col
from .registry import to_markdown
from .sources import binance as bn
from .sources.coinbase import Coinbase
from .sources.deribit import Deribit
from .sources.publications import Publications, farside_daily
from .sources.tardis import Tardis
from .storage import RawStore, atomic_write_json, atomic_write_text, read_json, read_parquet, write_parquet

DOCS_VERIFIED_ON = "2026-10-20"
FAMILIES = [basic, flow, structural, crossmarket, external, options, liquidations, book_features, quotes]


class Pipeline:
    def __init__(self, cfg, offline: bool = False):
        self.cfg = cfg
        self.offline = offline
        self.root = Path(cfg["storage"]["root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = instruments.from_config(cfg)
        st = cfg["storage"]
        self.http = Http(st["rate_limit_per_s"], st["request_timeout_s"], st["retries"], st["backoff_s"])
        self.raw = RawStore(self.root)
        self.archive = bn.BinanceArchive(self.http, self.raw)
        self.rest = bn.BinanceRest(self.http, self.raw)
        self.coinbase = Coinbase(self.http, self.raw)
        self.deribit = Deribit(self.http, self.raw)
        self.pubs = Publications(self.http, self.raw)
        self.tardis = Tardis(self.http, self.raw, cfg["api_keys"]["tardis_env"])
        self.start = tu.parse_time(cfg["period"]["start"])
        self.end = tu.parse_time(cfg["period"]["end"])
        self.origin = tu.parse_time(cfg["period"]["bar_origin"])
        wi = int(cfg["warmup"]["intraday_days"])
        self.trade_start = min(self.origin, (self.start // tu.DAY - wi) * tu.DAY)
        self.daily_start = (self.start // tu.DAY - int(cfg["warmup"]["daily_days"])) * tu.DAY
        self.source_status: dict = {}
        self.inputs: list = []
        self.tables: dict = {}

    # ------------------------------------------------------------------ helpers
    def _status(self, key, ok, detail):
        self.source_status[key] = {"ok": bool(ok), "detail": str(detail)[:400]}

    def days(self, start_us, end_us):
        return tu.days_between(start_us, end_us)

    def _fetch_archive(self, rel, allow_fetch):
        p = self.raw.latest("binance", rel)
        if p is not None:
            meta = self.raw.meta(p)
            self.inputs.append({"source": "binance_archive", "relpath": rel, "sha256": meta.get("sha256"),
                                "checksum_verified": meta.get("checksum_verified"), "version": meta.get("version")})
            return p.read_bytes()
        if not allow_fetch or self.offline:
            return None
        data, meta = self.archive.fetch(rel)
        self.inputs.append({"source": "binance_archive", "relpath": rel, "sha256": meta.get("sha256") or hashlib.sha256(data).hexdigest(),
                            "checksum_verified": meta.get("checksum_verified")})
        return data

    # ------------------------------------------------------------------ dry run
    def dry_run(self) -> dict:
        rep = {"generated_at": tu.to_iso(tu.now_us()), "docs_verified_on": DOCS_VERIFIED_ON,
               "instrument": self.spec.to_dict(), "trade_days": [str(self.days(self.trade_start, self.end)[0]),
                                                                 str(self.days(self.trade_start, self.end)[-1])],
               "probes": [], "archive": {}, "blockers": [], "families": {}}
        probes = [
            ("binance_archive", "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-01.zip.CHECKSUM"),
            ("binance_rest", "https://api.binance.com/api/v3/time"),
            ("binance_futures_rest", "https://fapi.binance.com/fapi/v1/time"),
            ("coinbase", "https://api.exchange.coinbase.com/time"),
            ("deribit", "https://www.deribit.com/api/v2/public/get_time"),
            ("alternative_me", "https://api.alternative.me/fng/?limit=1"),
            ("farside", "https://farside.co.uk/btc/"),
            ("wayback", "https://web.archive.org/cdx/search/cdx?url=farside.co.uk/btc/&limit=1"),
            ("tardis", "https://datasets.tardis.dev/v1/binance/trades/2024/01/01/BTCUSDT.csv.gz"),
            ("bybit", "https://api.bybit.com/v5/market/time"),
        ]
        for name, url in probes:
            r = self.http.probe(url, name)
            r["provider"] = name
            rep["probes"].append(r)
            if not r["reachable"] or (r["status"] and r["status"] >= 400 and name not in ("tardis",)):
                rep["blockers"].append(f"{name}: {r.get('error') or r.get('status')}")
        reach = {p["provider"]: p["reachable"] for p in rep["probes"]}
        market = self.spec.archive_market()
        dtype = self.cfg["bars"]["trade_source"]
        if reach.get("binance_archive"):
            base = "data/spot" if market == "spot" else f"data/futures/{market}"
            prefix = f"{base}/daily/{dtype}/{self.spec.symbol}/"
            try:
                objs = [o for o in self.archive.list_prefix(prefix) if o["key"].endswith(".zip")]
                want = {d.isoformat() for d in self.days(self.trade_start, self.end)}
                have = {o["key"].rsplit("-", 3)[-3] + "-" + o["key"].rsplit("-", 3)[-2] + "-" + o["key"].rsplit("-", 3)[-1][:2]: o for o in objs}
                sel = [have[d] for d in sorted(want) if d in have]
                rep["archive"] = {"prefix": prefix, "days_requested": len(want), "days_available": len(sel),
                                  "missing_days": sorted(want - set(have))[:200],
                                  "bytes_compressed": int(sum(o["size"] for o in sel)),
                                  "first_listed": min(have) if have else None, "last_listed": max(have) if have else None}
                gb = rep["archive"]["bytes_compressed"] / 1e9
                if gb > float(self.cfg["storage"]["max_download_gb"]):
                    rep["blockers"].append(f"trade archive {gb:.1f} GB exceeds storage.max_download_gb")
            except Exception as e:
                rep["archive"] = {"error": repr(e)[:300]}
        else:
            rep["archive"] = {"error": "binance archive not reachable from this environment"}
        rep["families"] = family_expectations(self.cfg, reach)
        out = self.root / "reports" / f"dry_run_{tu.now_us()}.json"
        atomic_write_json(out, rep)
        rep["report_path"] = str(out)
        return rep

    # ------------------------------------------------------------------ downloads
    def download(self, include_trades=True) -> dict:
        res = {}
        spec = self.spec
        market = spec.archive_market()
        dtype = self.cfg["bars"]["trade_source"]
        if include_trades:
            n_ok = n_missing = 0
            for d in self.days(self.trade_start, self.end):
                rel = bn.archive_rel(market, "daily", dtype, spec.symbol, d)
                try:
                    if self._fetch_archive(rel, True) is not None:
                        n_ok += 1
                except NotFound:
                    n_missing += 1
                except SourceUnreachable as e:
                    self._status("binance_archive_trades", False, e)
                    break
            res["trades_days_ok"], res["trades_days_missing"] = n_ok, n_missing
        for key, fn in self._aux_jobs():
            try:
                fn(True)
                self._status(key, True, "job finished (rows actually obtained: see aux_tables / sub-statuses)")
            except SourceUnreachable as e:
                self._status(key, False, f"source_unreachable: {e}")
            except Exception as e:  # keep going: one family failing must not stop the others
                self._status(key, False, f"{type(e).__name__}: {e}")
        res["sources"] = self.source_status
        res["aux_tables"] = table_inventory(self.tables)
        atomic_write_json(self.root / "reports" / f"download_{tu.now_us()}.json", res)
        return res

    # ------------------------------------------------------------------ aux tables
    def _aux_jobs(self):
        return [("klines", self._load_klines), ("funding", self._load_funding), ("coinbase", self._load_coinbase),
                ("dvol", self._load_dvol), ("publications", self._load_publications), ("quotes", self._load_quotes)]

    def _load_quotes(self, allow_fetch=False):
        """USD-M perpetual: Binance ``bookTicker`` daily archive (best bid/ask with transaction time).
        Spot has no such archive: spot quotes come only from live collection."""
        if self.spec.market_type != "perpetual":
            return
        frames = []
        for d in self.days(self.start - tu.DAY, self.end):
            rel = bn.archive_rel("um", "daily", "bookTicker", self.spec.symbol, d)
            try:
                data = self._fetch_archive(rel, allow_fetch)
            except NotFound:
                continue
            if data is not None:
                frames.append(bn.parse_book_ticker_archive(data))
        if frames:
            self.tables["quotes"] = pd.concat(frames, ignore_index=True).sort_values("ts_us").reset_index(drop=True)
            self.tables["quotes_source"] = "binance USD-M bookTicker archive (transaction_time)"

    def _klines_range(self, market, symbol, interval, start_us, end_us, kind="klines", allow_fetch=False):
        iv = tu.parse_duration(interval.replace("m", "min") if interval.endswith("m") else interval)
        frames = []
        for d in self.days(start_us, end_us):
            rel = bn.archive_rel(market, "daily", kind, symbol, d, interval)
            try:
                data = self._fetch_archive(rel, allow_fetch)
            except NotFound:
                continue
            if data is None:
                continue
            frames.append(bn.parse_klines_zip(data, market, d, iv))
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True).drop_duplicates("period_start_us").sort_values("period_start_us").reset_index(drop=True)

    def _load_klines(self, allow_fetch=False):
        T = self.tables
        s, e = (self.start // tu.DAY - 31) * tu.DAY, self.end
        sym = self.spec.symbol
        T["kl_spot_1m"] = self._klines_range("spot", sym, "1m", s, e, allow_fetch=allow_fetch)
        T["kl_perp_1m"] = self._klines_range("um", sym, "1m", s, e, allow_fetch=allow_fetch)
        T["mark_1m"] = self._klines_range("um", sym, "1m", s, e, "markPriceKlines", allow_fetch)
        T["index_1m"] = self._klines_range("um", sym, "1m", s, e, "indexPriceKlines", allow_fetch)
        T["premium_index_1m"] = self._klines_range("um", sym, "1m", s, e, "premiumIndexKlines", allow_fetch)
        dmarket = "spot" if self.spec.market_type == "spot" else "um"
        T["daily_close"] = self._klines_range(dmarket, sym, "1d", self.daily_start, e, allow_fetch=allow_fetch)
        usdt = {}
        ps = (self.start // tu.DAY - int(self.cfg["features"]["premium"]["z_window_days"]) - 5) * tu.DAY
        for a in list(dict.fromkeys(list(self.cfg["features"]["premium"]["context_assets"]) + [self.spec.base])):
            usdt[a] = self._klines_range("spot", f"{a}USDT", "1m", ps, e, allow_fetch=allow_fetch)
        T["usdt_1m"] = usdt

    def _load_funding(self, allow_fetch=False):
        rel = f"rest/fundingRate_cache/{self.spec.symbol}_{self.daily_start}_{self.end}.parquet"
        p = self.root / "normalized" / rel
        if p.exists():
            self.tables["funding"] = read_parquet(p)
            return
        if not allow_fetch or self.offline:
            self.tables["funding"] = None
            return
        df = self.rest.funding_rate_history(self.spec.symbol, self.start - 30 * tu.DAY, self.end)
        write_parquet(df, p)
        self.tables["funding"] = df

    def _load_coinbase(self, allow_fetch=False):
        pc = self.cfg["features"]["premium"]
        s = (self.start // tu.DAY - int(pc["z_window_days"]) - 5) * tu.DAY
        out = {}
        if self.offline or not allow_fetch:
            # rebuild from cached raw pages only
            allow = False
        else:
            allow = True
        for a in list(dict.fromkeys(list(pc["context_assets"]) + [self.spec.base])):
            out[a] = self._coinbase_cached(f"{a}-USD", s, self.end, allow)
        self.tables["usd_1m"] = out
        self.tables["fx_1m"] = self._coinbase_cached(pc["fx_product"], s, self.end, allow)

    def _coinbase_cached(self, product, s, e, allow):
        p = self.root / "normalized" / "coinbase" / f"{product}_60_{s}_{e}.parquet"
        if p.exists():
            return read_parquet(p)
        if not allow:
            return None
        df = self.coinbase.candles(product, s, e, 60)
        write_parquet(df, p)
        return df

    def _load_dvol(self, allow_fetch=False):
        dc = self.cfg["features"]["dvol"]
        out = {}
        for cur in dc["currencies"]:
            p = self.root / "normalized" / "deribit" / f"dvol_{cur}_{dc['resolution']}_{self.start}_{self.end}.parquet"
            if p.exists():
                out[cur] = read_parquet(p)
            elif allow_fetch and not self.offline:
                df = self.deribit.dvol(cur, self.start - 2 * tu.DAY, self.end, dc["resolution"])
                write_parquet(df, p)
                out[cur] = df
        self.tables["dvol"] = out

    def _load_publications(self, allow_fetch=False):
        """Strict mode needs vintages.  Evidence of availability: (a) own live poll logs (received_at),
        (b) Wayback captures (capture time = value was public no later than then).  A table fetched NOW
        has no availability evidence: it is kept only for the research zone (assumed lag)."""
        from .live import replay as lr
        from .sources.publications import parse_farside, parse_fng, vintages_from_snapshots
        fng_v, etf_v = lr.publications(self.root)
        etf_v = dict(etf_v or {})
        T = self.tables
        fng_now, etf_now = None, {}
        if allow_fetch and not self.offline:
            try:
                fng_now, _ = self.pubs.fetch_fng()
            except Exception as e:
                self._status("fear_greed_now", False, e)
            for a in ("BTC", "ETH"):
                try:
                    etf_now[a] = farside_daily(self.pubs.fetch_farside(a)[0])
                except Exception as e:
                    self._status(f"farside_{a}", False, e)
            for a in ("BTC", "ETH"):
                try:
                    wv = self.pubs.wayback_vintages("farside", a, self.daily_start, self.end)
                    if len(wv):
                        frames = [(farside_daily(g), int(c)) for c, g in wv.groupby("available_at_us")]
                        v = vintages_from_snapshots(frames, ["total_musd", "sum_funds_musd", "n_funds_reported"],
                                                    ["economic_date"]).assign(evidence="wayback_capture")
                        etf_v[a] = pd.concat([x for x in (etf_v.get(a), v) if x is not None], ignore_index=True)
                except Exception as e:
                    self._status(f"wayback_farside_{a}", False, e)
            try:
                wf = self.pubs.wayback_vintages("fng", "BTC", self.daily_start, self.end)
                if len(wf):
                    frames = [(g, int(c)) for c, g in wf.groupby("available_at_us")]
                    v = vintages_from_snapshots(frames, ["value"], ["economic_date"]).assign(evidence="wayback_capture")
                    fng_v = pd.concat([x for x in (fng_v, v) if x is not None and len(x)], ignore_index=True)
            except Exception as e:
                self._status("wayback_fear_greed", False, e)

        def no_evidence(df):
            if df is None or not len(df):
                return None
            d = df.copy()
            d["available_at_us"] = np.nan
            d["evidence"] = "fetched_now_no_publication_time"
            return d
        # strict tables: evidence-backed vintages only; otherwise values without evidence (-> null in strict)
        T["fng"] = fng_v if fng_v is not None and len(fng_v) else no_evidence(fng_now)
        T["etf"] = {a: (etf_v[a] if a in etf_v and len(etf_v[a]) else no_evidence(etf_now.get(a))) for a in ("BTC", "ETH")}
        # research zone: the most complete raw history (fetched now), used only with an ASSUMED lag
        T["research_raw"] = {"fng": fng_now, "etf": etf_now}

    # ------------------------------------------------------------------ build
    def build(self, allow_fetch=False, live_trades: pd.DataFrame | None = None) -> dict:
        cfg = self.cfg
        spec = self.spec
        market = spec.archive_market()
        dtype = cfg["bars"]["trade_source"]
        self.tables = {}
        builders = {}
        resolver = None
        if allow_fetch and not self.offline:
            def resolver(a, b):
                try:
                    return self.rest.historical_trades(spec, a, b)
                except Exception:
                    return None
        for bt in cfg["bars"]["types"]:
            sid = cfgmod.series_id(cfg, bt)
            thr = 0
            if bt == "volume":
                thr = instruments.decimal_to_scaled(cfg["bars"]["volume"]["threshold"], spec.qty_decimals)
            elif bt == "notional":
                thr = instruments.decimal_to_scaled(cfg["bars"]["notional"]["threshold"], spec.price_decimals + spec.qty_decimals)
            builders[bt] = BarBuilder(sid, bt, spec, self.root / "state", interval_us=tu.parse_duration(cfg["bars"]["time"]["interval"]),
                                      threshold_i=thr, origin_us=self.origin, gap_policy=cfg["bars"]["gap_policy"],
                                      adaptive=cfg["bars"]["adaptive"], gap_resolver=resolver,
                                      max_ts_regressions=int(cfg["bars"]["max_ts_regression_us"]))
        if cfg["storage"].get("qa_kline_crosscheck", True) and dtype == "trades":
            # internal 1-minute series used only to cross-check against venue klines (not exported)
            builders["_qa1m"] = BarBuilder(cfgmod.series_id(cfg, "time") + "_qa1m", "time", spec, self.root / "state",
                                           interval_us=tu.MIN, gap_policy=cfg["bars"]["gap_policy"],
                                           gap_resolver=resolver, max_ts_regressions=int(cfg["bars"]["max_ts_regression_us"]))
        px_dir = self.root / "normalized" / "px1s" / f"{spec.exchange}_{spec.market_type}_{spec.symbol}_{dtype}"
        qual_path = self.root / "normalized" / "trades_quality" / f"{spec.exchange}_{spec.market_type}_{spec.symbol}_{dtype}.jsonl"
        qual_path.parent.mkdir(parents=True, exist_ok=True)
        data_start = None
        missing_days = []
        last_complete_wm = None
        day_iter = [] if live_trades is not None else self.days(self.trade_start, self.end)
        for d in day_iter:
            d_us = tu.date_to_us(d)
            rel = bn.archive_rel(market, "daily", dtype, spec.symbol, d)
            # skip days fully processed by every builder (restart / update)
            px_file = px_dir / f"{d.isoformat()}.parquet"
            done = all(b.meta.get("last_day_done", -1) >= d_us for b in builders.values()) and px_file.exists()
            if done:
                if data_start is None:
                    data_start = d_us
                last_complete_wm = d_us + tu.DAY
                continue
            try:
                data = self._fetch_archive(rel, allow_fetch)
            except NotFound:
                data = None
            except SourceUnreachable as e:
                self._status("binance_archive_trades", False, e)
                data = None
            if data is None:
                missing_days.append(d.isoformat())
                for b in builders.values():
                    b.mark_day_incomplete(d_us)
                continue
            parse = bn.parse_aggtrades_zip if dtype == "aggTrades" else bn.parse_trades_zip
            df, q = parse(data, spec, d, rel)
            with open(qual_path, "a") as fh:
                fh.write(json.dumps({"day": d.isoformat(), **{k: v for k, v in q.items() if k != "internal_gap_ranges"},
                                     "gap_ranges": q.get("internal_gap_ranges")}) + "\n")
            if data_start is None:
                data_start = d_us
            if cfg["storage"].get("keep_normalized_trades"):
                write_parquet(df, self.root / "normalized" / "trades" / f"{spec.exchange}_{spec.market_type}_{spec.symbol}_{dtype}"
                              / f"{d.isoformat()}.parquet", metadata={"instrument": spec.to_dict(), "source_file": rel,
                                                                       "ts_unit_original": q.get("ts_unit_original")})
            self._process_day(builders, df, d_us, d_us + tu.DAY)
            self._write_px1s(df, px_file)
            last_complete_wm = d_us + tu.DAY
        self.recv_map = None
        if live_trades is not None and len(live_trades) and "received_at_us" in live_trades:
            lt = live_trades.sort_values("trade_id")
            self.recv_map = (lt["trade_id"].to_numpy(np.int64), lt["received_at_us"].to_numpy(np.float64))
        if live_trades is not None and len(live_trades):
            data_start = int(live_trades["ts_us"].min())
            wm = int(live_trades["ts_us"].max())  # live: only up to the last observed trade (+ grace applied in timing)
            self._process_day(builders, live_trades, None, wm)
            self._write_px1s(live_trades, px_dir / "live.parquet")
        summary = {"missing_trade_days": missing_days, "series": {}}
        for bt, b in builders.items():
            b.save()
        # ---------------- aux / external tables (from cache, or fetched when allowed)
        for key, fn in self._aux_jobs():
            try:
                fn(allow_fetch)
            except SourceUnreachable as e:
                self._status(key, False, f"source_unreachable: {e}")
            except Exception as e:
                self._status(key, False, f"{type(e).__name__}: {e}")
        self._load_live_tables()
        px1s = self._read_px1s(px_dir)
        out_dir = self.root / "datasets" / cfgmod.config_hash(cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        qa = builders.pop("_qa1m", None)
        manifest = self._manifest_base(builders, missing_days)
        if qa is not None:
            from .qa import compare_with_klines
            kl = self.tables.get("kl_spot_1m") if spec.market_type == "spot" else self.tables.get("kl_perp_1m")
            qb = self._read_bars(qa)
            manifest["kline_crosscheck_1m"] = (compare_with_klines(qb, kl) if kl is not None and qb is not None
                                               else "not run: venue klines not available in cache")
        for bt, b in builders.items():
            bars = self._read_bars(b)
            if bars is None or len(bars) == 0:
                summary["series"][bt] = {"rows": 0, "note": "no bars (no trade data)"}
                continue
            res = self._assemble(bt, b, bars, px1s, data_start, out_dir)
            summary["series"][bt] = res
            manifest["datasets"][bt] = res
        manifest["source_status"] = self.source_status
        manifest["aux_tables"] = table_inventory(self.tables)
        export_observations(self.tables, out_dir / "observations", self.cfg)
        sc = source_coverage({k: v for k, v in self.tables.items() if not callable(v)})
        write_parquet(sc, out_dir / "source_coverage.parquet")
        manifest["family_status"] = {k: {"status": v[0], "source": v[1], "note": v[2]} for k, v in status_mod.FAMILY_STATUS.items()}
        manifest["not_implemented"] = status_mod.NOT_IMPLEMENTED
        manifest["methodology_breaks"] = {bt: {"segments": r.get("segments"), "gaps": r.get("gaps")}
                                          for bt, r in manifest["datasets"].items()}
        atomic_write_json(out_dir / "manifest.json", manifest)
        summary["out_dir"] = str(out_dir)
        return summary

    def _process_day(self, builders, df, d_us, watermark):
        """``d_us`` = start of a complete daily archive file, or None for a stream without proof of
        completeness before its first trade (live logs / fixtures): then the first time bucket is
        marked incomplete and event-bar accumulation starts at the first observed trade."""
        from .bars import S_INCOMPLETE_UNTIL, S_LAST_CLOSE, S_PROCESSED
        first_ts = int(df["ts_us"].min()) if len(df) else None
        for bt, b in builders.items():
            sub = df
            is_time = b.bar_type == "time"
            if not is_time:
                sub = df[df["ts_us"] >= self.origin]
                if not len(sub):
                    if d_us is not None:
                        b.meta["last_day_done"] = d_us
                    continue
                if d_us is None and b.st[S_PROCESSED] < 0:
                    f0 = int(sub["ts_us"].min())
                    if f0 > self.origin:
                        b.st[S_LAST_CLOSE] = f0
                        b.meta["segments"][0] = {"segment_id": 0, "origin_us": f0,
                                                 "reason": "first_observed_trade (no completeness proof before it)"}
            else:
                if b.st[S_NEXT_BUCKET] < 0 and len(df):
                    if d_us is None:
                        b.start_series_at(first_ts)
                        if first_ts % b.interval != 0:
                            b.st[S_INCOMPLETE_UNTIL] = first_ts // b.interval
                    else:
                        b.start_series_at(d_us)
            try:
                out = b.process(sub, watermark_us=watermark if is_time else -1)
            except GapPolicyHalt:
                raise
            if len(out):
                self._append_bars(b, out)
            if d_us is not None:
                b.meta["last_day_done"] = d_us
            b.save()

    def _append_bars(self, b: BarBuilder, out: pd.DataFrame):
        first, last = int(out["bar_seq"].min()), int(out["bar_seq"].max())
        p = b.state_dir / "bars" / f"part-{first:012d}-{last:012d}.parquet"
        write_parquet(out, p)
        b.meta.setdefault("parts", []).append(p.name)

    def _read_bars(self, b: BarBuilder) -> pd.DataFrame | None:
        parts = [b.state_dir / "bars" / x for x in b.meta.get("parts", [])]
        parts = [p for p in parts if p.exists()]
        if not parts:
            return None
        df = pd.concat([read_parquet(p) for p in parts], ignore_index=True)
        df = df.sort_values("bar_seq").drop_duplicates("bar_seq", keep="last").reset_index(drop=True)
        return bars_to_float(df, self.spec)

    def _write_px1s(self, df: pd.DataFrame, path: Path):
        if not len(df):
            return
        sec = (df["ts_us"].to_numpy(np.int64) // tu.SEC) * tu.SEC
        tmp = pd.DataFrame({"sec_start_us": sec, "last_ts_us": df["ts_us"].to_numpy(np.int64),
                            "price": df["price_i"].to_numpy(np.float64) / self.spec.price_scale,
                            "trade_id": df["trade_id"].to_numpy(np.int64)})
        g = tmp.sort_values("trade_id").groupby("sec_start_us", as_index=False).last()
        write_parquet(g[["sec_start_us", "last_ts_us", "price"]], path)

    def _read_px1s(self, d: Path):
        fs = sorted(d.glob("*.parquet")) if d.exists() else []
        if not fs:
            return None
        df = pd.concat([read_parquet(f) for f in fs], ignore_index=True)
        return df.sort_values("sec_start_us").drop_duplicates("sec_start_us", keep="last").reset_index(drop=True)

    def _load_live_tables(self):
        from .live import replay as lr
        T = self.tables
        liq = lr.liquidations(self.root)
        if any(len(v["events"]) or v["uptime"] for v in liq.values()):
            T["liquidations"] = liq
        dv = lr.dvol(self.root, self.cfg["timing"]["replay_mode"])
        if dv:
            for k, v in dv.items():
                if T.get("dvol") is None:
                    T["dvol"] = {}
                T["dvol"][k] = pd.concat([T["dvol"].get(k), v], ignore_index=True) if T["dvol"].get(k) is not None else v
        if T.get("quotes") is None and self.spec.market_type == "spot":
            q = lr.quotes(self.root, self.spec)
            if len(q):
                T["quotes"] = q
                T["quotes_source"] = "binance spot bookTicker (live; no exchange timestamp -> receipt time)"
        opt = lr.options(self.root)
        if len(opt):
            T["options"] = opt
        else:
            T["options_missing_reason"] = "not_available_in_period"
        depth_files = lr.files(self.root, "binance_spot_depth" if self.spec.market_type == "spot" else "binance_um_depth")
        if depth_files:
            from .book.adapters import binance_live_messages
            tag = "binance_spot" if self.spec.market_type == "spot" else "binance_um"
            mode = self.cfg["timing"]["replay_mode"]
            T["book_messages"] = lambda: binance_live_messages(depth_files, lr.files(self.root, f"{tag}_depth_snapshot"),
                                                               lr.files(self.root, "_conn"), self.spec, mode,
                                                               futures=self.spec.market_type != "spot")
            T["book_venue_rule"] = "binance_spot" if self.spec.market_type == "spot" else "binance_futures"
        wb = self.cfg["features"]["book"]["wall"].get("bin_size") or instruments.KNOWN.get(
            (self.spec.exchange, self.spec.market_type, self.spec.symbol), {}).get("wall_bin_size")
        if wb is not None:
            T["wall_bin_size_i"] = instruments.decimal_to_scaled(wb, self.spec.price_decimals)

    # ------------------------------------------------------------------ assembly
    def timing(self, bars: pd.DataFrame, bar_type: str = "volume"):
        """bar_end (market time) -> bar_available -> decision_time.

        market_time replay: available = bar_end + bar_finalization_delay (declared assumption; no
        measured latency).  observed_availability (live logs): event bars become available when the
        closing trade was RECEIVED; time bars at max(bar_end + live_finalization_grace, receipt of the
        last trade inside the interval).  Both add bar_finalization_delay (processing)."""
        t = self.cfg["timing"]
        fin = tu.parse_duration(t["bar_finalization_delay"])
        lat = tu.parse_duration(t["decision_latency"])
        end = bars["bar_end_us"].to_numpy(np.int64)
        avail = end + fin
        if t["replay_mode"] == "observed_availability":
            rm = getattr(self, "recv_map", None)
            if rm is None:
                raise ValueError("observed_availability needs receipt logs (replay-live); archives only support market_time")
            ids, rec = rm
            lt = bars["last_trade_id"].to_numpy(np.int64)
            j = np.clip(np.searchsorted(ids, lt), 0, len(ids) - 1)
            r = np.where(ids[j] == lt, rec[j], np.nan)
            if bar_type == "time":
                grace = tu.parse_duration(t["live_finalization_grace"])
                base = end + grace
                r = np.where(np.isfinite(r), np.maximum(r, base), base)
            avail = np.where(np.isfinite(r), r, np.nan)
            if np.isnan(avail).any():
                raise ValueError("missing receipt time for some closing trades")
            avail = avail.astype(np.int64) + fin
        return end, avail, avail + lat

    def _assemble(self, bt, b: BarBuilder, bars: pd.DataFrame, px1s, data_start, out_dir: Path) -> dict:
        cfg = self.cfg
        valid_rows = bars["status"].to_numpy() != 1  # gap-invalidated event bars are not bars
        inv = bars[~valid_rows]
        bars = bars[valid_rows].reset_index(drop=True)
        end, avail, dec = self.timing(bars, bt)
        gaps = [(g["start_ts_us"], g["end_ts_us"]) for g in b.meta.get("gaps", [])]
        ctx = Ctx(cfg=cfg, spec=self.spec, bar_type=bt, bars=bars, decision_us=dec, ref_us=end, px1s=px1s,
                  gaps=gaps, data_start_us=data_start, tables=self.tables)
        for fam in FAMILIES:
            fam.compute(ctx)
        # ---------------- rows in the export period
        if bt == "time":
            sel = (bars["bar_start_us"] >= self.start) & (bars["bar_start_us"] < self.end)
        else:
            sel = (bars["bar_end_us"] >= self.start) & (bars["bar_end_us"] < self.end)
        sel = sel.to_numpy()
        base_cols = ["series_id", "bar_type", "bar_seq", "segment_id", "bar_start_us", "bar_end_us", "first_trade_ts_us",
                     "last_trade_ts_us", "first_trade_id", "last_trade_id", "open", "high", "low", "close", "volume",
                     "notional", "n_trades", "n_messages", "threshold", "complete", "status_name"]
        out = bars.loc[sel, base_cols].reset_index(drop=True)
        out.insert(0, "bar_id", out["series_id"] + ":" + out["bar_seq"].astype(str))
        out = out.rename(columns={"notional": f"notional_{self.spec.notional_currency.lower()}",
                                  "threshold": "threshold_frozen", "status_name": "bar_status"})
        comp = out["complete"].to_numpy() == 1
        for c in ("open", "high", "low", "close", "volume", f"notional_{self.spec.notional_currency.lower()}", "n_trades", "n_messages"):
            v = out[c].to_numpy(np.float64).copy()
            v[~comp] = np.nan  # partially observed bars are never presented as complete
            out[c] = v
        out["bar_available_us"] = avail[sel]
        out["decision_time_us"] = dec[sel]
        out["bar_provenance"] = "trades" if cfg["bars"]["trade_source"] == "trades" else "aggTrades"
        for c in ("bar_start_us", "bar_end_us", "bar_available_us", "decision_time_us"):
            out[c.replace("_us", "")] = pd.to_datetime(out[c], unit="us", utc=True)
        strict = cfg["timing"]["strictness"] == "strict"
        omit = bool(cfg["storage"]["omit_all_null_columns"])
        omitted, research_cols = [], []
        quality = []
        feats = {}
        for name, col in ctx.out.items():
            spec_ = next(s for s in ctx.specs if s.name == name)
            v = col.values[sel]
            r = col.reason[sel]
            if not spec_.strict_eligible:
                research_cols.append(name)
                continue
            if omit and not (r == VALID).any():
                omitted.append(name)
                continue
            feats[name] = v
            quality.append(compress_reasons(name, r, out["bar_seq"].to_numpy()))
        fdf = pd.DataFrame(feats)
        ds = pd.concat([out, fdf], axis=1)
        sid = b.series_id
        fn = out_dir / f"{bt}.parquet"
        write_parquet(ds, fn, metadata={"series_id": sid, "bar_type": bt, "strict": strict})
        if cfg["storage"]["csv_export"]:
            ds.to_csv(out_dir / f"{bt}.csv", index=False)
        q = pd.concat(quality, ignore_index=True) if quality else pd.DataFrame()
        write_parquet(q, out_dir / f"{bt}_quality.parquet")
        if research_cols:
            rdf = pd.concat([out[["bar_id", "decision_time_us"]], pd.DataFrame({c: ctx.out[c].values[sel] for c in research_cols})], axis=1)
            write_parquet(rdf, out_dir / "research" / f"{bt}_assumed_lag.parquet",
                          metadata={"point_in_time_certified": False})
        part = b.partial()
        if len(part):
            write_parquet(bars_to_float(part, self.spec), out_dir / f"{bt}_partial.parquet")
        if len(inv):
            write_parquet(inv, out_dir / f"{bt}_invalidated_bars.parquet")
        month = pd.to_datetime(out["bar_end_us"], unit="us", utc=True).dt.strftime("%Y-%m").to_numpy()
        sub_cols = {k: Col.from_arrays(v.values[sel], v.reason[sel]) for k, v in ctx.out.items()}
        cov = feature_month_coverage(month, sub_cols)
        write_parquet(cov, out_dir / f"{bt}_coverage.parquet")
        status_mod.apply(ctx.specs)
        reg = [s.to_dict() for s in ctx.specs]
        atomic_write_json(out_dir / f"{bt}_registry.json", reg)
        atomic_write_text(out_dir / f"{bt}_registry.md", to_markdown(ctx.specs))
        summ = summarize(cov)
        prev = self._invariance_check(out_dir, bt, ds)
        reg_hash = hashlib.sha256(json.dumps(reg, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return {"series_id": sid, "identity": cfgmod.series_identity(cfg, bt), "rows": int(len(ds)),
                "registry_sha256_16": reg_hash,
                "columns": int(ds.shape[1]), "features": int(len(feats)), "omitted_all_null_columns": omitted,
                "research_columns": research_cols, "file": str(fn), "partial_bar": bool(len(part)),
                "invalidated_bars": int(len(inv)), "segments": b.meta.get("segments"),
                "gaps": b.meta.get("gaps"), "recovered_gaps": b.meta.get("recovered_gaps", []),
                "feature_valid_share": dict(zip(summ["feature"], summ["valid_share"].round(4))),
                "quality_notes": ctx.quality, "prefix_invariance_vs_previous_build": prev}

    def _invariance_check(self, out_dir: Path, bt: str, ds: pd.DataFrame):
        """On update: rows already present in the previous build (same config hash) must be identical."""
        prev_p = out_dir / f"{bt}.prev.parquet"
        cur_p = out_dir / f"{bt}.parquet"
        res = None
        if prev_p.exists():
            prev = read_parquet(prev_p)
            common = prev.merge(ds[["bar_id"]], on="bar_id")
            a = prev.set_index("bar_id").sort_index()
            cur = ds.set_index("bar_id")
            # rows can legitimately leave the export only if the export period changed; then the config
            # hash (and directory) changes, so within one directory a missing row is a violation
            missing_now = int(len(a.index.difference(cur.index)))
            bb = cur.loc[a.index.intersection(cur.index)].sort_index()
            cols = [c for c in a.columns if c in bb.columns]
            diff = 0
            for c in cols:
                x, y = a.loc[bb.index, c], bb[c]
                if x.dtype.kind in "fc":
                    diff += int((~((x == y) | (x.isna() & y.isna()))).sum())
                else:
                    diff += int((x.astype(str) != y.astype(str)).sum())
            res = {"rows_compared": int(len(bb)), "cells_different": diff, "rows_in_previous": int(len(prev)),
                   "common_rows": int(len(common)), "previous_rows_missing_now": missing_now}
        # keep this build as the reference for the next update
        import shutil
        shutil.copyfile(cur_p, prev_p)
        return res

    def _manifest_base(self, builders, missing_days):
        try:
            deps = {d: md.version(d) for d in ("numpy", "pandas", "pyarrow", "numba", "scipy", "statsmodels",
                                                "sortedcontainers", "requests", "websockets", "PyYAML", "lxml")}
        except md.PackageNotFoundError:
            deps = {}
        for d in list(deps):
            try:
                deps[d] = md.version(d)
            except md.PackageNotFoundError:
                deps[d] = None
        src = Path(__file__).parent
        h = hashlib.sha256()
        for f in sorted(src.rglob("*.py")):
            h.update(f.read_bytes())
        return {
            "generated_at": tu.to_iso(tu.now_us()), "package_version": __version__, "code_sha256": h.hexdigest(),
            "python": sys.version, "platform": platform.platform(), "dependencies": deps,
            "config": json.loads(json.dumps(self.cfg, default=str)), "config_hash": cfgmod.config_hash(self.cfg),
            "docs_verified_on": DOCS_VERIFIED_ON, "strictness": self.cfg["timing"]["strictness"],
            "replay_mode": self.cfg["timing"]["replay_mode"],
            "temporal_evidence_level": ("market_time_replay: exchange timestamps/sequence; no measured live latency"
                                        if self.cfg["timing"]["replay_mode"] == "market_time"
                                        else "observed_availability: receipt logs"),
            "instrument": self.spec.to_dict(),
            "series": {bt: {"series_id": b.series_id, "identity": cfgmod.series_identity(self.cfg, bt)} for bt, b in builders.items()},
            "endpoints": ENDPOINTS, "missing_trade_days": missing_days, "inputs": self.inputs[:5000],
            "n_inputs": len(self.inputs), "datasets": {},
        }


ENDPOINTS = {
    "binance_archive": "https://data.binance.vision/data/{spot|futures/um}/daily/{trades|aggTrades|klines|markPriceKlines|indexPriceKlines|premiumIndexKlines}/...",
    "binance_rest": ["https://api.binance.com/api/v3/historicalTrades", "https://api.binance.com/api/v3/depth",
                     "https://fapi.binance.com/fapi/v1/fundingRate"],
    "coinbase": "https://api.exchange.coinbase.com/products/{BTC-USD|ETH-USD|USDT-USD}/candles?granularity=60",
    "deribit": ["https://www.deribit.com/api/v2/public/get_volatility_index_data", "public/get_instruments", "public/ticker"],
    "alternative_me": "https://api.alternative.me/fng/?limit=0",
    "farside": ["https://farside.co.uk/btc/", "https://farside.co.uk/eth/"],
    "wayback": "https://web.archive.org/cdx/search/cdx",
    "tardis": "https://datasets.tardis.dev/v1/{exchange}/{data_type}/{YYYY}/{MM}/{DD}/{symbol}.csv.gz",
    "live": ["wss://stream.binance.com:9443/stream", "wss://fstream.binance.com/stream",
             "wss://stream.bybit.com/v5/public/linear", "wss://www.deribit.com/ws/api/v2"],
}


def export_observations(T: dict, d: Path, cfg):
    """Level-2 observation tables with the full time contract: event/period times, published_at
    (null when not verifiable), received_at, ingested_at, available_at (rule of the current mode),
    version and evidence."""
    d.mkdir(parents=True, exist_ok=True)
    now = tu.now_us()

    def std(df, kind, avail_rule):
        x = df.copy()
        for c in ("published_at_us", "received_at_us"):
            if c not in x:
                x[c] = np.nan
        x["ingested_at_us"] = now
        if "available_at_us" not in x:
            x["available_at_us"] = x["period_end_us"] if "period_end_us" in x else np.nan
        x["availability_rule"] = avail_rule
        x["kind"] = kind
        return x
    for asset, v in (T.get("etf") or {}).items():
        if v is not None and len(v):
            write_parquet(std(v, f"etf_{asset}", "first verifiable availability (poll receipt / archive capture); "
                              "null = unknown_publication_time"), d / f"etf_{asset.lower()}.parquet")
    if T.get("fng") is not None and len(T["fng"]):
        write_parquet(std(T["fng"], "fear_greed", "first verifiable availability; null = unknown"), d / "fear_greed.parquet")
    for cur, v in (T.get("dvol") or {}).items():
        if v is not None and len(v):
            write_parquet(std(v, f"dvol_{cur}", "candle end (market time)"), d / f"dvol_{cur.lower()}.parquet")
    if T.get("funding") is not None and len(T["funding"]):
        f = T["funding"].assign(period_end_us=T["funding"]["funding_time_us"])
        write_parquet(std(f, "funding", "funding time (market time)"), d / "funding.parquet")


def table_inventory(T: dict) -> dict:
    """Rows per auxiliary table actually available to the build (None/0 = absent from cache/source)."""
    out = {}
    for k, v in T.items():
        if isinstance(v, pd.DataFrame):
            out[k] = int(len(v))
        elif isinstance(v, dict):
            out[k] = {kk: (int(len(vv)) if isinstance(vv, pd.DataFrame) else (None if vv is None else "present"))
                      for kk, vv in v.items()}
        elif callable(v):
            out[k] = "stream"
        elif v is None:
            out[k] = None
        else:
            out[k] = str(v)[:80]
    return out


def family_expectations(cfg, reach: dict) -> dict:
    """What each family can deliver given reachability (used by dry-run)."""
    r = lambda k: "reachable" if reach.get(k) else "NOT reachable from this environment"
    return {
        "bars+base+flow+lambdas+structural": f"Binance trades archive ({r('binance_archive')}); gap recovery via REST ({r('binance_rest')})",
        "premium": f"Coinbase candles ({r('coinbase')}) + Binance 1m klines; FX = Coinbase USDT-USD",
        "trend_30d": "Binance 1d klines (archive)",
        "basis/premium_index/funding": f"Binance USD-M archive + REST ({r('binance_futures_rest')})",
        "etf_flows": f"Farside ({r('farside')}); strict mode needs vintages: live polls or Wayback ({r('wayback')}); otherwise null",
        "fear_greed": f"Alternative.me ({r('alternative_me')}); strict mode needs vintages (live polls / Wayback)",
        "dvol": f"Deribit ({r('deribit')}); history availability to be checked by download",
        "skew_25d": "historical: Tardis free first-of-month options_chain only; otherwise live collection",
        "liquidations": "live collection only (Binance sampled, Bybit all); Tardis optional",
        "l2_book/spread/cost": "live collection (Binance diff depth + snapshots); Tardis free days optional",
    }

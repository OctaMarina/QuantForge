"""Live collector.

Every message is written verbatim to ``live/<stream>/<day>/<HHMM>.jsonl.gz`` with ``received_at_us``
(wall clock at receipt).  Connection open/close events go to ``live/_conn``.  These logs are the raw
input for later reconstruction (``python crypto_dataset.py replay-live``), which runs the SAME bar
builder and feature code as historical builds, in either market-time or observed-availability mode.

Endpoints (verified 2026-10-20 unless noted):
* Binance spot: wss://stream.binance.com:9443/stream?streams=<sym>@trade/<sym>@depth@100ms/<sym>@bookTicker
  (+ timeUnit=MICROSECOND); connections are dropped after 24h; REST snapshot /api/v3/depth?limit=5000.
* Binance USD-M: wss://fstream.binance.com/stream?streams=<sym>@forceOrder/<sym>@depth@100ms/<sym>@aggTrade;
  REST snapshot /fapi/v1/depth?limit=1000.
* Bybit v5 linear: wss://stream.bybit.com/v5/public/linear, subscribe allLiquidation.<SYM>; ping {"op":"ping"}.
* Deribit: wss://www.deribit.com/ws/api/v2, public/subscribe to deribit_volatility_index.<cur>_usd and
  ticker.<instrument>.100ms (channel names from Deribit docs; not re-verified in this build).
* Polls: Alternative.me F&G and Farside pages every ``live.poll_every`` (each snapshot stored).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import requests

from .. import timeutil as tu
from ..storage import JsonlGzWriter

log = logging.getLogger("cryptods.live")


class Collector:
    def __init__(self, cfg, root, spec):
        self.cfg = cfg
        self.root = root
        self.spec = spec
        self.rotate = tu.parse_duration(cfg["live"]["rotate"])
        self.writers: dict[str, JsonlGzWriter] = {}
        self.stop = asyncio.Event()

    def w(self, stream: str) -> JsonlGzWriter:
        if stream not in self.writers:
            self.writers[stream] = JsonlGzWriter(self.root, stream, self.rotate)
        return self.writers[stream]

    def log_msg(self, stream: str, payload, extra=None):
        self.w(stream).write(tu.now_us(), payload, extra)

    def conn(self, stream: str, event: str, detail: str = ""):
        self.log_msg("_conn", {"stream": stream, "event": event, "detail": detail})

    async def _ws_loop(self, name: str, url: str, on_open=None, on_msg=None, ping=None):
        import websockets
        backoff = 1
        while not self.stop.is_set():
            try:
                async with websockets.connect(url, max_size=2 ** 24, ping_interval=20, ping_timeout=60) as ws:
                    self.conn(name, "open", url)
                    backoff = 1
                    if on_open:
                        await on_open(ws)
                    ping_task = asyncio.create_task(self._pinger(ws, ping)) if ping else None
                    try:
                        async for raw in ws:
                            rec = tu.now_us()
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                msg = {"_raw": raw if isinstance(raw, str) else raw.decode("utf-8", "replace")}
                            if on_msg:
                                await on_msg(msg, rec)
                            if self.stop.is_set():
                                break
                    finally:
                        if ping_task:
                            ping_task.cancel()
            except Exception as e:  # network errors, server shutdowns, 24h drops
                self.conn(name, "close", repr(e)[:300])
                log.warning("%s disconnected: %r", name, e)
            else:
                self.conn(name, "close", "normal")
            await asyncio.sleep(min(backoff, 60))
            backoff *= 2

    async def _pinger(self, ws, payload):
        while True:
            await asyncio.sleep(20)
            await ws.send(json.dumps(payload))

    # ---------------------------------------------------------------- Binance
    async def binance(self, futures: bool):
        sym = self.spec.symbol.lower()
        if futures:
            url = f"wss://fstream.binance.com/stream?streams={sym}@forceOrder/{sym}@depth@100ms/{sym}@aggTrade"
            snap_url, limit, tag = "https://fapi.binance.com/fapi/v1/depth", 1000, "binance_um"
        else:
            url = (f"wss://stream.binance.com:9443/stream?streams={sym}@trade/{sym}@depth@100ms/{sym}@bookTicker"
                   "&timeUnit=MICROSECOND")
            snap_url, limit, tag = "https://api.binance.com/api/v3/depth", 5000, "binance_spot"

        async def on_open(ws):
            # snapshot AFTER subscribing so buffered diffs can bridge it (Binance procedure)
            await asyncio.sleep(1.0)
            loop = asyncio.get_running_loop()
            sent = tu.now_us()
            r = await loop.run_in_executor(None, lambda: requests.get(snap_url, params={"symbol": self.spec.symbol.upper(),
                                                                                     "limit": limit}, timeout=30))
            self.log_msg(f"{tag}_depth_snapshot", r.json(), {"limit": limit, "request_sent_us": sent})

        async def on_msg(msg, rec):
            stream = msg.get("stream", "")
            data = msg.get("data", msg)
            if "@depth" in stream:
                self.w(f"{tag}_depth").write(rec, data)
            elif "@forceOrder" in stream:
                self.w(f"{tag}_forceOrder").write(rec, data)
            elif "@bookTicker" in stream:
                self.w(f"{tag}_bookTicker").write(rec, data)
            elif "Trade" in stream or "@trade" in stream:
                self.w(f"{tag}_trades").write(rec, data)
            else:
                self.w(f"{tag}_other").write(rec, msg)

        await self._ws_loop(tag, url, on_open, on_msg)

    # ---------------------------------------------------------------- Bybit
    async def bybit_liquidations(self):
        sym = self.spec.symbol.upper()

        async def on_open(ws):
            await ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{sym}"]}))

        async def on_msg(msg, rec):
            self.w("bybit_allLiquidation").write(rec, msg)

        await self._ws_loop("bybit_linear", "wss://stream.bybit.com/v5/public/linear", on_open, on_msg, ping={"op": "ping"})

    # ---------------------------------------------------------------- Deribit
    async def deribit(self):
        lo, hi = self.cfg["live"]["deribit_option_expiry_window_days"]

        async def on_open(ws):
            chans = [f"deribit_volatility_index.{c.lower()}_usd" for c in self.cfg["features"]["dvol"]["currencies"]]
            cur = self.cfg["features"]["skew"]["currency"]
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(None, lambda: requests.get(
                "https://www.deribit.com/api/v2/public/get_instruments",
                params={"currency": cur, "kind": "option", "expired": "false"}, timeout=30))
            inst = r.json()["result"]
            self.log_msg("deribit_instruments", inst)
            now = tu.now_us()
            for x in inst:
                exp = int(x["expiration_timestamp"]) * 1000
                if now + lo * tu.DAY <= exp <= now + hi * tu.DAY:
                    chans.append(f"ticker.{x['instrument_name']}.100ms")
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "public/set_heartbeat", "params": {"interval": 30}}))
            for i in range(0, len(chans), 200):
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2 + i, "method": "public/subscribe",
                                          "params": {"channels": chans[i:i + 200]}}))

        async def on_msg(msg, rec, ws_holder={}):
            if msg.get("method") == "heartbeat":
                return
            ch = (msg.get("params") or {}).get("channel", "")
            if ch.startswith("deribit_volatility_index"):
                self.w("deribit_dvol").write(rec, msg)
            elif ch.startswith("ticker."):
                self.w("deribit_option_ticker").write(rec, msg)
            else:
                self.w("deribit_other").write(rec, msg)

        await self._ws_loop("deribit", "wss://www.deribit.com/ws/api/v2", on_open, on_msg)

    # ---------------------------------------------------------------- polls
    async def poll(self, name: str, url: str, every: str, params=None):
        period = tu.parse_duration(every) / tu.SEC
        loop = asyncio.get_running_loop()
        while not self.stop.is_set():
            sent = tu.now_us()
            try:
                r = await loop.run_in_executor(None, lambda: requests.get(url, params=params, timeout=30,
                                                                          headers={"User-Agent": "cryptods/0.1"}))
                self.log_msg(f"poll_{name}", {"status": r.status_code, "body": r.text}, {"request_sent_us": sent})
            except Exception as e:
                self.conn(f"poll_{name}", "error", repr(e)[:300])
            await asyncio.sleep(period)

    async def run(self, seconds: float | None = None):
        streams = self.cfg["live"]["streams"]
        tasks = []
        perp = self.spec.market_type == "perpetual"
        if "trades" in streams or "depth" in streams or "bookTicker" in streams:
            tasks.append(self.binance(futures=perp))
        if "forceOrder" in streams:
            tasks.append(self.binance(futures=True)) if not perp else None
        if "bybit_liquidations" in streams:
            tasks.append(self.bybit_liquidations())
        if "deribit_dvol" in streams or "deribit_options" in streams:
            tasks.append(self.deribit())
        pe = self.cfg["live"]["poll_every"]
        if "fear_greed_poll" in streams:
            tasks.append(self.poll("fear_greed", "https://api.alternative.me/fng/", pe["fear_greed"], {"limit": 10}))
        if "farside_poll" in streams:
            tasks.append(self.poll("farside_btc", "https://farside.co.uk/btc/", pe["farside"]))
            tasks.append(self.poll("farside_eth", "https://farside.co.uk/eth/", pe["farside"]))
        runners = [asyncio.create_task(t) for t in tasks if t is not None]
        flusher = asyncio.create_task(self._flusher())
        try:
            if seconds:
                await asyncio.sleep(seconds)
                self.stop.set()
            else:
                await asyncio.gather(*runners)
        finally:
            self.stop.set()
            for r in runners:
                r.cancel()
            flusher.cancel()
            for w in self.writers.values():
                w.close()

    async def _flusher(self):
        while True:
            await asyncio.sleep(5)
            for w in self.writers.values():
                w.flush()


def run(cfg, root, spec, seconds=None):
    logging.basicConfig(level=logging.INFO)
    c = Collector(cfg, root, spec)
    asyncio.run(c.run(seconds))
    _ = time

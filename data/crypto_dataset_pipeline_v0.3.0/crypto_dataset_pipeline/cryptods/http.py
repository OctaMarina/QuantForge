"""HTTP client: per-host rate limiting, retry with exponential backoff, explicit failure classes."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests


class SourceUnreachable(Exception):
    """Network policy / DNS / TLS / connection failure: the provider could not be reached."""


class NotFound(Exception):
    """HTTP 404: the object does not exist (e.g. archive day not published)."""


@dataclass
class Response:
    url: str
    status: int
    content: bytes
    headers: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


class RateLimiter:
    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.min_interval


class Http:
    def __init__(self, rate_limits: dict | None = None, timeout_s: float = 30, retries: int = 5,
                 backoff_s: float = 1.0, user_agent: str = "cryptods/0.1 (+research dataset builder)"):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.timeout = timeout_s
        self.retries = retries
        self.backoff = backoff_s
        self.rate_limits = rate_limits or {}
        self._limiters: dict[str, RateLimiter] = {}
        self.log: list[dict] = []

    def _limiter(self, provider: str) -> RateLimiter:
        if provider not in self._limiters:
            self._limiters[provider] = RateLimiter(float(self.rate_limits.get(provider, 2)))
        return self._limiters[provider]

    def get(self, url: str, provider: str, params: dict | None = None, headers: dict | None = None,
            stream: bool = False, allow_404: bool = False) -> Response:
        last_exc = None
        for attempt in range(self.retries + 1):
            self._limiter(provider).wait()
            t0 = time.monotonic()
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except (requests.exceptions.ProxyError, requests.exceptions.SSLError) as e:
                # proxy policy denials / TLS problems are not transient: do not hammer
                self._record(url, provider, None, str(e))
                raise SourceUnreachable(f"{urlparse(url).netloc}: {e}") from e
            except requests.exceptions.RequestException as e:
                last_exc = e
                self._record(url, provider, None, str(e))
                time.sleep(self.backoff * (2 ** attempt))
                continue
            el = time.monotonic() - t0
            self._record(url, provider, r.status_code, None)
            if r.status_code == 404:
                if allow_404:
                    return Response(r.url, 404, b"", dict(r.headers), el)
                raise NotFound(url)
            if r.status_code in (418, 429) or 500 <= r.status_code < 600:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff * (2 ** attempt)
                time.sleep(min(wait, 120))
                last_exc = RuntimeError(f"HTTP {r.status_code} for {url}")
                continue
            if r.status_code in (401, 403, 407):
                raise SourceUnreachable(f"HTTP {r.status_code} for {url}")
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:300]}")
            return Response(r.url, r.status_code, r.content, dict(r.headers), el)
        raise SourceUnreachable(f"giving up on {url}: {last_exc}")

    def get_json(self, url: str, provider: str, params: dict | None = None):
        import json
        return json.loads(self.get(url, provider, params=params).content)

    def _record(self, url, provider, status, error):
        self.log.append({"t": time.time(), "provider": provider, "url": url, "status": status, "error": error})

    def probe(self, url: str, provider: str) -> dict:
        """One cheap request for dry-run reachability; never retries policy denials."""
        try:
            r = self.session.get(url, timeout=min(self.timeout, 15))
            return {"url": url, "reachable": True, "status": r.status_code, "bytes": len(r.content)}
        except requests.exceptions.RequestException as e:
            return {"url": url, "reachable": False, "status": None, "error": type(e).__name__ + ": " + str(e)[:200]}

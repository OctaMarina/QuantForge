"""Storage: atomic writes, versioned raw store, parquet helpers.

Layout under ``storage.root``::

    raw/<source>/<relative path>              original bytes as downloaded / received
    raw/<source>/<relative path>.meta.json    url, sha256, headers, checksum status, ingested_at
    raw/<source>/<relative path>.v2 ...       later *different* versions (never overwritten)
    raw/<source>/_revisions.jsonl             one record per detected revision
    normalized/<table>/...                    normalised / reconstructed tables (parquet)
    state/<series_id>/...                     restartable builder state
    datasets/<config_hash>/...                final datasets, manifest, registry, coverage
    live/<stream>/<YYYY-MM-DD>/<HH>.jsonl.gz  raw live messages with received_at
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import timeutil as tu


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path, text: str):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_parquet(df: pd.DataFrame, path: Path, metadata: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if metadata:
        md = dict(table.schema.metadata or {})
        md[b"cryptods"] = json.dumps(metadata, default=str).encode()
        table = table.replace_schema_metadata(md)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_parquet(path: Path, columns=None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def parquet_metadata(path: Path) -> dict:
    md = pq.read_schema(path).metadata or {}
    raw = md.get(b"cryptods")
    return json.loads(raw) if raw else {}


class RawStore:
    """Content-addressed-by-path raw store that never silently overwrites a revised file."""

    def __init__(self, root: Path):
        self.root = Path(root) / "raw"

    def path(self, source: str, rel: str) -> Path:
        return self.root / source / rel

    def versions(self, source: str, rel: str) -> list[Path]:
        p = self.path(source, rel)
        out = [p] if p.exists() else []
        n = 2
        while True:
            q = p.with_name(p.name + f".v{n}")
            if not q.exists():
                break
            out.append(q)
            n += 1
        return out

    def latest(self, source: str, rel: str) -> Path | None:
        v = self.versions(source, rel)
        return v[-1] if v else None

    def meta(self, path: Path) -> dict:
        return read_json(Path(str(path) + ".meta.json"), {}) or {}

    def put(self, source: str, rel: str, data: bytes, meta: dict) -> tuple[Path, str]:
        """Store bytes. Returns (path, status) with status in {'new', 'unchanged', 'revision'}."""
        digest = sha256_bytes(data)
        vers = self.versions(source, rel)
        if vers:
            last = vers[-1]
            if self.meta(last).get("sha256") == digest or sha256_file(last) == digest:
                return last, "unchanged"
            target = self.path(source, rel)
            target = target.with_name(target.name + f".v{len(vers) + 1}")
            status = "revision"
        else:
            target = self.path(source, rel)
            status = "new"
        atomic_write_bytes(target, data)
        m = dict(meta)
        m.update({"sha256": digest, "size": len(data), "ingested_at": tu.to_iso(tu.now_us()),
                  "version": len(vers) + 1, "relpath": rel, "source": source})
        atomic_write_json(Path(str(target) + ".meta.json"), m)
        if status == "revision":
            rec = {"source": source, "relpath": rel, "new_version": len(vers) + 1,
                   "previous_sha256": self.meta(vers[-1]).get("sha256"), "new_sha256": digest,
                   "detected_at": tu.to_iso(tu.now_us())}
            rp = self.root / source / "_revisions.jsonl"
            rp.parent.mkdir(parents=True, exist_ok=True)
            with open(rp, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return target, status


class JsonlGzWriter:
    """Append-only gz JSONL writer with hourly rotation; each line carries received_at_us."""

    def __init__(self, root: Path, stream: str, rotate_us: int):
        self.root = Path(root) / "live" / stream
        self.rotate_us = rotate_us
        self._fh = None
        self._bucket = None

    def write(self, received_at_us: int, payload, extra: dict | None = None):
        bucket = int(tu.floor_to(received_at_us, self.rotate_us))
        if bucket != self._bucket:
            self.close()
            iso = tu.to_iso(bucket)
            day, hh = iso[:10], iso[11:13] + iso[14:16]
            p = self.root / day / f"{hh}.jsonl.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(p, "at")
            self._bucket = bucket
        rec = {"received_at_us": received_at_us, "payload": payload}
        if extra:
            rec.update(extra)
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def flush(self):
        if self._fh:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def iter_jsonl_gz(paths):
    for p in sorted(paths):
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            # truncated last line of a file being written / crashed collector
                            continue
        except EOFError:
            continue

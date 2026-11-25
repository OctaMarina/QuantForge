"""Read-only access to the raw CryptoHFTData hourly archives (outer Zstandard around Parquet).

Raw files are never modified. Readers return Arrow tables plus a provenance record
(path, compressed size, sha256, parquet created_by, key-value metadata, schema fingerprint).
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@dataclass
class RawFile:
    path: str
    size: int
    sha256: str
    created_by: str
    kv_metadata: dict = field(default_factory=dict)
    schema_fp: str = ""
    columns: list = field(default_factory=list)
    num_rows: int = 0
    num_row_groups: int = 0


def schema_fingerprint(schema: pa.Schema) -> str:
    desc = ";".join(f"{f.name}:{f.type}:{'N' if f.nullable else 'NN'}" for f in schema)
    return hashlib.sha1(desc.encode()).hexdigest()[:12]


def _decompress(raw: bytes) -> bytes:
    if raw[:4] == ZSTD_MAGIC:
        return zstd.ZstdDecompressor().decompressobj().decompress(raw)
    if raw[:4] == b"PAR1":
        return raw
    raise ValueError("not a zstd or parquet file")


def read_raw(path: str | Path, columns=None, with_hash: bool = True):
    """Return (pyarrow.Table, RawFile). Whole file is decompressed in memory (hourly files are small)."""
    path = Path(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest() if with_hash else ""
    dec = _decompress(raw)
    size = len(raw)
    del raw
    pf = pq.ParquetFile(pa.BufferReader(dec))
    md = pf.metadata
    kv = {}
    if pf.schema_arrow.metadata:
        for k, v in pf.schema_arrow.metadata.items():
            if k == b"ARROW:schema":
                continue
            kv[k.decode(errors="replace")] = v.decode(errors="replace")
    info = RawFile(path=str(path), size=size, sha256=sha, created_by=md.created_by or "",
                   kv_metadata=kv, schema_fp=schema_fingerprint(pf.schema_arrow),
                   columns=[f"{f.name}:{f.type}" for f in pf.schema_arrow],
                   num_rows=md.num_rows, num_row_groups=md.num_row_groups)
    present = set(pf.schema_arrow.names)
    cols = None if columns is None else [c for c in columns if c in present]
    table = pf.read(columns=cols, use_threads=False)
    return table, info


def col_i64(table: pa.Table, name: str, fill: int = 0):
    """int64 numpy array of a column; nulls replaced with `fill`; missing column -> all `fill`.
    Returns (values, null_mask or None)."""
    import numpy as np
    if name not in table.column_names:
        return np.full(table.num_rows, fill, np.int64), np.ones(table.num_rows, bool)
    c = table.column(name)
    if c.null_count:
        mask = c.is_null().to_numpy(zero_copy_only=False)
        vals = c.fill_null(fill).to_numpy().astype(np.int64, copy=False)
        return vals, mask
    return c.to_numpy().astype(np.int64, copy=False), None


def kv_json(info: RawFile) -> str:
    return json.dumps(info.kv_metadata, sort_keys=True)

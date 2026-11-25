"""A small, explicit disk cache for research pipelines.

Results are pickled under ``<root>/<hash of the configuration>/<name>.pkl``.  The hash covers whatever the
caller puts in ``key`` (parameters, data fingerprints, a version tag), so changing any of them gives a fresh
directory and stale results are never read by mistake.  Delete the directory to recompute.

    from quant_forge.utils import DiskCache, stable_hash, frame_fingerprint

    cache = DiskCache("~/.cache/quantforge/my_study", key={"windows": [...], "data": frame_fingerprint(bars)})
    feats = cache.get_or_compute("features", compute_features, bars)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

__all__ = ["DEFAULT_CACHE_ROOT", "DiskCache", "stable_hash", "frame_fingerprint"]

#: Root used by the QuantForge loaders when no cache directory is given.
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "quantforge"


def stable_hash(obj: Any, length: int = 12) -> str:
    """Hex digest of ``json.dumps(obj, sort_keys=True, default=str)``: equal configurations give equal hashes."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def frame_fingerprint(frames: pd.DataFrame | Mapping[str, pd.DataFrame], columns=None, length: int = 16) -> str:
    """Content hash of one DataFrame or of a mapping of DataFrames (keys and order included).

    ``columns`` restricts the hash to some columns (e.g. OHLCV) so that adding derived columns does not change it.
    """
    h = hashlib.sha256()
    items = frames.items() if isinstance(frames, Mapping) else [("", frames)]
    for name, df in items:
        sub = df if columns is None else df.loc[:, [c for c in columns if c in df.columns]]
        h.update(str(name).encode())
        h.update(",".join(map(str, sub.columns)).encode())
        h.update(pd.util.hash_pandas_object(sub, index=True).to_numpy().tobytes())
    return h.hexdigest()[:length]


class DiskCache:
    """Pickle cache in ``root`` (or ``root / stable_hash(key)`` when ``key`` is given).

    Writes are atomic (temporary file + rename), so an interrupted run never leaves a truncated entry.
    """

    def __init__(self, root: str | os.PathLike, key: Any = None):
        root = Path(root).expanduser()
        self.root = root / stable_hash(key) if key is not None else root
        self.key = key
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.pkl"

    def __contains__(self, name: str) -> bool:
        return self.path(name).exists()

    def load(self, name: str) -> Any:
        return pickle.loads(self.path(name).read_bytes())

    def save(self, name: str, value: Any) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.path(name))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def get_or_compute(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Return the cached ``name`` or compute ``fn(*args, **kwargs)``, store and return it."""
        if name in self:
            return self.load(name)
        value = fn(*args, **kwargs)
        self.save(name, value)
        return value

    def clear(self, name: str | None = None) -> None:
        """Remove one entry, or every ``*.pkl`` entry of this cache directory."""
        targets = [self.path(name)] if name is not None else list(self.root.glob("*.pkl"))
        for p in targets:
            if p.exists():
                p.unlink()

    def __repr__(self) -> str:
        return f"DiskCache({str(self.root)!r})"

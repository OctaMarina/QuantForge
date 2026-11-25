"""YAML configuration -> typed specs. Parameters that change semantics are explicit per bar set / family."""
from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .bars.build import BarSpec
from .book.replay import ReplayConfig
from .features.slow import SlowFeatureConfig
from .features.trades import TradeFeatureConfig

FAMILIES = ("bar", "trades", "mark", "oi", "liq", "ticker", "calendar", "book", "combos")


@dataclass
class DatasetConfig:
    name: str
    start: date
    end: date
    bars: list
    families: tuple = FAMILIES
    trades: TradeFeatureConfig = field(default_factory=TradeFeatureConfig)
    slow: SlowFeatureConfig = field(default_factory=SlowFeatureConfig)
    book: ReplayConfig = field(default_factory=ReplayConfig)
    book_start: date | None = None
    book_end: date | None = None
    select: list | None = None          # experiment column selection (globs); None = all computed columns
    workers: int = 6
    raw: dict = field(default_factory=dict)

    @property
    def config_hash(self):
        return hashlib.sha1(json.dumps(self.raw, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def selected(self, cols):
        if not self.select:
            return list(cols)
        keep = [c for c in cols if any(fnmatch.fnmatch(c, g) for g in self.select)]
        return keep


def _override(dc, d):
    if not d:
        return dc
    kw = {}
    for f in dataclasses.fields(dc):
        if f.name in d:
            v = d[f.name]
            if isinstance(getattr(dc, f.name), tuple):
                v = tuple(v)
            if f.name == "rv_grids":
                v = {float(k): tuple(x) for k, x in v.items()}
            kw[f.name] = v
    unknown = set(d) - {f.name for f in dataclasses.fields(dc)}
    if unknown:
        raise ValueError(f"unknown keys for {type(dc).__name__}: {sorted(unknown)}")
    return dataclasses.replace(dc, **kw)


def load_config(path: str | Path, overrides: dict | None = None) -> DatasetConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if overrides:
        for k, v in overrides.items():
            cur = raw
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = yaml.safe_load(str(v))
    ds = raw["dataset"]
    bars = []
    for b in raw["bars"]:
        bars.append(BarSpec(kind=b["kind"], threshold=b.get("threshold"), adaptive=b.get("adaptive"),
                            cusum_reset=b.get("cusum_reset", "triggered"), daily_reset=b.get("daily_reset", False),
                            origin_ns=int(b.get("origin_ns", 0)), name=b["name"]))
    feats = raw.get("features", {})
    cfg = DatasetConfig(
        name=ds["name"], start=date.fromisoformat(str(ds["start"])), end=date.fromisoformat(str(ds["end"])),
        bars=bars, families=tuple(feats.get("families", FAMILIES)),
        trades=_override(TradeFeatureConfig(), feats.get("trades")),
        slow=_override(SlowFeatureConfig(), feats.get("slow")),
        book=_override(ReplayConfig(), feats.get("book")),
        book_start=date.fromisoformat(str(feats["book_start"])) if feats.get("book_start") else None,
        book_end=date.fromisoformat(str(feats["book_end"])) if feats.get("book_end") else None,
        select=feats.get("select"), workers=int(raw.get("workers", 6)), raw=raw)
    bad = set(cfg.families) - set(FAMILIES)
    if bad:
        raise ValueError(f"unknown families {bad}")
    return cfg

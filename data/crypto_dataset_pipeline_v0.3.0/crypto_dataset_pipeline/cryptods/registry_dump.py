"""Produce the feature registry for a configuration without market data.

Feature families are run on a tiny synthetic bar table only to enumerate the configured columns and
their metadata; NO values from this run are written anywhere."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import instruments, status as status_mod
from .bars import BarBuilder, bars_to_float
from .features.context import Ctx
from .registry import to_markdown


def dump(cfg, out_dir: str | Path, bar_type: str = "volume"):
    from .pipeline import FAMILIES
    spec = instruments.from_config(cfg)
    n = 400
    rng = np.random.default_rng(0)
    ts = 1_700_000_000_000_000 + np.cumsum(rng.integers(1, 10_000_000, n))
    px = (100_000 * spec.price_scale + np.cumsum(rng.integers(-5, 6, n))).astype(np.int64)
    q = rng.integers(1, 10, n).astype(np.int64) * spec.qty_scale
    t = pd.DataFrame({"trade_id": np.arange(1, n + 1), "ts_us": ts, "price_i": px, "qty_i": q, "notional_i": px * q,
                      "aggressor": rng.choice([-1, 1], n).astype(np.int8)})
    import tempfile
    b = BarBuilder("registry", "volume", spec, tempfile.mkdtemp(), threshold_i=5 * spec.qty_scale)
    bars = bars_to_float(b.process(t), spec)
    end = bars["bar_end_us"].to_numpy(np.int64)
    ctx = Ctx(cfg=cfg, spec=spec, bar_type=bar_type, bars=bars, decision_us=end, ref_us=end, px1s=None, tables={})
    sec = (t.ts_us // 1_000_000) * 1_000_000
    ctx.px1s = pd.DataFrame({"sec_start_us": sec, "last_ts_us": t.ts_us, "price": t.price_i / spec.price_scale}) \
        .groupby("sec_start_us", as_index=False).last()
    for fam in FAMILIES:
        fam.compute(ctx)
    status_mod.apply(ctx.specs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "registry.json").write_text(json.dumps([s.to_dict() for s in ctx.specs], indent=1, default=str))
    (out / "registry.md").write_text(to_markdown(ctx.specs))
    return {"columns": len(ctx.specs), "path": str(out / "registry.json")}

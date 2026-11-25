"""EXPERIMENT (evidence for the open book decision D1; never used by the default datasets).

Two replays over the same period, evaluated on a 1 s grid:
  strict : every snapshot re-anchors; state valid only when anchored+chained (ground truth where valid)
  healed : anchor once, then never use snapshots again; continue through true gaps by accepting the sequence jump
Compares top of book and band depths where the strict replay is valid, by time since the healed replay's first heal.
Usage: .venv/bin/python scripts/heal_experiment.py 2026-05-14 2026-05-16 <out_dir> [workers]
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from chdds import paths as P
from chdds.book.driver import run_replay
from chdds.book.replay import ReplayConfig

s, e, out = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), Path(sys.argv[3])
w = int(sys.argv[4]) if len(sys.argv) > 4 else 2
out.mkdir(parents=True, exist_ok=True)
q = np.arange(P.hour_start_ns(s, 0) + P.NS, P.hour_start_ns(e, 23) + P.HOUR_NS, P.NS, dtype=np.int64)
base = dict(windows_s=(5.0,), mlofi_windows_s=(), basis_windows_s=(), es_windows_s=(), markout_h_s=(),
            markout_windows_s=(), minmax_max_window_s=0.0)
A, cols, st_a, ca, _ = run_replay(s, e, q, out / "tmp_a", ReplayConfig(**base), workers=w, write_grid=False,
                                  log=lambda *a: None)
B, _, st_b, cb, _ = run_replay(s, e, q, out / "tmp_b", ReplayConfig(snapshot_probe_only=2, experiment_emit_healed=True,
                                                                    **base),
                               workers=w, write_grid=False, heal=True, log=lambda *a: None)
ci = {c: i for i, c in enumerate(cols)}
ok = (A[:, ci["book_state"]] == 0) & np.isfinite(B[:, ci["mid"]])
res = dict(period=f"{s}..{e}", seconds=int(q.size), strict_valid_s=int((A[:, ci["book_state"]] == 0).sum()),
           healed_emitted_s=int(np.isfinite(B[:, ci["mid"]]).sum()), compared_s=int(ok.sum()),
           strict_counters=ca, healed_counters=cb)
for c in ("best_bid", "best_ask", "qty_bid1_btc", "qty_ask1_btc", "depth_bid_top10_btc", "depth_bid_5bps_btc",
          "depth_ask_5bps_btc", "depth_bid_25bps_btc", "depth_ask_25bps_btc", "cost_buy_1btc_bps"):
    a = A[ok, ci[c]]
    b = B[ok, ci[c]]
    m = np.isfinite(a) & np.isfinite(b)
    res[c] = dict(n=int(m.sum()), equal_frac=float(np.mean(a[m] == b[m])) if m.any() else None,
                  abs_err_p50=float(np.median(np.abs(a[m] - b[m]))) if m.any() else None,
                  abs_err_p99=float(np.quantile(np.abs(a[m] - b[m]), 0.99)) if m.any() else None,
                  rel_err_p99=float(np.quantile(np.abs(a[m] - b[m]) / np.maximum(np.abs(a[m]), 1e-12), 0.99)) if m.any() else None)
(out / "heal_vs_strict.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))

"""Finish the book stage after a completed replay (split per bar set + statistics + counters from the final
checkpoint) without re-running the replay. Usage: .venv/bin/python scripts/finish_book.py configs/btcusdt_initial.yaml"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.book import replay as RP  # noqa: E402
from chdds.config import load_config  # noqa: E402
from chdds.dataset import split_book  # noqa: E402

cfg = load_config(sys.argv[1])
b0, b1 = cfg.book_start or cfg.start, cfg.book_end or cfg.end
split_book(cfg, b0, b1)
statp = P.DERIVED / "book_state" / cfg.name
tag = f"{b0.isoformat()}_{b1.isoformat()}"
sd = statp / "_stats"
hrs = [pq.read_table(f) for f in sorted(sd.glob("hours_*.parquet"))]
pq.write_table(pa.concat_tables(hrs, promote_options="default"), statp / f"hour_stats_{tag}.parquet")
ver = [pq.read_table(f) for f in sorted(sd.glob("verify_*.parquet"))]
if ver:
    pq.write_table(pa.concat_tables(ver, promote_options="default"), statp / f"verify_log_{tag}.parquet")
z = np.load(statp / b1.isoformat() / "replay_state.npz")
si = z["si"]
counters = {k: int(si[getattr(RP, x)]) for k, x in [
    ("nmsg_applied", "I_NAPPLY"), ("nlevels", "I_NLEV"), ("nverify", "I_NVERIFY"), ("nverify_fail", "I_NVERIFY_FAIL"),
    ("ncross", "I_NCROSS"), ("ncross_inval", "I_NCROSS_INVAL"), ("noff_pos", "I_NOFFTICK_POS"), ("nanchor", "I_NANCHOR"),
    ("ninval", "I_NINVAL"), ("nes", "I_NES"), ("nmo", "I_NMO"), ("verify_mismatch_levels", "I_VERIFY_MISMATCH_LEVELS")]}
(statp / f"counters_{tag}.json").write_text(json.dumps(counters, indent=1))
print(json.dumps(counters))

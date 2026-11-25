"""D1: checks E_T(t) = R(t) restricted to levels reported since T, on a seeded sample of explicit runs.
The age pass predicts, from its own independent book, how many leading levels of the reference a book started at T holds; the explicit
run measured it. Output: derived/d1/results/identity_check.csv. Usage: .venv/bin/python scripts/d1_identity_check.py [n_intervals] [workers]"""
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402


def one(iv):
    import d1_age_pass as A
    rows = []
    ts = {}
    for s in iv["starts"]:
        f = pl.read_parquet(P.DERIVED / "d1" / "runs" / iv["id"] / f"{s['name']}_forced.parquet")
        if (f["state_e"] == 1).any():
            continue          # feed dead at T: the book really started later, so T is not its age
        f = f.filter(pl.col("state_r") == 0)
        if f.height == 0:
            continue
        ts[s["name"]] = (s["t"], f)
    if not ts:
        return rows
    q = np.unique(np.concatenate([f["t"].to_numpy() for _, f in ts.values()]))
    t, a, ok = A.run_segment("x", iv["t_anchor"] - iv["t_anchor"] % P.HOUR_NS, int(q[-1]), samples=q, write=False)
    for nm, (T, f) in ts.items():
        i = np.searchsorted(t, f["t"].to_numpy())
        el = (f["t"].to_numpy() - T) // 1_000_000
        for sd, o in (("b", 0), ("a", 50)):
            ag = a[i, o:o + 50]
            mr = f["mr_" + sd].to_numpy()
            nr = f["nr_" + sd].to_numpy()
            over = (ag > el[:, None]) | (ag < 0)
            pred = np.minimum(np.where(over.any(axis=1), over.argmax(axis=1), 50), nr)
            v = ok[i]
            rows.append(dict(interval=iv["id"], start=nm, side=sd, regime=iv["regime"], n=int(v.sum()), diff=int((pred[v] != mr[v]).sum()),
                             max_abs=int(np.abs(pred[v] - mr[v]).max()) if v.any() else 0))
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    plan = json.loads(Path("configs/d1_intervals.json").read_text())
    grid = [i for i in plan["intervals"] if i["cell"] == "grid"]
    rng = np.random.default_rng(20261106)
    pick = [grid[j] for j in rng.choice(len(grid), n, replace=False)]
    with ProcessPoolExecutor(max_workers=w, mp_context=multiprocessing.get_context("spawn")) as ex:
        rows = [r for x in ex.map(one, pick) for r in x]
    d = pl.from_dicts(rows)
    d.write_csv(P.DERIVED / "d1" / "results" / "identity_check.csv")
    print(d.select(pl.col("n").sum(), pl.col("diff").sum(), pl.col("max_abs").max()), d.group_by("regime").agg(pl.col("n").sum(), pl.col("diff").sum()).sort("regime"))


if __name__ == "__main__":
    main()

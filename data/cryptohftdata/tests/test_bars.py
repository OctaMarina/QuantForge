import math

import numpy as np
import pytest

from chdds.bars import engines as E
from conftest import requires_canonical, synthetic_trades

IC = E.IC


def run_event(mode, T, thr, chunks=1, both=False, daily_reset=False):
    st = E.new_event_state()
    outs = []
    n = T["seq"].size
    cuts = np.linspace(0, n, chunks + 1).astype(int)
    day0 = int(T["recv"][0] // E.DAY_NS)
    ndays = int(T["recv"][-1] // E.DAY_NS) - day0 + 1
    sched = np.full(ndays, thr, np.int64)
    for a, b in zip(cuts[:-1], cuts[1:]):
        sl = {k: v[a:b] for k, v in T.items()}
        oi, of, k = E.event_bars(mode, sl["seq"], sl["tid"], sl["recv"], sl["tt"], sl["px"], sl["qx"], sl["sign"],
                                 sl["seg"], sched, day0, st, both, daily_reset)
        outs.append((oi, of))
    oi = np.concatenate([o[0] for o in outs])
    of = np.concatenate([o[1] for o in outs])
    return oi, of, st


def covered_seqs(oi, st):
    """Every trade must belong to exactly one emitted bar or to the open bar in the state."""
    s = []
    for r in oi:
        if r[IC["n_trades"]] > 0:
            s.append(np.arange(r[IC["first_seq"]], r[IC["last_seq"]] + 1))
    if st[E.S_OPEN] == 1:
        s.append(np.arange(st[E.S_FSEQ], st[E.S_LSEQ] + 1))
    return np.concatenate(s)


@pytest.mark.parametrize("mode,thr", [(0, 50_000), (1, 5_000_000_000_000), (2, 500)])
def test_conservation_and_no_fragmentation(mode, thr):
    T = synthetic_trades(30000, seed=1, gaps_at=(12000, 25000))
    oi, of, st = run_event(mode, T, thr)
    cov = covered_seqs(oi, st)
    assert np.array_equal(cov, T["seq"]), "each trade exactly once, in order"
    # exact conservation of quantity and notional (integers)
    vol = oi[:, IC["vol_e3"]].sum() + (st[E.S_VOL] if st[E.S_OPEN] else 0)
    notl = oi[:, IC["notional_e5"]].sum() + (st[E.S_NOT] if st[E.S_OPEN] else 0)
    assert vol == T["qx"].sum()
    assert notl == (T["px"] * T["qx"]).sum()
    # at most one bar closes per trade
    last = oi[:, IC["last_seq"]]
    assert np.all(np.diff(last) > 0)
    # gap closes the open bar as partial and never mixes segments
    assert np.all(oi[oi[:, IC["reason"]] == E.R_GAP][:, IC["complete"]] == 0)
    for r in oi:
        segs = T["seg"][r[IC["first_seq"]]:r[IC["last_seq"]] + 1]
        assert segs.min() == segs.max()


def test_volume_threshold_semantics():
    T = synthetic_trades(5000, seed=2)
    thr = 40_000
    oi, of, st = run_event(0, T, thr)
    for r in oi[oi[:, IC["complete"]] == 1]:
        q = T["qx"][r[IC["first_seq"]]:r[IC["last_seq"]] + 1]
        assert q.sum() >= thr and q[:-1].sum() < thr          # closes at the FIRST trade reaching the threshold
        assert r[IC["overshoot_int"]] == q.sum() - thr


def cusum_reference(px, h_bps_c, both=False):
    """Textbook float recursion of the report (S+ = max(0, S+ + r), S- = min(0, S- + r))."""
    h = math.log1p(h_bps_c / 1e6)
    sp = sn = 0.0
    trig = []
    x = np.log(px.astype(float))
    for i in range(1, px.size):
        r = x[i] - x[i - 1]
        sp = max(0.0, sp + r)
        sn = min(0.0, sn + r)
        if sp >= h - 1e-12:
            trig.append((i, 1))
            sp = 0.0
            if both:
                sn = 0.0
        elif sn <= -h + 1e-12:
            trig.append((i, -1))
            sn = 0.0
            if both:
                sp = 0.0
    return trig


@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("thr", [100, 500, 2000])
def test_cusum_matches_float_recursion(thr, both):
    T = synthetic_trades(20000, seed=3)
    oi, of, st = run_event(2, T, thr, both=both)
    ref = cusum_reference(T["px"], thr, both)
    got = [(int(r[IC["last_seq"]]), int(r[IC["trigger"]])) for r in oi if r[IC["complete"]] == 1]
    assert got == ref
    # bar = all trades since previous boundary incl. the trigger trade, exactly once
    assert np.array_equal(covered_seqs(oi, st), T["seq"])


def test_cusum_equality_triggers():
    # price path hits exactly +b from the running minimum: p_min*(1+b/1e4) on an exact tick
    px = np.array([9990000, 9992000, 9993995, 9993996], np.int64)  # min 99900.00; +4 bp of 99900.00 = 99939.96 exactly
    n = px.size
    T = dict(seq=np.arange(n), tid=np.arange(n), recv=np.arange(1, n + 1) * 1_000_000_000 + 1_751_000_000_000_000_000,
             tt=np.arange(n), px=px, qx=np.ones(n, np.int64), sign=np.ones(n, np.int64), seg=np.zeros(n, np.int64))
    oi, of, st = run_event(2, T, 400)  # 4 bp
    assert oi.shape[0] == 1 and oi[0, IC["last_seq"]] == 3 and oi[0, IC["trigger"]] == 1   # 99939.95 does not


@pytest.mark.parametrize("mode,thr", [(0, 50_000), (1, 5_000_000_000_000), (2, 500)])
def test_partition_independence(mode, thr):
    T = synthetic_trades(30000, seed=4, gaps_at=(17000,))
    a = run_event(mode, T, thr, chunks=1)
    b = run_event(mode, T, thr, chunks=7)
    assert np.array_equal(a[0], b[0])
    assert np.allclose(a[1], b[1], equal_nan=True)
    assert np.array_equal(a[2], b[2])


def test_time_bar_boundaries():
    base = 1_751_000_000_000_000_000 - (1_751_000_000_000_000_000 % 60_000_000_000)
    recv = np.array([base + 1, base + 59_999_999_999, base + 60_000_000_000, base + 60_000_000_001,
                     base + 180_000_000_000], np.int64)
    n = recv.size
    T = dict(seq=np.arange(n), tid=np.arange(n), recv=recv, tt=recv // 1_000_000, px=np.full(n, 10000000, np.int64),
             qx=np.ones(n, np.int64), sign=np.ones(n, np.int64), seg=np.zeros(n, np.int64))
    st = E.new_event_state()
    oi, of, k = E.time_bars(T["seq"], T["tid"], T["recv"], T["tt"], T["px"], T["qx"], T["sign"], T["seg"],
                            60_000_000_000, 0, st, base, base + 240_000_000_000)
    ntr = oi[:, IC["n_trades"]]
    assert list(ntr) == [2, 2, 0, 1]                  # event exactly at T_{k+1} belongs to the next bar
    assert oi[0, IC["decision_time"]] == base + 60_000_000_000
    assert oi[0, IC["eval_time"]] == base + 60_000_000_000 - 1
    assert oi[2, IC["first_seq"]] == -1 and oi[2, IC["last_seq"]] == 3   # empty bar keeps the trade cursor


@requires_canonical
def test_real_canonical_conservation():
    import pyarrow.parquet as pq
    from chdds import paths as P
    t = pq.read_table(P.DERIVED / "canonical" / "trades" / "2025-06-28.parquet")
    T = dict(seq=t["seq"].to_numpy(), tid=t["trade_id"].to_numpy(), recv=t["recv_ns"].to_numpy(),
             tt=t["trade_time_ms"].to_numpy(), px=t["price_e2"].to_numpy(), qx=t["qty_e3"].to_numpy(),
             sign=t["sign"].to_numpy().astype(np.int64), seg=t["seg"].to_numpy().astype(np.int64))
    for mode, thr in ((0, 50_000), (1, 5_000_000_000_000), (2, 500)):
        oi, of, st = run_event(mode, T, thr, chunks=5)
        assert np.array_equal(covered_seqs(oi, st), T["seq"])


def test_time_bars_chunked_equivalence():
    """Chunked processing (t_until = next trade's receive time) equals a single call."""
    T = synthetic_trades(20000, seed=7, gaps_at=(9000,))
    dt = 1_000_000_000
    t_from = (T["recv"][0] // dt) * dt
    t_end = (T["recv"][-1] // dt + 2) * dt
    st = E.new_event_state()
    a, _, _ = E.time_bars(T["seq"], T["tid"], T["recv"], T["tt"], T["px"], T["qx"], T["sign"], T["seg"], dt, 0, st,
                          t_from, t_end)
    st2 = E.new_event_state()
    parts = []
    n = T["seq"].size
    for lo in range(0, n, 3333):
        hi = min(lo + 3333, n)
        tu = int(T["recv"][hi]) if hi < n else t_end
        sl = {k: v[lo:hi] for k, v in T.items()}
        o, _, _ = E.time_bars(sl["seq"], sl["tid"], sl["recv"], sl["tt"], sl["px"], sl["qx"], sl["sign"], sl["seg"], dt,
                              0, st2, t_from, tu)
        parts.append(o)
    b = np.concatenate(parts)
    assert np.array_equal(a, b)


def test_cusum_threshold_change_closes_partial_and_resets():
    """Daily threshold schedule: at a day boundary with a changed CUSUM threshold the open bar is closed as partial
    (reason threshold_change) and both accumulators restart; chunking across days gives the same result."""
    T = synthetic_trades(40000, seed=9)
    T["recv"] = (1_751_000_000_000_000_000 // E.DAY_NS) * E.DAY_NS + np.arange(T["seq"].size, dtype=np.int64) * 8_000_000_000
    day0 = int(T["recv"][0] // E.DAY_NS)
    ndays = int(T["recv"][-1] // E.DAY_NS) - day0 + 1
    sched = np.array([300 + 100 * (k % 3) for k in range(ndays)], np.int64)
    st = E.new_event_state()
    oi, of, k = E.event_bars(2, T["seq"], T["tid"], T["recv"], T["tt"], T["px"], T["qx"], T["sign"], T["seg"], sched,
                             day0, st, False, False)
    assert (oi[:, IC["reason"]] == E.R_THRESHOLD_CHANGE).sum() >= ndays - 2
    # per-day calls (as in bars.build) give identical output
    st2 = E.new_event_state()
    parts = []
    dd = T["recv"] // E.DAY_NS
    for day in range(day0, day0 + ndays):
        m = dd == day
        sl = {kk: v[m] for kk, v in T.items()}
        o2, f2, _ = E.event_bars(2, sl["seq"], sl["tid"], sl["recv"], sl["tt"], sl["px"], sl["qx"], sl["sign"], sl["seg"],
                                 np.array([sched[day - day0]], np.int64), day, st2, False, False)
        parts.append(o2)
    assert np.array_equal(oi, np.concatenate(parts))
    assert np.array_equal(covered_seqs(oi, st), T["seq"])

import copy

import numpy as np
import pytest

from chdds import paths as P
from chdds.book import chain as CH
from chdds.book.core import ReplayState, rebase, replay_kernel
from chdds.book.driver import _concat_batch, _take_pending
from chdds.book.reference import run_reference
from chdds.book.replay import BS_UNANCHORED, BS_VALID, ReplayConfig
from conftest import requires_raw

CFG = ReplayConfig(ring_size=1 << 16)
COLS = CFG.columns()
CI = {c: i for i, c in enumerate(COLS)}


def run_batches(batches, queries, cfg=CFG, state=None, cstate=None, timeout_s=60.0, trades=None, marks=None,
                return_seq=False, heal=0, probe=0, start_after=0):
    pa_ = cfg.param_arrays()
    state = state or ReplayState(cfg)
    cst, ru, rh = cstate or CH.new_state()
    carry = None
    out = np.full((queries.size, len(COLS)), np.nan)
    seqs = []
    vlog = np.zeros((1000, 5), np.int64)
    nver = 0
    for bi, cur in enumerate(batches):
        batch, ncar = _concat_batch(carry, cur)
        res = CH.resolve(batch["recv"], batch["et"], batch["U"], batch["u"], batch["pu"], batch["L"], batch["hash"],
                         cst, ru, rh, ncar, int(timeout_s * 1e9), heal, probe, 0, start_after)
        seq_msg, seq_time, seq_kind, seq_aux = res[:4]
        carry = _take_pending(batch, res[-1])
        seqs.append((batch, seq_msg, seq_kind, seq_aux, seq_time, res[4]))
        t_hi = batch["recv"].max() + 1 if bi < len(batches) - 1 else np.iinfo(np.int64).max
        t_lo = batch["recv"].min() if bi > 0 else np.iinfo(np.int64).min
        qa = np.searchsorted(queries, t_lo if bi > 0 else -1)
        qa = np.searchsorted(queries, seqs[bi - 1][0]["recv"].max() + 1) if bi > 0 else 0
        qb = np.searchsorted(queries, t_hi)
        qs = queries[qa:qb]
        o = np.full((qs.size, len(COLS)), np.nan)
        z = np.zeros(0, np.int64)
        zf = np.zeros(0)
        mk = marks or (z, z, zf, zf)
        tr = trades or (z, z, zf, zf, zf)
        nver += replay_kernel(seq_msg, seq_time, seq_kind, seq_aux, batch["evt"], batch["txn"], batch["synth"],
                              batch["row_start"], batch["side"], batch["idx"], batch["qty"], batch["off"],
                              *mk, *tr, qs, o, np.zeros((qs.size, 0), np.int64), np.zeros(0, np.int64),
                              np.zeros((0, 8), np.int64),
                              *state.kernel_args(), pa_["ip"], pa_["bands_c"], pa_["ks"], pa_["win_ns"],
                              pa_["win_mlofi"], pa_["win_minmax"], pa_["tw_bands"], pa_["cost_q"], pa_["cost_d"],
                              pa_["bwin_ns"], pa_["eswin_ns"], pa_["mo_h_ns"], pa_["mowin_ns"], pa_["min_cov"], vlog)
        out[qa:qb] = o
        rebase(state.cum, state.ring_c, state.si, state.es_cum, state.es_c, state.mo_cum, state.mo_c)
    if return_seq:
        return out, state, (cst, ru, rh), seqs, vlog[:nver]
    return out, state


def qgrid(h_from, h_to, step_s=5):
    a = P.hour_start_ns(P.HISTORY_START, h_from)
    b = P.hour_start_ns(P.HISTORY_START, h_to)
    return np.arange(a + step_s * 10**9, b, step_s * 10**9, dtype=np.int64)


def test_columns_layout_matches_kernel():
    # the kernel returns the number of columns it wrote; must equal the declared layout
    from chdds.book import core
    assert len(COLS) == len(set(COLS))


@requires_raw
def test_reconstruction_verified_by_every_snapshot(book_hours):
    out, st, cs, seqs, vlog = run_batches([book_hours[5], book_hours[6], book_hours[7]], qgrid(5, 8), return_seq=True)
    from chdds.book.replay import I_NVERIFY  # noqa
    assert vlog.shape[0] >= 30
    assert int(vlog[:, 1].sum()) == 0, "replayed book differs from an exchange REST snapshot"
    assert np.nanmean(out[:, CI["book_state"]] == BS_VALID) > 0.95


@requires_raw
def test_kernel_equals_reference(book_hours):
    out, st, cs, seqs, vlog = run_batches([book_hours[5], book_hours[6]], qgrid(5, 7, 60), return_seq=True)
    ref, results = run_reference([(b, m, k, a, t) for b, m, k, a, t, _ in seqs])
    lo, hi = st.si[2], st.si[3]   # certified domain
    kb = {int(i): int(st.qb[i]) for i in np.nonzero(st.qb)[0] if i >= lo}
    ka = {int(i): int(st.qa[i]) for i in np.nonzero(st.qa)[0] if i <= hi}
    rb = {i: q for i, q in ref.bids.items() if i >= lo}
    ra = {i: q for i, q in ref.asks.items() if i <= hi}
    assert kb == rb and ka == ra
    assert all(not m for _, m in results)


def _permute_rows(batch, seed):
    b = copy.deepcopy(batch)
    rng = np.random.default_rng(seed)
    rs = b["row_start"]
    for m in range(rs.size - 1):
        if rs[m + 1] - rs[m] > 1:
            perm = rs[m] + rng.permutation(rs[m + 1] - rs[m])
            for k in ("side", "idx", "qty", "off"):
                b[k][rs[m]:rs[m + 1]] = b[k][perm]
    return b


@requires_raw
def test_message_atomicity(book_hours):
    q = qgrid(5, 7, 2)
    a, _ = run_batches([book_hours[5], book_hours[6]], q)
    b, _ = run_batches([_permute_rows(book_hours[5], 1), _permute_rows(book_hours[6], 2)], q)
    assert np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99))


def _inject_copy(batch, m, delay_ns, alter=False):
    b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    rs = b["row_start"]
    rows = np.arange(rs[m], rs[m + 1])
    new_recv = b["recv"][m] + delay_ns
    pos = int(np.searchsorted(b["recv"], new_recv, side="right"))
    for k in ("recv", "evt", "txn", "et", "U", "u", "pu", "L", "hash", "synth"):
        val = b[k][m] if k != "recv" else new_recv
        if alter and k == "hash":
            val = np.uint64(int(val) ^ 12345)
        b[k] = np.insert(b[k], pos, val)
    sizes = np.diff(rs)
    sizes = np.insert(sizes, pos, rows.size)
    b["row_start"] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    ins_at = rs[pos] if pos < rs.size else rs[-1]
    for k in ("side", "idx", "qty", "off"):
        vals = b[k][rows].copy()
        if alter and k == "qty":
            vals[0] += 7
        b[k] = np.insert(b[k], ins_at, vals)
    return b


@requires_raw
def test_identical_retransmission_is_ignored(book_hours):
    q = qgrid(5, 7, 2)
    a, _ = run_batches([book_hours[5], book_hours[6]], q)
    h6 = book_hours[6]
    m = int(np.nonzero(h6["et"] == 0)[0][5000])
    b, _ = run_batches([book_hours[5], _inject_copy(h6, m, 300_000_000)], q)
    assert np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99))


@requires_raw
def test_conflicting_duplicate_invalidates(book_hours):
    h6 = book_hours[6]
    m = int(np.nonzero(h6["et"] == 0)[0][5000])
    q = qgrid(5, 7, 2)
    b, st, cs, seqs, _ = run_batches([book_hours[5], _inject_copy(h6, m, 300_000_000, alter=True)], q, return_seq=True)
    actions = seqs[1][5]
    assert (actions == CH.A_DUP_CONFLICT).sum() == 1
    t_conf = h6["recv"][m] + 300_000_000
    after = (q > t_conf) & (q < t_conf + 60 * 10**9)
    assert np.all(b[after, CI["book_state"]] != BS_VALID)


@requires_raw
def test_gap_is_not_auto_recovered(book_hours):
    h6 = book_hours[6]
    # drop one update message 30 s after a snapshot: chain cannot be completed -> true gap after the timeout
    snap_t = h6["recv"][h6["et"] == 1]
    t_pick = snap_t[3] + 30 * 10**9
    m = int(np.nonzero((h6["et"] == 0) & (h6["recv"] > t_pick))[0][0])
    keep = np.ones(h6["recv"].size, bool)
    keep[m] = False
    rs = h6["row_start"]
    rows = np.ones(rs[-1], bool)
    rows[rs[m]:rs[m + 1]] = False
    b = {k: h6[k][keep] for k in ("recv", "evt", "txn", "et", "U", "u", "pu", "L", "hash", "synth")}
    sizes = np.diff(rs)[keep]
    b["row_start"] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    for k in ("side", "idx", "qty", "off"):
        b[k] = h6[k][rows]
    q = qgrid(5, 7, 1)
    out, st, cs, seqs, _ = run_batches([book_hours[5], b], q, timeout_s=10.0, return_seq=True)
    t_gap = h6["recv"][m]
    # after the gap: not valid until a snapshot re-anchors (never just because messages keep arriving)
    snaps = h6["recv"][(h6["et"] == 1)]
    next_snap = snaps[snaps > t_gap].min()
    window = (q > t_gap) & (q < next_snap)
    assert window.sum() > 10
    assert np.all(out[window, CI["book_state"]] != BS_VALID)


@requires_raw
def test_checkpoint_resume_equivalence(book_hours, tmp_path):
    q = qgrid(5, 8, 3)
    full, _ = run_batches([book_hours[5], book_hours[6], book_hours[7]], q)
    out1, st, cs, seqs, _ = run_batches([book_hours[5], book_hours[6]], q[q < P.hour_start_ns(P.HISTORY_START, 7) - 1],
                                        return_seq=True)
    st.save(tmp_path / "s.npz")
    st2 = ReplayState(CFG)
    st2.load_into(tmp_path / "s.npz")
    cst = (cs[0].copy(), cs[1].copy(), cs[2].copy())
    q7 = q[q >= P.hour_start_ns(P.HISTORY_START, 7)]
    # resumed run: hour 7 only, starting from the saved state (pending carry is empty at hour boundaries here)
    out2, _ = run_batches([book_hours[7]], q7, state=st2, cstate=cst)
    ref = full[np.searchsorted(q, q7)]
    assert np.array_equal(np.nan_to_num(ref, nan=-9e99), np.nan_to_num(out2, nan=-9e99))


@requires_raw
def test_prefix_invariance(book_hours):
    q = qgrid(5, 7, 4)
    full, _ = run_batches([book_hours[5], book_hours[6]], q)
    # run only up to a cut inside hour 6 (truncate the batch) and compare queries before the cut
    h6 = book_hours[6]
    cut = int(h6["recv"].size * 0.4)
    tcut = h6["recv"][cut]
    b = {k: h6[k][:cut] for k in ("recv", "evt", "txn", "et", "U", "u", "pu", "L", "hash", "synth")}
    b["row_start"] = h6["row_start"][:cut + 1]
    for k in ("side", "idx", "qty", "off"):
        b[k] = h6[k][:h6["row_start"][cut]]
    qq = q[q < tcut]
    part, _ = run_batches([book_hours[5], b], qq)
    assert np.array_equal(np.nan_to_num(full[:qq.size], nan=-9e99), np.nan_to_num(part, nan=-9e99))


@requires_raw
def test_identities(book_hours):
    out, _ = run_batches([book_hours[5], book_hours[6]], qgrid(5, 7, 5))
    ok = out[:, CI["book_state"]] == BS_VALID
    o = out[ok]
    assert ok.sum() > 100
    wm = o[:, CI["weighted_mid_offset_bps"]]
    assert np.allclose(wm, o[:, CI["spread_bps"]] / 2 * o[:, CI["qi_1"]], atol=1e-9)
    for c in [c for c in COLS if c.startswith("qi_")]:
        v = o[:, CI[c]]
        v = v[np.isfinite(v)]
        assert np.all((v >= -1) & (v <= 1))
    a1 = o[:, CI["cost_buy_0p1btc_bps"]]
    a2 = o[:, CI["cost_buy_1btc_bps"]]
    m = np.isfinite(a1) & np.isfinite(a2)
    assert np.all(a2[m] >= a1[m] - 1e-12)
    assert np.all(o[:, CI["spread_bps"]] > 0)


# ---------------------------------------------------------------------------- synthetic exactness tests
def _synthetic_stream(updates, t0=1_751_000_000_000_000_000):
    """Anchor snapshot (bids 100.0..99.6, asks 100.1..100.5, qty 1 BTC) + updates [(dt_s, [(side, price_e1, qty_e3)])]."""
    msgs = [(t0, 1, -1, 1000, -1, 1000, [(0, 1000 - i, 1000) for i in range(5)] + [(1, 1001 + i, 1000) for i in range(5)])]
    u = 1000
    for dt, lv in updates:
        msgs.append((t0 + int(dt * 1e9), 0, u + 1, u + 10, u, -1, lv))
        u += 10
    n = len(msgs)
    rows = [r for m in msgs for r in m[6]]
    b = dict(recv=np.array([m[0] for m in msgs], np.int64), evt=np.array([m[0] // 10**6 for m in msgs], np.int64),
             txn=np.array([m[0] // 10**6 for m in msgs], np.int64), et=np.array([m[1] for m in msgs], np.int8),
             U=np.array([m[2] for m in msgs], np.int64), u=np.array([m[3] for m in msgs], np.int64),
             pu=np.array([m[4] for m in msgs], np.int64), L=np.array([m[5] for m in msgs], np.int64),
             hash=np.arange(n, dtype=np.uint64), synth=np.zeros(n, bool),
             row_start=np.concatenate([[0], np.cumsum([len(m[6]) for m in msgs])]).astype(np.int64),
             side=np.array([r[0] for r in rows], np.int8), idx=np.array([r[1] * 10000 for r in rows], np.int64) // 10000,
             qty=np.array([r[2] for r in rows], np.int64), off=np.zeros(len(rows), bool))
    # place prices in a realistic range (index units of 0.1 USDT): 100.0 -> idx 1000 is fine for the kernel
    return b, t0


def test_tw_spread_and_ofi_exact():
    # t0 anchor (spread 1 tick); +0.5s first update connects (no change); +1.0s bid 100.1 qty 2 -> spread 0 crossed?
    upd = [(0.5, [(0, 1000, 1000)]),            # connects; spread = 1 tick (101-100) on idx units
           (1.0, [(1, 1001, 0)]),               # best ask removed -> ask 1002, spread 2 ticks
           (3.0, [(0, 1000, 3000)])]            # bid qty 1 -> 3 BTC
    b, t0 = _synthetic_stream(upd)
    q = np.array([t0 + 4 * 10**9], np.int64)
    cfg = ReplayConfig(windows_s=(1.0, 2.0, 3.0), mlofi_windows_s=(), minmax_max_window_s=0.0, ring_size=1 << 12,
                       max_age_ms=10_000, basis_windows_s=(), es_windows_s=(), markout_h_s=(), markout_windows_s=(),
                       tw_band_bps=())
    cols = cfg.columns()
    ci = {c: i for i, c in enumerate(cols)}
    global COLS
    saved = COLS
    COLS = cols
    try:
        out, st = run_batches([b], q, cfg=cfg)
    finally:
        COLS = saved
    o = out[0]
    # spread in bps relative to mid: state after 1.0s has bid 1000, ask 1002 -> 2 ticks / 1001 * 1e4
    s_wide = 1e4 * 2 / 1001.0
    s_tight = 1e4 * 1 / 1000.5
    # window (1s,4s]: spread wide all the time
    assert o[ci["tw_spread_bps_3s"]] == pytest.approx(s_wide, rel=1e-12)
    # window (2s,4s] coverage full
    assert o[ci["book_coverage_2s"]] == pytest.approx(1.0)
    # OFI over (1s, 4s]: event at 3.0s: bid qty 1->3 at same price -> e = +2 BTC
    assert o[ci["ofi_3s_btc"]] == pytest.approx(2.0)
    # window (3s,4s] excludes the event at exactly 3.0s? No: (t-w, t] = (3.0, 4.0] excludes t=3.0 -> 0
    assert o[ci["ofi_1s_btc"]] == pytest.approx(0.0)
    # OFI for (2s,4s] includes the 3.0s event only
    assert o[ci["ofi_2s_btc"]] == pytest.approx(2.0)
    # tw qi1 over (2s,4s]: QI=(1-1)/2=0 until 3s, then (3-1)/(3+1)=0.5 -> mean 0.25
    assert o[ci["tw_qi_1_2s"]] == pytest.approx(0.25, rel=1e-12)
    assert s_tight < s_wide


@requires_raw
def test_skip_unanchored_is_exact(tmp_path):
    """Skipping decode of snapshot-free hours while unanchored must not change any output (strict mode)."""
    from datetime import date
    from chdds.book.driver import run_replay
    d = date(2025, 7, 2)            # unanchored by then; no snapshots in July 2025
    hours = [(d, h) for h in (12, 13, 14)]
    q = np.arange(P.hour_start_ns(d, 12) + 10**9, P.hour_start_ns(d, 15), 30 * 10**9, dtype=np.int64)
    cfg = ReplayConfig(ring_size=1 << 14)
    a, cols, sa, _, _ = run_replay(d, d, q, tmp_path / "a", cfg, workers=0, hours=hours, write_grid=True,
                                   skip_unanchored=False, log=lambda *x: None)
    b, _, sb, _, _ = run_replay(d, d, q, tmp_path / "b", cfg, workers=0, hours=hours, write_grid=True,
                                skip_unanchored=True, log=lambda *x: None)
    assert np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99))
    assert all(r.get("skipped_unanchored") for r in sb if r.get("kind") != "verify")


def test_far_levels_outside_tick_grid_are_ignored_safely():
    """Vendor checkpoints contain asks up to ~4.5M USDT (beyond the 0..400k grid): they must be ignored without
    out-of-bounds writes, both when anchoring and when verifying."""
    from chdds.book.replay import NT
    b, t0 = _synthetic_stream([(0.5, [(0, 1000, 1000)]), (1.0, [(0, 999, 2000)])])
    # second snapshot = verification at L == last_u (u of last update) with an extra far ask level
    last_u = int(b["u"][-1])
    snap_rows = [(0, 1000, 1000), (0, 999, 2000), (0, 998, 1000), (0, 997, 1000), (0, 996, 1000)] + \
                [(1, 1001 + i, 1000) for i in range(5)] + [(1, NT + 12345, 7), (1, 3 * NT, 9)]
    n0 = b["recv"].size
    b2 = {k: np.append(b[k], v) for k, v in dict(recv=t0 + 2 * 10**9, evt=(t0 + 2 * 10**9) // 10**6,
                                                txn=(t0 + 2 * 10**9) // 10**6, et=1, U=-1, u=last_u, pu=-1, L=last_u,
                                                hash=np.uint64(999), synth=False).items()}
    b2["et"] = b2["et"].astype(np.int8)
    b2["row_start"] = np.append(b["row_start"], b["row_start"][-1] + len(snap_rows)).astype(np.int64)
    b2["side"] = np.append(b["side"], [r[0] for r in snap_rows]).astype(np.int8)
    b2["idx"] = np.append(b["idx"], [r[1] for r in snap_rows]).astype(np.int64)
    b2["qty"] = np.append(b["qty"], [r[2] for r in snap_rows]).astype(np.int64)
    b2["off"] = np.append(b["off"], np.zeros(len(snap_rows), bool))
    # also anchor-from-far-levels: first snapshot gets a far ask too
    b2["idx"][b2["row_start"][0] + 9] = 1005  # keep anchor valid
    cfg = ReplayConfig(windows_s=(1.0,), mlofi_windows_s=(), minmax_max_window_s=0.0, ring_size=1 << 12,
                       max_age_ms=10_000, basis_windows_s=(), es_windows_s=(), markout_h_s=(), markout_windows_s=(),
                       tw_band_bps=())
    global COLS
    saved = COLS
    COLS = cfg.columns()
    try:
        out, st, cs, seqs, vlog = run_batches([b2], np.array([t0 + 3 * 10**9], np.int64), cfg=cfg, return_seq=True)
    finally:
        COLS = saved
    assert vlog.shape[0] == 1 and vlog[0, 1] == 0          # verification passed (far levels ignored)
    assert st.si[3] < NT                                   # certified ask domain stays inside the grid


@requires_raw
def test_outputs_independent_of_query_set(book_hours):
    """Comparability across bar configurations: values at a query time are bit-identical whatever other queries
    (other bar sets) are evaluated in the same replay."""
    q1 = qgrid(5, 7, 5)
    rng = np.random.default_rng(3)
    extra = np.sort(rng.integers(q1[0], q1[-1], 3000)).astype(np.int64)
    q2 = np.unique(np.concatenate([q1, extra]))
    a, _ = run_batches([book_hours[5], book_hours[6]], q1)
    b, _ = run_batches([book_hours[5], book_hours[6]], q2)
    bb = b[np.searchsorted(q2, q1)]
    assert np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(bb, nan=-9e99))


# ------------------------------------------------------------------ D1: snapshot parts, client buffer, rebuilt tier
def _snap_table(parts):
    """parts: list of (recv, [(side, price, qty), ...]) of ONE snapshot (event_time 1000, last_update_id 50)."""
    import pyarrow as pa
    cols = {k: [] for k in ("received_time", "event_time", "transaction_time", "event_type", "first_update_id",
                            "final_update_id", "prev_final_update_id", "last_update_id", "side", "price", "quantity")}
    for recv, lv in parts:
        for sd, px, q in lv:
            for k, v in (("received_time", recv), ("event_time", 1000), ("transaction_time", None),
                         ("event_type", "snapshot"), ("first_update_id", None), ("final_update_id", None),
                         ("prev_final_update_id", None), ("last_update_id", 50), ("side", sd), ("price", px),
                         ("quantity", q)):
                cols[k].append(v)
    ints = ("received_time", "event_time", "transaction_time", "first_update_id", "final_update_id",
            "prev_final_update_id", "last_update_id")
    return pa.table({k: pa.array(v, pa.int64() if k in ints else pa.string()) for k, v in cols.items()})


def test_snapshot_written_in_two_halves_is_one_message():
    from chdds.book.messages import build_messages
    bids = [("bid", "100.00", "1.000"), ("bid", "99.90", "2.000")]
    asks = [("ask", "100.10", "3.000"), ("ask", "100.20", "4.000")]
    bm = build_messages(_snap_table([(5_000_000_000, bids), (6_000_000_000, asks)]))
    assert bm.m_recv.size == 1, "halves with different levels form one message"
    assert bm.m_recv[0] == 6_000_000_000, "available only when the last half arrived"
    assert bm.m_nbid[0] == 2 and bm.m_nask[0] == 2


def test_identical_snapshot_copies_keep_earliest_real_reception():
    from chdds.book.messages import build_messages
    lv = [("bid", "100.00", "1.000"), ("ask", "100.10", "3.000")]
    bm = build_messages(_snap_table([(1000 * 1_000_000, lv), (5_000_000_000, lv), (6_000_000_000, lv)]))
    assert bm.m_recv.size == 1 and bm.m_recv[0] == 5_000_000_000
    assert bm.m_nbid[0] == 1 and bm.m_nask[0] == 1


def _msg(rows):
    """Build a batch dict from [(recv, U, u, pu, [(side, tick, qty_e3), ...]), ...]; snapshots have U = -2 and L = u."""
    n = len(rows)
    b = dict(recv=np.array([r[0] for r in rows], np.int64), evt=np.array([r[0] // 1_000_000 for r in rows], np.int64),
             txn=np.array([r[0] // 1_000_000 for r in rows], np.int64),
             et=np.array([CH.ET_SNAPSHOT if r[1] == -2 else CH.ET_UPDATE for r in rows], np.int8),
             U=np.array([r[1] for r in rows], np.int64), u=np.array([r[2] for r in rows], np.int64),
             pu=np.array([r[3] for r in rows], np.int64),
             L=np.array([r[2] if r[1] == -2 else -1 for r in rows], np.int64),
             hash=np.arange(1, n + 1).astype(np.uint64), synth=np.zeros(n, bool))
    sizes = [len(r[4]) for r in rows]
    b["row_start"] = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    lv = [x for r in rows for x in r[4]]
    b["side"] = np.array([x[0] for x in lv], np.int8)
    b["idx"] = np.array([x[1] for x in lv], np.int64)
    b["qty"] = np.array([x[2] for x in lv], np.int64)
    b["off"] = np.zeros(len(lv), bool)
    return b


def _res(b, **kw):
    cst, ru, rh = CH.new_state()
    args = dict(heal=0, probe_only=0, legacy=0, start_after=0)
    args.update(kw)
    r = CH.resolve(b["recv"], b["et"], b["U"], b["u"], b["pu"], b["L"], b["hash"], cst, ru, rh, 0, 60 * 10**9,
                   args["heal"], args["probe_only"], args["legacy"], args["start_after"])
    return r, cst


def test_snapshot_connects_to_messages_received_before_it():
    # the update containing L was received BEFORE the snapshot: a live client buffers the stream while the REST call runs
    s = 10**9
    rows = [(1 * s, 10, 19, 9, [(0, 1000, 1000)]), (2 * s, 20, 29, 19, [(1, 1001, 1000)]),
            (3 * s, -2, 25, -1, [(0, 1000, 500), (1, 1001, 700)]), (4 * s, 30, 39, 29, [(0, 1000, 900)])]
    b = _msg(rows)
    (r_new, c_new), (r_old, c_old) = _res(b), _res(b, legacy=1)
    assert c_old[CH.ST_MODE] != CH.M_CHAIN, "old behaviour: the connecting update was already gone"
    assert c_new[CH.ST_MODE] == CH.M_CHAIN and c_new[CH.ST_LAST_U] == 39
    kinds = list(zip(r_new[2].tolist(), r_new[0].tolist()))
    assert kinds == [(CH.Q_ANCHOR, 2), (CH.Q_UPDATE, 1), (CH.Q_UPDATE, 3)]
    assert (r_new[1] >= 3 * s).all(), "buffered messages are applied at the snapshot's availability, never earlier"


def test_snapshot_bridges_unpublished_update():
    # vendor: the update with u == L is not published in the hour of the snapshot; the chain jumps pu == L
    s = 10**9
    rows = [(1 * s, -2, 10, -1, [(0, 1000, 500), (1, 1001, 700)]), (2 * s, 11, 19, 10, [(0, 1000, 600)]),
            (3 * s, -2, 29, -1, [(0, 1000, 650), (1, 1001, 700)]), (4 * s, 30, 39, 29, [(1, 1001, 800)])]
    b = _msg(rows)
    (r_new, c_new), (r_old, c_old) = _res(b), _res(b, legacy=1)
    assert c_new[CH.ST_LAST_U] == 39 and c_new[CH.ST_GAP_SINCE] < 0
    assert c_old[CH.ST_LAST_U] == 19 and c_old[CH.ST_GAP_SINCE] >= 0, "old behaviour: pending gap"
    assert CH.Q_ANCHOR in r_new[2][1:].tolist()


def _run_rebuilt(rows, queries, forced=False, probe_only=3, start_after=0):
    cfg = ReplayConfig(ring_size=1 << 12, book_mode="rebuilt", experiment_emit_healed=forced)
    b = _msg(rows)
    (res, cst) = _res(b, heal=2, probe_only=probe_only, start_after=start_after)
    st = ReplayState(cfg)
    pa_ = cfg.param_arrays()
    z = np.zeros(0, np.int64)
    zf = np.zeros(0)
    out = np.full((queries.size, len(cfg.columns())), np.nan)
    xo = np.zeros((queries.size, 4 * 3), np.int64)
    replay_kernel(res[0], res[1], res[2], res[3], b["evt"], b["txn"], b["synth"], b["row_start"], b["side"], b["idx"],
                  b["qty"], b["off"], z, z, zf, zf, z, z, zf, zf, zf, queries, out, xo, z, np.zeros((0, 8), np.int64),
                  *st.kernel_args(), pa_["ip"], pa_["bands_c"], pa_["ks"], pa_["win_ns"], pa_["win_mlofi"],
                  pa_["win_minmax"], pa_["tw_bands"], pa_["cost_q"], pa_["cost_d"], pa_["bwin_ns"], pa_["eswin_ns"],
                  pa_["mo_h_ns"], pa_["mowin_ns"], pa_["min_cov"], np.zeros((16, 9), np.int64))
    return out, xo, st, {c: i for i, c in enumerate(cfg.columns())}


def test_rebuilt_top_certificate_only_when_no_hidden_tick_remains():
    from chdds.book.replay import BS_REBUILT, BS_REBUILT_UNC
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100005, 1000)]),      # gap of 4 ticks: a better bid/ask could hide
            (2 * s, 2, 2, 1, [(0, 100002, 1000)]),                         # better bid, still ticks in between
            (3 * s, 3, 3, 2, [(1, 100003, 2000)])]                         # 1-tick spread: exact by protocol
    q = np.array([1, 2, 3], np.int64) * s + s // 2
    out, _, st, ci = _run_rebuilt(rows, q)
    assert out[0, ci["book_state"]] == BS_REBUILT_UNC and np.isnan(out[0, ci["mid"]])
    assert out[1, ci["book_state"]] == BS_REBUILT_UNC and np.isnan(out[1, ci["spread_bps"]])
    assert out[2, ci["book_state"]] == BS_REBUILT
    assert out[2, ci["best_bid"]] == pytest.approx(10000.2) and out[2, ci["best_ask"]] == pytest.approx(10000.3)


def test_rebuilt_certificate_uses_reported_ticks_between_best_levels():
    from chdds.book.replay import BS_REBUILT, BS_REBUILT_UNC
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100003, 1000)]),
            (2 * s, 2, 2, 1, [(0, 100001, 0)]),                            # bid tick reported empty
            (3 * s, 3, 3, 2, [(0, 100002, 0)])]                            # all ticks between are now known on the bid side
    q = np.array([2, 3], np.int64) * s + s // 2
    out, _, st, ci = _run_rebuilt(rows, q)
    assert out[0, ci["book_state"]] == BS_REBUILT_UNC
    assert st.si[RP_CERT_LO()] <= 100000, "best bid certified"
    assert st.si[RP_CERT_HI()] < 100003, "asks not reported in between: best ask stays uncertified"
    assert out[1, ci["book_state"]] == BS_REBUILT_UNC


def RP_CERT_LO():
    from chdds.book.replay import I_CERT_LO
    return I_CERT_LO


def RP_CERT_HI():
    from chdds.book.replay import I_CERT_HI
    return I_CERT_HI


def test_rebuilt_restarts_empty_after_a_true_gap():
    from chdds.book.replay import I_NEMPTY
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100001, 1000)]),
            (2 * s, 5, 5, 4, [(0, 99990, 1000)]),                           # message 2..4 lost
            (70 * s, 6, 6, 5, [(1, 100010, 1000)]),                         # timeout: restart from the earliest buffered
            (71 * s, 7, 7, 6, [(0, 100009, 1000)])]
    q = np.array([1 * s + s // 2, 71 * s + s // 2], np.int64)
    out, _, st, ci = _run_rebuilt(rows, q)
    assert st.si[I_NEMPTY] == 2, "initial start + restart after the gap"
    assert out[1, ci["book_state"]] == 9 and out[1, ci["best_bid"]] == pytest.approx(10000.9)
    # nothing from before the gap survives
    assert st.qb[100000] == 0 and st.qa[100001] == 0


def test_rebuilt_book_returns_to_anchored_tier_at_a_snapshot():
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100010, 1000)]),
            (2 * s, -2, 1, -1, [(0, 100005, 700), (1, 100006, 800)]),       # hidden levels revealed by the snapshot
            (3 * s, 2, 2, 1, [(1, 100006, 900)])]
    q = np.array([3 * s + s // 2], np.int64)
    out, _, st, ci = _run_rebuilt(rows, q, probe_only=0)
    assert out[0, ci["book_state"]] == BS_VALID
    assert out[0, ci["best_bid"]] == pytest.approx(10000.5) and st.qb[100000] == 0


def test_empty_book_independent_of_snapshots_in_probe_mode():
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100001, 1000)]),
            (2 * s, -2, 1, -1, [(0, 100000, 1), (1, 100001, 1)]),
            (3 * s, 2, 2, 1, [(0, 99999, 1000)])]
    out, _, st, ci = _run_rebuilt(rows, np.array([3 * s + s // 2], np.int64))
    assert st.qb[100000] == 1000, "probe book must ignore the snapshot"


def test_strict_mode_never_starts_from_an_empty_book():
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100001, 1000)]), (2 * s, 2, 2, 1, [(0, 100000, 900)])]
    (res, cst) = _res(_msg(rows))
    assert CH.Q_EMPTY not in res[2].tolist() and cst[CH.ST_MODE] == CH.M_NONE


def test_rebuilt_burst_of_holes_costs_one_timeout():
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100001, 1000)]),
            (2 * s, 5, 5, 4, [(0, 100000, 900)]),      # hole 2..4
            (3 * s, 9, 9, 8, [(0, 100000, 800)]),      # hole 6..8, observable since t = 3 s
            (70 * s, 10, 10, 9, [(1, 100001, 900)]),   # first timeout: restart, reach the second hole
            (71 * s, 11, 11, 10, [(1, 100001, 700)])]  # second hole is already older than 60 s: restart now, not at 130 s
    res, cst = _res(_msg(rows), heal=2, probe_only=3)
    assert res[2].tolist().count(CH.Q_EMPTY) == 3
    assert cst[CH.ST_LAST_U] == 11 and cst[CH.ST_GAP_SINCE] < 0


@requires_raw
def test_rebuilt_certified_values_equal_the_anchored_book(book_hours):
    """Real data: a book started empty inside the hour never sees a snapshot, and everything it emits equals the anchored book."""
    from chdds.book.replay import BS_REBUILT
    cfg_r = ReplayConfig(ring_size=1 << 16, book_mode="rebuilt")
    cols = cfg_r.columns()
    ci = {c: i for i, c in enumerate(cols)}
    T = P.hour_start_ns(P.HISTORY_START, 6) + 137 * 10**9
    q = np.arange(T + 10**9, P.hour_start_ns(P.HISTORY_START, 7), 5 * 10**9, dtype=np.int64)

    def run(heal, probe, start_after):
        st = ReplayState(cfg_r)
        pa_ = cfg_r.param_arrays()
        cst, ru, rh = CH.new_state()
        carry = None
        out = np.full((q.size, len(cols)), np.nan)
        z = np.zeros(0, np.int64)
        zf = np.zeros(0)
        for hh in (5, 6):
            batch, ncar = _concat_batch(carry, book_hours[hh])
            r = CH.resolve(batch["recv"], batch["et"], batch["U"], batch["u"], batch["pu"], batch["L"], batch["hash"], cst, ru, rh, ncar,
                           60 * 10**9, heal, probe, 0, start_after)
            carry = _take_pending(batch, r[-1])
            a = np.searchsorted(q, P.hour_start_ns(P.HISTORY_START, hh))
            b = np.searchsorted(q, P.hour_start_ns(P.HISTORY_START, hh + 1))
            o = np.full((b - a, len(cols)), np.nan)
            replay_kernel(r[0], r[1], r[2], r[3], batch["evt"], batch["txn"], batch["synth"], batch["row_start"], batch["side"], batch["idx"],
                          batch["qty"], batch["off"], z, z, zf, zf, z, z, zf, zf, zf, q[a:b], o, np.zeros((b - a, 0), np.int64), z,
                          np.zeros((0, 8), np.int64), *st.kernel_args(), pa_["ip"], pa_["bands_c"], pa_["ks"], pa_["win_ns"], pa_["win_mlofi"],
                          pa_["win_minmax"], pa_["tw_bands"], pa_["cost_q"], pa_["cost_d"], pa_["bwin_ns"], pa_["eswin_ns"], pa_["mo_h_ns"],
                          pa_["mowin_ns"], pa_["min_cov"], np.zeros((1000, 9), np.int64))
            out[a:b] = o
            rebase(st.cum, st.ring_c, st.si, st.es_cum, st.es_c, st.mo_cum, st.mo_c)
        return out

    ref = run(0, 0, 0)
    emp = run(2, 3, T)
    assert (ref[:, ci["book_state"]] == BS_VALID).all()
    assert (emp[:, ci["book_state"]] == BS_REBUILT).mean() > 0.99
    skip = ("book_state", "book_recv_age_ms", "book_mkt_age_ms", "cert_bid_floor_dist_bps", "cert_ask_ceiling_dist_bps")
    both = 0
    for j, c in enumerate(cols):
        if c in skip or "coverage" in c:
            continue
        m = np.isfinite(emp[:, j]) & np.isfinite(ref[:, j])
        both += int(m.sum())
        assert np.isfinite(ref[np.isfinite(emp[:, j]), j]).all() or "markout" in c or "eff_spread" in c, c
        np.testing.assert_allclose(emp[m, j], ref[m, j], rtol=1e-7, atol=1e-7, err_msg=c)
    assert both > 50000


def test_rebuilt_crossing_voids_the_certificate_until_it_is_earned_again():
    from chdds.book.replay import BS_EMPTY, BS_REBUILT, BS_SETTLE, I_QUAR
    s = 10**9
    rows = [(1 * s, 1, 1, 0, [(0, 100000, 1000), (1, 100001, 1000)]),
            (2 * s, 2, 2, 1, [(0, 100002, 1000)]),                   # bid above the ask: the stream itself is crossed
            (3 * s, 3, 3, 2, [(1, 100001, 0), (1, 100003, 1000)]),   # uncrossed again, one-tick spread, but inside the quarantine
            (604 * s, 4, 4, 3, [(0, 100002, 900)]),                  # quarantine over: restart, one side only, nothing certified
            (605 * s, 5, 5, 4, [(1, 100010, 500)]),                  # both sides, 8 ticks apart: not certifiable
            (606 * s, 6, 6, 5, [(1, 100003, 700)])]                  # adjacent best levels reported after the restart: certified
    q = np.array([1, 3, 604, 605, 606], np.int64) * s + s // 2
    out, _, st, ci = _run_rebuilt(rows, q)
    states = out[:, ci["book_state"]].tolist()
    assert states == [BS_REBUILT, BS_SETTLE, BS_EMPTY, 10, BS_REBUILT], states
    assert np.isnan(out[1, ci["spread_bps"]]) and np.isnan(out[3, ci["spread_bps"]])
    assert st.si[I_QUAR] == 0
    assert st.qa[100001] == 0 and st.qb[100000] == 0, "nothing from before the restart survives"


@requires_raw
def test_rebuilt_checkpoint_resume_equivalence(book_hours, tmp_path):
    cfg = ReplayConfig(ring_size=1 << 16, book_mode="rebuilt")
    T = P.hour_start_ns(P.HISTORY_START, 5) + 77 * 10**9
    q = qgrid(5, 8, 3)
    kw = dict(cfg=cfg, heal=2, probe=3, start_after=T)
    full, _ = run_batches([book_hours[5], book_hours[6], book_hours[7]], q, **kw)
    _, st, cs, _, _ = run_batches([book_hours[5], book_hours[6]], q[q < P.hour_start_ns(P.HISTORY_START, 7) - 1], return_seq=True, **kw)
    st.save(tmp_path / "s.npz")
    st2 = ReplayState(cfg)
    st2.load_into(tmp_path / "s.npz")
    q7 = q[q >= P.hour_start_ns(P.HISTORY_START, 7)]
    out2, _ = run_batches([book_hours[7]], q7, state=st2, cstate=(cs[0].copy(), cs[1].copy(), cs[2].copy()), **kw)
    assert np.array_equal(np.nan_to_num(full[np.searchsorted(q, q7)], nan=-9e99), np.nan_to_num(out2, nan=-9e99))


def test_markouts_do_not_cross_a_batch_boundary_early():
    """A maturity after the end of a batch is evaluated in the next batch, on the state of that moment."""
    from chdds.book.replay import ReplayConfig as RC
    s = 10**9
    cfg = RC(windows_s=(1.0,), mlofi_windows_s=(), minmax_max_window_s=0.0, ring_size=1 << 12, max_age_ms=10_000, basis_windows_s=(),
             es_windows_s=(5.0,), markout_h_s=(1.0,), markout_windows_s=(4.0,), tw_band_bps=())
    cols = cfg.columns()
    ci = {c: i for i, c in enumerate(cols)}
    t0 = 1000 * s
    first = [(t0, -2, 1, -1, [(0, 100000, 1000), (1, 100001, 1000)]), (t0 + 1 * s, 2, 2, 1, [(0, 100000, 900)]), (t0 + 7 * s, 3, 3, 2, [(0, 100000, 800)])]
    second = [(t0 + 7 * s + s // 2, 4, 4, 3, [(0, 100010, 1000), (1, 100001, 0), (1, 100011, 1000)])]   # mid moves up by 10 ticks
    b1, b2 = _msg(first), _msg(second)
    tr_t = np.array([t0 + 7 * s + s // 10], np.int64)                 # trade close to the end of the first batch, maturity in the second
    z = np.zeros(0, np.int64)
    zf = np.zeros(0)
    st = ReplayState(cfg)
    pa_ = cfg.param_arrays()
    cst, ru, rh = CH.new_state()
    outs = []
    for b, trd, q in ((b1, (tr_t, tr_t // 10**6, np.array([10000.0]), np.array([1.0]), np.array([1.0])), z),
                      (b2, (z, z, zf, zf, zf), np.array([t0 + 9 * s], np.int64))):
        r = CH.resolve(b["recv"], b["et"], b["U"], b["u"], b["pu"], b["L"], b["hash"], cst, ru, rh, 0, 60 * s)
        o = np.full((q.size, len(cols)), np.nan)
        replay_kernel(r[0], r[1], r[2], r[3], b["evt"], b["txn"], b["synth"], b["row_start"], b["side"], b["idx"], b["qty"], b["off"], z, z, zf, zf, *trd, q, o,
                      np.zeros((q.size, 0), np.int64), z, np.zeros((0, 8), np.int64), *st.kernel_args(), pa_["ip"], pa_["bands_c"], pa_["ks"], pa_["win_ns"], pa_["win_mlofi"],
                      pa_["win_minmax"], pa_["tw_bands"], pa_["cost_q"], pa_["cost_d"], pa_["bwin_ns"], pa_["eswin_ns"], pa_["mo_h_ns"], pa_["mowin_ns"], pa_["min_cov"], np.zeros((16, 9), np.int64))
        outs.append(o)
        rebase(st.cum, st.ring_c, st.si, st.es_cum, st.es_c, st.mo_cum, st.mo_c)
    o = outs[1][0]
    assert o[ci["markout_1s_n_4s"]] == 1
    assert o[ci["markout_1s_qw_bps_4s"]] == pytest.approx(1e4 * np.log(10001.05 / 10000.05), rel=1e-9), "evaluated after the move of the next batch"

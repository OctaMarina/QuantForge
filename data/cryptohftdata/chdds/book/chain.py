"""Causal L2 sequence resolver for Binance USD-M diff-depth messages.

Messages are visited in reception (availability) order. A depth update is *applicable* only
when its prev_final_update_id equals the final_update_id of the last applied message. Updates
arriving early (pu > last_u, e.g. a fresh connection overtaking a lagging one) are buffered and
applied at the reception time of the message that completes the chain — never earlier.
A pending gap that is not filled within `timeout_ns` (reception clock) is a true gap: the book
becomes unanchored until a snapshot connects (Binance rule: first update with U <= L <= u, or
pu == L). Updates with u <= last_u are duplicates: verified against the content hash of the
applied copy (identical -> dropped; different -> conflict, invalidates the chain).

Snapshots while chained: L == last_u -> verification point (state must equal snapshot content);
L < last_u -> stale (ignored, logged); L > last_u -> re-anchor.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..audit.kernels import ET_SNAPSHOT, ET_UPDATE

# modes
M_NONE, M_CHAIN, M_AWAIT = 0, 1, 2
# sequence kinds (what the replay must do)
Q_UPDATE, Q_ANCHOR, Q_VERIFY, Q_INVALIDATE, Q_HEAL, Q_EMPTY = 0, 1, 2, 3, 4, 5
# per-message actions
A_APPLY, A_APPLY_BUFFERED, A_ANCHOR, A_VERIFY, A_SNAP_STALE, A_DUP_IDENT, A_DUP_CONFLICT, A_COVERED, \
    A_BUFFERED_DROPPED, A_UNANCHORED, A_DUP_UNKNOWN, A_OTHER = range(12)
ACTION_NAMES = ["apply", "apply_buffered", "anchor", "verify_snapshot", "snapshot_stale", "dup_identical",
                "dup_conflict", "covered_by_snapshot", "buffered_then_dropped", "unanchored", "dup_unverifiable",
                "other"]
# event kinds
E_GAP, E_ANCHOR_FAIL, E_CONFLICT, E_SHADOW_GAP, E_RESOLVED, E_ANCHOR_OK = range(6)
EVENT_NAMES = ["true_gap", "anchor_noconnect", "dup_conflict", "stream_gap_unanchored", "gap_resolved_by_buffer",
               "anchor_connected"]

# state layout
ST_MODE, ST_LAST_U, ST_ANCHOR_L, ST_SHADOW_U, ST_GAP_SINCE, ST_RING_POS, ST_ANCHOR_IDX, ST_ANCHOR_TIME, \
    ST_VER_IDX, ST_VER_L, ST_NVER_UNRESOLVED = range(11)
NST = 11
RING = 16384


def new_state():
    st = np.zeros(NST, np.int64)
    st[ST_MODE] = M_NONE
    st[ST_LAST_U] = -1
    st[ST_ANCHOR_L] = -1
    st[ST_SHADOW_U] = -1
    st[ST_GAP_SINCE] = -1
    st[ST_VER_IDX] = -1
    st[ST_VER_L] = -1
    return st, np.full(RING, -1, np.int64), np.zeros(RING, np.uint64)


@njit(cache=True)
def _ring_find(ring_u, ring_h, u):
    for j in range(ring_u.shape[0]):
        if ring_u[j] == u:
            return j
    return -1


@njit(cache=True)
def resolve(recv, et, U, u, pu, L, hsh, st, ring_u, ring_h, n_carried, timeout_ns, heal=0, probe_only=0, legacy=0,
            start_after=0):
    """Process messages [0, n) in reception order; the first `n_carried` are pending messages
    carried from the previous batch (already buffered).

    heal: 0 strict; 1 EXPERIMENT continue the old state over a gap (rejected); 2 rebuild: start from an EMPTY book
          (first message at/after `start_after`, and again after every true gap). A book started empty holds only levels
          reported since the restart, so it never holds a wrong size; what it lacks is tracked by the replay.
    probe_only: 2 = ignore every snapshot after the first anchor; 3 = ignore all snapshots (independent empty book).
    legacy: 1 reproduces the pre-2026-11-04 behaviour (no client-side buffer while unanchored; a snapshot ahead of the
          chain cannot bridge an unpublished update) — kept only to quantify the effect of the corrections.
    Returns
      seq_msg, seq_time, seq_kind : application sequence for the replay
      action                      : per-message action code
      ev_kind, ev_t0, ev_t1, ev_u0, ev_u1 : events
      pending                     : indices of still-buffered messages to carry forward
    """
    n = recv.shape[0]
    seq_msg = np.empty(2 * n + 4, np.int64)
    seq_time = np.empty(2 * n + 4, np.int64)
    seq_kind = np.empty(2 * n + 4, np.int8)
    seq_aux = np.full(2 * n + 4, -1, np.int64)
    ns = 0
    st[ST_VER_IDX] = -1    # deferred verifications do not survive a batch boundary (rows not carried)
    action = np.full(n, A_OTHER, np.int8)
    ev_kind = np.empty(n + 4, np.int8)
    ev_t0 = np.empty(n + 4, np.int64)
    ev_t1 = np.empty(n + 4, np.int64)
    ev_u0 = np.empty(n + 4, np.int64)
    ev_u1 = np.empty(n + 4, np.int64)
    ne = 0
    pend = np.empty(n + 1, np.int64)
    npend = 0
    for i in range(n_carried):
        pend[npend] = i
        npend += 1
    for i in range(n_carried, n):
        r = recv[i]
        # ---- timeout of a pending gap (checked on every new reception)
        if st[ST_MODE] != M_NONE and st[ST_GAP_SINCE] >= 0 and r - st[ST_GAP_SINCE] > timeout_ns:
            if st[ST_MODE] == M_CHAIN:
                ev_kind[ne] = E_GAP
            else:
                ev_kind[ne] = E_ANCHOR_FAIL
            ev_t0[ne] = st[ST_GAP_SINCE]
            ev_t1[ne] = r
            ev_u0[ne] = st[ST_LAST_U] if st[ST_MODE] == M_CHAIN else st[ST_ANCHOR_L]
            mnp = -1
            for j in range(npend):
                if mnp < 0 or pu[pend[j]] < mnp:
                    mnp = pu[pend[j]]
            ev_u1[ne] = mnp
            ne += 1
            if heal == 2 and st[ST_MODE] == M_CHAIN and npend > 0:
                # rebuild: discard the state, restart EMPTY from the earliest buffered message (available now)
                jmin = 0
                for j in range(npend):
                    if pu[pend[j]] < pu[pend[jmin]]:
                        jmin = j
                seq_msg[ns] = -1
                seq_time[ns] = r
                seq_kind[ns] = Q_EMPTY
                ns += 1
                st[ST_LAST_U] = pu[pend[jmin]]
                st[ST_GAP_SINCE] = -1
                st[ST_VER_IDX] = -1
            elif heal == 1 and st[ST_MODE] == M_CHAIN and npend > 0:
                # EXPERIMENTAL (never the default): accept the jump and continue from the earliest pending message;
                # the replay marks the state healed_unverified until a snapshot verifies or rebuilds it.
                jmin = 0
                for j in range(npend):
                    if pu[pend[j]] < pu[pend[jmin]]:
                        jmin = j
                seq_msg[ns] = -1
                seq_time[ns] = r
                seq_kind[ns] = Q_HEAL
                ns += 1
                st[ST_LAST_U] = pu[pend[jmin]]
                st[ST_GAP_SINCE] = -1
                if st[ST_VER_L] < st[ST_LAST_U]:
                    st[ST_VER_IDX] = -1
            else:
                seq_msg[ns] = -1
                seq_time[ns] = st[ST_GAP_SINCE]
                seq_kind[ns] = Q_INVALIDATE
                ns += 1
            continuing = (heal >= 1) and st[ST_MODE] == M_CHAIN and npend > 0
            if not continuing:
                # shadow chain continues from the newest buffered message
                for j in range(npend):
                    if u[pend[j]] > st[ST_SHADOW_U]:
                        st[ST_SHADOW_U] = u[pend[j]]
                if legacy == 1:
                    for j in range(npend):
                        action[pend[j]] = A_BUFFERED_DROPPED
                    npend = 0
                st[ST_MODE] = M_NONE
                st[ST_GAP_SINCE] = -1
                st[ST_VER_IDX] = -1
        if et[i] == ET_SNAPSHOT:
            Lk = L[i]
            if probe_only == 3:
                action[i] = A_SNAP_STALE      # EXPERIMENT: this book never sees a snapshot (not even as a verification)
                continue
            if st[ST_MODE] == M_CHAIN and st[ST_GAP_SINCE] < 0 and Lk == st[ST_LAST_U]:
                action[i] = A_VERIFY
                seq_msg[ns] = i
                seq_time[ns] = r
                seq_kind[ns] = Q_VERIFY
                ns += 1
                continue
            if st[ST_MODE] == M_CHAIN and Lk < st[ST_LAST_U]:
                action[i] = A_SNAP_STALE
                continue
            if probe_only == 2 and st[ST_LAST_U] >= 0:
                action[i] = A_SNAP_STALE      # EXPERIMENT: snapshots are not used by this book
                continue
            if probe_only == 1 and st[ST_LAST_U] >= 0:
                # EXPERIMENT: after the first anchor, snapshots never re-anchor (emulates a period without anchors);
                # each one becomes a deferred probe evaluated when the (healed) chain reaches L
                if Lk > st[ST_LAST_U]:
                    st[ST_VER_IDX] = i
                    st[ST_VER_L] = Lk
                    action[i] = A_VERIFY
                else:
                    action[i] = A_SNAP_STALE
                continue
            if st[ST_MODE] == M_CHAIN and st[ST_GAP_SINCE] < 0 and Lk > st[ST_LAST_U]:
                # snapshot ahead of an intact chain (typically a synthetic receive time): verify when the chain
                # reaches L instead of re-anchoring
                if st[ST_VER_IDX] >= 0:
                    st[ST_NVER_UNRESOLVED] += 1
                st[ST_VER_IDX] = i
                st[ST_VER_L] = Lk
                action[i] = A_VERIFY
                continue
            # anchor (from none / await / chain with gap or behind)
            action[i] = A_ANCHOR
            seq_msg[ns] = i
            seq_time[ns] = r
            seq_kind[ns] = Q_ANCHOR
            ns += 1
            st[ST_MODE] = M_AWAIT
            st[ST_ANCHOR_L] = Lk
            st[ST_LAST_U] = Lk
            st[ST_GAP_SINCE] = r
            st[ST_ANCHOR_IDX] = i
            st[ST_ANCHOR_TIME] = r
            # drop buffered messages already covered by the snapshot
            k = 0
            for j in range(npend):
                if u[pend[j]] <= Lk:
                    action[pend[j]] = A_COVERED
                else:
                    pend[k] = pend[j]
                    k += 1
            npend = k
        elif et[i] == ET_UPDATE:
            # shadow chain (raw stream continuity, independent of anchoring)
            if st[ST_SHADOW_U] >= 0 and u[i] > st[ST_SHADOW_U] and pu[i] != st[ST_SHADOW_U] and st[ST_MODE] == M_NONE:
                ev_kind[ne] = E_SHADOW_GAP
                ev_t0[ne] = r
                ev_t1[ne] = r
                ev_u0[ne] = st[ST_SHADOW_U]
                ev_u1[ne] = pu[i]
                ne += 1
            if u[i] > st[ST_SHADOW_U]:
                st[ST_SHADOW_U] = u[i]
            if st[ST_MODE] == M_NONE:
                if heal == 1 and st[ST_LAST_U] >= 0:
                    seq_msg[ns] = -1
                    seq_time[ns] = r
                    seq_kind[ns] = Q_HEAL
                    ns += 1
                    st[ST_MODE] = M_CHAIN
                    st[ST_LAST_U] = pu[i]
                elif heal == 2 and r >= start_after:
                    seq_msg[ns] = -1
                    seq_time[ns] = r
                    seq_kind[ns] = Q_EMPTY
                    ns += 1
                    st[ST_MODE] = M_CHAIN
                    st[ST_LAST_U] = pu[i]
                    npend = 0
                else:
                    action[i] = A_UNANCHORED
                    if legacy == 0:
                        # client-side buffer: a live client keeps the stream while it waits for the REST snapshot, so a
                        # later snapshot can connect to messages received BEFORE it (bounded by timeout_ns)
                        pend[npend] = i
                        npend += 1
                        if (i & 511) == 0:
                            k = 0
                            for j in range(npend):
                                if recv[pend[j]] >= r - timeout_ns:
                                    pend[k] = pend[j]
                                    k += 1
                            npend = k
                    continue
            if u[i] <= st[ST_LAST_U]:
                if st[ST_MODE] == M_AWAIT and u[i] <= st[ST_ANCHOR_L]:
                    action[i] = A_COVERED
                    continue
                j = _ring_find(ring_u, ring_h, u[i])
                if j < 0:
                    action[i] = A_DUP_UNKNOWN
                elif ring_h[j] == hsh[i]:
                    action[i] = A_DUP_IDENT
                else:
                    action[i] = A_DUP_CONFLICT
                    ev_kind[ne] = E_CONFLICT
                    ev_t0[ne] = r
                    ev_t1[ne] = r
                    ev_u0[ne] = u[i]
                    ev_u1[ne] = u[i]
                    ne += 1
                    seq_msg[ns] = -1
                    seq_time[ns] = r
                    seq_kind[ns] = Q_INVALIDATE
                    ns += 1
                    st[ST_MODE] = M_NONE
                    npend = 0
                    st[ST_GAP_SINCE] = -1
                continue
            # candidate for application or buffering
            pend[npend] = i
            npend += 1
            action[i] = A_APPLY_BUFFERED
        else:
            action[i] = A_OTHER
            continue
        # ---- drain: apply every buffered message that connects
        progress = True
        while progress and npend > 0 and st[ST_MODE] != M_NONE:
            progress = False
            for j in range(npend):
                m = pend[j]
                ok = False
                if st[ST_MODE] == M_AWAIT:
                    Lk = st[ST_ANCHOR_L]
                    ok = (U[m] <= Lk and Lk <= u[m]) or pu[m] == Lk
                else:
                    ok = pu[m] == st[ST_LAST_U]
                    if (not ok) and legacy == 0 and probe_only == 0 and st[ST_VER_IDX] >= 0:
                        Lv = st[ST_VER_L]
                        if Lv > st[ST_LAST_U] and ((U[m] <= Lv and Lv <= u[m]) or pu[m] == Lv):
                            # the chain cannot reach L (the vendor does not publish the update with u == L in the hour
                            # of the snapshot, or messages are missing): the snapshot replaces the state (Binance rule)
                            sv = st[ST_VER_IDX]
                            action[sv] = A_ANCHOR
                            seq_msg[ns] = sv
                            seq_time[ns] = r
                            seq_kind[ns] = Q_ANCHOR
                            ns += 1
                            st[ST_VER_IDX] = -1
                            st[ST_MODE] = M_AWAIT
                            st[ST_ANCHOR_L] = Lv
                            st[ST_LAST_U] = Lv
                            st[ST_ANCHOR_IDX] = sv
                            st[ST_ANCHOR_TIME] = r
                            ok = True
                if ok:
                    vpost = -1
                    if st[ST_MODE] == M_CHAIN and st[ST_VER_IDX] >= 0:
                        Lv = st[ST_VER_L]
                        if pu[m] == Lv:
                            seq_msg[ns] = st[ST_VER_IDX]
                            seq_time[ns] = r
                            seq_kind[ns] = Q_VERIFY
                            seq_aux[ns] = -1
                            ns += 1
                            st[ST_VER_IDX] = -1
                        elif U[m] <= Lv and Lv <= u[m]:
                            vpost = st[ST_VER_IDX]
                            st[ST_VER_IDX] = -1
                        elif U[m] > Lv:
                            st[ST_NVER_UNRESOLVED] += 1
                            st[ST_VER_IDX] = -1
                    if st[ST_MODE] == M_AWAIT:
                        ev_kind[ne] = E_ANCHOR_OK
                        ev_t0[ne] = st[ST_ANCHOR_TIME]
                        ev_t1[ne] = r
                        ev_u0[ne] = st[ST_ANCHOR_L]
                        ev_u1[ne] = u[m]
                        ne += 1
                    elif st[ST_GAP_SINCE] >= 0:
                        ev_kind[ne] = E_RESOLVED
                        ev_t0[ne] = st[ST_GAP_SINCE]
                        ev_t1[ne] = r
                        ev_u0[ne] = st[ST_LAST_U]
                        ev_u1[ne] = u[m]
                        ne += 1
                    seq_msg[ns] = m
                    seq_time[ns] = r if r > recv[m] else recv[m]
                    seq_kind[ns] = Q_UPDATE
                    ns += 1
                    if vpost >= 0:
                        seq_msg[ns] = vpost
                        seq_time[ns] = seq_time[ns - 1]
                        seq_kind[ns] = Q_VERIFY
                        seq_aux[ns] = m
                        ns += 1
                    action[m] = A_APPLY if (recv[m] == r) else A_APPLY_BUFFERED
                    st[ST_LAST_U] = u[m]
                    st[ST_MODE] = M_CHAIN
                    pos = st[ST_RING_POS]
                    ring_u[pos] = u[m]
                    ring_h[pos] = hsh[m]
                    st[ST_RING_POS] = (pos + 1) % ring_u.shape[0]
                    pend[j] = pend[npend - 1]
                    npend -= 1
                    progress = True
                    break
        # remove buffered messages made obsolete (u <= last_u) after draining
        if npend > 0:
            k = 0
            for j in range(npend):
                if u[pend[j]] <= st[ST_LAST_U]:
                    action[pend[j]] = A_DUP_UNKNOWN
                else:
                    pend[k] = pend[j]
                    k += 1
            npend = k
        if st[ST_MODE] == M_CHAIN:
            if npend == 0:
                st[ST_GAP_SINCE] = -1
            elif st[ST_GAP_SINCE] < 0:
                st[ST_GAP_SINCE] = r
                if heal == 2:
                    # rebuild: a hole is timed from when it became observable (first message waiting behind it), so a burst
                    # of holes costs one timeout and not one per hole; the rule itself (60 s of reception time) is unchanged
                    for j in range(npend):
                        if recv[pend[j]] < st[ST_GAP_SINCE]:
                            st[ST_GAP_SINCE] = recv[pend[j]]
    if st[ST_VER_IDX] >= 0:
        st[ST_NVER_UNRESOLVED] += 1
        st[ST_VER_IDX] = -1
    return (seq_msg[:ns], seq_time[:ns], seq_kind[:ns], seq_aux[:ns], action, ev_kind[:ne], ev_t0[:ne],
            ev_t1[:ne], ev_u0[:ne], ev_u1[:ne], pend[:npend].copy())

"""Numba kernels used by the exhaustive per-file audit (and reused by the L2 replay)."""
from __future__ import annotations

import numpy as np
from numba import njit

# event_type codes
ET_UPDATE, ET_SNAPSHOT, ET_OTHER = 0, 1, 2
# side codes
SD_BID, SD_ASK, SD_NOOP, SD_OTHER = 0, 1, 2, 3


@njit(cache=True)
def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def group_book_messages(recv, etype, U, u, pu, L, evt, txn, side, price, qty):
    """Group consecutive rows into messages. Key = (recv, etype, u). Rows of one message must be
    contiguous in file order (non-contiguity is detected later as a duplicate key).

    Returns arrays sized n_msgs:
      start, end, nbid, nask, nnoop, nzero, hash (order-independent content hash of levels),
      inconsistent (U/pu/L/evt/txn vary inside the message), duplevel (same side+price twice),
      best_bid, best_ask (max bid / min ask price among positive-quantity rows), min_bid, max_ask.
    """
    n = recv.shape[0]
    starts = np.empty(n + 1, np.int64)
    m = 0
    for i in range(n):
        if i == 0 or recv[i] != recv[i - 1] or etype[i] != etype[i - 1] or u[i] != u[i - 1]:
            starts[m] = i
            m += 1
    starts[m] = n
    nbid = np.zeros(m, np.int32)
    nask = np.zeros(m, np.int32)
    nnoop = np.zeros(m, np.int32)
    nzero = np.zeros(m, np.int32)
    h = np.zeros(m, np.uint64)
    inconsistent = np.zeros(m, np.bool_)
    duplevel = np.zeros(m, np.bool_)
    best_bid = np.full(m, -1, np.int64)
    best_ask = np.full(m, -1, np.int64)
    min_bid = np.full(m, -1, np.int64)
    max_ask = np.full(m, -1, np.int64)
    for k in range(m):
        s = starts[k]
        e = starts[k + 1]
        acc = np.uint64(0)
        keys = np.empty(e - s, np.int64)
        for i in range(s, e):
            if U[i] != U[s] or pu[i] != pu[s] or L[i] != L[s] or evt[i] != evt[s] or txn[i] != txn[s]:
                inconsistent[k] = True
            sd = side[i]
            if sd == SD_BID:
                nbid[k] += 1
            elif sd == SD_ASK:
                nask[k] += 1
            else:
                nnoop[k] += 1
            if qty[i] == 0:
                nzero[k] += 1
            else:
                if sd == SD_BID:
                    if best_bid[k] < 0 or price[i] > best_bid[k]:
                        best_bid[k] = price[i]
                    if min_bid[k] < 0 or price[i] < min_bid[k]:
                        min_bid[k] = price[i]
                elif sd == SD_ASK:
                    if best_ask[k] < 0 or price[i] < best_ask[k]:
                        best_ask[k] = price[i]
                    if max_ask[k] < 0 or price[i] > max_ask[k]:
                        max_ask[k] = price[i]
            x = np.uint64(sd) * np.uint64(0x100000001B3) ^ np.uint64(price[i]) * np.uint64(0x9E3779B1) \
                ^ splitmix64(np.uint64(qty[i]) + np.uint64(sd) * np.uint64(7919))
            acc += splitmix64(x)
            keys[i - s] = price[i] * 4 + sd
        h[k] = acc
        if e - s > 1:
            ks = np.sort(keys)
            for j in range(1, ks.shape[0]):
                if ks[j] == ks[j - 1]:
                    duplevel[k] = True
                    break
    return starts[:m + 1].copy(), nbid, nask, nnoop, nzero, h, inconsistent, duplevel, best_bid, best_ask, min_bid, max_ask


@njit(cache=True)
def group_copies(order, recv, etype, U, u, pu, L, evt, txn, side, price, qty):
    """Group rows (visited in `order` = lexsort by etype, u, recv, row) into reception groups
    (etype, u, recv) and copies (etype, u). Rows of a group need not be contiguous in the file.

    Returns per-group arrays (g_*) and per-message arrays (canonical copy = earliest reception):
      g_start/g_end index into `order`; g_minrow = first file row of the group.
      msg_group = index of the canonical group; msg_ncopies; msg_copies_identical; msg_copy_conflict.
    """
    n = order.shape[0]
    gstarts = np.empty(n + 1, np.int64)
    ng = 0
    for j in range(n):
        i = order[j]
        if j == 0:
            gstarts[ng] = j
            ng += 1
        else:
            p = order[j - 1]
            if etype[i] != etype[p] or u[i] != u[p] or recv[i] != recv[p]:
                gstarts[ng] = j
                ng += 1
    gstarts[ng] = n
    g_nbid = np.zeros(ng, np.int32)
    g_nask = np.zeros(ng, np.int32)
    g_nnoop = np.zeros(ng, np.int32)
    g_nzero = np.zeros(ng, np.int32)
    g_hash = np.zeros(ng, np.uint64)
    g_incons = np.zeros(ng, np.bool_)
    g_duplev = np.zeros(ng, np.bool_)
    g_minrow = np.empty(ng, np.int64)
    g_bbid = np.full(ng, -1, np.int64)
    g_bask = np.full(ng, -1, np.int64)
    g_mbid = np.full(ng, -1, np.int64)
    g_mask = np.full(ng, -1, np.int64)
    for k in range(ng):
        s = gstarts[k]
        e = gstarts[k + 1]
        i0 = order[s]
        acc = np.uint64(0)
        mr = i0
        keys = np.empty(e - s, np.int64)
        for j in range(s, e):
            i = order[j]
            if i < mr:
                mr = i
            if U[i] != U[i0] or pu[i] != pu[i0] or L[i] != L[i0] or evt[i] != evt[i0] or txn[i] != txn[i0]:
                g_incons[k] = True
            sd = side[i]
            if sd == SD_BID:
                g_nbid[k] += 1
            elif sd == SD_ASK:
                g_nask[k] += 1
            else:
                g_nnoop[k] += 1
            if qty[i] == 0:
                g_nzero[k] += 1
            else:
                if sd == SD_BID:
                    if g_bbid[k] < 0 or price[i] > g_bbid[k]:
                        g_bbid[k] = price[i]
                    if g_mbid[k] < 0 or price[i] < g_mbid[k]:
                        g_mbid[k] = price[i]
                elif sd == SD_ASK:
                    if g_bask[k] < 0 or price[i] < g_bask[k]:
                        g_bask[k] = price[i]
                    if g_mask[k] < 0 or price[i] > g_mask[k]:
                        g_mask[k] = price[i]
            x = np.uint64(sd) * np.uint64(0x100000001B3) ^ np.uint64(price[i]) * np.uint64(0x9E3779B1) \
                ^ splitmix64(np.uint64(qty[i]) + np.uint64(sd) * np.uint64(7919))
            acc += splitmix64(x)
            keys[j - s] = price[i] * 4 + sd
        g_hash[k] = acc
        g_minrow[k] = mr
        if e - s > 1:
            ks = np.sort(keys)
            for t in range(1, ks.shape[0]):
                if ks[t] == ks[t - 1]:
                    g_duplev[k] = True
                    break
    # copies: consecutive groups with equal (etype, u); first group has the earliest recv (sorted)
    mstarts = np.empty(ng + 1, np.int64)
    nm = 0
    for k in range(ng):
        i = order[gstarts[k]]
        if k == 0:
            mstarts[nm] = k
            nm += 1
        else:
            p = order[gstarts[k - 1]]
            if etype[i] != etype[p] or u[i] != u[p]:
                mstarts[nm] = k
                nm += 1
    mstarts[nm] = ng
    msg_group = np.empty(nm, np.int64)
    msg_ncopies = np.empty(nm, np.int32)
    msg_ident = np.ones(nm, np.bool_)
    msg_synth_only = np.zeros(nm, np.bool_)
    for t in range(nm):
        a = mstarts[t]
        b = mstarts[t + 1]
        # canonical copy: earliest reception whose received_time is not synthetic (== event_time*1e6)
        chosen = -1
        for k in range(a, b):
            i = order[gstarts[k]]
            if recv[i] != evt[i] * 1000000:
                chosen = k
                break
        if chosen < 0:
            chosen = a
            msg_synth_only[t] = True
        msg_group[t] = chosen
        msg_ncopies[t] = b - a
        for k in range(a + 1, b):
            if (g_hash[k] != g_hash[a] or g_nbid[k] != g_nbid[a] or g_nask[k] != g_nask[a]
                    or g_nnoop[k] != g_nnoop[a]):
                msg_ident[t] = False
    return (gstarts[:ng + 1].copy(), g_nbid, g_nask, g_nnoop, g_nzero, g_hash, g_incons, g_duplev, g_minrow,
            g_bbid, g_bask, g_mbid, g_mask, msg_group, msg_ncopies, msg_ident, msg_synth_only)


# chain classification codes per message
CH_FIRST = 0          # first update message in file with no prior reference
CH_OK = 1             # pu == previous u
CH_SNAP = 2           # snapshot message
CH_CONNECT = 3        # first update after a snapshot satisfying U <= L <= u or pu == L
CH_COVERED = 4        # update with u <= snapshot L (already contained in snapshot) -> drop
CH_DUP = 5            # u == previous u (retransmission candidate)
CH_STALE = 6          # u < previous u (out of order / replay)
CH_BREAK = 7          # pu != previous u and u > previous u: missing updates
CH_NOCONNECT = 8      # first update after snapshot that neither connects nor is covered (gap after snapshot)


@njit(cache=True)
def chain_classify(m_etype, m_U, m_u, m_pu, m_L, init_last_u, init_mode):
    """Classify each message against the Binance USD-M sequencing rules.

    init_mode: 0 = no reference, 1 = reference is an update chain ending at init_last_u,
               2 = reference is a snapshot with lastUpdateId = init_last_u (not yet connected).
    Returns per-message code and the reference value used, plus final (last_u, mode).
    """
    m = m_etype.shape[0]
    code = np.empty(m, np.int8)
    ref = np.full(m, -1, np.int64)
    last_u = init_last_u
    mode = init_mode
    for k in range(m):
        if m_etype[k] == ET_SNAPSHOT:
            code[k] = CH_SNAP
            ref[k] = last_u
            last_u = m_L[k]
            mode = 2
            continue
        if m_etype[k] != ET_UPDATE:
            code[k] = CH_BREAK
            continue
        ref[k] = last_u
        if mode == 0:
            code[k] = CH_FIRST
            last_u = m_u[k]
            mode = 1
        elif mode == 2:
            if m_u[k] <= last_u:
                code[k] = CH_COVERED
            elif (m_U[k] <= last_u and last_u <= m_u[k]) or m_pu[k] == last_u:
                code[k] = CH_CONNECT
                last_u = m_u[k]
                mode = 1
            else:
                code[k] = CH_NOCONNECT
                last_u = m_u[k]
                mode = 1
        else:
            if m_pu[k] == last_u:
                code[k] = CH_OK
                last_u = m_u[k]
            elif m_u[k] == last_u:
                code[k] = CH_DUP
            elif m_u[k] < last_u:
                code[k] = CH_STALE
            else:
                code[k] = CH_BREAK
                last_u = m_u[k]
    return code, ref, last_u, mode


@njit(cache=True)
def id_gaps(sorted_unique_ids):
    """Return indices i where ids[i+1] - ids[i] > 1."""
    n = sorted_unique_ids.shape[0]
    cnt = 0
    for i in range(n - 1):
        if sorted_unique_ids[i + 1] - sorted_unique_ids[i] > 1:
            cnt += 1
    out = np.empty(cnt, np.int64)
    j = 0
    for i in range(n - 1):
        if sorted_unique_ids[i + 1] - sorted_unique_ids[i] > 1:
            out[j] = i
            j += 1
    return out

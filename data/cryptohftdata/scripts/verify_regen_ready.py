"""Checks required before regenerating the dataset, on selected intervals (no full scan). Writes derived/regen_check/report.json.

  markout   independent recomputation of matured markouts across hourly batch boundaries, on an interval (2026-09-17) that was
            not used while correcting the code, plus a second regime (2026-06-14)
  strict    corrected code vs the existing strict values on 2025-06-28/29: only effective spread and markout columns may change
  crossing  real crossed episode: nothing is certified between the first crossing and the end of the quarantine
  oi        unaffected observations unchanged; no rejected version used around the incident
  liq       real hours without file: unknown, never zero, in every aggregate and window
  mark      real outage and vendor mark gaps: nothing invented, dependent features null, book basis null
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402
from chdds.book import chain as CH  # noqa: E402
from chdds.book.driver import _concat_batch, _take_pending, load_book_hour, run_replay  # noqa: E402
from chdds.book.replay import ReplayConfig  # noqa: E402
from chdds.features import slow as S  # noqa: E402

OUT = P.DERIVED / "regen_check"
NT = 4_000_000
SEC = 10**9


@njit(cache=True)
def _quotes(seq_msg, seq_time, seq_kind, row_start, side, idx, qty, off, m_evt, m_txn, qb, qa, s, T, E, X, M, V):
    """Independent book: one record per applied update (s: 0 bb, 1 ba, 2 anchored, 3 mode ok, 4 n)."""
    for k in range(seq_msg.shape[0]):
        kind = seq_kind[k]
        m = seq_msg[k]
        if kind == 3:
            s[2] = 0
            continue
        if kind == 1:
            qb[:] = 0
            qa[:] = 0
            bb = -1
            ba = NT
            for r in range(row_start[m], row_start[m + 1]):
                if off[r] or idx[r] < 0 or idx[r] >= NT:
                    continue
                if side[r] == 0:
                    qb[idx[r]] = qty[r]
                    if qty[r] > 0 and idx[r] > bb:
                        bb = idx[r]
                elif side[r] == 1:
                    qa[idx[r]] = qty[r]
                    if qty[r] > 0 and idx[r] < ba:
                        ba = idx[r]
            s[0] = bb
            s[1] = ba
            s[2] = 1
            s[3] = 0
            continue
        if kind != 0 or s[2] == 0:
            continue
        bb = s[0]
        ba = s[1]
        for r in range(row_start[m], row_start[m + 1]):
            if off[r] or idx[r] < 0 or idx[r] >= NT:
                continue
            p = idx[r]
            if side[r] == 0:
                qb[p] = qty[r]
                if qty[r] > 0 and p > bb:
                    bb = p
            elif side[r] == 1:
                qa[p] = qty[r]
                if qty[r] > 0 and p < ba:
                    ba = p
        while bb >= 0 and qb[bb] == 0:
            bb -= 1
        while ba < NT and qa[ba] == 0:
            ba += 1
        s[0] = bb
        s[1] = ba
        s[3] = 1
        n = s[4]
        T[n] = seq_time[k]
        E[n] = m_evt[m]
        X[n] = m_txn[m]
        M[n] = bb + ba
        V[n] = 1
        s[4] = n + 1


@njit(cache=True)
def _mpre(tq, ttm, T, X, M, V):
    n = tq.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        j = np.searchsorted(T, tq[i], side="left") - 1
        back = 0
        while j >= 0 and back < 1023:
            if X[j] < ttm[i]:
                if V[j] == 1 and ttm[i] - X[j] <= 1000:
                    out[i] = M[j] * 0.05
                break
            j -= 1
            back += 1
    return out


def markout_check(day, h_from, h_to):
    # the book starts at the last snapshot before the target hours; only the target hours are compared
    tgt = P.hour_start_ns(day, h_from)
    tr_ = pl.read_parquet(P.DERIVED / "d1" / "chain_pass" / "transitions_strict.parquet").filter(pl.col("kind").is_in(["anchor", "verify"]))["t"].to_numpy()
    a0 = int(tr_[tr_ <= tgt].max())
    h0 = a0 - a0 % P.HOUR_NS
    h1 = P.hour_start_ns(day, h_to - 1) + P.HOUR_NS
    hours = []
    x = h0
    while x < h1:
        dt = np.datetime64(x, "ns").astype("datetime64[h]")
        d_ = dt.astype("datetime64[D]").astype(object)
        hours.append((d_, int((dt - dt.astype("datetime64[D]")).astype(int))))
        x += P.HOUR_NS
    warm = tgt + 1200 * SEC
    q = np.arange(warm, h1 - SEC, SEC, dtype=np.int64)
    cfg = ReplayConfig()
    out, cols, _, _, _ = run_replay(hours[0][0], day, q, OUT / "_tmp", cfg, workers=0, write_grid=False, hours=hours, skip_unanchored=False, log=lambda *_: None)
    ci = {c: i for i, c in enumerate(cols)}
    n = 300_000 * len(hours)
    T = np.zeros(n, np.int64)
    E = np.zeros(n, np.int64)
    X = np.zeros(n, np.int64)
    M = np.zeros(n, np.int64)
    V = np.zeros(n, np.int8)
    qb = np.zeros(NT, np.int64)
    qa = np.zeros(NT, np.int64)
    s = np.array([-1, NT, 0, 0, 0], np.int64)
    cst, ru, rh = CH.new_state()
    carry = None
    for d, h in hours:
        cur = load_book_hour(d.isoformat(), h)
        b, nc = _concat_batch(carry, cur)
        r = CH.resolve(b["recv"], b["et"], b["U"], b["u"], b["pu"], b["L"], b["hash"], cst, ru, rh, nc, 60 * SEC, 0, 0, 0, 0)
        carry = _take_pending(b, r[-1])
        _quotes(r[0], r[1], r[2], b["row_start"], b["side"], b["idx"], b["qty"], b["off"], b["evt"], b["txn"], qb, qa, s, T, E, X, M, V)
    k = int(s[4])
    T, E, X, M, V = T[:k], E[:k], X[:k], M[:k], V[:k]
    days = sorted({d for d, _ in hours})
    tr = pa.concat_tables([pq.read_table(P.DERIVED / "canonical" / "trades" / f"{d.isoformat()}.parquet", columns=["recv_ns", "trade_time_ms", "price_e2", "qty_e3", "sign"]) for d in days])
    rv = tr["recv_ns"].to_numpy()
    a, b_ = np.searchsorted(rv, h0), np.searchsorted(rv, h1)
    tq = rv[a:b_]
    ttm = tr["trade_time_ms"].to_numpy()[a:b_]
    qq = tr["qty_e3"].to_numpy()[a:b_] / 1000.0
    sg = tr["sign"].to_numpy()[a:b_].astype(float)
    mpre = _mpre(tq, ttm, T, X, M, V)
    res = dict(day=day.isoformat(), hours=[h_from, h_to], replay_hours=len(hours), trades=int(tq.size), queries=int(q.size), series={})
    bnd = np.zeros(q.size, bool)
    for hh in range(h_from + 1, h_to):
        e = P.hour_start_ns(day, hh)
        bnd |= (q >= e) & (q < e + 1000 * SEC)
    for hz in cfg.markout_h_s:
        tm = tq + int(hz * SEC)
        j = np.searchsorted(T, tm, side="right") - 1
        fresh = (j >= 0) & (tm - E[np.maximum(j, 0)] * 1_000_000 <= cfg.max_age_ms * 1_000_000) & (tm < T[-1])
        ok = np.isfinite(mpre) & fresh
        mo = np.where(ok, 1e4 * sg * np.log(np.where(ok, M[np.maximum(j, 0)] * 0.05, 1) / np.where(ok, mpre, 1)), 0.0)
        o = np.argsort(tm, kind="stable")
        tms, mos, oks, qs = tm[o], mo[o], ok[o], qq[o]
        c_n = np.concatenate([[0], np.cumsum(oks)])
        c_q = np.concatenate([[0], np.cumsum(qs * oks)])
        c_m = np.concatenate([[0], np.cumsum(mos * qs * oks)])
        for w in cfg.markout_windows_s:
            nm = f"markout_{int(hz)}s_n_{int(w // 60)}m"
            qw = f"markout_{int(hz)}s_qw_bps_{int(w // 60)}m"
            hi = np.searchsorted(tms, q, side="right")
            lo = np.searchsorted(tms, q - int(w * SEC), side="right")
            bn = c_n[hi] - c_n[lo]
            bq = c_q[hi] - c_q[lo]
            bm = np.where(bq > 0, (c_m[hi] - c_m[lo]) / np.maximum(bq, 1e-12), np.nan)
            e_n, e_q = out[:, ci[nm]], out[:, ci[qw]]
            m = np.isfinite(e_n) & (q >= tgt + int((w + hz + 2) * SEC))
            dn = np.abs(e_n[m] - bn[m])
            dq = np.abs(e_q[m] - bm[m])
            res["series"][nm] = dict(compared=int(m.sum()), compared_near_boundary=int((m & bnd).sum()), count_diff=int((dn > 0).sum()), max_count_diff=float(dn.max()) if dn.size else 0.0,
                                     max_value_diff_bps=float(np.nanmax(dq)) if dq.size else 0.0, null_share=float(np.isnan(e_n).mean()))
    return res


def strict_check():
    from chdds.config import load_config
    cfg = load_config("configs/demo_3days.yaml")
    d0, d1 = date(2025, 6, 28), date(2025, 6, 29)
    ref = pl.concat([pl.read_parquet(P.DERIVED / "book_union" / "demo_3days" / f"{d.isoformat()}.parquet") for d in (d0, d1)]).sort("eval_time_ns")
    q = ref["eval_time_ns"].to_numpy()
    out, cols, _, _, _ = run_replay(d0, d1, q, OUT / "_tmp", cfg.book, workers=3, write_grid=False, log=lambda *_: None)
    diff = {}
    for j, c in enumerate(cols):
        a = out[:, j]
        b = ref[c].to_numpy()
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            diff[c] = int((~same).sum())
    other = {c: n for c, n in diff.items() if not c.startswith(("markout_", "eff_spread_"))}
    return dict(values=int(out.size), columns_changed=len(diff), rows_changed_max=max(diff.values()) if diff else 0, outside_markout_effspread=other)


def crossing_check():
    d = date(2025, 8, 29)
    h6 = P.hour_start_ns(d, 6)
    q = np.arange(h6 + 30 * 60 * SEC, h6 + 2 * P.HOUR_NS, SEC, dtype=np.int64)
    cfg = ReplayConfig(book_mode="rebuilt", checkpoint_cert_levels=500)
    out, cols, _, cnt, st = run_replay(d, d, q, OUT / "_tmp", cfg, workers=0, write_grid=False, hours=[(d, h) for h in range(4, 8)], skip_unanchored=False, log=lambda *_: None)
    ci = {c: i for i, c in enumerate(cols)}
    s = out[:, ci["book_state"]].astype(int)
    first = int(np.argmax(s == 3))
    back = first + int(np.argmax(s[first:] == 9)) if (s[first:] == 9).any() else -1
    last_cross = int(np.nonzero(s == 3)[0][-1])
    sp = (out[:, ci["best_ask"]] - out[:, ci["best_bid"]]) / 0.1
    return dict(first_crossed=str(np.datetime64(int(q[first]), "ns")), certified_again=str(np.datetime64(int(q[back]), "ns")), last_crossed_query=str(np.datetime64(int(q[last_cross]), "ns")),
                certified_rows_in_between=int((s[first:back] == 9).sum()), gap_s=float((q[back] - q[last_cross]) / 1e9), ncross=cnt["ncross"], restarts=cnt["nempty"],
                spread_ticks_p50_after=float(np.nanmedian(sp[back:])))


def oi_check():
    data = S.SlowData()
    o = data.oi
    st = o["status"].to_numpy()
    ts = o["ts_ms"].to_numpy()
    multi = o["n_versions"].to_numpy() > 1
    single_rejected = int(((~multi) & (st != 0)).sum())
    t = np.arange(P.hour_start_ns(date(2026, 10, 15), 8), P.hour_start_ns(date(2026, 10, 17), 12), 60 * SEC, dtype=np.int64)
    f = S.oi_features(t, S.SlowFeatureConfig(), data)
    acc = set(o.filter(pa.array(st == 0))["oi_e8"].to_numpy().tolist())
    v = f["oi_btc_asof"]
    used = np.round(v[np.isfinite(v)] * 1e8).astype(np.int64)
    states, counts = np.unique(f["oi_state"], return_counts=True)
    log_ = json.loads((P.DERIVED / "canonical" / "oi_build.json").read_text())
    return dict(status=log_["status"], contested=log_["ts_contested"], resolved=log_["ts_resolved"], quarantined=log_["ts_quarantined"], single_version_rows_rejected=single_rejected,
                rejected_value_used=int(sum(1 for x in np.unique(used) if int(x) not in acc)), max_oi_asof=float(np.nanmax(v)), state_minutes={str(int(a)): int(b) for a, b in zip(states, counts)})


def liq_check():
    data = S.SlowData()
    cfg = S.SlowFeatureConfig()
    h0 = P.hour_start_ns(P.HISTORY_START, 0)
    miss = sorted(set(range(int((P.hour_start_ns(P.HISTORY_END, 23) - h0) // P.HOUR_NS) + 1)) - data.liq_hours)
    rng = np.random.default_rng(1)
    pick = np.array(miss)[rng.choice(len(miss), min(40, len(miss)), replace=False)]
    t = np.sort(h0 + pick * P.HOUR_NS + 1800 * SEC)
    f = S.liq_features(t, cfg, data, h0)
    keys = [k for k in f if k.startswith(("liq_", "log1p_liq")) and not k.startswith("liq_state")]
    finite = {k: int(np.isfinite(f[k]).sum()) for k in keys if np.isfinite(f[k]).any()}
    zero = sum(int((f[k] == 0).sum()) for k in keys)
    ts_ = int(np.isfinite(f["time_since_buy_short_liq_s"]).sum() + np.isfinite(f["time_since_sell_long_liq_s"]).sum())
    states = {w: sorted(set(f[f"liq_state_{w}"].tolist())) for w in ("30s", "2m", "15m", "1h")}
    return dict(missing_hours=len(miss), sampled=int(t.size), states=states, finite_values_in_missing_hours=finite, zeros=zero, time_since_finite=ts_, policy=cfg.liq_missing_file_policy)


def mark_check():
    data = S.SlowData()
    cfg = S.SlowFeatureConfig()
    res = {}
    d = date(2026, 4, 24)
    for lab, day, a, b in (("outage_2026-04-24", d, 0, 24), ("vendor_gaps_2026-09-17", date(2026, 9, 17), 0, 24)):
        t = np.arange(P.hour_start_ns(day, a), P.hour_start_ns(day, 23) + P.HOUR_NS, 5 * SEC, dtype=np.int64)
        f = S.mark_features(day, t, cfg, data)
        st = f["mark_state"]
        bad = st != 0
        leak = {k: int(np.isfinite(f[k][bad]).sum()) for k in ("basis_mark_index_bps",)}
        slow_bad = f["mark_mkt_age_ms"] > cfg.mark_slow_age_s * 1000
        leak2 = {k: int(np.isfinite(f[k][slow_bad]).sum()) for k in ("mark_price", "index_price", "funding_rate_asof", "time_to_funding_s", "funding_phase_sin", "funding_change_1h")}
        res[lab] = dict(queries=int(t.size), stale_share=float(bad.mean()), over_10s_share=float(slow_bad.mean()), fast_leak=leak, slow_leak=leak2,
                        windows_null_share={k: float(np.isnan(f[k]).mean()) for k in ("tw_basis_mark_index_bps_15m", "ret_index_1h", "funding_change_1h")})
    # book-side basis in a real outage: mid exists but mark does not
    day = date(2026, 9, 17)
    q = np.arange(P.hour_start_ns(day, 10), P.hour_start_ns(day, 12), SEC, dtype=np.int64)
    cfgb = ReplayConfig(book_mode="rebuilt", checkpoint_cert_levels=500)
    out, cols, _, _, _ = run_replay(day, day, q, OUT / "_tmp", cfgb, workers=0, write_grid=False, hours=[(day, 10), (day, 11)], skip_unanchored=False, log=lambda *_: None)
    ci = {c: i for i, c in enumerate(cols)}
    age = out[:, ci["basis_mark_mkt_age_ms"]]
    res["book_basis"] = dict(queries=int(q.size), mark_too_old=int((age > cfgb.mark_max_age_ms).sum()), basis_values_with_old_mark=int(np.isfinite(out[age > cfgb.mark_max_age_ms, ci["basis_mid_mark_bps"]]).sum()))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["markout", "strict", "crossing", "oi", "liq", "mark"]
    p = OUT / "report.json"
    rep = json.loads(p.read_text()) if p.exists() else {}
    for k in which:
        if k == "markout":
            rep[k] = [markout_check(date(2026, 9, 17), 9, 13), markout_check(date(2026, 6, 14), 9, 12)]
        else:
            rep[k] = globals()[f"{k}_check"]()
        p.write_text(json.dumps(rep, indent=1, default=str))
        print(k, json.dumps(rep[k], default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()

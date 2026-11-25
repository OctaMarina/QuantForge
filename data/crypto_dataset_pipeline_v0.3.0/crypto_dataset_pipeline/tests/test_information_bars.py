"""Information-driven bars (AFML 2018, section 2.3.2): TIB, VIB, DIB, TRB, VRB, DRB.

* hand-computed examples (tick imbalance, volume runs) incl. the boundary condition b0 = terminal b of
  the previous bar and the tick-rule carry on equal prices;
* equality with an independent, deliberately naive pure-Python reference (all 6 types, both signs,
  with and without the declared non-AFML stabilisers);
* EWMA definition == pandas ewm(adjust=True);
* thresholds frozen at bar start / built from prior bars only (prefix and future-perturbation invariance);
* batch / restart invariance; a gap restarts the expectations with a warm-up bar;
* the declared non-AFML RMS imbalance threshold (imbalance_threshold=rms: empirical RMS of the signed flow over
  H-tick windows): hand-computed example, the same independent reference, E[T] ~ H for independent signs,
  invariances, old-state compatibility; the fixed-E0[T] runs variant.
"""
import math

import numpy as np
import pandas as pd
import pytest

from conftest import mk_trades, random_trades
from cryptods.bars import BarBuilder, bars_to_float, INFO_BAR_TYPES, STATUS_GAP_INVALIDATED

INFO = {"sign": "tick_rule", "ewma_span_bars": 5, "ewma_span_ticks": 200, "warmup_ticks": 150,
        "expected_ticks_bounds": None, "min_expected_imbalance_ratio": None}


def flow_trades(spec, n=12000, seed=0, stay=0.97):
    """Trades with persistent order flow (Markov aggressor regimes, price moves with the aggressor): the
    situation information bars are designed for.  With i.i.d. signs the literal imbalance threshold is
    ill-conditioned (see docs/INFORMATION_BARS.md)."""
    rng = np.random.default_rng(seed)
    side = np.empty(n, np.int8)
    side[0] = 1
    flips = rng.random(n) > stay
    for i in range(1, n):
        side[i] = -side[i - 1] if flips[i] else side[i - 1]
    step = np.where(rng.random(n) < 0.4, side, 0) + rng.choice([-1, 0, 0, 0, 0, 0, 1], n)
    price = 30000_00 + np.cumsum(step)
    qty = (rng.lognormal(-3, 1.2, n) * 1e8).astype(np.int64) + 1
    ts = 1_700_000_000_000_000 + np.cumsum(rng.exponential(50_000, n).astype(np.int64))
    return pd.DataFrame({"trade_id": np.arange(1, n + 1, dtype=np.int64), "ts_us": ts, "price_i": price.astype(np.int64),
                         "qty_i": qty, "notional_i": price.astype(np.int64) * qty, "aggressor": side})


def ib(spec, tmp, bt, name="ib", **over):
    return BarBuilder(name, bt, spec, tmp, info={**INFO, **over})


# ---------------------------------------------------------------------------------------------- reference
class _E:
    """Bias-corrected EWMA (pandas adjust=True) written independently of the kernel."""

    def __init__(self, span):
        self.a = 2.0 / (span + 1.0)
        self.num = 0.0
        self.den = 0.0

    def add(self, x):
        self.num = x + (1.0 - self.a) * self.num
        self.den = 1.0 + (1.0 - self.a) * self.den

    def val(self):
        return self.num / self.den if self.den > 0 else math.nan


def reference(trades: pd.DataFrame, bt: str, p: dict):
    """Naive AFML information bars: one python loop, lists of (b_t, v_t) per bar."""
    runs = bt.endswith("_runs")
    unit = bt.split("_")[0]
    e_t, e_p, e_vb, e_vs = (_E(p["ewma_span_bars"]) for _ in range(4))
    e_bv, e_v = _E(p["ewma_span_ticks"]), _E(p["ewma_span_ticks"])
    rms = (not runs) and p.get("imbalance_threshold") == "rms"
    if rms:                                   # H-tick windows = 4 sub-blocks of m ticks, one window per sub-block
        m = int(p["rms_horizon_ticks"]) // 4
        e_blk = _E(p["ewma_span_ticks"] / m)
        subs, sub_cur, sub_cnt, sub_done = [0.0, 0.0, 0.0, 0.0], 0.0, 0, 0
    init = False
    last_px, last_b = None, 0
    cur, out = [], []
    thr = et = ex = math.nan
    warm = True
    for r in trades.itertuples(index=False):
        if not cur:                                          # a bar opens: freeze E0[theta_T]
            warm = not init
            if warm:
                thr = et = ex = math.nan
            elif rms:
                thr = math.sqrt(e_blk.num / e_blk.den)
                et = float(4 * m)
                ex = thr / et
            else:
                et = e_t.val()
                if p.get("expected_ticks_bounds"):
                    lo, hi = p["expected_ticks_bounds"]
                    et = min(max(et, lo), hi)
                if runs:
                    pb = e_p.val()
                    a1 = pb * e_vb.val() if not math.isnan(e_vb.val()) else 0.0
                    a2 = (1.0 - pb) * e_vs.val() if not math.isnan(e_vs.val()) else 0.0
                    ex = a1 if a1 > a2 else a2
                else:
                    ex = abs(e_bv.val())
                    if p.get("min_expected_imbalance_ratio"):
                        ex = max(ex, p["min_expected_imbalance_ratio"] * e_v.val())
                thr = et * ex
        px = int(r.price_i)
        if last_px is None:
            b = 0
        elif px > last_px:
            b = 1
        elif px < last_px:
            b = -1
        else:
            b = last_b                                       # tick rule: equal price keeps the sign
        last_px = px
        if b != 0:
            last_b = b
        bs = int(r.aggressor) if p["sign"] == "aggressor" else b
        v = {"tick": 1, "volume": int(r.qty_i), "notional": int(r.notional_i)}[unit]
        cur.append((bs, v))
        if bs != 0:
            e_bv.add(float(bs) * float(v))
            e_v.add(float(v))
        if rms:
            sub_cur += float(bs) * float(v)
            sub_cnt += 1
            if sub_cnt >= m:
                subs[sub_done % 4] = sub_cur
                sub_cur, sub_cnt, sub_done = 0.0, 0, sub_done + 1
                if sub_done >= 4:
                    w = subs[0] + subs[1] + subs[2] + subs[3]
                    e_blk.add(w * w)
        theta = sum(s * x for s, x in cur)
        buyv = sum(x for s, x in cur if s > 0)
        sellv = sum(x for s, x in cur if s < 0)
        if warm:
            close = len(cur) >= p["warmup_ticks"]
        elif runs:
            close = float(max(buyv, sellv)) >= thr
        else:
            close = abs(float(theta)) >= thr
        if close:
            out.append({"last_trade_id": int(r.trade_id), "n_trades": len(cur), "thr": thr, "et": et, "ex": ex,
                        "theta": float(max(buyv, sellv) if runs else theta), "warm": warm})
            nb = sum(1 for s, _ in cur if s > 0)
            ns = sum(1 for s, _ in cur if s < 0)
            ready = nb + ns > 0 and (runs or e_bv.den > 0) and (not rms or e_blk.den > 0)
            if not (warm and not ready):
                e_t.add(float(len(cur)))
                if nb + ns:
                    e_p.add(nb / (nb + ns))
                if nb:
                    e_vb.add(buyv / nb)
                if ns:
                    e_vs.add(sellv / ns)
                init = True
            cur = []
    return pd.DataFrame(out)


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


# ---------------------------------------------------------------------------------------------- hand-made
def test_tick_imbalance_hand_computed(spec, tmp_path):
    """warm-up 3 ticks, spans 1 (alpha = 1: the EWMA is the last observation).
    t1 100 (b unknown) t2 101 (+1) t3 102 (+1)            -> warm-up bar, theta = 2
    E0[T] = 3, E0[b] = +1 -> threshold 3
    t4 102 (= -> +1, carried ACROSS the bar boundary: b0) t5 103 (+1) t6 104 (+1) -> |theta| = 3 closes
    t7 103 (-1) t8 103 (= -> -1) t9 102 (-1)                -> theta = -3 closes (threshold 3*|-1| = 3)"""
    t = mk_trades(spec, range(1, 10), range(1, 10), [100, 101, 102, 102, 103, 104, 103, 103, 102], ["1"] * 9)
    b = ib(spec, tmp_path, "tick_imbalance", ewma_span_bars=1, ewma_span_ticks=1, warmup_ticks=3)
    out = bars_to_float(b.process(t), spec)
    assert list(out.last_trade_id) == [3, 6, 9]
    assert list(out.ib_warmup) == [True, False, False]
    assert same(out.threshold, [np.nan, 3, 3]) and list(out.ib_theta) == [2, 3, -3]
    assert same(out.ib_expected_ticks, [np.nan, 3, 3]) and same(out.ib_expected_per_tick, [np.nan, 1, 1])


def test_volume_runs_hand_computed(spec, tmp_path):
    """warm-up 2 ticks, spans 1.
    t1 (100, 1) unknown, t2 (101, 2) buy               -> warm-up: E0[T]=2, P=1, E[v|buy]=2, no sell yet
    threshold = 2 * max(1*2, 0) = 4
    t3 (100,1) sell, t4 (99,1) sell, t5 (100,3) buy, t6 (101,1) buy -> buy volume 4 >= 4 closes
    next: E0[T]=4, P=2/4, E[v|buy]=4/2, E[v|sell]=2/2 -> threshold 4 * max(0.5*2, 0.5*1) = 4"""
    t = mk_trades(spec, range(1, 8), range(1, 8), [100, 101, 100, 99, 100, 101, 102], [1, 2, 1, 1, 3, 1, 1])
    b = ib(spec, tmp_path, "volume_runs", ewma_span_bars=1, ewma_span_ticks=1, warmup_ticks=2)
    out = bars_to_float(b.process(t), spec)
    assert list(out.last_trade_id) == [2, 6]
    assert same(out.threshold, [np.nan, 4.0]) and list(out.ib_theta) == [2.0, 4.0]
    part = bars_to_float(b.partial(), spec)
    assert part.threshold.iloc[0] == 4.0 and part.ib_expected_ticks.iloc[0] == 4.0
    assert part.ib_expected_per_tick.iloc[0] == 1.0          # max(0.5 * 2, 0.5 * 1) in BTC


def test_warmup_without_classified_tick_repeats(spec, tmp_path):
    """The first trades of a segment have no tick-rule sign: no expectation can be formed from them."""
    t = mk_trades(spec, range(1, 7), range(1, 7), [100, 100, 100, 100, 101, 102], ["1"] * 6)
    b = ib(spec, tmp_path, "tick_imbalance", ewma_span_bars=1, ewma_span_ticks=1, warmup_ticks=2)
    out = bars_to_float(b.process(t), spec)
    # bars (1,2) and (3,4) have no classified tick -> still warm-up; (5,6) is the first informative bar
    assert list(out.last_trade_id) == [2, 4, 6] and out.ib_warmup.all()
    assert b.info_state()["initialised"] and b.info_state()["E0_T"] == 2.0


# ---------------------------------------------------------------------------------------------- reference
CASES = ([(bt, sign, stab) for bt in INFO_BAR_TYPES for sign in ("tick_rule", "aggressor") for stab in (False, True)]
         + [(bt, sign, "rms") for bt in INFO_BAR_TYPES[:3] for sign in ("tick_rule", "aggressor")])
RMS = {"imbalance_threshold": "rms", "rms_horizon_ticks": 152, "warmup_ticks": 600}


@pytest.mark.parametrize("bt,sign,stab", CASES)
def test_equals_independent_reference(spec, tmp_path, bt, sign, stab):
    t = flow_trades(spec, 12000, seed=11)
    over = {"sign": sign}
    if stab == "rms":
        over.update(RMS)
    elif stab:
        over.update(expected_ticks_bounds=[40, 400], min_expected_imbalance_ratio=0.2)
    out = bars_to_float(ib(spec, tmp_path, bt, **over).process(t), spec)
    ref = reference(t, bt, {**INFO, **over})
    assert len(out) >= 5 and len(out) == len(ref)
    assert list(out.last_trade_id) == list(ref.last_trade_id)
    assert list(out.n_trades) == list(ref.n_trades)
    assert list(out.ib_warmup) == list(ref.warm)
    unit = 1.0 if bt.startswith("tick") else (spec.qty_scale if bt.startswith("volume") else spec.notional_scale)
    assert same(out.ib_threshold_raw, ref.thr) and same(out.ib_expected_ticks, ref.et)
    assert same(out.ib_expected_per_tick_raw, ref.ex) and same(out.ib_theta_raw, ref.theta)
    assert same(out.threshold, np.asarray(ref.thr) / unit)
    if stab == "rms":
        assert (out.ib_expected_ticks[~out.ib_warmup] == 152).all()
    elif stab:
        e = out.ib_expected_ticks[~out.ib_warmup]
        assert ((e >= 40) & (e <= 400)).all()
    # closing rule: every non-warm-up bar reached its frozen threshold, and not one tick earlier
    nw = out[~out.ib_warmup]
    if bt.endswith("_imbalance"):
        assert (nw.ib_theta_raw.abs() >= nw.ib_threshold_raw).all()
    else:
        assert (nw.ib_theta_raw >= nw.ib_threshold_raw).all()


def test_ewma_is_pandas_adjusted_ewm(spec, tmp_path):
    t = flow_trades(spec, 8000, seed=3)
    b = ib(spec, tmp_path, "tick_runs")
    out = b.process(t)
    lengths = out.n_trades.to_numpy(float)          # every bar (incl. warm-up) feeds E0[T]
    exp = pd.Series(lengths).ewm(span=INFO["ewma_span_bars"], adjust=True).mean().iloc[-1]
    assert b.info_state()["E0_T"] == pytest.approx(exp, rel=1e-12)
    # the threshold of each bar uses the EWMA of the PREVIOUS bars only
    et = pd.Series(lengths).ewm(span=INFO["ewma_span_bars"], adjust=True).mean().shift(1)
    nw = ~bars_to_float(out, spec).ib_warmup.to_numpy()
    np.testing.assert_allclose(out.ib_expected_ticks.to_numpy()[nw], et.to_numpy()[nw], rtol=1e-12)


# ---------------------------------------------------------------------------------------------- causality
VARIANTS = [(bt, {}) for bt in INFO_BAR_TYPES] + [(bt, RMS) for bt in INFO_BAR_TYPES[:3]]
VIDS = [bt + ("_rms" if o else "") for bt, o in VARIANTS]


@pytest.mark.parametrize("bt,over", VARIANTS, ids=VIDS)
def test_prefix_and_future_perturbation_invariance(spec, tmp_path, bt, over):
    t = flow_trades(spec, 15000, seed=21)
    full = ib(spec, tmp_path / "f", bt, **over).process(t)
    cut = 9000
    pre = ib(spec, tmp_path / "p", bt, **over).process(t.iloc[:cut])
    cols = [c for c in full.columns if c != "series_id"]
    pd.testing.assert_frame_equal(pre[cols], full[cols].iloc[:len(pre)], check_exact=True)
    pert = t.copy()
    late = pert.trade_id > t.trade_id.iloc[cut - 1]
    pert.loc[late, "qty_i"] *= 7
    pert.loc[late, "price_i"] += np.where(np.arange(late.sum()) % 2 == 0, 5, -5)
    pert["notional_i"] = pert.price_i * pert.qty_i
    pert.loc[late, "aggressor"] = -pert.loc[late, "aggressor"]
    p = ib(spec, tmp_path / "x", bt, **over).process(pert)
    pd.testing.assert_frame_equal(p[cols].iloc[:len(pre)], pre[cols], check_exact=True)


@pytest.mark.parametrize("bt,over", VARIANTS, ids=VIDS)
def test_batch_and_restart_invariance(spec, tmp_path, bt, over):
    t = flow_trades(spec, 20000, seed=7)
    full = ib(spec, tmp_path / "a", bt, name="a", **over).process(t)
    rng = np.random.default_rng(2)
    cuts = np.sort(rng.choice(np.arange(1, len(t)), 10, replace=False))
    bounds = [0] + list(cuts) + [len(t)]
    b = ib(spec, tmp_path / "b", bt, name="b", **over)
    outs = []
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        if i in (3, 7):
            b.save()
            b = ib(spec, tmp_path / "b", bt, name="b", **over)      # restart from the persisted state
        outs.append(b.process(t.iloc[lo:hi]))
    inc = pd.concat(outs, ignore_index=True)
    cols = [c for c in full.columns if c != "series_id"]
    pd.testing.assert_frame_equal(full[cols], inc[cols], check_exact=True)


def test_gap_restarts_expectations_with_a_warmup_bar(spec, tmp_path):
    t = flow_trades(spec, 6000, seed=4)
    holed = t[(t.trade_id < 3000) | (t.trade_id >= 3100)]
    out = bars_to_float(ib(spec, tmp_path, "volume_runs").process(holed), spec)
    seg1 = out[out.segment_id == 1]
    assert (out.status == STATUS_GAP_INVALIDATED).sum() == 1
    assert seg1.ib_warmup.iloc[0] and not seg1.ib_warmup.iloc[1:].any()
    assert np.isnan(seg1.threshold.iloc[0]) and seg1.first_trade_id.iloc[0] == 3100
    # the new segment equals a fresh series started at the first trade after the gap
    fresh = bars_to_float(ib(spec, tmp_path / "n", "volume_runs").process(t[t.trade_id >= 3100]), spec)
    np.testing.assert_array_equal(seg1.last_trade_id.to_numpy(), fresh.last_trade_id.to_numpy()[:len(seg1)])


def test_information_types_have_their_own_series_identity():
    from cryptods import config as cfgmod
    a = cfgmod.load(None, ["bars.types=[tick_imbalance,volume_runs]"])
    b = cfgmod.load(None, ["bars.types=[tick_imbalance,volume_runs]", "bars.information.ewma_span_bars=50"])
    c = cfgmod.load(None, ["bars.types=[tick_imbalance,volume_runs]", "bars.information.guard_max_bars_per_day=10"])
    assert cfgmod.series_id(a, "tick_imbalance") != cfgmod.series_id(a, "volume_runs")
    assert cfgmod.series_id(a, "tick_imbalance") != cfgmod.series_id(b, "tick_imbalance")
    assert cfgmod.series_id(a, "tick_imbalance") == cfgmod.series_id(c, "tick_imbalance")   # guards: not identity
    assert cfgmod.series_id(a, "volume") == cfgmod.series_id(b, "volume")                   # other types untouched


# ---------------------------------------------------------------------------------------------- RMS variant
def test_rms_tick_imbalance_hand_computed(spec, tmp_path):
    """H = 4 (sub-blocks of 1 tick, windows of the last 4 ticks), ewma_span_ticks 3 -> 3 windows (alpha 1/2),
    warm-up 6 ticks, tick rule.
    t1 100 (x=0, unknown) t2 101 (+1) t3 102 (+1) t4 103 (+1) t5 102 (-1) t6 103 (+1)   -> warm-up bar
      windows: t1..t4 = 3, t2..t5 = 2, t3..t6 = 2 -> EWMA of w^2 with weights 1/4, 1/2, 1:
      (4 + 4/2 + 9/4) / (1 + 1/2 + 1/4) = 33/7 -> threshold sqrt(33/7) = 2.171
    t7 104 (+1) t8 105 (+1) t9 106 (+1)  -> theta 1, 2 (< 2.171), 3 closes at t9
      windows meanwhile: t4..t7 = 2, t5..t8 = 2, t6..t9 = 4 -> EWMA = 641/63 -> next threshold sqrt(641/63)"""
    t = mk_trades(spec, range(1, 11), range(1, 11), [100, 101, 102, 103, 102, 103, 104, 105, 106, 107], ["1"] * 10)
    b = ib(spec, tmp_path, "tick_imbalance", ewma_span_ticks=3, warmup_ticks=6, imbalance_threshold="rms",
           rms_horizon_ticks=4)
    out = bars_to_float(b.process(t), spec)
    assert list(out.last_trade_id) == [6, 9] and list(out.ib_warmup) == [True, False]
    r = out.iloc[1]
    assert r.threshold == pytest.approx(math.sqrt(33 / 7), rel=1e-15) and r.ib_theta == 3
    assert r.ib_expected_ticks == 4 and r.ib_expected_per_tick == pytest.approx(math.sqrt(33 / 7) / 4, rel=1e-15)
    part = bars_to_float(b.partial(), spec)
    assert part.threshold.iloc[0] == pytest.approx(math.sqrt(641 / 63), rel=1e-15)


@pytest.mark.parametrize("bt,lo,hi", [("tick_imbalance", 0.85, 1.4), ("volume_imbalance", 0.85, 2.5)])
def test_rms_expected_length_is_the_horizon_for_independent_signs(spec, tmp_path, bt, lo, hi):
    """Independent signs: theta is a random walk; the threshold estimates sqrt(E[theta_H^2]) = sigma sqrt(H), and
    the exit time from |theta| < sigma sqrt(H) has mean H (ticks: barrier ceil(sqrt(H)) -> mean <= (sqrt(H)+1)^2,
    plus estimation noise).  Heavier tails (volume) overshoot the barrier: mean >= H, never a collapse."""
    t = random_trades(spec, 200_000, seed=9)          # i.i.d. aggressor sides and lognormal sizes
    H = 400
    out = bars_to_float(ib(spec, tmp_path, bt, sign="aggressor", ewma_span_ticks=5000, warmup_ticks=H,
                           imbalance_threshold="rms", rms_horizon_ticks=H).process(t), spec)
    nw = out[~out.ib_warmup]
    assert len(nw) > 150
    ratio = nw.n_trades.mean() / H
    assert lo <= ratio <= hi, ratio
    assert (nw.n_trades <= 2).mean() < 0.05


def test_rms_gap_restarts_and_old_state_files_load(spec, tmp_path):
    import json
    t = flow_trades(spec, 8000, seed=5)
    over = dict(RMS)
    holed = t[(t.trade_id < 4000) | (t.trade_id >= 4050)]
    out = bars_to_float(ib(spec, tmp_path / "g", "notional_imbalance", **over).process(holed), spec)
    seg1 = out[out.segment_id == 1]
    assert seg1.ib_warmup.iloc[0] and np.isnan(seg1.threshold.iloc[0]) and not seg1.ib_warmup.iloc[1:].any()
    # a state written by 0.2.0 (15 float slots) continues an AFML series unchanged
    full = ib(spec, tmp_path / "a", "volume_runs", name="a").process(t)
    b = ib(spec, tmp_path / "b", "volume_runs", name="b")
    first = b.process(t.iloc[:5000])
    b.save()
    st_path = b.state_path()
    st = json.loads(st_path.read_text())
    st["ibf_hex"] = st["ibf_hex"][:15]
    st_path.write_text(json.dumps(st))
    rest = ib(spec, tmp_path / "b", "volume_runs", name="b").process(t.iloc[5000:])
    cols = [c for c in full.columns if c != "series_id"]
    pd.testing.assert_frame_equal(pd.concat([first, rest], ignore_index=True)[cols], full[cols], check_exact=True)


def test_rms_does_not_collapse_where_afml_does(spec, tmp_path):
    """Same i.i.d. trades: the literal AFML threshold collapses to 1-tick bars, the RMS one does not."""
    t = random_trades(spec, 60_000, seed=1)
    afml = ib(spec, tmp_path / "a", "tick_imbalance", sign="aggressor", warmup_ticks=200, ewma_span_ticks=2000).process(t)
    rms = ib(spec, tmp_path / "r", "tick_imbalance", sign="aggressor", warmup_ticks=200, ewma_span_ticks=2000,
             imbalance_threshold="rms", rms_horizon_ticks=200).process(t)
    assert afml.n_trades.iloc[1:].median() <= 2
    assert rms.n_trades.iloc[1:].median() > 50


@pytest.mark.parametrize("bt", ["tick_runs", "volume_runs", "volume_imbalance"])
def test_fixed_expected_ticks_equals_reference_with_constant_E0T(spec, tmp_path, bt):
    """fixed_expected_ticks = H: the AFML threshold with E0[T] replaced by the constant H (P, E[v|b] and
    E[b v] still EWMAs of prior bars / ticks)."""
    t = flow_trades(spec, 12000, seed=13)
    over = {"fixed_expected_ticks": 120, "warmup_ticks": 120, "sign": "aggressor"}
    if bt.endswith("imbalance"):
        over["min_expected_imbalance_ratio"] = 0.2
    out = bars_to_float(ib(spec, tmp_path, bt, **over).process(t), spec)
    ref = reference(t, bt, {**INFO, **over, "expected_ticks_bounds": [120, 120]})
    assert list(out.last_trade_id) == list(ref.last_trade_id) and same(out.ib_threshold_raw, ref.thr)
    assert (out.ib_expected_ticks[~out.ib_warmup] == 120).all()


def test_fixed_expected_ticks_removes_the_drift_of_runs_bars(spec, tmp_path):
    """With E0[T] = EWMA of past bar lengths, nothing anchors the scale of runs bars: E0[T] follows the
    realised lengths, which follow E0[T] (a random walk in log scale).  A fixed E0[T] keeps the scale."""
    t = flow_trades(spec, 120_000, seed=2)
    lit = ib(spec, tmp_path / "l", "tick_runs", sign="aggressor", warmup_ticks=300, ewma_span_bars=10).process(t)
    fix = ib(spec, tmp_path / "f", "tick_runs", sign="aggressor", warmup_ticks=300, ewma_span_bars=10,
             fixed_expected_ticks=300).process(t)
    q = lambda d: np.array_split(d.n_trades.iloc[1:].to_numpy(), 4)
    fq = [x.mean() / 300 for x in q(fix)]
    assert max(fq) / min(fq) < 1.3                                   # stationary scale
    assert all(0.6 < x < 1.4 for x in fq)

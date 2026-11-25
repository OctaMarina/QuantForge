"""D1: summaries of the no-reference days and of the crossing scan. Outputs in derived/d1/results/."""
import glob
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

R = P.DERIVED / "d1" / "results"
KEEP = ["spread_bps", "qi_1", "depth_bid_top5_btc", "depth_bid_top10_btc", "depth_bid_top20_btc", "depth_bid_1bps_btc", "depth_bid_5bps_btc", "depth_bid_10bps_btc",
        "depth_bid_25bps_btc", "depth_bid_50bps_btc", "cost_buy_1btc_bps", "cost_buy_5btc_bps", "cost_buy_1000000usdt_atmid_bps", "ofi_1s_btc", "ofi_5s_btc", "ofi_30s_btc", "ofi_2m_btc",
        "ofi_15m_btc", "mlofi_l5_30s_btc", "tw_spread_bps_2m", "tw_spread_bps_15m", "tw_depth_bid_5bps_2m_btc", "tw_depth_bid_25bps_15m_btc", "add_bid_10bps_2m_btc",
        "add_bid_10bps_15m_btc", "eff_spread_qw_bps_2m", "markout_30s_qw_bps_15m", "basis_mid_index_bps"]


def no_ref():
    rows, fe = [], []
    for f in sorted(glob.glob(str(P.DERIVED / "d1" / "no_ref" / "*.json"))):
        r = json.load(open(f))
        s, n = r["states"], r["n"]
        rows.append(dict(day=r["day"], group=r["group"], certified=s.get("9", 0) / n, uncertified=s.get("10", 0) / n, stale=s.get("2", 0) / n,
                         quarantine=s.get("5", 0) / n, gaps=r["gaps"], restarts=r["restarts"], crossings=r["crossings"], floor_p50=r["floor_q"][1], ceil_p50=r["ceil_q"][1]))
        fe.append(pl.read_parquet(f.replace(".json", ".parquet")).with_columns(group=pl.lit(r["group"])))
    pl.from_dicts(rows).write_csv(R / "no_ref_days.csv")
    x = pl.concat(fe).filter(pl.col("feature").is_in(KEEP)).group_by("feature", "group").agg(pl.col("share").mean())
    x.pivot(on="group", index="feature", values="share", sort_columns=True).sort("feature").write_csv(R / "no_ref_feature_availability.csv")


def cross():
    parts, msgs, crossed, gaps, days = [], 0, 0, 0, 0
    for f in sorted(glob.glob(str(P.DERIVED / "d1" / "cross" / "*.parquet"))):
        a, b, c = open(f.replace(".parquet", ".meta")).read().split()
        msgs, crossed, gaps, days = msgs + int(a), crossed + int(b), gaps + int(c), days + 1
        z = pl.read_parquet(f)
        if z.height:
            parts.append(z.with_columns(day=pl.lit(Path(f).stem)))
    x = pl.concat(parts).with_columns(dur_s=(pl.col("t1") - pl.col("t0")) / 1e9)
    g = x.group_by("day").agg(pl.len().alias("episoade"), pl.col("dur_s").sum().alias("secunde"), pl.col("dur_s").max().alias("max_s"), (pl.col("worst_ticks").max() / 10).alias("max_usdt"),
                              pl.col("t0").min().alias("t0"), pl.col("t1").max().alias("t1")).sort("day")
    g = g.with_columns(pl.from_epoch("t0", time_unit="ns").dt.strftime("%H:%M:%S"), pl.from_epoch("t1", time_unit="ns").dt.strftime("%H:%M:%S"),
                       zile_scanate=pl.lit(days), mesaje=pl.lit(msgs), mesaje_incrucisate=pl.lit(crossed), goluri_reale=pl.lit(gaps))
    g.write_csv(R / "cross_days.csv")


if __name__ == "__main__":
    no_ref()
    cross()

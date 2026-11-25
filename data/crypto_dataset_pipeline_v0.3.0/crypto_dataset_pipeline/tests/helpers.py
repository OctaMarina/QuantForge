import numpy as np
import pandas as pd

from cryptods import config as cfgmod
from cryptods.bars import BarBuilder, bars_to_float
from cryptods.features.context import Ctx


def make_cfg(**over):
    cfg = cfgmod.deep_merge(cfgmod.DEFAULTS, over)
    cfgmod.validate(cfg)
    return cfgmod.Cfg(cfg)


def build_bars(spec, trades, tmp_path, bar_type="volume", thr=1.0, interval_us=60_000_000, name="x", watermark=-1):
    kw = {}
    if bar_type == "volume":
        kw["threshold_i"] = int(thr * spec.qty_scale)
    elif bar_type == "notional":
        kw["threshold_i"] = int(thr * spec.notional_scale)
    else:
        kw["interval_us"] = interval_us
    b = BarBuilder(name, bar_type, spec, tmp_path, **kw)
    if bar_type == "time":
        b.start_series_at(int(trades.ts_us.min()) // interval_us * interval_us)
    out = b.process(trades, watermark_us=watermark)
    return bars_to_float(out, spec), b


def px1s_from_trades(trades, spec):
    sec = (trades.ts_us.to_numpy() // 1_000_000) * 1_000_000
    tmp = pd.DataFrame({"sec_start_us": sec, "last_ts_us": trades.ts_us.to_numpy(),
                        "price": trades.price_i.to_numpy() / spec.price_scale, "trade_id": trades.trade_id.to_numpy()})
    return tmp.sort_values("trade_id").groupby("sec_start_us", as_index=False).last()[["sec_start_us", "last_ts_us", "price"]]


def ctx_for(cfg, spec, bars, bar_type="volume", px1s=None, tables=None, gaps=None, data_start=None):
    end = bars["bar_end_us"].to_numpy(np.int64)
    return Ctx(cfg=cfg, spec=spec, bar_type=bar_type, bars=bars.reset_index(drop=True), decision_us=end, ref_us=end,
               px1s=px1s, gaps=gaps or [], data_start_us=data_start, tables=tables or {})

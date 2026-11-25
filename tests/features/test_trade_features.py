import numpy as np
import pandas as pd

from quant_forge import features as qf
from tests.synthetic import make_trade_panel


def test_track_record_is_point_in_time():
    tr = make_trade_panel(400, seed=2)
    base = qf.track_record_features(tr)
    assert list(base.columns) == ["s_swr_30", "s_mr_30", "s_swr_100", "s_mr_100", "s_awr_100", "s_wr_asset_10",
                                  "s_wr_asset_30", "s_open_positions", "s_signals_24h", "s_closed_30d"]
    j = 200
    flipped = tr.copy()
    flipped.loc[j, "y"] = 1 - flipped.loc[j, "y"]
    flipped.loc[j, "trade_return"] = -flipped.loc[j, "trade_return"]
    after = qf.track_record_features(flipped)
    unaffected = tr["decision_time"] <= tr.loc[j, "exit_time"]
    pd.testing.assert_frame_equal(base.loc[unaffected], after.loc[unaffected])
    assert not base.loc[~unaffected, "s_swr_30"].equals(after.loc[~unaffected, "s_swr_30"])


def test_track_record_window_rate():
    tr = make_trade_panel(300, seed=5)
    f = qf.track_record_features(tr)
    i = 250
    closed = tr.loc[tr["exit_time"] < tr.loc[i, "decision_time"]].sort_values("exit_time")
    assert f.loc[i, "s_swr_30"] == closed["y"].iloc[-30:].mean()


def test_orient_by_side():
    df = pd.DataFrame({"side": [1, -1], "t_ret_1d": [0.1, 0.1], "t_rsi14": [0.7, 0.7], "lib_dc_high_rel": [0.2, 0.2],
                       "lib_dc_high_level": [5.0, 5.0], "v_rvol_1d": [0.3, 0.3]})
    out = qf.orient_by_side(df)
    assert out["t_ret_1d"].tolist() == [0.1, -0.1]
    np.testing.assert_allclose(out["t_rsi14"], [0.2, -0.2])
    assert out["lib_dc_high_rel"].tolist() == [0.2, -0.2]
    assert out["lib_dc_high_level"].tolist() == [5.0, 5.0] and out["v_rvol_1d"].tolist() == [0.3, 0.3]
    assert df["t_ret_1d"].tolist() == [0.1, 0.1]                                   # input untouched


def test_relative_levels_and_families():
    df = pd.DataFrame({"raw_close": [100.0], "dc_high": [110.0], "atr_value": [2.0], "st_vol_ma": [9.0], "adx_value": [25.0]})
    out = qf.relative_levels(df, ["dc_high", "atr_value", "st_vol_ma", "adx_value"])
    assert out.columns.tolist() == ["lib_dc_high_rel", "lib_dc_high_level", "lib_atr_value_rel", "lib_atr_value_level",
                                    "lib_adx_value"]
    assert out["lib_dc_high_rel"].iloc[0] == 110 / 100 - 1 and out["lib_atr_value_rel"].iloc[0] == 0.02
    assert qf.feature_family("xr_rvol_30d") == "X" and qf.feature_family("x_rs_7d") == "R"
    assert qf.feature_family("b_dd_90d") == "M" and qf.feature_family("trade_return") is None
    assert qf.feature_families(["side", "v_a", "s_b"]) == {"v_a": "V", "s_b": "S"}

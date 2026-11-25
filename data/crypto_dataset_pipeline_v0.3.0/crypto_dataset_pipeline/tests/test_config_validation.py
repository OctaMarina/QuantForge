"""Configuration checks that prevent silently wrong datasets."""
import pytest

from conftest import ROOT
from cryptods import config as cfgmod


def test_symbol_must_match_base_and_quote():
    with pytest.raises(ValueError, match="does not match instrument.base"):
        cfgmod.load(None, ["instrument.symbol=ETHUSDT"])               # base still BTC
    c = cfgmod.load(None, ["instrument.base=ETH"])
    assert c["instrument"]["symbol"] == "ETHUSDT"                      # derived
    c = cfgmod.load(None, ["instrument.base=ETH", "instrument.symbol=ethusdt"])
    assert c["instrument"]["base"] == "ETH"
    cfgmod.load(None, ["instrument.market_type=perpetual", "instrument.symbol=BTCUSDT_PERP"])
    with pytest.raises(ValueError):
        cfgmod.load(None, ["instrument.symbol=BTCUSDT_PERP"])           # _PERP only for perpetuals


def test_dvol_max_age_must_cover_resolution():
    with pytest.raises(ValueError, match="shorter than the DVOL resolution"):
        cfgmod.load(None, ["features.dvol.resolution=3600"])           # default max_age 5min
    cfgmod.load(None, ["features.dvol.resolution=3600", "features.dvol.max_age=2h"])
    with pytest.raises(ValueError):
        cfgmod.load(None, ["features.dvol.resolution=7"])


@pytest.mark.parametrize("bad", ["bars.information.sign=majority", "bars.information.ewma_span_bars=0",
                                 "bars.information.expected_ticks_bounds=[100,10]",
                                 "bars.information.min_expected_imbalance_ratio=1.5",
                                 "bars.information.guard_max_bars_per_day=0", "bars.on_unconfirmed_missing_day=skip",
                                 "bars.archive_publication_lag_days=-1", "bars.types=[dollar_imbalance]"])
def test_information_and_missing_day_options_are_validated(bad):
    with pytest.raises(ValueError):
        cfgmod.load(None, [bad])


def test_warnings():
    w = cfgmod.warnings_for(cfgmod.load(None, ["instrument.base=ETH", "period.bar_origin=2024-06-01T00:00:00Z",
                                               "bars.types=[volume,tick_imbalance]"]))
    text = " ".join(w)
    assert "bar_origin == period.start" in text and "skew currency BTC differs" in text and "literal AFML" in text
    assert cfgmod.warnings_for(cfgmod.load(str(ROOT / "configs" / "eth_spot.yaml"))) == []


def test_guards_and_missing_day_policy_do_not_change_series_identity_but_stabilisers_do():
    a = cfgmod.load(None, ["bars.types=[volume,tick_runs]"])
    b = cfgmod.load(None, ["bars.types=[volume,tick_runs]", "bars.information.guard_max_bars_per_day=null",
                           "bars.on_unconfirmed_missing_day=gap"])
    c = cfgmod.load(None, ["bars.types=[volume,tick_runs]", "bars.information.expected_ticks_bounds=[10,100]"])
    assert cfgmod.series_id(a, "tick_runs") == cfgmod.series_id(b, "tick_runs")
    assert cfgmod.series_id(a, "tick_runs") != cfgmod.series_id(c, "tick_runs")
    assert cfgmod.series_id(a, "volume") == cfgmod.series_id(c, "volume")


# ------------------------------------------------------------------ information bars: rms variant and per-type params
@pytest.mark.parametrize("bad", [
    ["bars.types=[tick_imbalance]", "bars.information.imbalance_threshold=rms"],               # horizon missing
    ["bars.types=[tick_imbalance]", "bars.information.imbalance_threshold=median"],
    ["bars.types=[tick_imbalance]", "bars.information.imbalance_threshold=rms", "bars.information.rms_horizon_ticks=500",
     "bars.information.min_expected_imbalance_ratio=0.1"],                                     # floor meaningless
    ["bars.types=[tick_imbalance,tick_runs]", "bars.information.imbalance_threshold=rms",
     "bars.information.rms_horizon_ticks=500", "bars.information.expected_ticks_bounds=[10,100]"],
    ["bars.information.per_type={volume: {warmup_ticks: 10}}"],                               # not an information type
    ["bars.information.per_type={tick_runs: {threshold: 10}}"],                               # unknown key
    ["bars.types=[tick_runs]", "bars.information.per_type={tick_runs: {warmup_ticks: 0}}"],
    ["bars.types=[tick_runs]", "bars.information.fixed_expected_ticks=100", "bars.information.expected_ticks_bounds=[10,200]"],
    ["bars.types=[tick_imbalance]", "bars.information.imbalance_threshold=rms", "bars.information.rms_horizon_ticks=500",
     "bars.information.fixed_expected_ticks=500"],
    ["bars.types=[tick_runs]", "bars.information.fixed_expected_ticks=0"],
    ["bars.types=[tick_imbalance]", "bars.information.imbalance_threshold=rms", "bars.information.rms_horizon_ticks=1001"],
])
def test_rms_and_per_type_validation(bad):
    with pytest.raises(ValueError):
        cfgmod.load(None, bad)


def test_per_type_overrides_and_bounds_only_for_runs():
    c = cfgmod.load(None, ["bars.types=[tick_runs,tick_imbalance]", "bars.information.imbalance_threshold=rms",
                           "bars.information.per_type={tick_runs: {warmup_ticks: 777, expected_ticks_bounds: [100, 2000]},"
                           " tick_imbalance: {rms_horizon_ticks: 3000, warmup_ticks: 3000}}"])
    r, i = cfgmod.effective_info(c, "tick_runs"), cfgmod.effective_info(c, "tick_imbalance")
    assert r["warmup_ticks"] == 777 and r["expected_ticks_bounds"] == [100, 2000]
    assert i["warmup_ticks"] == 3000 and i["rms_horizon_ticks"] == 3000 and i["expected_ticks_bounds"] is None
    assert "per_type" not in r


def test_series_identity_contains_exactly_the_parameters_that_change_the_bars():
    base = ["bars.types=[tick_runs,volume_imbalance]"]
    sid = lambda extra, bt: cfgmod.series_id(cfgmod.load(None, base + extra), bt)
    # 0.2.0 ids of the AFML imbalance types are unchanged (default config, BTCUSDT spot)
    assert cfgmod.series_id(cfgmod.load(None, ["bars.types=[tick_imbalance]"]), "tick_imbalance") == \
        "binance_spot_BTCUSDT_tick_imbalance_10ba2511d02e"
    assert cfgmod.series_id(cfgmod.load(None, ["bars.types=[volume_imbalance]",
                                               "bars.information.expected_ticks_bounds=[200,5000]",
                                               "bars.information.min_expected_imbalance_ratio=0.1"]),
                            "volume_imbalance") == "binance_spot_BTCUSDT_volume_imbalance_65c73dfb438e"
    # runs bars: the tick-level EWMA span and the imbalance rule do not enter their threshold
    assert sid(["bars.information.ewma_span_ticks=77"], "tick_runs") == sid([], "tick_runs")
    rms = ["bars.information.imbalance_threshold=rms", "bars.information.rms_horizon_ticks=900"]
    assert sid(rms, "tick_runs") == sid([], "tick_runs")
    assert sid(rms, "volume_imbalance") != sid([], "volume_imbalance")
    # rms imbalance: the bar-level span does not enter; horizon, tick span, warm-up and sign do
    assert sid(rms + ["bars.information.ewma_span_bars=99"], "volume_imbalance") == sid(rms, "volume_imbalance")
    for k in ("rms_horizon_ticks=904", "ewma_span_ticks=77", "warmup_ticks=5", "sign=aggressor"):
        assert sid(rms[:1] + ["bars.information." + k] + (rms[1:] if "horizon" not in k else []),
                   "volume_imbalance") != sid(rms, "volume_imbalance")
    # per-type override changes only that type; 1000.0 == 1000
    assert sid(["bars.information.per_type={tick_runs: {warmup_ticks: 1000}}"], "volume_imbalance") == \
        sid([], "volume_imbalance")
    assert sid(["bars.information.per_type={tick_runs: {warmup_ticks: 1000}}"], "tick_runs") != sid([], "tick_runs")
    assert sid(["bars.information.per_type={tick_runs: {warmup_ticks: 1000.0}}"], "tick_runs") == \
        sid(["bars.information.per_type={tick_runs: {warmup_ticks: 1000}}"], "tick_runs")

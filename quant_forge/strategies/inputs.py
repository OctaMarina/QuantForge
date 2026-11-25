"""Input-column requirements shared by every strategy (OHLC bars with microsecond timestamps)."""

from .spec import ColumnRequirement

BAR_INPUTS = (
    ColumnRequirement("open", True, "ohlc", "bar open"),
    ColumnRequirement("high", True, "ohlc", "bar high"),
    ColumnRequirement("low", True, "ohlc", "bar low"),
    ColumnRequirement("close", True, "ohlc", "bar close"),
    ColumnRequirement("bar_start_us", True, "timestamp", "bar start, integer microseconds since UTC epoch; strictly increasing"),
    ColumnRequirement("bar_end_us", True, "timestamp", "bar end, integer microseconds >= bar_start_us"),
    ColumnRequirement("first_trade_ts_us", False, "timestamp", "first trade of the bar; fill time of next_open orders", "bar_start_us"),
    ColumnRequirement("bar_available_us", False, "timestamp", "when the bar became available", "bar_end_us"),
    ColumnRequirement("decision_time_us", False, "timestamp", "when decisions on the bar may be taken", "bar_end_us"),
    ColumnRequirement("complete", False, "validity", "falsy marks the bar invalid", "all bars complete"),
    ColumnRequirement("bar_seq", False, "segmentation", "must increase by 1 between adjacent bars", "not checked"),
    ColumnRequirement("segment_id", False, "segmentation", "a change starts a new segment", "not checked"),
    ColumnRequirement("bar_id", False, "reference", "copied into event references", "omitted"),
    ColumnRequirement("series_id", False, "reference", "must hold a single value if present", "not checked"),
    ColumnRequirement("bar_type", False, "reference", "must hold a single value if present", "not checked"),
)

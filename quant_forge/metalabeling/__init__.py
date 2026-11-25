"""Meta-labeling on top of the primary strategies: datasets, walk-forward fitting and run reports.

    from quant_forge import metalabeling as ml

    trades = ml.build_trade_panel(bars, feats, {S.RSI_CROSS_ABOVE: "RSI"}, feature_columns=cols, tick_sizes=TICKS)
    ds = ml.prepare_meta_dataset(trades)                         # + track record, side orientation, candidates
    cfg = ml.MetaLabelingConfig(train_assets=tuple(SEEN_ASSETS))
    res = ml.run_walk_forward(ds.trades, ["RSI"], windows, {"RSI": ds.candidates_for("RSI")}, cfg, cache=cache)
    pred = ml.collect_predictions(res, ds.trades, ["strategy", "asset", "trade_return", "y", ...])

See ``docs/metalabeling.md`` for the full pipeline.
"""

from .dataset import (TRADE_COLUMNS, MetaDataset, assign_groups, assign_periods, build_trade_panel, candidates_for,
                      prepare_meta_dataset)
from .reports import (family_selection_table, feature_selection_records, probability_columns, sizing_constants_table,
                      window_summary)
from .walk_forward import (MetaLabelingConfig, WindowFit, collect_predictions, fit_window, run_walk_forward,
                           window_key)

__all__ = ["build_trade_panel", "assign_groups", "assign_periods", "prepare_meta_dataset", "MetaDataset",
           "candidates_for", "TRADE_COLUMNS", "MetaLabelingConfig", "WindowFit", "fit_window", "run_walk_forward",
           "collect_predictions", "window_key", "window_summary", "feature_selection_records",
           "family_selection_table", "sizing_constants_table", "probability_columns"]

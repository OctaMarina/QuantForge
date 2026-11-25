"""Model-validation utilities (Lopez de Prado, *Advances in Financial Machine Learning*, chapter 7).

* :class:`PurgedKFold` - purged K-fold with embargo for ONE series of events (sklearn cross-validator);
* :class:`CalendarPurgedKFold` - the same overlap rule on common calendar folds for a PANEL of events
  (several assets, repeated timestamps);
* :func:`calendar_windows`, :func:`expanding_split` - walk-forward windows (train on events ended before the
  window, test on events decided inside it).

    from quant_forge.validation import PurgedKFold, CalendarPurgedKFold
    cv = PurgedKFold(n_splits=5, samples_info_sets=t1, pct_embargo=0.01)
    folds = list(CalendarPurgedKFold(4, 0.01).split(trades["decision_time"], trades["exit_time"]))
"""

from .calendar_kfold import CalendarFold, CalendarPurgedKFold, to_int64_ns
from .purged_kfold import PurgedFold, PurgedKFold
from .walk_forward import calendar_windows, expanding_split

__all__ = ["PurgedKFold", "PurgedFold", "CalendarPurgedKFold", "CalendarFold", "to_int64_ns",
           "calendar_windows", "expanding_split"]

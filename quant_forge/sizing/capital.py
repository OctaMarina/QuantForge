"""Capital-constrained replay of a sized trade list.

Signals cluster in time, so rules with equal *expected* exposure per signal can still exceed a capital budget.
:func:`capital_constrained_sizes` processes trades in order of entry and cuts each new size to the capital not
already committed: ``size = min(requested, (K - open exposure) / notional)``.  Only earlier entries and exits
are used (point-in-time).
"""

from __future__ import annotations

import heapq

import numpy as np

from ..validation.calendar_kfold import to_int64_ns

__all__ = ["capital_constrained_sizes"]


def capital_constrained_sizes(entry_time, exit_time, sizes, budget: float, notional: float = 10_000.0) -> np.ndarray:
    """Sizes (multiples of ``notional``) after enforcing a gross-exposure budget ``budget`` (same currency).

    A position frees its capital at its exit time; a trade entering exactly when another exits can reuse it.
    """
    en_all, ex_all = to_int64_ns(entry_time), to_int64_ns(exit_time)
    order = np.argsort(en_all, kind="stable")
    en, ex = en_all[order], ex_all[order]
    ws = np.asarray(sizes, dtype=float)[order]
    out = np.zeros(len(ws))
    heap: list = []
    open_exp = 0.0
    for i in range(len(ws)):
        while heap and heap[0][0] <= en[i]:
            open_exp -= heapq.heappop(heap)[1]
        room = max(budget - open_exp, 0.0)
        size = min(ws[i], room / notional)
        if size > 0:
            heapq.heappush(heap, (ex[i], size * notional))
            open_exp += size * notional
        out[order[i]] = size
    return out

"""cryptods - reproducible, causal crypto dataset pipeline (time / volume / notional bars + features).

The package is organised in layers:

* ``sources``   - provider adapters (download, parse, normalise; raw files are kept and versioned)
* ``bars``      - exact bar construction from individual trades (scaled-integer arithmetic)
* ``book``      - L2 order-book reconstruction with sequence checks and gap invalidation
* ``features``  - feature families; every feature is declared in ``registry``
* ``asof``      - backward as-of joins on availability time with staleness limits
* ``pipeline``  - orchestration of the CLI modes (dry-run, download, build, update, live, replay)

Nothing in this package places orders.
"""

__version__ = "0.1.0"

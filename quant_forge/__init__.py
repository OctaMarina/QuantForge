"""QuantForge: systematic-trading research library.

Sub-packages (import them explicitly; ``import quant_forge`` loads nothing heavy):

* :mod:`quant_forge.data`              market data (Binance klines + cache -> bar frames), research universes
* :mod:`quant_forge.features`          point-in-time bar features, fractional differentiation, trade features
* :mod:`quant_forge.strategies`        the primary strategies
* :mod:`quant_forge.validation`        purged K-fold (series and panel), walk-forward windows
* :mod:`quant_forge.feature_selection` clustering, placebos, clustered MDA / MDI          (scikit-learn)
* :mod:`quant_forge.models`            model specs, purged tuning, calibration           (scikit-learn)
* :mod:`quant_forge.sizing`            bet sizing rules and capital-budget replay        (scipy)
* :mod:`quant_forge.evaluation`        probability quality and economic evaluation
* :mod:`quant_forge.metalabeling`      trade panels, walk-forward meta-labeling, reports
* :mod:`quant_forge.utils`             disk cache and hashing helpers

Documentation: ``docs/README.md`` in the repository.
"""

__version__ = "0.2.0"

"""chdds — CryptoHFTData dataset library (Binance USD-M BTCUSDT perpetual).

Layers (see docs/ARCHITECTURE.md):
  1. raw (immutable .parquet.zst files, never modified)
  2. audit + staged/canonical events (exact scaled integers, provenance)
  3. state engines (L2 replay, slow-stream as-of state) and primitives
  4. bars (time / volume / dollar / CUSUM) and feature snapshots at decision times
  5. targets/labels — intentionally NOT part of this package
"""

__version__ = "0.1.0"

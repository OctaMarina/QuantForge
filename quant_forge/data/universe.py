"""Asset universes of the QuantForge cross-asset research (Binance spot, quoted in USDT).

* :data:`SEEN_ASSETS`: the 20 assets every meta-model of the research notebooks is trained on.
* :data:`NEW_ASSETS`: 20 further assets used only for out-of-sample transfer (models never train on them).
* :data:`TICK_SIZES`: price tick per asset.  The strategy library requires one; it only matters with
  ``slippage_ticks > 0``.
"""

from __future__ import annotations

__all__ = ["SEEN_ASSETS", "NEW_ASSETS", "ALL_ASSETS", "TICK_SIZES"]

SEEN_ASSETS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "BCH", "DOGE", "DOT", "LINK", "LTC", "UNI",
               "AAVE", "ATOM", "ETC", "FIL", "NEAR", "TRX", "XLM"]
NEW_ASSETS = ["ALGO", "VET", "THETA", "SAND", "MANA", "AXS", "ICP", "HBAR", "CRV", "GRT", "ENJ", "CHZ", "ZEC", "DASH",
              "XTZ", "COMP", "SNX", "IOTA", "ZIL", "QTUM"]
ALL_ASSETS = SEEN_ASSETS + NEW_ASSETS
TICK_SIZES = {"BTC": 1e-2, "ETH": 1e-2, "BNB": 1e-4, "SOL": 1e-4, "XRP": 1e-5, "ADA": 1e-5, "AVAX": 1e-4,
              "BCH": 1e-2, "DOGE": 1e-7, "DOT": 1e-4, "LINK": 1e-4, "LTC": 1e-2, "UNI": 1e-4, "AAVE": 1e-3,
              "ATOM": 1e-3, "ETC": 1e-4, "FIL": 1e-4, "NEAR": 1e-4, "TRX": 1e-5, "XLM": 1e-5,
              "ALGO": 1e-4, "VET": 1e-6, "THETA": 1e-4, "SAND": 1e-5, "MANA": 1e-4, "AXS": 1e-3, "ICP": 1e-3,
              "HBAR": 1e-5, "CRV": 1e-4, "GRT": 1e-5, "ENJ": 1e-5, "CHZ": 1e-5, "ZEC": 1e-2, "DASH": 1e-2,
              "XTZ": 1e-4, "COMP": 1e-2, "SNX": 1e-4, "IOTA": 1e-4, "ZIL": 1e-6, "QTUM": 1e-3}

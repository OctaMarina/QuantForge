"""Instrument identity, units and contract conversion.

Quantities are stored as exact scaled integers.  ``notional`` always means *quote-currency notional*
(e.g. USDT for BTCUSDT) unless a column explicitly says ``usd``.

Linear (spot, USDT-margined perpetual): qty is in base units, notional_quote = price * qty.
Inverse (coin-margined): qty is in contracts of ``contract_size`` USD; notional_usd = contracts *
contract_size and base_qty = contracts * contract_size / price.  Inverse support requires an explicit
contract size (and its history) in the configuration; otherwise the instrument is rejected as
unsupported rather than silently treated as spot.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal

# Decimals known to cover the venue's historical tick/step sizes (strings in Binance archives carry
# 8 decimals, but the economically used precision is lower; we keep the sum small enough that
# price_i * qty_i fits in int64 for single trades).  These are *representation* choices: any value
# needing more decimals raises PrecisionError instead of being rounded.
KNOWN = {
    ("binance", "spot", "BTCUSDT"): {"price_decimals": 2, "qty_decimals": 8, "wall_bin_size": "10"},
    ("binance", "spot", "ETHUSDT"): {"price_decimals": 2, "qty_decimals": 8, "wall_bin_size": "0.5"},
    ("binance", "perpetual", "BTCUSDT"): {"price_decimals": 2, "qty_decimals": 4, "wall_bin_size": "10"},
    ("binance", "perpetual", "ETHUSDT"): {"price_decimals": 2, "qty_decimals": 4, "wall_bin_size": "0.5"},
}


class UnsupportedInstrument(Exception):
    pass


@dataclass(frozen=True)
class InstrumentSpec:
    exchange: str
    market_type: str          # spot | perpetual
    symbol: str
    base: str
    quote: str
    contract_type: str = "linear"   # linear | inverse
    contract_size: str | None = None
    contract_size_currency: str | None = None
    price_decimals: int | None = None
    qty_decimals: int | None = None

    @property
    def price_scale(self) -> int:
        return 10 ** self.price_decimals

    @property
    def qty_scale(self) -> int:
        return 10 ** self.qty_decimals

    @property
    def notional_scale(self) -> int:
        return 10 ** (self.price_decimals + self.qty_decimals)

    @property
    def notional_currency(self) -> str:
        if self.contract_type == "inverse":
            return self.contract_size_currency or "USD"
        return self.quote

    def to_dict(self):
        return asdict(self)

    def archive_market(self) -> str:
        """Binance public-data trading type path component."""
        if self.market_type == "spot":
            return "spot"
        return "um" if self.contract_type == "linear" else "cm"


def from_config(cfg) -> InstrumentSpec:
    ins = cfg["instrument"]
    c = ins.get("contract") or {}
    ctype = c.get("type", "linear")
    if ins["market_type"] == "spot" and ctype != "linear":
        raise UnsupportedInstrument("spot instruments are linear")
    if ctype == "inverse" and not c.get("contract_size"):
        raise UnsupportedInstrument(
            "inverse contract requires instrument.contract.contract_size (and its currency); "
            "refusing to treat contracts as base units")
    known = KNOWN.get((ins["exchange"], ins["market_type"], ins["symbol"]), {})
    pdp = ins.get("price_decimals") if ins.get("price_decimals") is not None else known.get("price_decimals")
    qdp = ins.get("qty_decimals") if ins.get("qty_decimals") is not None else known.get("qty_decimals")
    if pdp is None or qdp is None:
        raise UnsupportedInstrument(
            f"no decimal specification for {ins['exchange']}/{ins['market_type']}/{ins['symbol']}: set "
            "instrument.price_decimals and instrument.qty_decimals to the venue's tick/step decimals")
    if pdp + qdp > 12:
        raise UnsupportedInstrument("price_decimals + qty_decimals > 12 risks int64 overflow of trade notional")
    return InstrumentSpec(
        exchange=ins["exchange"], market_type=ins["market_type"], symbol=ins["symbol"],
        base=ins["base"], quote=ins["quote"], contract_type=ctype,
        contract_size=str(c["contract_size"]) if c.get("contract_size") else None,
        contract_size_currency=c.get("contract_size_currency"),
        price_decimals=int(pdp), qty_decimals=int(qdp))


def decimal_to_scaled(value, decimals: int) -> int:
    d = Decimal(str(value))
    scaled = d * (Decimal(10) ** decimals)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"{value} not representable with {decimals} decimals")
    return int(scaled)

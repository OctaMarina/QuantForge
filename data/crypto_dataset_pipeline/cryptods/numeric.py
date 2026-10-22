"""Exact decimal-string -> scaled int64 conversion (no float round trip)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class PrecisionError(ValueError):
    """A value needs more decimals than the instrument's declared representation."""


def scaled_int(arr, decimals: int) -> np.ndarray:
    """Convert an array of decimal strings to int64 at scale 10**decimals, exactly.

    Raises PrecisionError if any value has non-zero digits beyond ``decimals`` or does not fit int64.
    """
    if not isinstance(arr, (pa.Array, pa.ChunkedArray)):
        arr = pa.array(np.asarray(arr).astype(str))
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if pa.types.is_integer(arr.type):
        return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.int64) * (10 ** decimals)
    if not pa.types.is_string(arr.type) and not pa.types.is_large_string(arr.type):
        arr = pc.cast(arr, pa.string())
    if arr.null_count:
        raise PrecisionError("null values in numeric column")
    try:
        dec = pc.cast(arr, pa.decimal128(38, decimals))
    except pa.ArrowInvalid as e:
        raise PrecisionError(f"value not representable with {decimals} decimals: {e}") from e
    words = np.frombuffer(dec.buffers()[1], dtype=np.int64)[2 * dec.offset: 2 * (dec.offset + len(dec))]
    words = words.reshape(-1, 2)
    lo, hi = words[:, 0], words[:, 1]
    # fits in int64 iff the high word is the sign extension of the low word
    if not np.array_equal(hi, np.where(lo < 0, -1, 0)):
        raise PrecisionError("value overflows int64 at this scale")
    return lo.copy()


def safe_mul_int64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise int64 product with an explicit overflow check."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    approx = a.astype(np.float64) * b.astype(np.float64)
    if np.any(np.abs(approx) >= 9.0e18):
        raise PrecisionError("int64 overflow computing price*qty; reduce price/qty decimals")
    return a * b

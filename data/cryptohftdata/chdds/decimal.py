"""Exact decimal-string -> scaled int64 parsing (numba), straight from Arrow string buffers.

Monetary fields are decimal strings in the raw Parquet files. We keep them exact as
integers in a fixed scale (e.g. price * 10**2, quantity * 10**3). A value with more
significant decimals than the scale is flagged (status=1) and the result is truncated —
callers must quarantine such rows; no silent rounding is ever applied.

status codes: 0 ok, 1 inexact (non-zero digits beyond scale), 2 malformed, 3 null, 4 overflow
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
from numba import njit

ST_OK, ST_INEXACT, ST_MALFORMED, ST_NULL, ST_OVERFLOW = 0, 1, 2, 3, 4


@njit(cache=True)
def _parse_kernel(data, offsets, base, n, scale, valid, out, status, ndec_out):
    for i in range(n):
        if valid is not None and valid.shape[0] > 0 and not valid[i]:
            out[i] = 0
            status[i] = ST_NULL
            ndec_out[i] = -1
            continue
        s = offsets[base + i]
        e = offsets[base + i + 1]
        neg = False
        if s < e and data[s] == 45:  # '-'
            neg = True
            s += 1
        if s >= e:
            out[i] = 0
            status[i] = ST_MALFORMED
            ndec_out[i] = -1
            continue
        v = 0
        seen_dot = False
        ndec = 0
        st = ST_OK
        ndig = 0
        sig_dec = 0  # number of decimals up to last non-zero decimal digit
        for j in range(s, e):
            c = data[j]
            if c == 46:  # '.'
                if seen_dot:
                    st = ST_MALFORMED
                    break
                seen_dot = True
                continue
            if c < 48 or c > 57:
                st = ST_MALFORMED
                break
            d = c - 48
            ndig += 1
            if seen_dot:
                ndec += 1
                if d != 0:
                    sig_dec = ndec
                if ndec > scale:
                    if d != 0:
                        st = ST_INEXACT
                    continue
            if v > 922337203685477580:
                st = ST_OVERFLOW
                break
            v = v * 10 + d
        if ndig == 0 and st == ST_OK:
            st = ST_MALFORMED
        if st == ST_OK or st == ST_INEXACT:
            k = ndec if ndec < scale else scale
            for _ in range(scale - k):
                v = v * 10
        out[i] = -v if neg else v
        status[i] = st
        ndec_out[i] = sig_dec


def parse_decimal(arr, scale: int):
    """Parse an Arrow (chunked) string array into (int64 values, uint8 status, int8 significant decimals)."""
    if isinstance(arr, pa.ChunkedArray):
        chunks = arr.chunks
    else:
        chunks = [arr]
    n = sum(len(c) for c in chunks)
    out = np.empty(n, np.int64)
    status = np.empty(n, np.uint8)
    ndec = np.empty(n, np.int8)
    pos = 0
    for c in chunks:
        m = len(c)
        if m == 0:
            continue
        if pa.types.is_dictionary(c.type):
            c = c.cast(c.type.value_type)
        if pa.types.is_large_string(c.type):
            off_dtype = np.int64
        elif pa.types.is_string(c.type):
            off_dtype = np.int32
        else:
            raise TypeError(f"expected string array, got {c.type}")
        bufs = c.buffers()
        offsets = np.frombuffer(bufs[1], dtype=off_dtype)
        data = np.frombuffer(bufs[2], dtype=np.uint8) if bufs[2] is not None else np.zeros(1, np.uint8)
        if c.null_count:
            valid = np.asarray(c.is_valid().to_numpy(zero_copy_only=False), dtype=np.bool_)
        else:
            valid = np.zeros(0, np.bool_)
        _parse_kernel(data, offsets, c.offset, m, scale, valid,
                      out[pos:pos + m], status[pos:pos + m], ndec[pos:pos + m])
        pos += m
    return out, status, ndec

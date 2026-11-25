import numpy as np
import pyarrow as pa

from chdds.decimal import ST_INEXACT, ST_MALFORMED, ST_NULL, ST_OK, parse_decimal


def test_exact_scaled_integers():
    arr = pa.array(["77826.80", "0.001", "0", "0.000", "12", "-0.00003447", "107142.6", None, "1.2.3", "", "0.0015"])
    v, st, nd = parse_decimal(arr, 3)
    assert v[0] == 77826800 and st[0] == ST_OK
    assert v[1] == 1 and v[2] == 0 and v[3] == 0 and v[4] == 12000
    assert v[5] == 0 and st[5] == ST_INEXACT          # more significant decimals than scale -> flagged, never rounded
    assert v[6] == 107142600
    assert st[7] == ST_NULL and st[8] == ST_MALFORMED and st[9] == ST_MALFORMED
    assert st[10] == ST_INEXACT


def test_trailing_zeros_are_exact():
    v, st, nd = parse_decimal(pa.array(["98146.61100000", "7620564167.11170000"]), 8)
    assert v[0] == 9814661100000 and st[0] == ST_OK
    assert v[1] == 762056416711170000 and st[1] == ST_OK
    v, st, nd = parse_decimal(pa.array(["98146.61100000"]), 3)
    assert v[0] == 98146611 and st[0] == ST_OK and nd[0] == 3


def test_chunked_and_sliced():
    arr = pa.chunked_array([pa.array(["1.5", "2.25"]), pa.array(["x", "3"]).slice(1)])
    v, st, _ = parse_decimal(arr, 2)
    assert list(v) == [150, 225, 300] and list(st) == [0, 0, 0]

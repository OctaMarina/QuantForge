"""Coverage reports: per feature x month (valid share, internal null runs) and per source x day."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nulls import REASONS, VALID


def feature_month_coverage(month: np.ndarray, cols: dict) -> pd.DataFrame:
    """``cols``: name -> Col (finalised).  ``month``: 'YYYY-MM' per row."""
    rows = []
    month = np.asarray(month)
    for name, c in cols.items():
        valid = c.reason == VALID
        for m in np.unique(month):
            sel = month == m
            v = valid[sel]
            n = int(sel.sum())
            nv = int(v.sum())
            internal = 0
            if nv:
                idx = np.flatnonzero(v)
                inner = ~v[idx[0]:idx[-1] + 1]
                # number of null runs strictly inside the first..last valid rows of the month
                if inner.any():
                    d = np.diff(np.concatenate([[0], inner.astype(np.int8), [0]]))
                    internal = int((d == 1).sum())
            reasons = pd.Series(c.reason[sel][~v]).map(REASONS).value_counts()
            rows.append({"feature": name, "month": m, "rows": n, "valid": nv,
                         "valid_share": nv / n if n else np.nan, "internal_null_runs": internal,
                         "top_null_reason": reasons.index[0] if len(reasons) else None,
                         "null_reasons": "; ".join(f"{k}:{int(v)}" for k, v in reasons.items())})
    return pd.DataFrame(rows)


def summarize(cov: pd.DataFrame) -> pd.DataFrame:
    g = cov.groupby("feature").agg(rows=("rows", "sum"), valid=("valid", "sum"),
                                   months_with_data=("valid", lambda s: int((s > 0).sum())),
                                   internal_null_runs=("internal_null_runs", "sum")).reset_index()
    g["valid_share"] = g["valid"] / g["rows"]
    return g


def source_coverage(tables: dict) -> pd.DataFrame:
    """Per auxiliary source x month: rows, first/last time and, for 1-minute candle tables, expected
    minutes and the number of internal missing minutes (gaps), so coverage is not reduced to first/last."""
    rows = []

    def one(name, df):
        if not isinstance(df, pd.DataFrame) or not len(df):
            rows.append({"source": name, "month": None, "rows": 0})
            return
        tcol = "period_start_us" if "period_start_us" in df else ("ts_us" if "ts_us" in df else
               ("available_at_us" if "available_at_us" in df else None))
        if tcol is None:
            rows.append({"source": name, "month": None, "rows": int(len(df))})
            return
        t = pd.to_numeric(df[tcol], errors="coerce").dropna().astype("int64")
        if not len(t):
            rows.append({"source": name, "month": None, "rows": int(len(df)), "note": f"{tcol} all null"})
            return
        month = pd.to_datetime(t, unit="us", utc=True).dt.strftime("%Y-%m")
        step = None
        if tcol == "period_start_us" and "period_end_us" in df:
            step = int((df["period_end_us"] - df["period_start_us"]).median())
        for m, g in t.groupby(month.values):
            rec = {"source": name, "month": m, "rows": int(len(g)), "first_us": int(g.min()), "last_us": int(g.max())}
            if step:
                exp = (int(g.max()) - int(g.min())) // step + 1
                rec["expected_between_first_last"] = int(exp)
                rec["internal_missing"] = int(exp - g.nunique())
            rows.append(rec)

    for k, v in tables.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, pd.DataFrame):
                    one(f"{k}.{kk}", vv)
                elif isinstance(vv, dict) and isinstance(vv.get("events"), pd.DataFrame):
                    one(f"{k}.{kk}.events", vv["events"])
        else:
            one(k, v)
    return pd.DataFrame(rows)

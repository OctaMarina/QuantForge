"""Controlled stop of long runs. A stage asks at a safe point (daily checkpoint, finished day, finished month) whether it must stop:
free disk below CHDDS_MIN_FREE_GB, or the STOP file exists. Nothing is ever deleted here."""
from __future__ import annotations

import os
import shutil

from . import paths as P

STOP_FILE = P.DERIVED / "regen_check" / "STOP"
EXIT_STOPPED = 75


class RunStopped(Exception):
    pass


def stop_reason() -> str | None:
    if STOP_FILE.exists():
        return f"stop file {STOP_FILE}"
    limit = float(os.environ.get("CHDDS_MIN_FREE_GB", "0"))
    free = shutil.disk_usage(P.DERIVED).free / 2**30
    if limit > 0 and free < limit:
        return f"free disk {free:.1f} GiB below the limit of {limit:.1f} GiB"
    return None

"""Disk needed by the regeneration, temporary files included, projected from the regen_dry_* runs of the same configuration.
Peak = the larger of two moments: end of split_book (book union and per-day book files both on disk), end of assemble (union removed,
book files removed month by month). Exit 1 if the peak with margin does not fit above the reserve. Writes derived/regen_check/preflight.json."""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

DRY = {"cross": 3, "liq": 3, "storm": 2, "marks": 2, "oi": 4}
LAYERS = ("features", "book_union", "book_features", "book_grid", "datasets")
MARGIN = 1.2
DAYS = (P.HISTORY_END - P.HISTORY_START).days + 1


def size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main():
    n = sum(DRY.values())
    per_day = {L: sum(size(P.DERIVED / L / f"regen_dry_{k}") for k in DRY) / n for L in LAYERS}
    gb = {L: per_day[L] * DAYS / 2**30 for L in LAYERS}
    state = size(P.DERIVED / "book_state" / "btcusdt_initial") / 2**30
    at_split = gb["features"] + gb["book_union"] + gb["book_features"] + gb["book_grid"] + state
    at_end = gb["features"] + gb["book_grid"] + state + gb["datasets"]
    peak = max(at_split, at_end)
    free = shutil.disk_usage(P.DERIVED).free / 2**30
    reserve = float(os.environ.get("CHDDS_MIN_FREE_GB", "10"))
    rep = dict(days=DAYS, projected_gib={k: round(v, 1) for k, v in gb.items()}, state_gib=round(state, 1), at_split_gib=round(at_split, 1), at_end_gib=round(at_end, 1), peak_gib=round(peak, 1),
               margin=MARGIN, peak_with_margin_gib=round(peak * MARGIN, 1), free_gib=round(free, 1), reserve_gib=reserve, without_dropping_gib=round(sum(gb.values()) + state, 1),
               fits=bool(free - peak * MARGIN >= reserve))
    (P.DERIVED / "regen_check").mkdir(exist_ok=True)
    (P.DERIVED / "regen_check" / "preflight.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return 0 if rep["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())

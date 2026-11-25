"""Writes docs/DATASET_V2.md from the measured coverage of a dataset. Usage: .venv/bin/python scripts/dataset_summary.py btcusdt_v2"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chdds import paths as P  # noqa: E402

NAMES = {"F1": "Preț și traiectorie (F1)", "F1/F7": "Trades vs. index", "F1/bar": "Bară", "F2": "Volatilitate (F2)", "F3": "Activitate și flux (F3)", "F4": "Book, stare (F4)", "F5": "Book, ferestre și OFI (F5)",
         "F6": "Execuție (F6)", "F6/F11": "Book + flux (F6/F11)", "F7": "Mark, basis, funding (F7)", "F7/F12": "Combinații funding (F7/F12)", "F8": "Open interest (F8)", "F9": "Lichidări (F9)", "F10": "Ticker, calendar (F10)"}
BARS = ["time_1m", "time_5m", "volume_50btc", "volume_adapt60s", "dollar_5musdt", "dollar_adapt60s", "cusum_5bps", "cusum_10bps", "cusum_adapt_c1"]
SCOPES = [("all", "Tot istoricul"), ("no_anchors", "Perioada fără ancore (2025-07-01 – 2026-04-10)"), ("crash_window_2025-10-09_11", "Crash, 9–11 octombrie 2025"), ("crash_2025-10-10", "Crash, doar 10 octombrie 2025")]


def n_(x):
    return f"{x:,}".replace(",", ".")


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}".replace(".", ",")


def table(df, first):
    head = [first] + BARS
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in df.iter_rows(named=True):
        out.append("| " + " | ".join([NAMES.get(r[df.columns[0]], r[df.columns[0]])] + [pct(r.get(b)) for b in BARS]) + " |")
    return out


def main():
    name = sys.argv[1]
    d = P.DERIVED / "datasets" / name
    man = json.loads((d / "manifest.json").read_text())
    ver = json.loads((d / "verification.json").read_text())
    c = pl.read_csv(d / "coverage.csv")
    k = pl.read_csv(d / "coverage_key.csv")
    st = pl.read_csv(d / "coverage_book_state.csv")
    inc = pl.read_csv(d / "coverage_incidents.csv")
    cnt = json.loads(next((P.DERIVED / "book_state" / name).glob("counters_*.json")).read_text())
    size = subprocess.run(["du", "-sh", str(d)], capture_output=True, text=True).stdout.split()[0]
    checks = [x for b in ver["bar_sets"].values() for x in b["checks"]]
    L = [f"# Dataset `{name}`: rezultate măsurate", "", f"Generat de `scripts/dataset_summary.py`. Locație: `derived/datasets/{name}/`, {size}, {n_(sum(b['rows'] for b in man['bar_sets'].values()))} bare" +
         f", {len(man['bar_sets'])} seturi. Mod `{man['book_provenance']['book_mode']}`, rezolver corectat. Datasetul `btcusdt_initial` și datele brute sunt neatinse.", "",
         "Procentele sunt cote din barele setului cu valoare prezentă, media pe coloanele familiei. Identitatea barei, coloanele `*_state` și metadatele de calitate nu intră în medie.", "",
         "## Bare", "", "| Set | Bare | Coloane | Coloane pentru model |", "|---|---|---|---|"]
    for b in man["bar_sets"].values():
        L.append(f"| {b['name']} | {n_(b['rows'])} | {b['n_columns']} | {len(b['model_feature_columns'])} |")
    L += ["", "## Verificări", "",
          f"- Book: {n_(cnt['nverify'])} verificări cu snapshot-uri, {cnt['nverify_fail']} eșuate, {cnt['verify_mismatch_levels']} niveluri nepotrivite; {n_(cnt['nanchor'])} ancore, {n_(cnt['nempty'])} porniri din gol, {cnt['ncross']} încrucișări (toate pe 2025-08-29).",
          f"- `verify_dataset.py` pe 9 zile: {n_(sum(x['bars_checked'] for x in checks))} bare, {n_(sum(x['bruteforce_comparisons'] for x in checks))} recalculări brute-force, {sum(x['bruteforce_errors'] for x in checks)} erori, {sum(x['conservation_mismatches'] for x in checks)} erori de conservare, {sum(x['book_values_when_invalid'] for x in checks)} valori sub măști invalide, {sum(len(b['unregistered']) for b in ver['bar_sets'].values())} coloane neînregistrate.",
          "", "## Proveniența book-ului (cotă din bare)", ""]
    s = st.with_columns((pl.col("rows") / pl.col("rows").sum().over("bars", "scope")).alias("share"))
    lab = {0: "ancorat (0)", 9: "reconstruit certificat (9)", 10: "necertificat (10)", 2: "vechi (2)", 1: "fără book (1)"}
    for sc, title in SCOPES:
        x = s.filter(pl.col("scope") == sc).with_columns(pl.col("book_state").replace_strict(lab, default="altele", return_dtype=pl.Utf8).alias("stare")).group_by("stare", "bars").agg(pl.col("share").sum())
        L += [f"**{title}**", ""] + table(x.pivot(on="bars", index="stare", values="share").sort("stare"), "Stare") + [""]
    L += ["## Acoperire pe familii", ""]
    for sc, title in SCOPES:
        L += [f"**{title}**", ""] + table(c.filter(pl.col("scope") == sc).pivot(on="bars", index="family", values="mean_coverage").sort("family"), "Familie") + [""]
    L += ["## Features reprezentative", ""]
    for sc, title in SCOPES[:3]:
        L += [f"**{title}**", ""] + table(k.filter(pl.col("scope") == sc).pivot(on="bars", index="feature", values="coverage").sort("feature"), "Feature") + [""]
    L += ["## Metadate de calitate", "", f"Coloane păstrate în fișiere, excluse implicit din features pentru model (`chdds.dataset.model_columns`, `model_feature_columns` în manifest): {', '.join('`' + x + '`' for x in man['quality_columns'])}.", "",
          "Incidente pentru excludere în experimente (`quality_incidents.json`, `chdds.dataset.incident_rows`). Nu sunt coloane în dataset și nu se folosesc ca input:", ""]
    for i in man["quality_incidents"]:
        L.append(f"- `{i['id']}`: {i['start']} – {i['end']}. {i['reason']}")
    L += ["", "Bare care ating intervalul: " + ", ".join(f"{r['bars']} {r['rows_touching_incident']}" for r in inc.iter_rows(named=True)) + ".", ""]
    Path("docs/DATASET_V2.md").write_text("\n".join(L))
    print("written", len(L))


if __name__ == "__main__":
    main()

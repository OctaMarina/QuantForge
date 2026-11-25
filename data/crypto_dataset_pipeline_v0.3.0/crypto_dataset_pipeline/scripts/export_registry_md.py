#!/usr/bin/env python3
"""Render a build's feature registry (``<bar>_registry.json``) as Markdown (docs/FEATURES.md)."""
import json
import sys
from pathlib import Path


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def main(reg_path: str, out: str):
    reg = json.loads(Path(reg_path).read_text())
    fams: dict = {}
    for r in reg:
        fams.setdefault(r["family"], []).append(r)
    lines = ["# Registrul feature-urilor (generat)", "",
             f"Generat din `{Path(reg_path).name}` al eșantionului real (bare de timp). Numele coloanelor care conțin "
             "parametri (ferestre, praguri, benzi, mărimi) se schimbă odată cu configurația; registrul JSON scris la "
             "fiecare build este sursa autoritativă pentru acel build.", ""]
    for fam, rows in fams.items():
        lines += [f"## {fam}", "", "| coloană | formulă / variantă | unitate | clock | disponibilitate / staleness | natură | strict | statut | motive null |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            av = r["availability"] + (f"; {r['staleness']}" if r.get("staleness") else "")
            frm = r["formula"] + (f" — *{r['variant']}*" if r.get("variant") else "")
            lines.append(f"| `{r['name']}` | {esc(frm)} | {esc(r['unit'])} | {esc(r['clock'])} | {esc(av)} | {r['nature']} | "
                         f"{'da' if r['strict_eligible'] else 'nu'} | {r['status']} | {esc(', '.join(r['null_reasons']))} |")
        notes = {r["notes"] for r in rows if r.get("notes")}
        refs = {r["reference"] for r in rows if r.get("reference")}
        if refs:
            lines += ["", "Referință: " + " / ".join(sorted(refs))]
        if notes:
            lines += ["", "Note:"] + [f"- {esc(n)}" for n in sorted(notes)]
        lines.append("")
    Path(out).write_text("\n".join(lines))
    print(f"wrote {out} ({len(reg)} columns)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

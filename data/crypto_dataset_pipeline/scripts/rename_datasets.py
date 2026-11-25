#!/usr/bin/env python
"""
Redenumește dataseturile vechi data/datasets/<config_hash>/ în formatul lizibil folosit acum de build:
<SYMBOL>_<market>_<bară>_<prag>_<start>-<end>_<hash8>  (numele se citește din manifest.json).

    python scripts/rename_datasets.py            # doar afișează ce ar redenumi
    python scripts/rename_datasets.py --apply    # redenumește
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryptods import config as cfgmod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/datasets")
    ap.add_argument("--apply", action="store_true", help="redenumește efectiv (implicit: dry-run)")
    a = ap.parse_args()
    root = Path(a.root)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        man = d / "manifest.json"
        if not man.exists():
            print(f"SKIP {d.name}: fără manifest.json (build neterminat?)")
            continue
        m = json.loads(man.read_text())
        name = cfgmod.dataset_name(m["config"], m.get("config_hash"))
        if name == d.name:
            continue
        dst = root / name
        if dst.exists():
            print(f"SKIP {d.name}: {name} există deja")
            continue
        print(f"{d.name} -> {name}")
        if a.apply:
            d.rename(dst)
    if not a.apply:
        print("(dry-run; rulează cu --apply pentru a redenumi)")


if __name__ == "__main__":
    main()

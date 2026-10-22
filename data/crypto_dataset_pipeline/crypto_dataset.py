#!/usr/bin/env python3
"""Entry point: ``python crypto_dataset.py <command> [--config configs/btc_demo.yaml] [--set k=v]``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryptods.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

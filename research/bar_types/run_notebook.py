"""Execute the bar-type notebook with nbclient (from this folder).  QF_BAR_TYPES=time restricts it to time bars (tests).

    python run_notebook.py [output.ipynb]
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
src = HERE / "bar_type_frequency_meta_labeling.ipynb"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else src
nb = nbformat.read(src, as_version=4)
t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(HERE)}},
                        allow_errors=False)
try:
    client.execute()
finally:
    nbformat.write(nb, out)
    print(f"executed in {(time.time() - t0) / 60:.1f} min -> {out}")

import hashlib
import json

import pytest

from .conftest import NOTEBOOK, REFERENCE_FILE


def test_frozen_reference_matches_notebook():
    if not NOTEBOOK.exists():
        pytest.skip("original notebook not available")
    cell = ''.join(json.loads(NOTEBOOK.read_text())["cells"][3]["source"])
    text = REFERENCE_FILE.read_text()
    assert cell in text, "tests/strategies/reference no longer matches code cell 3 of the notebook"
    assert hashlib.sha256(cell.encode()).hexdigest() in text.splitlines()[1] + text.splitlines()[2]

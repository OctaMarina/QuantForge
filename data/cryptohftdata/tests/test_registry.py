import glob

import pyarrow.parquet as pq
import pytest

from chdds import paths as P
from chdds import registry as R
from chdds.book.replay import ReplayConfig


def test_book_columns_registered():
    cols = ReplayConfig().columns()
    assert not R.check_columns(cols)
    assert all(len(R.match(c)) == 1 for c in cols)


def test_registry_patterns_unique():
    pats = [f.pattern for f in R.REGISTRY]
    assert len(pats) == len(set(pats))


@pytest.mark.skipif(not glob.glob(str(P.DERIVED / "datasets" / "*" / "*" / "*.parquet")), reason="no dataset built")
def test_every_dataset_column_is_registered_exactly_once():
    cols = set()
    for f in glob.glob(str(P.DERIVED / "datasets" / "*" / "*" / "*.parquet"))[:200]:
        cols |= set(pq.read_schema(f).names)
    assert not R.check_columns(sorted(cols))
    assert not [c for c in cols if len(R.match(c)) > 1]

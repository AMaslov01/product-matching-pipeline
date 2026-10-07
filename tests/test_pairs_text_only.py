from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.pairs import prepare_pairs
from matchcup.parser import canonicalize_record

TRAINER_COLUMNS = {"id1", "id2", "text_a", "text_b", "target", "category"}


def _canonical_items(tmp_path: Path) -> Path:
    raw = [
        {"id": 1, "name": "мяч футбольный adidas", "attributes": '{"бренд": "adidas"}',
         "category": "Спорт и отдых"},
        {"id": 2, "name": "мяч футбольный адидас", "attributes": '{"бренд": "адидас"}',
         "category": "Спорт и отдых"},
        {"id": 3, "name": "чайник bosch", "attributes": '{"бренд": "bosch"}',
         "category": "Бытовая техника"},
        {"id": 4, "name": "чайник электрический bosch", "attributes": '{"бренд": "bosch"}',
         "category": "Бытовая техника"},
    ]
    path = tmp_path / "items.parquet"
    pq.write_table(pa.Table.from_pylist([canonicalize_record(r) for r in raw]), path)
    return path


def _matches(tmp_path: Path) -> Path:
    path = tmp_path / "matches.parquet"
    pq.write_table(
        pa.table({"id1": [1, 3], "id2": [2, 4], "target": [1.0, 0.0], "row_id": [0, 1]}), path
    )
    return path


def test_text_only_keeps_every_column_the_trainer_reads(tmp_path: Path):
    items, matches = _canonical_items(tmp_path), _matches(tmp_path)
    lean = tmp_path / "lean.parquet"
    prepare_pairs(items, matches, lean, include_features=False)
    table = pq.read_table(lean)
    assert TRAINER_COLUMNS <= set(table.column_names)
    assert table.num_rows == 2


def test_text_only_drops_the_catboost_features(tmp_path: Path):
    items, matches = _canonical_items(tmp_path), _matches(tmp_path)
    full, lean = tmp_path / "full.parquet", tmp_path / "lean.parquet"
    prepare_pairs(items, matches, full)
    prepare_pairs(items, matches, lean, include_features=False)

    full_columns = set(pq.read_table(full).column_names)
    lean_columns = set(pq.read_table(lean).column_names)
    assert "name_token_jaccard" in full_columns
    assert "name_token_jaccard" not in lean_columns
    # Everything dropped is a feature; nothing the trainer needs went with it.
    assert TRAINER_COLUMNS <= lean_columns
    assert lean_columns < full_columns


def test_both_paths_agree_on_the_shared_columns(tmp_path: Path):
    """The lean path must not change text or category, only omit features."""
    items, matches = _canonical_items(tmp_path), _matches(tmp_path)
    full, lean = tmp_path / "full.parquet", tmp_path / "lean.parquet"
    prepare_pairs(items, matches, full)
    prepare_pairs(items, matches, lean, include_features=False)

    a = pq.read_table(full).to_pydict()
    b = pq.read_table(lean).to_pydict()
    for column in sorted(TRAINER_COLUMNS):
        assert a[column] == b[column], f"{column} differs between the two paths"

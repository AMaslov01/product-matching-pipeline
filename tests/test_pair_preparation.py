import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import matchcup.transductive as transductive
from matchcup.pairs import prepare_pairs
from matchcup.parser import canonicalize_parquet


def test_end_to_end_pair_preparation(tmp_path: Path) -> None:
    raw_items = tmp_path / "items.parquet"
    canonical = tmp_path / "canonical.parquet"
    matches = tmp_path / "matches.parquet"
    output = tmp_path / "pairs.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2],
                "name": ["Шампунь 1 л", "Шампунь 1000 мл"],
                "attributes": [
                    json.dumps({"Бренд": "Тест", "Объем товара, мл": "1000"}),
                    json.dumps({"Бренд": "Test", "Объем товара, мл": "1000"}),
                ],
                "category": ["Красота и гигиена", "Красота и гигиена"],
            }
        ),
        raw_items,
    )
    pq.write_table(
        pa.table({"id1": [1], "id2": [2], "target": [1.0], "fold": [0], "row_id": [0]}),
        matches,
    )
    canonicalize_parquet(raw_items, canonical, workers=1, batch_size=2)
    result = prepare_pairs(canonical, matches, output, chunk_size=2)
    assert result == {"rows": 1, "missing_items": 0}
    row = pq.read_table(output).to_pylist()[0]
    assert row["target"] == 1.0
    assert row["fold"] == 0
    assert row["measurement_compatible_types"] >= 1.0
    assert "name_a:" in row["text_a"]


def test_large_catalogue_pair_preparation_adds_transductive_features(
    tmp_path: Path, monkeypatch
) -> None:
    """The public pair-preparation seam keeps H12 features on the DuckDB path."""
    small_items = tmp_path / "small_items.parquet"
    large_items = tmp_path / "large_items.parquet"
    matches = tmp_path / "matches.parquet"
    transductive_items = tmp_path / "transductive_items.parquet"
    in_memory_output = tmp_path / "in_memory.parquet"
    duckdb_output = tmp_path / "duckdb.parquet"
    item_columns = {
        "id": [1, 2],
        "name_norm": ["rare 123", "rare 123"],
        "brand_norm": ["brand", "brand"],
        "identifier_tokens": ['["123"]', '["123"]'],
    }
    pq.write_table(pa.table(item_columns), small_items)
    pq.write_table(pa.table(item_columns), transductive_items)
    pq.write_table(
        pa.table({"id1": [1], "id2": [2], "target": [1.0], "fold": [0], "row_id": [0]}),
        matches,
    )

    rows = 1_000_001
    pq.write_table(
        pa.table(
            {
                "id": pa.array(range(1, rows + 1), type=pa.int64()),
                "name_norm": pa.array(["rare 123"] * rows),
                "brand_norm": pa.array(["brand"] * rows),
                "identifier_tokens": pa.array(['["123"]'] * rows),
            }
        ),
        large_items,
        row_group_size=8192,
        compression="zstd",
    )
    assert pq.ParquetFile(large_items).metadata.num_rows > 1_000_000

    original_statistics = transductive.build_transductive_statistics
    calls: list[Path] = []

    def counted_statistics(path, **kwargs):
        calls.append(Path(path))
        return original_statistics(path, **kwargs)

    monkeypatch.setattr(transductive, "build_transductive_statistics", counted_statistics)
    prepare_pairs(
        small_items,
        matches,
        in_memory_output,
        transductive_items_path=transductive_items,
    )
    calls.clear()
    prepare_pairs(
        large_items,
        matches,
        duckdb_output,
        transductive_items_path=transductive_items,
    )
    assert calls == [transductive_items]

    in_memory = pq.read_table(in_memory_output).to_pylist()[0]
    duckdb = pq.read_table(duckdb_output).to_pylist()[0]
    assert duckdb == in_memory


def test_transductive_features_are_symmetric_for_swapped_pair_ids(tmp_path: Path) -> None:
    items = tmp_path / "items.parquet"
    forward_matches = tmp_path / "forward.parquet"
    swapped_matches = tmp_path / "swapped.parquet"
    forward_output = tmp_path / "forward_features.parquet"
    swapped_output = tmp_path / "swapped_features.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3],
                "name_norm": ["rare 123", "rare 123", "common 999"],
                "brand_norm": ["brand", "brand", "other"],
                "identifier_tokens": ['["123"]', '["123"]', '["999"]'],
            }
        ),
        items,
    )
    pq.write_table(
        pa.table({"id1": [1], "id2": [2], "target": [1.0], "fold": [0], "row_id": [0]}),
        forward_matches,
    )
    pq.write_table(
        pa.table({"id1": [2], "id2": [1], "target": [1.0], "fold": [0], "row_id": [0]}),
        swapped_matches,
    )
    prepare_pairs(items, forward_matches, forward_output, transductive_items_path=items)
    prepare_pairs(items, swapped_matches, swapped_output, transductive_items_path=items)
    forward = pq.read_table(forward_output).to_pandas()
    swapped = pq.read_table(swapped_output).to_pandas()
    columns = [name for name in forward.columns if name.startswith("transductive_")]
    assert columns
    assert forward[columns].equals(swapped[columns])


def test_pair_preparation_rejects_statistics_from_a_different_profile(tmp_path: Path) -> None:
    items = tmp_path / "items.parquet"
    matches = tmp_path / "matches.parquet"
    output = tmp_path / "pairs.parquet"
    rows = [
        {
            "id": 1,
            "category": "A",
            "name_raw": "one",
            "name_norm": "one",
            "name_translit": "one",
            "attribute_count": 1,
            "parse_error": False,
            "brand_norm": "brand",
            "identifier_tokens": "[]",
            "remaining_attributes": "material=steel",
        },
        {
            "id": 2,
            "category": "A",
            "name_raw": "two",
            "name_norm": "two",
            "name_translit": "two",
            "attribute_count": 1,
            "parse_error": False,
            "brand_norm": "brand",
            "identifier_tokens": "[]",
            "remaining_attributes": "material=glass",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), items)
    pq.write_table(
        pa.table({"id1": [1], "id2": [2], "target": [1.0], "fold": [0], "row_id": [0]}),
        matches,
    )
    statistics = transductive.build_transductive_statistics(items, profile="h12-v1-c1-c2")

    with pytest.raises(ValueError, match="does not match pair preparation"):
        prepare_pairs(items, matches, output, transductive_statistics=statistics)

    prepare_pairs(
        items,
        matches,
        output,
        transductive_statistics=statistics,
        transductive_profile="h12-v1-c1-c2",
    )
    assert "aligned_attribute_key_idf_weighted_jaccard" in pq.read_table(output).schema.names

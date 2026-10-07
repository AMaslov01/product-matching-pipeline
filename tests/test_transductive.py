from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.features import pair_features
from matchcup.parser import canonicalize_record
from matchcup.transductive import (
    build_disk_backed_transductive_statistics,
    build_transductive_statistics,
)
from matchcup.transductive_profile import C1_FEATURE_NAMES, C2_FEATURE_NAMES


def _item(item_id: int, name: str, brand: str, article: str) -> dict:
    return canonicalize_record(
        {
            "id": item_id,
            "name": name,
            "attributes": json.dumps({"Бренд": brand, "Артикул": article}),
            "category": "A",
        }
    )


def test_item_statistics_are_scale_invariant_and_pair_symmetric(tmp_path: Path) -> None:
    raw = tmp_path / "items.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3],
                "name": ["Rare 123", "Rare 123", "Common"],
                "attributes": [
                    json.dumps({"Бренд": "A", "Артикул": "123"}),
                    json.dumps({"Бренд": "A", "Артикул": "123"}),
                    json.dumps({"Бренд": "B", "Артикул": "999"}),
                ],
                "category": ["A", "A", "A"],
            }
        ),
        raw,
    )
    statistics = build_transductive_statistics(raw, batch_size=1)
    left = statistics.annotate(_item(1, "Rare 123", "A", "123"))
    right = statistics.annotate(_item(2, "Rare 123", "A", "123"))

    forward = pair_features(left, right)
    backward = pair_features(right, left)

    assert forward == backward
    assert forward["transductive_rarest_shared_token_idf"] > 0.0
    assert 0.0 < forward["transductive_identifier_df"] < 1.0
    assert forward["transductive_brand_frequency_percentile_min"] == 1.0


def test_disk_backed_statistics_match_in_memory_statistics(tmp_path: Path) -> None:
    raw = tmp_path / "items.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3],
                "name": ["Rare 123", "Rare 123", "Common"],
                "attributes": [
                    json.dumps({"Бренд": "A", "Артикул": "123"}),
                    json.dumps({"Бренд": "A", "Артикул": "123"}),
                    json.dumps({"Бренд": "B", "Артикул": "999"}),
                ],
                "category": ["A", "A", "A"],
            }
        ),
        raw,
    )
    canonical = tmp_path / "canonical.parquet"
    rows = [
        _item(1, "Rare 123", "A", "123"),
        _item(2, "Rare 123", "A", "123"),
        _item(3, "Common", "B", "999"),
    ]
    pq.write_table(pa.Table.from_pylist(rows), canonical)
    in_memory = build_transductive_statistics(raw)
    disk_backed = build_disk_backed_transductive_statistics(canonical, tmp_path / "stats.duckdb")
    try:
        assert disk_backed.annotate(rows[0]) == in_memory.annotate(rows[0])
        assert disk_backed.annotate(rows[2]) == in_memory.annotate(rows[2])
    finally:
        disk_backed.close()


def test_c1_c2_statistics_are_category_local_symmetric_and_match_duckdb(tmp_path: Path) -> None:
    raw = tmp_path / "items.parquet"
    rows = [
        canonicalize_record(
            {
                "id": 1,
                "name": "Common Flash",
                "attributes": json.dumps(
                    {"Бренд": "A", "Артикул": "1", "Материал": "Стекло"}
                ),
                "category": "A",
            }
        ),
        canonicalize_record(
            {
                "id": 2,
                "name": "Common Lamp",
                "attributes": json.dumps(
                    {"Бренд": "A", "Артикул": "2", "Материал": "Стекло", "Сезон": "Лето"}
                ),
                "category": "A",
            }
        ),
        canonicalize_record(
            {
                "id": 3,
                "name": "Only A",
                "attributes": json.dumps({"Бренд": "A", "Артикул": "3", "Сезон": "Лето"}),
                "category": "A",
            }
        ),
        canonicalize_record(
            {
                "id": 4,
                "name": "Common Flash",
                "attributes": json.dumps({"Бренд": "B", "Артикул": "4", "Материал": "Сталь"}),
                "category": "B",
            }
        ),
        canonicalize_record(
            {
                "id": 5,
                "name": "No Category",
                "attributes": json.dumps({"Бренд": "C", "Артикул": "5", "Материал": "Дерево"}),
                "category": "",
            }
        ),
    ]
    pq.write_table(pa.Table.from_pylist(rows), raw)

    in_memory = build_transductive_statistics(raw, profile="h12-v1-c1-c2", batch_size=2)
    disk_backed = build_disk_backed_transductive_statistics(
        raw, tmp_path / "stats-c1-c2.duckdb", profile="h12-v1-c1-c2", batch_size=2
    )
    try:
        annotated = [in_memory.annotate(row) for row in rows]
        assert annotated[4]["_transductive_category_name_idf"] == {}
        assert annotated[0]["_transductive_category_name_idf"]["flash"] == 1.0
        assert annotated[0]["_transductive_category_name_idf"]["common"] < 1.0
        for row, expected in zip(rows, annotated, strict=True):
            assert disk_backed.annotate(row) == expected

        forward = pair_features(annotated[0], annotated[1])
        backward = pair_features(annotated[1], annotated[0])
        assert forward == backward
        assert set(C1_FEATURE_NAMES) <= set(forward)
        assert set(C2_FEATURE_NAMES) <= set(forward)
        assert forward["transductive_category_rarest_shared_token_idf"] > 0.0
        assert forward["aligned_attribute_common_key_idf_sum"] > 0.0
        assert forward["aligned_attribute_key_idf_weighted_jaccard"] > 0.0
        no_category = pair_features(annotated[4], annotated[4])
        assert set(C1_FEATURE_NAMES) <= set(no_category)
        assert all(no_category[name] == 0.0 for name in C1_FEATURE_NAMES)
    finally:
        disk_backed.close()


def test_legacy_profile_neither_counts_nor_emits_c1_c2_features(tmp_path: Path) -> None:
    raw = tmp_path / "items.parquet"
    rows = [_item(1, "Rare 123", "A", "123"), _item(2, "Rare 123", "A", "123")]
    pq.write_table(pa.Table.from_pylist(rows), raw)

    statistics = build_transductive_statistics(raw)
    left, right = (statistics.annotate(row) for row in rows)
    features = pair_features(left, right)

    assert statistics.category_item_count == {}
    assert statistics.category_name_token_document_frequency == {}
    assert statistics.attribute_key_document_frequency == {}
    assert "_transductive_category_name_idf" not in left
    assert "_transductive_attribute_key_idf" not in left
    assert not (set(C1_FEATURE_NAMES) | set(C2_FEATURE_NAMES)) & set(features)

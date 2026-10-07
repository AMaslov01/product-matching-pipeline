from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.proxy_shift import diagnose_proxy_domain_shift


def _write(path, rows) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_proxy_shift_uses_only_source_covariates_and_excludes_seen_llm_ids(tmp_path) -> None:
    human_items = tmp_path / "items_human.parquet"
    llm_items = tmp_path / "items.parquet"
    human_matches = tmp_path / "matches.parquet"
    llm_matches = tmp_path / "matches_llm.parquet"
    silver_matches = tmp_path / "silver.parquet"
    report = tmp_path / "proxy.json"
    _write(
        human_items,
        [
            {"id": 1, "name": "alpha brand a", "attributes": '{"бренд":"a"}', "category": "A"},
            {"id": 2, "name": "alpha brand b", "attributes": '{"бренд":"b"}', "category": "A"},
            {"id": 3, "name": "alpha brand c", "attributes": '{"бренд":"c"}', "category": "A"},
            {"id": 4, "name": "beta brand a", "attributes": '{"бренд":"a"}', "category": "B"},
            {"id": 5, "name": "beta brand b", "attributes": '{"бренд":"b"}', "category": "B"},
            {"id": 6, "name": "beta brand c", "attributes": '{"бренд":"c"}', "category": "B"},
        ],
    )
    _write(
        llm_items,
        [
            {
                "id": 101,
                "name": "long alpha listing one",
                "attributes": '{"цвет":"красный"}',
                "category": "A",
            },
            {
                "id": 102,
                "name": "long alpha listing two",
                "attributes": '{"цвет":"синий"}',
                "category": "A",
            },
            {
                "id": 103,
                "name": "long alpha listing three",
                "attributes": '{"цвет":"зелёный"}',
                "category": "A",
            },
            {
                "id": 104,
                "name": "long beta listing one",
                "attributes": '{"размер":"m"}',
                "category": "B",
            },
            {
                "id": 105,
                "name": "long beta listing two",
                "attributes": '{"размер":"l"}',
                "category": "B",
            },
            {
                "id": 106,
                "name": "long beta listing three",
                "attributes": '{"размер":"xl"}',
                "category": "B",
            },
            {"id": 107, "name": "seen alpha", "attributes": '{}', "category": "A"},
        ],
    )
    _write(
        human_matches,
        [
            {"id1": 1, "id2": 2, "target": 0.0},
            {"id1": 1, "id2": 3, "target": 1.0},
            {"id1": 4, "id2": 5, "target": 0.0},
            {"id1": 4, "id2": 6, "target": 1.0},
        ],
    )
    _write(
        llm_matches,
        [
            {"id1": 101, "id2": 102, "target": 0.0},
            {"id1": 101, "id2": 103, "target": 1.0},
            {"id1": 104, "id2": 105, "target": 0.0},
            {"id1": 104, "id2": 106, "target": 1.0},
            {"id1": 101, "id2": 107, "target": 1.0},
        ],
    )
    _write(silver_matches, [{"id1": 107, "id2": 108, "target": 0.5}])

    result = diagnose_proxy_domain_shift(
        human_items,
        human_matches,
        llm_items,
        llm_matches,
        silver_matches,
        report,
        max_pairs_per_source_category=2,
        min_pairs_per_source_category=2,
        temp_dir=tmp_path / "scratch",
    )

    assert result["target_labels_used"] is False
    assert result["llm_pair_labels_used"] is False
    assert result["rows"] == {"total": 8, "per_source": 4}
    assert set(result["categories"]) == {"A", "B"}
    assert all(record["fully_new_llm_pairs"] == 2 for record in result["categories"].values())
    assert json.loads(report.read_text(encoding="utf-8"))["purpose"].startswith("proxy_")

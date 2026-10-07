from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.holdout import (
    _balanced_quotas,
    describe_new_item_llm_slices,
    evaluate_novel_item_holdout,
    make_natural_rate_public_proxy,
    make_novel_item_holdout,
    select_score_mode,
)
from matchcup.pairs import prepare_pairs
from matchcup.parser import canonicalize_parquet


def _write_items(path: Path) -> None:
    ids = list(range(1, 15))
    categories = ["A", "A", "B", "B", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C"]
    pq.write_table(
        pa.table(
            {
                "id": ids,
                "name": [f"Товар {item_id}" for item_id in ids],
                "attributes": [json.dumps({"Бренд": "Тест"}) for _ in ids],
                "category": categories,
            }
        ),
        path,
    )


def test_novel_holdout_excludes_training_items_and_nonbinary_labels(tmp_path: Path) -> None:
    items = tmp_path / "items.parquet"
    llm = tmp_path / "matches_llm.parquet"
    gold = tmp_path / "gold.parquet"
    silver = tmp_path / "silver.parquet"
    holdout = tmp_path / "holdout.parquet"
    _write_items(items)
    pq.write_table(pa.table({"id1": [1], "id2": [2], "target": [1.0]}), gold)
    pq.write_table(pa.table({"id1": [3], "id2": [4], "target": [0.4]}), silver)
    pq.write_table(
        pa.table(
            {
                "id1": [5, 7, 9, 11, 13, 5, 3, 7],
                "id2": [6, 8, 10, 12, 14, 2, 6, 8],
                "target": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.4],
            }
        ),
        llm,
    )

    result = make_novel_item_holdout(
        items, llm, gold, silver, holdout, max_rows=4, temp_dir=tmp_path
    )
    rows = pq.read_table(holdout).to_pylist()

    assert result == {
        "rows": 4,
        "eligible_rows": 5,
        "categories": 2,
        "excluded_incomplete_categories": 1,
    }
    assert {row["target"] for row in rows} == {0.0, 1.0}
    assert all({row["id1"], row["id2"]}.isdisjoint({1, 2, 3, 4}) for row in rows)
    assert {row["category"] for row in rows} == {"A", "B"}
    assert [row["row_id"] for row in rows] == [0, 1, 2, 3]

    canonical = tmp_path / "canonical.parquet"
    prepared = tmp_path / "prepared.parquet"
    ids = {int(value) for row in rows for value in (row["id1"], row["id2"])}
    canonicalize_parquet(items, canonical, accepted_ids=ids, workers=1, batch_size=4)
    assert prepare_pairs(canonical, holdout, prepared) == {"rows": 4, "missing_items": 0}
    prepared_rows = pq.read_table(prepared).to_pylist()
    assert {row["category"] for row in prepared_rows} == {"A", "B"}
    assert all("name_a:" in row["text_a"] and "name_b:" in row["text_b"] for row in prepared_rows)


def test_balanced_quotas_and_score_selection() -> None:
    quotas = _balanced_quotas(
        {("A", 0.0): 100, ("A", 1.0): 1, ("B", 0.0): 100, ("B", 1.0): 100},
        max_rows=16,
    )
    assert sum(quotas.values()) == 16
    assert quotas[("A", 1.0)] == 1
    assert max(quotas.values()) - min(value for value in quotas.values() if value) <= 4
    small = _balanced_quotas(
        {("A", 0.0): 100, ("A", 1.0): 100, ("B", 0.0): 100, ("B", 1.0): 100},
        max_rows=5,
    )
    assert sum(small.values()) == 5
    assert select_score_mode(0.50, 0.5019) == "text"
    assert select_score_mode(0.50, 0.503) == "hybrid"


def test_public_proxy_keeps_natural_rate_and_excludes_pretraining_items(tmp_path: Path) -> None:
    items = tmp_path / "items.parquet"
    llm = tmp_path / "matches_llm.parquet"
    gold = tmp_path / "gold.parquet"
    silver = tmp_path / "silver.parquet"
    proxy = tmp_path / "proxy.parquet"
    _write_items(items)
    pq.write_table(pa.table({"id1": [1], "id2": [2], "target": [1.0]}), gold)
    pq.write_table(pa.table({"id1": [3], "id2": [4], "target": [0.5]}), silver)
    pq.write_table(
        pa.table(
            {
                "id1": [5, 7, 9, 11, 13, 5, 3],
                "id2": [6, 8, 10, 12, 14, 8, 6],
                "target": [1.0, 8 / 9, 0.0, 1.0, 0.0, 4 / 9, 1.0],
            }
        ),
        llm,
    )

    report = make_natural_rate_public_proxy(
        items, llm, gold, silver, proxy, max_rows_per_category=10, temp_dir=tmp_path
    )
    rows = pq.read_table(proxy).to_pylist()

    assert report["rows"] == 5
    assert report["categories"] == 2
    assert report["excluded_incomplete_categories"] == 1
    assert report["label"] == "target == 1.0"
    assert {row["target"] for row in rows} == {0.0, 1.0}
    assert all({row["id1"], row["id2"]}.isdisjoint({1, 2, 3, 4}) for row in rows)
    assert [row["row_id"] for row in rows] == list(range(5))
    persisted = json.loads(proxy.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert persisted["sample_macro_prevalence"] == report["sample_macro_prevalence"]


def test_new_item_llm_diagnostics_keep_weak_labels_out_of_model_selection(tmp_path: Path) -> None:
    items = tmp_path / "items.parquet"
    llm = tmp_path / "matches_llm.parquet"
    gold = tmp_path / "gold.parquet"
    silver = tmp_path / "silver.parquet"
    report = tmp_path / "diagnostic.json"
    _write_items(items)
    pq.write_table(pa.table({"id1": [1], "id2": [2], "target": [1.0]}), gold)
    pq.write_table(pa.table({"id1": [3], "id2": [4], "target": [0.0]}), silver)
    pq.write_table(
        pa.table(
            {
                "id1": [5, 7, 9, 11, 13, 5, 3],
                "id2": [6, 8, 10, 12, 14, 6, 6],
                "target": [0.0, 1.0, 4 / 9, 5 / 9, 1.0, 4 / 9, 5 / 9],
            }
        ),
        llm,
    )

    result = describe_new_item_llm_slices(items, llm, gold, silver, report, temp_dir=tmp_path)

    assert result["model_selection_allowed"] is False
    assert result["rows"] == 6
    assert result["hard_slice"]["rows"] == 3
    assert json.loads(report.read_text(encoding="utf-8"))["purpose"].endswith("model_selection")


def test_evaluate_novel_holdout_selects_text_and_writes_submission_gate(
    tmp_path: Path, monkeypatch
) -> None:
    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    report = tmp_path / "report.json"
    pq.write_table(
        pa.table(
            {
                "id1": [1, 2, 3, 4],
                "id2": [11, 12, 13, 14],
                "target": [0.0, 1.0, 0.0, 1.0],
                "category": ["A", "A", "B", "B"],
                "feature": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        pairs,
    )
    pq.write_table(
        pa.table(
            {
                "row_index": [0, 1, 2, 3],
                "id1": [1, 2, 3, 4],
                "id2": [11, 12, 13, 14],
                "text_score": [0.1, 0.9, 0.2, 0.8],
            }
        ),
        scores,
    )

    class FakeFusion:
        def __init__(self, _directory: Path) -> None:
            pass

        def predict(self, records: list[dict[str, object]]) -> np.ndarray:
            assert len(records) == 4
            return np.array([0.2, 0.8, 0.3, 0.7])

    monkeypatch.setattr("matchcup.holdout.FusionPredictor", FakeFusion)
    result = evaluate_novel_item_holdout(
        pairs, scores, tmp_path / "fusion", report, min_hybrid_gain=0.002
    )

    assert result["selected_score_mode"] == "text"
    assert result["submission_allowed"] is True
    assert json.loads(report.read_text(encoding="utf-8"))["rows"] == 4

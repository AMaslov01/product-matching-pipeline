from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from matchcup.schema_robustness_gate import (
    combined_gate,
    frozen_stress_gate,
    ordinary_oof_gate,
    paired_fusion_gate,
    strict_oof_frame,
    write_combined_oof,
    write_probability_ensemble_oof,
)


def _gold(path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"row_id": 0, "id1": 1, "id2": 2, "target": 0.0, "category": "A", "fold": 0},
                {"row_id": 1, "id1": 3, "id2": 4, "target": 1.0, "category": "A", "fold": 1},
                {"row_id": 2, "id1": 5, "id2": 6, "target": 0.0, "category": "B", "fold": 0},
                {"row_id": 3, "id1": 7, "id2": 8, "target": 1.0, "category": "B", "fold": 1},
            ]
        ),
        path,
    )


def _scores(path: Path, values: list[float]) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_index": index,
                    "id1": 2 * index + 1,
                    "id2": 2 * index + 2,
                    "text_score": value,
                }
                for index, value in enumerate(values)
            ]
        ),
        path,
    )


def test_strict_oof_and_gates_cover_all_rows_with_exact_keys(tmp_path: Path) -> None:
    gold = tmp_path / "gold.parquet"
    control = tmp_path / "control.parquet"
    candidate = tmp_path / "candidate.parquet"
    combined = tmp_path / "combined.parquet"
    _gold(gold)
    _scores(control, [0.1, 0.9, 0.2, 0.8])
    _scores(candidate, [0.05, 0.95, 0.1, 0.9])

    assert len(strict_oof_frame(gold, [candidate])) == 4
    assert write_combined_oof([candidate], combined)["rows"] == 4
    ordinary = ordinary_oof_gate(gold, [control], [combined])
    assert ordinary["passed"] is True
    assert ordinary["macro_gain"] == pytest.approx(0.0)

    stress = tmp_path / "stress.parquet"
    pq.write_table(pq.read_table(gold), stress)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "stress_macro_ap": 1.0,
                "stress_per_category_ap": {"A": 1.0, "B": 1.0},
            }
        ),
        encoding="utf-8",
    )
    frozen = frozen_stress_gate(stress, baseline, [candidate])
    assert frozen["passed"] is False
    assert combined_gate(ordinary, frozen)["passed"] is False


def test_probability_ensemble_and_paired_fusion_gate_are_key_aligned(tmp_path: Path) -> None:
    gold = tmp_path / "gold.parquet"
    control = tmp_path / "control.parquet"
    candidate = tmp_path / "candidate.parquet"
    ensemble = tmp_path / "ensemble.parquet"
    _gold(gold)
    _scores(control, [0.1, 0.9, 0.2, 0.8])
    _scores(candidate, [0.3, 0.7, 0.4, 0.6])

    result = write_probability_ensemble_oof([control], [candidate], ensemble)

    assert result["control_weight"] == 0.5
    assert pq.read_table(ensemble).column("text_score").to_pylist() == pytest.approx(
        [0.2, 0.8, 0.3, 0.7]
    )
    gate = paired_fusion_gate(
        gold,
        [control],
        [control],
        minimum_macro_gain=0.0,
        samples=20,
        seed=7,
    )
    assert gate["macro_gain"] == pytest.approx(0.0)
    assert gate["ci95_low"] == pytest.approx(0.0)
    assert gate["passed"] is True

    misaligned = tmp_path / "misaligned.parquet"
    _scores(misaligned, [0.3, 0.7, 0.4, 0.6])
    table = pq.read_table(misaligned).to_pandas()
    table.loc[3, "id2"] = 999
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), misaligned)
    with pytest.raises(ValueError, match="keys differ"):
        write_probability_ensemble_oof([control], [misaligned], tmp_path / "bad.parquet")


@pytest.mark.parametrize("bad_kind", ["duplicate", "missing", "nonfinite", "misaligned"])
def test_strict_oof_rejects_invalid_score_contract(tmp_path: Path, bad_kind: str) -> None:
    gold = tmp_path / "gold.parquet"
    scores = tmp_path / "scores.parquet"
    _gold(gold)
    rows = [
        {"row_index": 0, "id1": 1, "id2": 2, "text_score": 0.1},
        {"row_index": 1, "id1": 3, "id2": 4, "text_score": 0.9},
        {"row_index": 2, "id1": 5, "id2": 6, "text_score": 0.2},
        {"row_index": 3, "id1": 7, "id2": 8, "text_score": 0.8},
    ]
    if bad_kind == "duplicate":
        rows[-1]["row_index"] = 2
    elif bad_kind == "missing":
        rows.pop()
    elif bad_kind == "nonfinite":
        rows[-1]["text_score"] = float("nan")
    else:
        rows[-1]["id2"] = 999
    pq.write_table(pa.Table.from_pylist(rows), scores)

    with pytest.raises(ValueError):
        strict_oof_frame(gold, [scores])

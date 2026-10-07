from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from backbone_fixtures import write_saved_backbone

from matchcup.augmentation import (
    materialize_attribute_dropout,
    materialize_hard_identifier_negatives,
)
from matchcup.cross_encoder import oof_indices_for_fold, training_indices_for_fold
from matchcup.folds import validate_component_folds


def _folded_table() -> pa.Table:
    return pa.table(
        {
            "id1": [1, 3, 5, 7],
            "id2": [2, 4, 6, 8],
            "fold": [0, 0, 1, 1],
            "is_synthetic": [False, True, False, True],
        }
    )


def test_synthetic_rows_inherit_fold_train_but_never_enter_oof() -> None:
    table = _folded_table()

    assert np.array_equal(training_indices_for_fold(table, 0), np.array([2, 3]))
    assert np.array_equal(oof_indices_for_fold(table, 0), np.array([0]))
    assert np.array_equal(oof_indices_for_fold(table, 1), np.array([2]))


def test_score_cross_encoder_uses_real_only_oof_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public fold-scoring path must not silently score synthetic descendants."""
    import matchcup.cross_encoder as cross_encoder

    data_path = tmp_path / "folded.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [10, 11, 12, 13],
                "id1": [1, 3, 5, 7],
                "id2": [2, 4, 6, 8],
                "text_a": ["a", "b", "c", "d"],
                "text_b": ["e", "f", "g", "h"],
                "target": [1.0, 1.0, 0.0, 0.0],
                "category": ["A", "A", "B", "B"],
                "fold": [0, 0, 1, 1],
                "is_synthetic": [False, True, False, True],
            }
        ),
        data_path,
    )

    class _StopBeforeModelLoad(RuntimeError):
        pass

    class _Tokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            raise _StopBeforeModelLoad

    class _Accelerator:
        def __init__(self, **_kwargs: object) -> None:
            pass

    selected: dict[str, np.ndarray] = {}
    real_selector = cross_encoder.oof_indices_for_fold

    def spy_selector(table: pa.Table, fold: int) -> np.ndarray:
        selected["indices"] = real_selector(table, fold)
        return selected["indices"]

    monkeypatch.setattr(cross_encoder, "oof_indices_for_fold", spy_selector)
    monkeypatch.setattr(
        cross_encoder,
        "_require_training_dependencies",
        lambda: (object(), _Accelerator, object(), object(), object(), _Tokenizer, object()),
    )

    with pytest.raises(_StopBeforeModelLoad):
        cross_encoder.score_cross_encoder(
            data_path, write_saved_backbone(tmp_path / "model"), tmp_path / "scores.parquet", fold=0
        )

    assert np.array_equal(selected["indices"], np.array([0]))


def test_component_fold_validation_stays_green_on_augmented_table(tmp_path: Path) -> None:
    source = tmp_path / "augmented.parquet"
    pq.write_table(
        pa.table(
            {
                "id1": [1, 1, 10, 10],
                "id2": [2, 2, 11, 11],
                "fold": [0, 0, 1, 1],
                "is_synthetic": [False, True, False, True],
            }
        ),
        source,
    )

    validate_component_folds(source)


def test_attribute_dropout_materializes_one_train_only_synthetic_per_positive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gold.parquet"
    output = tmp_path / "gold_attr_dropout.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [10, 11, 12, 13],
                "id1": [1, 3, 5, 7],
                "id2": [2, 4, 6, 8],
                "target": [1.0, 1.0, 1.0, 0.0],
                "category": ["A", "A", "A", "A"],
                "fold": [0, 1, 1, 0],
                "text_a": [
                    "name_a: one\nattributes_a: color=red",
                    "name_a: two\nattributes_a: color=red; size=m; material=cotton; width=1",
                    "name_a: three\nattributes_a: color=red; size=m; material=cotton; width=1",
                    "name_a: negative\nattributes_a: color=red",
                ],
                "text_b": [
                    "name_b: one\nattributes_b: color=blue; size=l; material=wool; width=2",
                    "name_b: two\nattributes_b: color=blue; size=l; material=wool; width=2",
                    "name_b: three\nattributes_b: color=blue; size=l; material=wool; width=2",
                    "name_b: negative\nattributes_b: color=blue",
                ],
            }
        ),
        source,
    )

    report = materialize_attribute_dropout(source, output, seed=17)
    result = pq.read_table(output).to_pandas()
    synthetic = result[result.is_synthetic]

    assert report["source_positive_rows"] == 3
    assert report["synthetic_rows"] == 3
    assert len(result) == 7
    assert synthetic.source_row_id.tolist() == [10, 11, 12]
    assert synthetic.fold.tolist() == [0, 1, 1]
    assert (synthetic.target == 1.0).all()
    assert any(
        left != right
        for left, right in zip(
            synthetic.text_b.tolist(),
            result.loc[result.row_id.isin([10, 11, 12]), "text_b"].tolist(),
            strict=True,
        )
    )
    for text in [*synthetic.text_a.tolist(), *synthetic.text_b.tolist()]:
        for line in text.splitlines():
            if line.startswith("attributes_"):
                assert line.partition(": ")[2]

    # The materialized file is safe to hand to the strict component-fold and
    # OOF paths: source-fold descendants are train-only for their own fold.
    validate_component_folds(output)
    table = pq.read_table(output)
    assert np.array_equal(oof_indices_for_fold(table, 0), np.array([0, 3]))


def test_hard_identifier_negatives_use_real_catalogue_near_misses(tmp_path: Path) -> None:
    source = tmp_path / "gold.parquet"
    catalogue = tmp_path / "items.parquet"
    output = tmp_path / "gold_hard_identifiers.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [100, 101],
                "id1": [1, 10],
                "id2": [2, 11],
                "target": [1.0, 0.0],
                "category": ["A", "A"],
                "fold": [0, 1],
                "text_a": ["source a", "negative a"],
                "text_b": ["source b", "negative b"],
            }
        ),
        source,
    )
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 10, 11],
                "category": ["A", "A", "A", "A", "A"],
                "name_norm": ["rs07c", "two", "rs07cst4", "ten", "eleven"],
                "identifier_tokens": [
                    '["rs07c"]',
                    "[]",
                    '["rs07cst4"]',
                    '["other10"]',
                    '["other11"]',
                ],
            }
        ),
        catalogue,
    )

    report = materialize_hard_identifier_negatives(source, catalogue, output, seed=9)
    result = pq.read_table(output).to_pandas()
    synthetic = result[result.is_synthetic]

    assert report["eligible_pairs"] == 1
    assert report["synthetic_rows"] == 1
    assert synthetic[["id1", "id2", "target", "fold", "source_row_id"]].to_dict("records") == [
        {"id1": 1, "id2": 3, "target": 0.0, "fold": 0, "source_row_id": 100}
    ]
    assert "rs07c" in synthetic.iloc[0].text_a
    assert "rs07cst4" in synthetic.iloc[0].text_b
    assert np.array_equal(oof_indices_for_fold(pq.read_table(output), 0), np.array([0]))


def test_hard_identifier_donor_cannot_come_from_positive_source_component(tmp_path: Path) -> None:
    source = tmp_path / "gold.parquet"
    catalogue = tmp_path / "items.parquet"
    output = tmp_path / "hard.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [1],
                "id1": [1],
                "id2": [2],
                "target": [1.0],
                "category": ["A"],
                "fold": [0],
                "text_a": ["source a"],
                "text_b": ["source b"],
            }
        ),
        source,
    )
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "category": ["A", "A", "A", "A"],
                "name_norm": ["rs07c", "rs07cst4", "other3", "other4"],
                "identifier_tokens": [
                    '["rs07c"]',
                    '["rs07cst4"]',
                    '["other3"]',
                    '["other4"]',
                ],
            }
        ),
        catalogue,
    )

    report = materialize_hard_identifier_negatives(source, catalogue, output, seed=9)

    assert report["eligible_pairs"] == 0
    assert report["synthetic_rows"] == 0
    assert pq.read_table(output).column("is_synthetic").to_pylist() == [False]


def test_hard_identifier_without_an_eligible_anchor_writes_source_only(tmp_path: Path) -> None:
    """A valid source with no rare identifier is a zero-row augmentation, not an error."""
    source = tmp_path / "gold.parquet"
    catalogue = tmp_path / "items.parquet"
    output = tmp_path / "hard.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [1],
                "id1": [1],
                "id2": [2],
                "target": [1.0],
                "category": ["A"],
                "fold": [0],
                "text_a": ["source a"],
                "text_b": ["source b"],
            }
        ),
        source,
    )
    pq.write_table(
        pa.table(
            {
                "id": [1, 2],
                "category": ["A", "A"],
                "name_norm": ["one", "two"],
                "identifier_tokens": ['["shared"]', '["shared"]'],
            }
        ),
        catalogue,
    )

    report = materialize_hard_identifier_negatives(source, catalogue, output, seed=9)

    assert report["eligible_pairs"] == 0
    assert report["synthetic_rows"] == 0
    assert pq.read_table(output).column("is_synthetic").to_pylist() == [False]


def test_hard_identifier_donor_from_another_positive_component_is_excluded(
    tmp_path: Path,
) -> None:
    """A synthetic row must not attach two already assigned Gold components."""
    source = tmp_path / "gold.parquet"
    catalogue = tmp_path / "items.parquet"
    output = tmp_path / "hard.parquet"
    pq.write_table(
        pa.table(
            {
                "row_id": [1, 2],
                "id1": [1, 3],
                "id2": [2, 4],
                "target": [1.0, 1.0],
                "category": ["A", "A"],
                "fold": [0, 1],
                "text_a": ["source a", "other source a"],
                "text_b": ["source b", "other source b"],
            }
        ),
        source,
    )
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "category": ["A", "A", "A", "A"],
                "name_norm": ["rs07c", "two", "rs07cst4", "four"],
                "identifier_tokens": [
                    '["rs07c"]',
                    "[]",
                    '["rs07cst4"]',
                    "[]",
                ],
            }
        ),
        catalogue,
    )

    report = materialize_hard_identifier_negatives(source, catalogue, output, seed=9)

    assert report["eligible_pairs"] == 0
    assert report["synthetic_rows"] == 0
    validate_component_folds(output)

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from matchcup.research import (
    assess_candidate_gate,
    compare_score_distributions,
    describe_fusion_score_shift,
    evaluate_fold_scores,
    graph_diagnostics,
    mine_training_hard_negatives,
    paired_bootstrap_difference,
    run_fold_ablation,
)
from matchcup.score_transform import BatchCategoryPercentileScoreTransform


def _write_gold(path: Path) -> None:
    rows = []
    for category, start in (("A", 0), ("B", 10)):
        for offset in range(6):
            rows.append(
                {
                    "row_id": start + offset,
                    "id1": 100 + start + offset,
                    "id2": 200 + start + offset,
                    "target": float(offset >= 3),
                    "category": category,
                    "fold": 0 if offset in {0, 1, 3, 4} else 1,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_scores(path: Path, values: list[float]) -> None:
    rows = []
    for index, score in enumerate(values[:6]):
        rows.append(
            {
                "row_index": index,
                "id1": 100 + index,
                "id2": 200 + index,
                "text_score": score,
            }
        )
    for index, score in enumerate(values[6:]):
        rows.append(
            {
                "row_index": 10 + index,
                "id1": 110 + index,
                "id2": 210 + index,
                "text_score": score,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _fold_zero_scores(source: Path, output: Path) -> None:
    table = pq.read_table(source)
    allowed = {0, 1, 3, 4, 10, 11, 13, 14}
    rows = [row for row in table.to_pylist() if row["row_index"] in allowed]
    pq.write_table(pa.Table.from_pylist(rows), output)


def test_research_evaluation_and_train_only_hard_negative_mining(tmp_path: Path) -> None:
    gold = tmp_path / "gold.parquet"
    scores = tmp_path / "scores.parquet"
    fold_scores = tmp_path / "fold_scores.parquet"
    hard = tmp_path / "hard.parquet"
    _write_gold(gold)
    _write_scores(scores, [0.9, 0.8, 0.7, 0.2, 0.3, 0.4] * 2)
    _fold_zero_scores(scores, fold_scores)

    evaluation = evaluate_fold_scores(gold, fold_scores, fold=0)
    assert evaluation["rows"] == 8
    assert evaluation["macro_ap"] == pytest.approx(5.0 / 12.0)

    mining = mine_training_hard_negatives(
        gold, scores, hard, validation_fold=0, top_fraction=1.0
    )
    assert mining["training_rows"] == 4
    assert pq.read_table(hard).column("row_id").to_pylist() == [2, 12]


def test_research_score_drift_bootstrap_and_graph_reports(tmp_path: Path) -> None:
    gold = tmp_path / "gold.parquet"
    control = tmp_path / "control.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_gold(gold)
    _write_scores(control, [0.9, 0.8, 0.7, 0.2, 0.3, 0.4] * 2)
    _write_scores(candidate, [0.1, 0.2, 0.9, 0.1, 0.2, 0.9] * 2)

    drift = compare_score_distributions(gold, control, candidate, top_fraction=0.5)
    assert drift["purpose"] == "diagnostic_only_not_a_model_selection_metric"
    assert drift["catboost_input_shift"]["changing_feature"] == "text_score"
    assert set(drift["categories"]) == {"A", "B"}

    bootstrap = paired_bootstrap_difference(
        gold, control, candidate, fold=0, samples=20, seed=7
    )
    assert bootstrap["samples"] == 20
    assert bootstrap["difference"] > 0

    gate = assess_candidate_gate(
        gold,
        control,
        candidate,
        fold=0,
        minimum_macro_gain=0.01,
        maximum_category_drop=0.02,
        samples=20,
        seed=7,
    )
    assert gate["passed"] is True
    assert gate["failing_categories"] == {}

    graph = graph_diagnostics(gold, control)
    assert graph["isolated_edges"] == 12
    assert graph["nonisolated"]["rows"] == 0


def test_fusion_shift_is_explicitly_diagnostic_and_per_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold = tmp_path / "gold.parquet"
    oof = tmp_path / "oof.parquet"
    final = tmp_path / "final.parquet"
    _write_gold(gold)
    _write_scores(oof, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 2)
    _write_scores(final, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 2)

    class FakeFusion:
        def __init__(self, _directory: Path) -> None:
            pass

        def predict(self, records: list[dict[str, object]]) -> np.ndarray:
            return np.asarray([2.0 * float(record["text_score"]) for record in records])

    monkeypatch.setattr("matchcup.inference.FusionPredictor", FakeFusion)
    transform = BatchCategoryPercentileScoreTransform()
    (tmp_path / "feature_schema.json").write_text(
        json.dumps({"score_transform": transform.to_schema()}), encoding="utf-8"
    )
    report = describe_fusion_score_shift(gold, oof, final, tmp_path, top_fraction=0.5)
    assert report["purpose"] == "diagnostic_only_in_sample_final_score_no_model_selection"
    assert report["rows"] == 12
    assert set(report["categories"]) == {"A", "B"}
    assert report["overall"]["hybrid_output_rank_correlation"] == pytest.approx(-1.0)
    assert report["score_transform"]["kind"] == "batch_category_percentile"
    assert report["overall"]["rank_input"]["available"] is True


def test_schema_robustness_runner_records_spec_and_rejects_other_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = tmp_path / "attribute_permutation.yaml"
    spec_path.write_text(
        """schema_robustness:
  version: 1
  variant: silver_control
  transform: attribute_permutation
  pair_probability: 0.5
  seed: 20260815
""",
        encoding="utf-8",
    )

    def fake_train(*_args: object, **kwargs: object) -> dict[str, object]:
        config = _args[2]
        assert config.train_text_transform is not None
        return {"train_text_transform_summary": {"changed_pairs": 4}}

    monkeypatch.setattr("matchcup.cross_encoder.train_cross_encoder", fake_train)
    monkeypatch.setattr(
        "matchcup.cross_encoder.score_cross_encoder", lambda *_args, **_kwargs: {"rows": 8}
    )
    monkeypatch.setattr(
        "matchcup.research.evaluate_fold_scores", lambda *_args, **_kwargs: {"macro_ap": 0.5}
    )
    report = run_fold_ablation(
        tmp_path / "gold.parquet",
        tmp_path / "result",
        variant="silver_control",
        fold=0,
        base_model="base",
        silver_model=tmp_path / "silver",
        model_revision=None,
        max_length=32,
        train_batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.0,
        ranking_weight=0.0,
        mixed_precision="no",
        num_workers=0,
        seed=1,
        schema_robustness_spec=spec_path,
    )

    schema = report["schema_robustness"]
    assert schema["spec"]["transform"] == "attribute_permutation"
    assert schema["training_summary"] == {"changed_pairs": 4}
    assert schema["source_yaml_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    persisted = json.loads((tmp_path / "result" / "variant_report.json").read_text())
    assert persisted["schema_robustness"] == schema

    with pytest.raises(ValueError, match="only allowed"):
        run_fold_ablation(
            tmp_path / "gold.parquet",
            tmp_path / "invalid",
            variant="gold_only",
            fold=0,
            base_model="base",
            silver_model=None,
            model_revision=None,
            max_length=32,
            train_batch_size=2,
            eval_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_ratio=0.0,
            ranking_weight=0.0,
            mixed_precision="no",
            num_workers=0,
            seed=1,
            schema_robustness_spec=spec_path,
        )

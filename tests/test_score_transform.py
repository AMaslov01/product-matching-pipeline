import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from matchcup.fusion import ExportedFusionModel, _matrix, _strict_oof_frame, train_fusion
from matchcup.inference import FusionPredictor
from matchcup.score_transform import (
    BatchCategoryPercentileScoreTransform,
    score_transform_from_feature_schema,
)
from matchcup.transductive_profile import (
    C1_FEATURE_NAMES,
    H12_V1_FEATURE_NAMES,
    feature_schema_digest,
)


def test_batch_category_percentile_handles_ties_unknown_and_singleton_groups() -> None:
    transform = BatchCategoryPercentileScoreTransform()
    frame = pd.DataFrame(
        {
            "row": list(range(7)),
            "category": ["A", "A", "A", "B", "missing", "missing", "singleton"],
            "text_score": [0.2, 0.2, 0.8, 0.4, 0.1, 0.9, 0.3],
        }
    )

    ranked = transform.add_features(frame)

    assert ranked["text_score_category_rank"].tolist() == pytest.approx(
        [0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0]
    )
    assert ranked["row"].tolist() == frame["row"].tolist()
    assert np.isfinite(ranked["text_score_category_rank"]).all()


def test_batch_category_percentile_is_monotone_and_row_order_invariant() -> None:
    transform = BatchCategoryPercentileScoreTransform()
    frame = pd.DataFrame(
        {
            "row": [0, 1, 2, 3],
            "category": ["A", "A", "B", "B"],
            "text_score": [0.1, 0.8, 0.3, 0.4],
        }
    )
    expected = transform.add_features(frame).set_index("row")["text_score_category_rank"]
    shifted = frame.assign(text_score=np.exp(frame["text_score"] * 3.0))
    permuted = shifted.iloc[[2, 0, 3, 1]].copy()
    actual = transform.add_features(permuted).set_index("row")["text_score_category_rank"]

    assert actual.sort_index().tolist() == pytest.approx(expected.sort_index().tolist())


def test_batch_category_percentile_supports_fold_category_oof_groups() -> None:
    transform = BatchCategoryPercentileScoreTransform()
    frame = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "category": ["A", "A", "A", "A"],
            "text_score": [0.1, 0.9, 0.2, 0.8],
        }
    )

    ranked = transform.add_features(frame, group_columns=("fold", "category"))

    assert ranked["text_score_category_rank"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])


def test_score_transform_schema_round_trip() -> None:
    transform = BatchCategoryPercentileScoreTransform()
    schema = json.loads(json.dumps(transform.to_schema()))
    restored = BatchCategoryPercentileScoreTransform.from_schema(schema)

    assert restored == transform
    assert schema == {
        "version": 2,
        "kind": "batch_category_percentile",
        "source_feature": "text_score",
        "output_feature": "text_score_category_rank",
        "grouping": ["category"],
        "rank_method": "average",
        "percentile": True,
        "score_source": "single_probability",
    }
    assert score_transform_from_feature_schema({"feature_names": []}) is None


def _write_pair_features(path: Path) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": 0,
                    "id1": 10,
                    "id2": 20,
                    "target": 0.0,
                    "category": "A",
                    "fold": 0,
                    "name_token_jaccard": 0.1,
                },
                {
                    "row_id": 1,
                    "id1": 11,
                    "id2": 21,
                    "target": 1.0,
                    "category": "A",
                    "fold": 0,
                    "name_token_jaccard": 0.9,
                },
                {
                    "row_id": 2,
                    "id1": 12,
                    "id2": 22,
                    "target": 0.0,
                    "category": "A",
                    "fold": 1,
                    "name_token_jaccard": 0.2,
                },
                {
                    "row_id": 3,
                    "id1": 13,
                    "id2": 23,
                    "target": 1.0,
                    "category": "A",
                    "fold": 1,
                    "name_token_jaccard": 0.8,
                },
            ]
        ),
        path,
    )


def _write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _valid_score_rows() -> list[dict[str, object]]:
    return [
        {"row_index": 0, "id1": 10, "id2": 20, "text_score": 0.1},
        {"row_index": 1, "id1": 11, "id2": 21, "text_score": 0.9},
        {"row_index": 2, "id1": 12, "id2": 22, "text_score": 0.2},
        {"row_index": 3, "id1": 13, "id2": 23, "text_score": 0.8},
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "row_index": 0}), "duplicate row_index"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "id1": 10, "id2": 20}), "duplicate pairs"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "id2": 999}), "do not cover"),
        (
            lambda rows: rows.__setitem__(1, {**rows[1], "text_score": float("nan")}),
            "must be finite",
        ),
    ],
)
def test_strict_oof_frame_rejects_invalid_score_contract(
    tmp_path: Path, mutate, message: str
) -> None:
    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    _write_pair_features(pairs)
    rows = _valid_score_rows()
    mutate(rows)
    _write_scores(scores, rows)

    with pytest.raises(ValueError, match=message):
        _strict_oof_frame(pairs, scores)


def test_strict_oof_frame_rejects_missing_score(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    _write_pair_features(pairs)
    _write_scores(scores, _valid_score_rows()[:-1])

    with pytest.raises(ValueError, match="exactly one score"):
        _strict_oof_frame(pairs, scores)


def test_train_fusion_uses_fold_category_rank_and_excludes_raw_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    output = tmp_path / "fusion"
    _write_pair_features(pairs)
    _write_scores(scores, _valid_score_rows())

    class FakeCatBoostRegressor:
        prediction_inputs: list[pd.DataFrame] = []

        def __init__(self, **_kwargs) -> None:
            self.value = 0.0

        def fit(self, _x: pd.DataFrame, y: np.ndarray) -> None:
            self.value = float(np.mean(y))

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            self.prediction_inputs.append(x.copy())
            return np.full(len(x), self.value, dtype=np.float32)

        def save_model(self, path: Path, **_kwargs) -> None:
            Path(path).write_text("fake", encoding="utf-8")

    monkeypatch.setattr("matchcup.fusion._require_catboost", lambda: FakeCatBoostRegressor)
    train_fusion(
        pairs,
        scores,
        output,
        iterations=1,
        learning_rate=0.1,
        depth_candidates=[1],
        l2_candidates=[1.0],
        seed=7,
        score_contract="category_rank",
    )

    first_validation = FakeCatBoostRegressor.prediction_inputs[0]
    assert "text_score" not in first_validation
    assert first_validation["text_score_category_rank"].tolist() == pytest.approx([0.5, 1.0])
    schema = json.loads((output / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["feature_names"][:2] == ["name_token_jaccard", "text_score_category_rank"]
    assert "text_score" not in schema["feature_names"]
    assert schema["score_transform"]["kind"] == "batch_category_percentile"


def test_ranked_fusion_sorts_category_queries_and_restores_oof_row_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pool prediction order must never leak into key-aligned fusion_oof.parquet."""
    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    output = tmp_path / "fusion"
    rows = [
        {
            "row_id": 0,
            "id1": 10,
            "id2": 20,
            "target": 0.0,
            "category": "B",
            "fold": 0,
            "name_token_jaccard": 0.10,
        },
        {
            "row_id": 1,
            "id1": 11,
            "id2": 21,
            "target": 1.0,
            "category": "A",
            "fold": 0,
            "name_token_jaccard": 0.90,
        },
        {
            "row_id": 2,
            "id1": 12,
            "id2": 22,
            "target": 1.0,
            "category": "B",
            "fold": 0,
            "name_token_jaccard": 0.80,
        },
        {
            "row_id": 3,
            "id1": 13,
            "id2": 23,
            "target": 0.0,
            "category": "A",
            "fold": 0,
            "name_token_jaccard": 0.20,
        },
        {
            "row_id": 4,
            "id1": 14,
            "id2": 24,
            "target": 0.0,
            "category": "B",
            "fold": 1,
            "name_token_jaccard": 0.30,
        },
        {
            "row_id": 5,
            "id1": 15,
            "id2": 25,
            "target": 1.0,
            "category": "A",
            "fold": 1,
            "name_token_jaccard": 0.70,
        },
        {
            "row_id": 6,
            "id1": 16,
            "id2": 26,
            "target": 1.0,
            "category": "B",
            "fold": 1,
            "name_token_jaccard": 0.60,
        },
        {
            "row_id": 7,
            "id1": 17,
            "id2": 27,
            "target": 0.0,
            "category": "A",
            "fold": 1,
            "name_token_jaccard": 0.40,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), pairs)
    _write_scores(
        scores,
        [
            {
                "row_index": row["row_id"],
                "id1": row["id1"],
                "id2": row["id2"],
                "text_score": row["name_token_jaccard"],
            }
            for row in rows
        ],
    )

    class FakePool:
        def __init__(self, data: pd.DataFrame, **kwargs) -> None:
            self.data = data.copy()
            self.group_id = np.asarray(kwargs["group_id"])

    class FakeCatBoostRanker:
        prediction_groups: list[np.ndarray] = []

        def __init__(self, **_kwargs) -> None:
            pass

        def fit(self, _pool: FakePool) -> None:
            pass

        def predict(self, pool: FakePool) -> np.ndarray:
            self.prediction_groups.append(pool.group_id)
            return pool.data["name_token_jaccard"].to_numpy(dtype=np.float64)

        def save_model(self, path: Path, **_kwargs) -> None:
            Path(path).write_text("fake", encoding="utf-8")

    monkeypatch.setattr("matchcup.fusion._require_catboost_ranker", lambda: FakeCatBoostRanker)
    monkeypatch.setattr("matchcup.fusion._require_catboost_pool", lambda: FakePool)
    monkeypatch.setattr(
        "matchcup.fusion._verify_exported_fusion_round_trip",
        lambda *_args: {"passed": True, "max_abs_delta": 0.0},
    )

    report = train_fusion(
        pairs,
        scores,
        output,
        iterations=1,
        learning_rate=0.1,
        depth_candidates=[1],
        l2_candidates=[1.0],
        seed=7,
        objective="YetiRank",
    )

    oof = pq.read_table(output / "fusion_oof.parquet").to_pandas().sort_values("row_index")
    assert oof["text_score"].tolist() == pytest.approx([row["name_token_jaccard"] for row in rows])
    assert report["objective"] == "YetiRank"
    assert report["grouped_by_category"] is True
    assert report["export_round_trip"] == {"passed": True, "max_abs_delta": 0.0}
    assert FakeCatBoostRanker.prediction_groups
    assert all(np.all(groups[:-1] <= groups[1:]) for groups in FakeCatBoostRanker.prediction_groups)


def test_ranked_fusion_exports_a_python_model_that_matches_predict(tmp_path: Path) -> None:
    """The archive serves fusion_model.py, so YetiRank must round-trip exactly."""
    from catboost import CatBoostRanker

    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    output = tmp_path / "fusion"
    _write_pair_features(pairs)
    _write_scores(scores, _valid_score_rows())

    report = train_fusion(
        pairs,
        scores,
        output,
        iterations=7,
        learning_rate=0.1,
        depth_candidates=[2],
        l2_candidates=[1.0],
        seed=7,
        objective="YetiRank",
    )

    frame = _strict_oof_frame(pairs, scores)
    matrix, _, _ = _matrix(frame)
    native = CatBoostRanker()
    native.load_model(output / "fusion.cbm")
    exported = ExportedFusionModel(output / "fusion_model.py", output / "feature_schema.json")
    delta = np.max(
        np.abs(native.predict(matrix) - exported.predict(frame.to_dict(orient="records")))
    )

    assert report["objective"] == "YetiRank"
    assert report["export_round_trip"]["passed"] is True
    assert report["export_round_trip"]["max_abs_delta"] < 1e-6
    assert delta < 1e-6


def test_profiled_fusion_persists_the_exact_schema_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    output = tmp_path / "fusion"
    _write_pair_features(pairs)
    records = pq.read_table(pairs).to_pylist()
    for record in records:
        record.update(
            {
                **{name: 0.1 for name in H12_V1_FEATURE_NAMES},
                **{name: 0.2 for name in C1_FEATURE_NAMES},
            }
        )
    pq.write_table(pa.Table.from_pylist(records), pairs)
    _write_scores(scores, _valid_score_rows())

    class FakeCatBoostRegressor:
        def __init__(self, **_kwargs) -> None:
            self.value = 0.0

        def fit(self, _x: pd.DataFrame, y: np.ndarray) -> None:
            self.value = float(np.mean(y))

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return np.full(len(x), self.value, dtype=np.float32)

        def save_model(self, path: Path, **_kwargs) -> None:
            Path(path).write_text("fake", encoding="utf-8")

    monkeypatch.setattr("matchcup.fusion._require_catboost", lambda: FakeCatBoostRegressor)
    train_fusion(
        pairs,
        scores,
        output,
        iterations=1,
        learning_rate=0.1,
        depth_candidates=[1],
        l2_candidates=[1.0],
        seed=7,
        transductive_profile="h12-v1-c1",
    )

    schema = json.loads((output / "feature_schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "fusion_manifest.json").read_text(encoding="utf-8"))
    assert schema["transductive_profile"] == manifest["transductive_profile"] == "h12-v1-c1"
    assert schema["feature_schema_sha256"] == manifest["feature_schema_sha256"]
    assert schema["feature_schema_sha256"] == feature_schema_digest(schema["feature_names"])

    monkeypatch.setattr(
        "matchcup.fusion._require_catboost",
        lambda: pytest.fail("CatBoost must not start before the schema lock is validated"),
    )
    with pytest.raises(ValueError, match="Predeclared feature schema digest"):
        train_fusion(
            pairs,
            scores,
            tmp_path / "blocked",
            transductive_profile="h12-v1-c1",
            expected_feature_schema_sha256="wrong",
        )


def test_unprofiled_fusion_still_records_a_feature_schema_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H39's bare two_score fusion (no transductive family) needs profile=None
    to avoid resolve_transductive_profile demanding features it deliberately
    dropped, but two such archives (F-5, F-6) still have to prove they trained
    on the exact same feature set. The digest must not disappear along with
    the profile it used to be nested under."""
    pairs = tmp_path / "pairs.parquet"
    scores = tmp_path / "scores.parquet"
    output = tmp_path / "fusion"
    _write_pair_features(pairs)
    _write_scores(scores, _valid_score_rows())

    class FakeCatBoostRegressor:
        def __init__(self, **_kwargs) -> None:
            self.value = 0.0

        def fit(self, _x: pd.DataFrame, y: np.ndarray) -> None:
            self.value = float(np.mean(y))

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return np.full(len(x), self.value, dtype=np.float32)

        def save_model(self, path: Path, **_kwargs) -> None:
            Path(path).write_text("fake", encoding="utf-8")

    monkeypatch.setattr("matchcup.fusion._require_catboost", lambda: FakeCatBoostRegressor)
    train_fusion(
        pairs,
        scores,
        output,
        iterations=1,
        learning_rate=0.1,
        depth_candidates=[1],
        l2_candidates=[1.0],
        seed=7,
        transductive_profile=None,
    )

    schema = json.loads((output / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["feature_schema_sha256"] == feature_schema_digest(schema["feature_names"])
    assert "transductive_profile" not in schema
    assert not (output / "fusion_manifest.json").is_file()


def test_train_and_runtime_matrices_share_rank_contract_and_drop_raw_score(tmp_path: Path) -> None:
    transform = BatchCategoryPercentileScoreTransform()
    records = [
        {"category": "A", "text_score": 0.3},
        {"category": "A", "text_score": 0.7},
        {"category": "unknown", "text_score": 0.3},
    ]
    train_matrix, feature_names, categories = _matrix(
        pd.DataFrame(records), categories=["A", "B"], score_transform=transform
    )
    schema = {
        "feature_names": feature_names,
        "categories": categories,
        "score_transform": transform.to_schema(),
    }
    predictor = FusionPredictor.__new__(FusionPredictor)
    predictor.schema = schema
    predictor.score_transform = score_transform_from_feature_schema(schema)

    assert np.array_equal(train_matrix.to_numpy(), predictor._matrix(records))
    assert "text_score" not in train_matrix
    assert train_matrix["text_score_category_rank"].tolist() == pytest.approx([0.5, 1.0, 1.0])

    exported_model = tmp_path / "fusion_model.py"
    exported_model.write_text(
        "def apply_catboost_model(features):\n    return features[0]\n", encoding="utf-8"
    )
    schema_path = tmp_path / "feature_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    exported = ExportedFusionModel(exported_model, schema_path)
    assert exported.predict(records) == pytest.approx([0.5, 1.0, 1.0])


def test_runtime_matrix_keeps_legacy_schema_compatible() -> None:
    predictor = FusionPredictor.__new__(FusionPredictor)
    predictor.schema = {"feature_names": ["text_score", "category__A"], "categories": ["A"]}
    predictor.score_transform = score_transform_from_feature_schema(predictor.schema)

    assert np.allclose(predictor._matrix([{"category": "A", "text_score": 0.3}]), [[0.3, 1.0]])


def test_score_contract_defaults_to_the_batch_independent_one() -> None:
    """The rank contract has to be asked for; leaving it as a default cost 0.0174 twice.

    candidate A shipped batch-category percentile and scored 0.4227 against a raw
    control at 0.4401, because a rank computed inside the inference batch depends
    on what else is in that batch. On 2026-08-24 an emergency rebuild called the
    CLI without the flag and lost the same margin again on two archives.
    """
    import inspect

    from matchcup.cli import build_parser
    from matchcup.fusion import train_fusion

    assert inspect.signature(train_fusion).parameters["score_contract"].default == "raw"

    parser = build_parser()
    parsed = parser.parse_args(["train-fusion"])
    assert parsed.score_contract == "raw"
    explicit = parser.parse_args(["train-fusion", "--score-contract", "category_rank"])
    assert explicit.score_contract == "category_rank"

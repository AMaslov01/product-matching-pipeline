"""The two-score serving contract, and a lock on the shapes that predate it.

Archive R3 shipped a prevalence-tuned model and a 0.50-sampler one combined by a
nominal 0.5/0.5 mean of raw probabilities and scored 0.5111099373, below the
single-model R2 at 0.5136507571.  The build was clean; the calibration was not.
Under ``two_score`` both models still score every row, so the runtime profile is
R3's, but the fusion receives the two scores as separate features and learns the
combination from OOF instead of averaging in probability space.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from backbone_fixtures import write_saved_backbone

from matchcup import inference
from matchcup.inference import FusionPredictor, _resolve_score_source
from matchcup.packaging import _run_py, build_submission
from matchcup.transductive_profile import H12_V1

# Every keyword an archive hands to run_inference, in order.  Pinned as literal
# text because run.py is the only executable the contest runs: a silent change
# here is a change to what every future archive does.
_SINGLE_MODEL_RUN_PY = '''from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchcup.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", "-i", required=True)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True)
    parser.add_argument("--output_path", "--output-path", "-o", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    model_batch_size = int(metadata["matchcup"]["model_batch_size"])
    run_inference(
        args.items_path,
        args.matches_path,
        args.output_path,
        root / "models" / "cross_encoder",
        root / "models" / "fusion",
        secondary_model_path=None,
        model_weights=(1.0,),
        score_mode='hybrid',
        score_composition='mean_probability',
        cuda_precision='fp16_amp',
        constant_categories=None,
        model_batch_size=model_batch_size,
    )


if __name__ == "__main__":
    main()
'''

_TWO_MODEL_MEAN_RUN_PY = _SINGLE_MODEL_RUN_PY.replace(
    "secondary_model_path=None,\n        model_weights=(1.0,),",
    'secondary_model_path=root / "models" / "cross_encoder_secondary",\n'
    "        model_weights=(0.5, 0.5),",
)


def _items_parquet(path: Path) -> Path:
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alpha 10", "Alpha 10 Pro", "Beta 20", "Beta 20 Pro"],
                "attributes": ["{}", "{}", "{}", "{}"],
                "category": ["A", "A", "B", "B"],
            }
        ),
        path,
    )
    return path


def _matches_parquet(path: Path) -> Path:
    pq.write_table(pa.table({"id1": [1, 3], "id2": [2, 4]}), path)
    return path


class _FakeTensor:
    """Just enough of the tensor surface run_inference actually touches."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def to(self, _device: object) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return self

    def view(self, _shape: int) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeTokenizer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.special_tokens_map = {"pad_token": "[PAD]"}
        self.model_max_length = 384
        self.padding_side = "right"
        self.truncation_side = "right"

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0}

    def __call__(self, left: list[str], _right: list[str], **_kwargs: object):
        return {"input_ids": _FakeTensor(np.arange(len(left)))}


class _FakeDevice:
    def __init__(self, name: str) -> None:
        self.type = name

    def __str__(self) -> str:
        return self.type


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return False


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=_FakeDevice,
        inference_mode=_NullContext,
        autocast=lambda **_kwargs: _NullContext(),
        sigmoid=lambda tensor: tensor,
    )


def _install_fake_runtime(monkeypatch, scores_by_model: dict[str, float]) -> None:
    """Give each packaged model a constant, distinguishable score."""

    class _FakeModel:
        def __init__(self, model_path: Path) -> None:
            self.score = scores_by_model[Path(model_path).name]

        def to(self, _device: object) -> _FakeModel:
            return self

        def eval(self) -> None:
            return None

        def __call__(self, **encoded: object) -> SimpleNamespace:
            rows = len(encoded["input_ids"].values)
            return SimpleNamespace(
                logits=_FakeTensor(np.full(rows, self.score, dtype=np.float32))
            )

    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda path, **_kwargs: _FakeTokenizer(Path(path).name)
        ),
        AutoModelForSequenceClassification=SimpleNamespace(
            from_pretrained=lambda path, **_kwargs: _FakeModel(path)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _install_recording_fusion(monkeypatch, score_source: str) -> list[dict[str, object]]:
    """Capture exactly the records the runtime hands the fusion."""
    captured: list[dict[str, object]] = []

    class _RecordingFusion:
        def __init__(self, _directory: str | Path) -> None:
            self.schema = {"feature_names": ["text_score", "second_score"], "categories": []}
            self.transductive_profile = H12_V1
            self.score_transform = None
            self.score_source = score_source

        def predict(self, records: list[dict[str, object]]) -> np.ndarray:
            captured.extend(dict(record) for record in records)
            return np.arange(len(records), dtype=np.float64)

    monkeypatch.setattr(inference, "FusionPredictor", _RecordingFusion)
    return captured


def _run_two_pair_inference(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    return inference.run_inference(
        _items_parquet(tmp_path / "items.parquet"),
        _matches_parquet(tmp_path / "matches.parquet"),
        tmp_path / "submission.csv",
        tmp_path / "primary",
        tmp_path / "fusion",
        secondary_model_path=tmp_path / "secondary",
        workers=1,
        **kwargs,
    )


def test_two_score_gives_each_model_its_own_column_and_never_averages(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_runtime(monkeypatch, {"primary": 0.25, "secondary": 0.75})
    records = _install_recording_fusion(monkeypatch, "two_score")

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("two_score must not average in probability space")

    monkeypatch.setattr(inference, "_combine_probability_scores", refuse)

    result = _run_two_pair_inference(tmp_path, score_composition="two_score")

    assert result["score_composition"] == "two_score"
    assert result["score_source"] == "two_score"
    assert result["model_count"] == 2
    assert len(records) == 2
    assert [record["text_score"] for record in records] == [0.25, 0.25]
    assert [record["second_score"] for record in records] == [0.75, 0.75]


def test_default_composition_still_averages_before_the_fusion(
    tmp_path: Path, monkeypatch
) -> None:
    """The control for the test above: the shipped path does call the combiner."""
    _install_fake_runtime(monkeypatch, {"primary": 0.25, "secondary": 0.75})
    records = _install_recording_fusion(monkeypatch, "mean_probability")
    calls: list[int] = []
    combine = inference._combine_probability_scores

    def counting_combine(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return combine(*args, **kwargs)

    monkeypatch.setattr(inference, "_combine_probability_scores", counting_combine)

    result = _run_two_pair_inference(tmp_path)

    assert result["score_composition"] == "mean_probability"
    assert result["score_source"] == "mean_probability"
    assert calls == [1]
    assert [record["text_score"] for record in records] == [0.5, 0.5]
    assert all("second_score" not in record for record in records)


def test_mean_probability_keeps_its_two_historical_contracts() -> None:
    assert _resolve_score_source("mean_probability", 1, "hybrid", None) == "single_probability"
    assert _resolve_score_source("mean_probability", 2, "hybrid", None) == "mean_probability"
    assert _resolve_score_source("mean_probability", 2, "text", (0.5, 0.5)) == "mean_probability"


def test_two_score_refuses_anything_other_than_two_models() -> None:
    for model_count in (1, 3):
        with pytest.raises(ValueError, match="exactly two models"):
            _resolve_score_source("two_score", model_count, "hybrid", None)


def test_two_score_refuses_text_only_scoring() -> None:
    """Without the fusion nothing combines the two columns, so the mode is meaningless."""
    with pytest.raises(ValueError, match="hybrid fusion"):
        _resolve_score_source("two_score", 2, "text", None)


def test_two_score_refuses_probability_space_weights() -> None:
    with pytest.raises(ValueError, match="no probability-space weights"):
        _resolve_score_source("two_score", 2, "hybrid", (0.5, 0.5))


def test_unknown_composition_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported score composition"):
        _resolve_score_source("rank_average", 2, "hybrid", None)


def _two_score_predictor() -> FusionPredictor:
    predictor = FusionPredictor.__new__(FusionPredictor)
    predictor.schema = {
        "feature_names": ["text_score", "second_score", "category__A"],
        "categories": ["A"],
        "score_source": "two_score",
    }
    predictor.score_transform = None
    return predictor


def test_two_score_fusion_refuses_a_record_missing_the_second_column() -> None:
    """A wiring slip here would otherwise serve a silently halved ensemble.

    ``_matrix`` resolves features with a ``0.0`` default, so an unpopulated
    ``second_score`` would build, run and pass every phase before showing up as
    a worse leaderboard number days later.
    """
    predictor = _two_score_predictor()

    with pytest.raises(ValueError, match="second_score"):
        predictor._matrix([{"category": "A", "text_score": 0.4}])


def test_two_score_fusion_accepts_a_record_carrying_both_columns() -> None:
    predictor = _two_score_predictor()

    matrix = predictor._matrix([{"category": "A", "text_score": 0.4, "second_score": 0.6}])

    assert matrix.tolist() == [[pytest.approx(0.4), pytest.approx(0.6), 1.0]]


def test_shipped_contracts_keep_the_permissive_feature_default() -> None:
    """R2 and its predecessors were built against the 0.0 default; do not move it."""
    predictor = FusionPredictor.__new__(FusionPredictor)
    predictor.schema = {
        "feature_names": ["text_score", "absent_feature"],
        "categories": [],
        "score_source": "single_probability",
    }
    predictor.score_transform = None

    assert predictor._matrix([{"text_score": 0.4}]).tolist() == [[pytest.approx(0.4), 0.0]]


def _fusion_dir(path: Path, *, score_source: str | None) -> Path:
    path.mkdir()
    (path / "fusion.cbm").write_bytes(b"model")
    schema: dict[str, object] = {
        "feature_names": ["text_score", "second_score"],
        "categories": ["A"],
    }
    if score_source is not None:
        schema["score_source"] = score_source
        schema["score_contract"] = {
            "version": 2,
            "kind": "raw_probability",
            "score_source": score_source,
        }
    (path / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return path


def test_packaging_accepts_a_two_score_schema_with_two_models(tmp_path: Path) -> None:
    primary = write_saved_backbone(tmp_path / "primary")
    secondary = write_saved_backbone(tmp_path / "secondary")
    output = tmp_path / "two_score.zip"

    result = build_submission(
        primary,
        _fusion_dir(tmp_path / "fusion", score_source="two_score"),
        output,
        secondary_cross_encoder_dir=secondary,
        score_composition="two_score",
        allow_ungated_build=True,
    )

    assert result["score_composition"] == "two_score"
    assert result["model_count"] == 2
    with zipfile.ZipFile(output) as archive:
        run_source = archive.read("run.py").decode("utf-8")
        metadata = json.loads(archive.read("metadata.json"))
    assert "score_composition='two_score'" in run_source
    assert "model_weights=None," in run_source
    assert metadata["matchcup"]["score_composition"] == "two_score"
    assert metadata["matchcup"]["model_weights"] is None


def test_packaging_rejects_a_two_score_schema_with_one_model(tmp_path: Path) -> None:
    cross_encoder = write_saved_backbone(tmp_path / "primary")
    fusion = _fusion_dir(tmp_path / "fusion", score_source="two_score")

    with pytest.raises(ValueError, match="exactly two models"):
        build_submission(
            cross_encoder,
            fusion,
            tmp_path / "asked.zip",
            score_composition="two_score",
            allow_ungated_build=True,
        )
    with pytest.raises(ValueError, match="score source does not match"):
        build_submission(
            cross_encoder,
            fusion,
            tmp_path / "implied.zip",
            allow_ungated_build=True,
        )


def test_packaging_refuses_two_score_beside_a_probability_space_weight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no probability-space weights"):
        build_submission(
            write_saved_backbone(tmp_path / "primary"),
            _fusion_dir(tmp_path / "fusion", score_source="two_score"),
            tmp_path / "weighted.zip",
            secondary_cross_encoder_dir=write_saved_backbone(tmp_path / "secondary"),
            score_composition="two_score",
            model_weights=(0.5, 0.5),
            allow_ungated_build=True,
        )


def test_two_score_run_py_carries_the_composition_and_no_weights() -> None:
    source = _run_py("hybrid", has_secondary_model=True, score_composition="two_score")

    assert "score_composition='two_score'" in source
    assert "model_weights=None," in source
    assert "model_weights=(0.5, 0.5)" not in source


def _matchcup_metadata(archive_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive_path) as archive:
        return json.loads(archive.read("metadata.json"))["matchcup"]


def test_default_single_model_archive_is_unchanged(tmp_path: Path) -> None:
    cross_encoder = write_saved_backbone(tmp_path / "cross_encoder")
    output = tmp_path / "single.zip"

    build_submission(
        cross_encoder,
        _fusion_dir(tmp_path / "fusion", score_source="single_probability"),
        output,
        allow_ungated_build=True,
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.read("run.py").decode("utf-8") == _SINGLE_MODEL_RUN_PY
    metadata = _matchcup_metadata(output)
    assert len(metadata.pop("backbone_contracts")) == 1
    assert metadata == {
        "model_count": 1,
        "model_weights": [1.0],
        "score_composition": "mean_probability",
        "cuda_precision": "fp16_amp",
        "model_batch_size": 1024,
        "score_contract": {
            "version": 2,
            "kind": "raw_probability",
            "score_source": "single_probability",
        },
        "transductive_profile": "h12-v1",
        "feature_schema_sha256": None,
        "constant_categories": [],
    }


def test_default_two_model_mean_probability_archive_is_unchanged(tmp_path: Path) -> None:
    primary = write_saved_backbone(tmp_path / "primary")
    secondary = write_saved_backbone(tmp_path / "secondary")
    output = tmp_path / "ensemble.zip"

    build_submission(
        primary,
        _fusion_dir(tmp_path / "fusion", score_source="mean_probability"),
        output,
        secondary_cross_encoder_dir=secondary,
        allow_ungated_build=True,
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.read("run.py").decode("utf-8") == _TWO_MODEL_MEAN_RUN_PY
    metadata = _matchcup_metadata(output)
    assert len(metadata.pop("backbone_contracts")) == 2
    assert metadata == {
        "model_count": 2,
        "model_weights": [0.5, 0.5],
        "score_composition": "mean_probability",
        "cuda_precision": "fp16_amp",
        "model_batch_size": 1024,
        "score_contract": {
            "version": 2,
            "kind": "raw_probability",
            "score_source": "mean_probability",
        },
        "transductive_profile": "h12-v1",
        "feature_schema_sha256": None,
        "constant_categories": [],
    }

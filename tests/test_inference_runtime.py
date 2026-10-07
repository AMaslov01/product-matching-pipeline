import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from matchcup import inference
from matchcup.features import pair_features
from matchcup.inference import (
    FusionPredictor,
    _canonicalize_items,
    _combine_probability_scores,
    _cuda_precision_config,
    _disable_modernbert_reference_compile,
    _prepare_pair_chunk,
    _tokenizers_compatible,
    _write_predictions,
)
from matchcup.parser import canonicalize_record
from matchcup.transductive import build_transductive_statistics


def test_cuda_precision_defaults_to_oof_fp16_autocast_contract() -> None:
    torch = SimpleNamespace(float16="fp16", bfloat16="bf16")

    model_dtype, autocast_dtype = _cuda_precision_config(torch, "cuda", "fp16_amp")

    assert model_dtype is None
    assert autocast_dtype == "fp16"
    assert _cuda_precision_config(torch, "cuda", "bf16_weights") == ("bf16", "bf16")
    with pytest.raises(ValueError, match="Unsupported CUDA precision"):
        _cuda_precision_config(torch, "cuda", "automatic")


def test_probability_ensemble_is_precommitted_equal_weight_mean() -> None:
    control = np.asarray([0.1, 0.9], dtype=np.float32)
    candidate = np.asarray([0.3, 0.7], dtype=np.float32)

    assert _combine_probability_scores([control, candidate]) == pytest.approx([0.2, 0.8])
    assert _combine_probability_scores([control, candidate], (0.5, 0.5)) == pytest.approx(
        [0.2, 0.8]
    )
    with pytest.raises(ValueError, match="sum to one"):
        _combine_probability_scores([control, candidate], (0.75, 0.75))
    with pytest.raises(ValueError, match="equal length"):
        _combine_probability_scores([control, candidate[:1]])


def _fusion_directory(tmp_path):
    directory = tmp_path / "fusion"
    directory.mkdir()
    (directory / "feature_schema.json").write_text(
        json.dumps({"feature_names": [], "categories": []}), encoding="utf-8"
    )
    (directory / "fusion.cbm").write_bytes(b"not-a-catboost-model")
    return directory


def test_fusion_predictor_requires_native_catboost(tmp_path, monkeypatch) -> None:
    directory = _fusion_directory(tmp_path)
    monkeypatch.setitem(sys.modules, "catboost", None)

    with pytest.raises(RuntimeError, match="native CatBoost is required"):
        FusionPredictor(directory)


def test_fusion_predictor_reports_invalid_native_model(tmp_path) -> None:
    directory = _fusion_directory(tmp_path)

    with pytest.raises(RuntimeError, match="could not load native CatBoost model"):
        FusionPredictor(directory)


def test_fusion_predictor_reports_missing_native_model(tmp_path) -> None:
    directory = _fusion_directory(tmp_path)
    (directory / "fusion.cbm").unlink()

    with pytest.raises(RuntimeError, match="could not load native CatBoost model"):
        FusionPredictor(directory)


def test_prediction_csv_preserves_pair_order_and_required_columns(tmp_path) -> None:
    matches = pd.DataFrame({"id1": [42, 7], "id2": [9, 11]})
    path = tmp_path / "submission.csv"

    output = _write_predictions(matches, np.asarray([0.25, 1.25]), path)

    assert output.columns.tolist() == ["id1", "id2", "predict"]
    assert output.to_dict("records") == [
        {"id1": 42, "id2": 9, "predict": 0.25},
        {"id1": 7, "id2": 11, "predict": 1.25},
    ]
    assert pd.read_csv(path).to_dict("records") == output.to_dict("records")


def test_prediction_csv_rejects_nonfinite_scores(tmp_path) -> None:
    matches = pd.DataFrame({"id1": [1], "id2": [2]})
    with pytest.raises(RuntimeError, match="finite score"):
        _write_predictions(matches, np.asarray([np.nan]), tmp_path / "submission.csv")


def test_modernbert_runtime_uses_eager_embedding_path() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="modernbert", reference_compile=None)
    )

    _disable_modernbert_reference_compile(model)

    assert model.config.reference_compile is False


def test_runtime_compile_setting_does_not_touch_other_models() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="bert"))

    _disable_modernbert_reference_compile(model)

    assert not hasattr(model.config, "reference_compile")


def test_runtime_canonicalization_streams_only_pair_item_ids(tmp_path) -> None:
    items = tmp_path / "items.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3],
                "name": ["one", "two", "three"],
                "attributes": ["{}", "{}", "{}"],
                "category": ["A", "A", "B"],
            }
        ),
        items,
    )

    result, statistics = _canonicalize_items(items, workers=1, batch_size=1, accepted_ids={1, 3})

    assert set(result) == {1, 3}
    assert result[1]["name_raw"] == "one"
    assert statistics is None


def test_parallel_pair_preparation_keeps_symmetric_features_and_text() -> None:
    left = canonicalize_record(
        {"id": 1, "name": "Samsung S24", "attributes": '{"Бренд":"Samsung"}', "category": "A"}
    )
    right = canonicalize_record(
        {"id": 2, "name": "Самсунг S24", "attributes": '{"Бренд":"Самсунг"}', "category": "A"}
    )

    records, texts = _prepare_pair_chunk([(left, right), (right, left)])

    assert records[0] == records[1]
    assert texts[0] == texts[1]


def test_tokenizer_compatibility_requires_identical_encoding_contract() -> None:
    class FakeTokenizer:
        def __init__(self, vocabulary, *, padding_side="right") -> None:
            self._vocabulary = vocabulary
            self.special_tokens_map = {"pad_token": "[PAD]"}
            self.model_max_length = 384
            self.padding_side = padding_side
            self.truncation_side = "right"

        def get_vocab(self):
            return self._vocabulary

    assert _tokenizers_compatible(FakeTokenizer({"a": 0}), FakeTokenizer({"a": 0}))
    assert not _tokenizers_compatible(
        FakeTokenizer({"a": 0}), FakeTokenizer({"a": 0}, padding_side="left")
    )


def _serving_items_parquet(path):
    """Items file whose catalogue is deliberately wider than its pair universe."""
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "name": [
                    "Rare 123 Gadget",
                    "Rare 123 Gadget",
                    "Common thing",
                    "Common thing rare",
                    "Unpaired filler",
                ],
                "attributes": [
                    json.dumps({"Бренд": "Alpha", "Артикул": "AB-123"}, ensure_ascii=False),
                    json.dumps({"Бренд": "Alpha", "Артикул": "AB-123"}, ensure_ascii=False),
                    json.dumps({"Бренд": "Beta", "Артикул": "ZZ-999"}, ensure_ascii=False),
                    json.dumps({"Бренд": "Alpha", "Артикул": "QQ-777"}, ensure_ascii=False),
                    "{}",
                ],
                "category": ["A", "A", "B", "B", "C"],
            }
        ),
        path,
    )
    return path


@pytest.mark.parametrize("workers", [1, 2])
def test_single_pass_statistics_equal_the_separate_corpus_sweep(tmp_path, workers) -> None:
    """The fused pass must reproduce the sweep it replaced, bit for bit.

    Statistics describe the whole submitted items file while only pair items are
    returned, so the fixture keeps ids 4 and 5 out of ``accepted_ids``.
    """
    items_path = _serving_items_parquet(tmp_path / "items.parquet")
    expected = build_transductive_statistics(items_path, batch_size=2)

    items, statistics = _canonicalize_items(
        items_path, workers, 2, accepted_ids={1, 2, 3}, collect_statistics=True
    )

    assert set(items) == {1, 2, 3}
    assert statistics == expected
    assert statistics.total_items == expected.total_items == 5
    assert statistics.name_token_document_frequency == expected.name_token_document_frequency
    assert statistics.brand_document_frequency == expected.brand_document_frequency
    assert statistics.identifier_document_frequency == expected.identifier_document_frequency
    assert statistics.sorted_brand_frequencies == expected.sorted_brand_frequencies


@pytest.mark.parametrize("workers", [1, 2])
def test_single_pass_transductive_feature_values_are_unchanged(tmp_path, workers) -> None:
    items_path = _serving_items_parquet(tmp_path / "items.parquet")
    expected_statistics = build_transductive_statistics(items_path, batch_size=2)

    items, statistics = _canonicalize_items(
        items_path, workers, 2, accepted_ids={1, 2, 3}, collect_statistics=True
    )
    annotated = {item_id: statistics.annotate(item) for item_id, item in items.items()}
    reference = {item_id: expected_statistics.annotate(item) for item_id, item in items.items()}

    for left, right in ((1, 2), (1, 3), (2, 3)):
        produced = pair_features(annotated[left], annotated[right])
        transductive = {
            name: value for name, value in produced.items() if name.startswith("transductive_")
        }
        assert transductive
        assert all(math.isfinite(value) for value in transductive.values())
        assert transductive == {
            name: value
            for name, value in pair_features(reference[left], reference[right]).items()
            if name.startswith("transductive_")
        }
        assert produced == pair_features(annotated[right], annotated[left])


def test_corpus_statistics_ignore_the_accepted_id_filter(tmp_path) -> None:
    """Narrowing the pair universe must not change what the catalogue statistics see."""
    items_path = _serving_items_parquet(tmp_path / "items.parquet")

    _, wide = _canonicalize_items(
        items_path, 1, 2, accepted_ids={1, 2, 3, 4, 5}, collect_statistics=True
    )
    narrow_items, narrow = _canonicalize_items(
        items_path, 1, 2, accepted_ids={1}, collect_statistics=True
    )

    assert set(narrow_items) == {1}
    assert narrow == wide


def test_statistics_do_not_cost_a_second_read_of_the_items_file(tmp_path, monkeypatch) -> None:
    """Guard the regression this stage was rewritten to remove.

    The corpus sweep used to be its own sequential, unfiltered read of the same
    parquet, which on a serving-sized catalogue cost more than the parallel pass
    it duplicated.
    """
    items_path = _serving_items_parquet(tmp_path / "items.parquet")
    reads = 0
    original = inference.iter_item_batches

    def counting_iter_item_batches(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "iter_item_batches", counting_iter_item_batches)

    _, statistics = _canonicalize_items(
        items_path, 1, 2, accepted_ids={1, 2, 3}, collect_statistics=True
    )

    assert statistics.total_items == 5
    assert reads == 1

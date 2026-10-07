from __future__ import annotations

import inspect
from argparse import Namespace
from pathlib import Path

import pytest

from matchcup.schema_robustness import (
    AttributePermutationSpec,
    load_attribute_permutation_spec,
    raw_text_pairs,
    summarize_training_pairs,
    training_text_pairs,
    transform_training_pair,
)
from matchcup.serialize import serialize_pair


def _spec(*, probability: float = 1.0) -> AttributePermutationSpec:
    return AttributePermutationSpec(
        version=1,
        variant="silver_control",
        transform="attribute_permutation",
        pair_probability=probability,
        seed=20260815,
    )


def _texts() -> tuple[str, str]:
    return (
        "category: test\nname_a: left\nattributes_a: color=red; size=m; material=cotton",
        "name_b: right\nattributes_b: color=blue; size=l; material=wool",
    )


def _attribute_fragments(text: str, prefix: str) -> list[str]:
    line = next(line for line in text.splitlines() if line.startswith(prefix))
    return line.removeprefix(prefix).split("; ")


def test_attribute_permutation_is_deterministic_and_preserves_fragments() -> None:
    text_a, text_b = _texts()
    spec = _spec()

    first = transform_training_pair(text_a, text_b, row_id=17, spec=spec)
    second = transform_training_pair(text_a, text_b, row_id=17, spec=spec)

    assert first == second
    assert first.selected is True
    assert first.applicable_fields == 2
    assert first.changed_fields == 2
    assert first.text_a.splitlines()[:2] == text_a.splitlines()[:2]
    assert first.text_b.splitlines()[0] == text_b.splitlines()[0]
    assert sorted(_attribute_fragments(first.text_a, "attributes_a: ")) == sorted(
        _attribute_fragments(text_a, "attributes_a: ")
    )
    assert sorted(_attribute_fragments(first.text_b, "attributes_b: ")) == sorted(
        _attribute_fragments(text_b, "attributes_b: ")
    )
    assert first.text_a != text_a
    assert first.text_b != text_b


def test_attribute_permutation_handles_probability_and_short_fields() -> None:
    text_a = "name_a: left\nattributes_a: color=red"
    text_b = "name_b: right\nattributes_b: size=l"

    disabled = transform_training_pair(text_a, text_b, row_id=3, spec=_spec(probability=0.0))
    short = transform_training_pair(text_a, text_b, row_id=3, spec=_spec())

    assert disabled.text_a == text_a
    assert disabled.text_b == text_b
    assert disabled.selected is False
    assert short.text_a == text_a
    assert short.text_b == text_b
    assert short.selected is True
    assert short.applicable_fields == 0
    assert short.changed_fields == 0


def test_training_only_inputs_leave_raw_validation_and_serving_texts_unchanged() -> None:
    text_a, text_b = _texts()
    rows = [{"row_id": 9, "text_a": text_a, "text_b": text_b, "target": 1.0, "category": "A"}]

    augmented = training_text_pairs(rows, _spec())
    raw = raw_text_pairs(rows)

    assert augmented[0] != raw[0]
    assert raw == [(text_a, text_b)]
    assert rows[0]["text_a"] == text_a
    assert rows[0]["text_b"] == text_b
    assert set(inspect.signature(transform_training_pair).parameters) == {
        "text_a",
        "text_b",
        "row_id",
        "spec",
    }


def test_attribute_permutation_preserves_canonical_pair_symmetry() -> None:
    item_a = {
        "id": 10,
        "category": "test",
        "name_norm": "left",
        "remaining_attributes": "color=red; size=m; material=cotton",
    }
    item_b = {
        "id": 20,
        "category": "test",
        "name_norm": "right",
        "remaining_attributes": "color=blue; size=l; material=wool",
    }
    forward = serialize_pair(item_a, item_b)
    reverse = serialize_pair(item_b, item_a)

    assert forward == reverse
    assert transform_training_pair(*forward, row_id=42, spec=_spec()) == transform_training_pair(
        *reverse, row_id=42, spec=_spec()
    )


def test_schema_spec_round_trip_and_transform_summary(tmp_path: Path) -> None:
    source = tmp_path / "schema.yaml"
    source.write_text(
        """schema_robustness:
  version: 1
  variant: silver_control
  transform: attribute_permutation
  pair_probability: 1.0
  seed: 11
""",
        encoding="utf-8",
    )
    spec = load_attribute_permutation_spec(source)
    text_a, text_b = _texts()

    assert spec.normalized() == {
        "version": 1,
        "variant": "silver_control",
        "transform": "attribute_permutation",
        "pair_probability": 1.0,
        "seed": 11,
    }
    assert len(spec.fingerprint()) == 64
    assert summarize_training_pairs([(text_a, text_b, 1), ("x", "y", 2)], spec) == {
        "rows": 2,
        "selected_pairs": 2,
        "changed_pairs": 1,
        "applicable_attribute_fields": 2,
        "changed_attribute_fields": 2,
        "clean_pairs": 1,
    }


def test_checked_in_schema_spec_is_the_fixed_fold_candidate() -> None:
    spec_path = (
        Path(__file__).parents[1] / "configs" / "schema_robustness_attribute_permutation.yaml"
    )
    spec = load_attribute_permutation_spec(spec_path)

    assert spec.normalized() == {
        "version": 1,
        "variant": "silver_control",
        "transform": "attribute_permutation",
        "pair_probability": 0.5,
        "seed": 20260815,
    }


def test_schema_spec_rejects_non_control_variant(tmp_path: Path) -> None:
    source = tmp_path / "bad.yaml"
    source.write_text(
        """schema_robustness:
  version: 1
  variant: gold_only
  transform: attribute_permutation
  pair_probability: 0.5
  seed: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="silver_control"):
        load_attribute_permutation_spec(source)


def test_full_training_transform_requires_explicit_final_opt_in() -> None:
    from matchcup.cross_encoder import CrossEncoderConfig

    config = CrossEncoderConfig(model_name="base", train_text_transform=_spec())

    assert config.allow_full_training_transform is False


def test_final_cli_explicitly_enables_transform_and_rejects_other_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matchcup.cli import train_cross_encoder_command

    captured = {}

    def fake_train(*_args, **kwargs):
        captured["config"] = _args[2]
        captured["kwargs"] = kwargs
        return {"rows": 1}

    monkeypatch.setattr("matchcup.cross_encoder.train_cross_encoder", fake_train)
    spec = Path(__file__).parents[1] / "configs" / "schema_robustness_attribute_permutation.yaml"
    args = Namespace(
        config=str(Path(__file__).parents[1] / "configs" / "default.yaml"),
        stage="final",
        fold=None,
        input=str(tmp_path / "gold.parquet"),
        output=str(tmp_path / "model"),
        base_model="silver",
        hard_negatives=None,
        epochs=1,
        positive_prevalence=0.25677,
        disable_hard_negatives=True,
        schema_robustness_spec=str(spec),
        checkpoint_dir=None,
        checkpoint_gcs=None,
        checkpoint_interval_seconds=1200.0,
    )

    train_cross_encoder_command(args)

    config = captured["config"]
    assert config.allow_full_training_transform is True
    assert config.train_text_transform is not None
    assert config.train_text_transform_provenance["spec"]["pair_probability"] == 0.5
    assert config.epochs == 1
    assert config.positive_prevalence == 0.25677
    assert captured["kwargs"]["hard_negative_path"] is None

    args.stage = "silver"
    args.disable_hard_negatives = False
    with pytest.raises(ValueError, match="only for the explicit final refit"):
        train_cross_encoder_command(args)


def test_fold_cli_saves_the_completed_model_for_oof_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matchcup.cli import train_cross_encoder_command

    captured = {}

    def fake_train(*args, **_kwargs):
        captured["config"] = args[2]
        return {"rows": 1}

    monkeypatch.setattr("matchcup.cross_encoder.train_cross_encoder", fake_train)
    train_cross_encoder_command(
        Namespace(
            config=str(Path(__file__).parents[1] / "configs" / "default.yaml"),
            stage="fold",
            fold=0,
            input=str(tmp_path / "gold.parquet"),
            output=str(tmp_path / "fold_0"),
            base_model="silver",
            hard_negatives=None,
            epochs=1,
            positive_prevalence=0.0625,
            disable_hard_negatives=False,
            schema_robustness_spec=None,
            checkpoint_dir=None,
            checkpoint_gcs=None,
            checkpoint_interval_seconds=1200.0,
        )
    )

    assert captured["config"].save_final_checkpoint is True

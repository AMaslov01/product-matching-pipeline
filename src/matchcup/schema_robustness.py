"""Deterministic, train-only text transforms for schema-robustness research.

The module intentionally accepts only serialized texts and a stable ``row_id``.
It has no access to labels, categories, folds, or prediction-time inputs.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SPEC_KEYS = {"version", "variant", "transform", "pair_probability", "seed"}
_ATTRIBUTE_PREFIXES = ("attributes_a: ", "attributes_b: ")


@dataclass(frozen=True)
class AttributePermutationSpec:
    """The complete, serializable contract for the attribute permutation ablation."""

    version: int
    variant: str
    transform: str
    pair_probability: float
    seed: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AttributePermutationSpec:
        unknown = set(values) - _SPEC_KEYS
        missing = _SPEC_KEYS - set(values)
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing keys: {sorted(missing)}")
            if unknown:
                details.append(f"unknown keys: {sorted(unknown)}")
            raise ValueError("Invalid schema-robustness spec (" + "; ".join(details) + ")")
        if values["version"] != 1:
            raise ValueError("Unsupported schema-robustness spec version")
        if values["variant"] != "silver_control":
            raise ValueError("schema-robustness is restricted to variant='silver_control'")
        if values["transform"] != "attribute_permutation":
            raise ValueError("Unsupported schema-robustness transform")
        probability = values["pair_probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("pair_probability must be a number")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("pair_probability must be in [0, 1]")
        seed = values["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        return cls(
            version=1,
            variant="silver_control",
            transform="attribute_permutation",
            pair_probability=float(probability),
            seed=seed,
        )

    def normalized(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "variant": self.variant,
            "transform": self.transform,
            "pair_probability": self.pair_probability,
            "seed": self.seed,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.normalized(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PairTextTransformResult:
    """A transformed pair plus auditable transform counts for that pair."""

    text_a: str
    text_b: str
    selected: bool
    applicable_fields: int
    changed_fields: int


def load_attribute_permutation_spec(path: str | Path) -> AttributePermutationSpec:
    """Load the versioned standalone YAML spec used by the research runner."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or set(document) != {"schema_robustness"}:
        raise ValueError("Schema-robustness YAML must contain only 'schema_robustness'")
    values = document["schema_robustness"]
    if not isinstance(values, dict):
        raise ValueError("schema_robustness must be a mapping")
    return AttributePermutationSpec.from_mapping(values)


def _stable_uint64(*parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def _is_selected(row_id: int | str, spec: AttributePermutationSpec) -> bool:
    threshold = _stable_uint64(spec.seed, row_id, "pair-selection") / 2**64
    return threshold < spec.pair_probability


def _nonidentity_permutation(
    size: int, *, row_id: int | str, prefix: str, occurrence: int, seed: int
) -> list[int]:
    order = list(range(size))
    random.Random(_stable_uint64(seed, row_id, prefix, occurrence)).shuffle(order)
    if order == list(range(size)):
        order = order[1:] + order[:1]
    return order


def _permuted_attribute_text(
    text: str, *, row_id: int | str, prefix: str, seed: int
) -> tuple[str, int, int]:
    """Return text, number of applicable lines, and number visibly changed."""
    transformed: list[str] = []
    applicable = 0
    changed = 0
    occurrence = 0
    for line in text.splitlines(keepends=True):
        if not line.startswith(prefix):
            transformed.append(line)
            continue
        value = line[len(prefix) :]
        content = value.rstrip("\r\n")
        line_ending = value[len(content) :]
        fragments = content.split("; ")
        if len(fragments) < 2:
            transformed.append(line)
            continue
        applicable += 1
        order = _nonidentity_permutation(
            len(fragments), row_id=row_id, prefix=prefix, occurrence=occurrence, seed=seed
        )
        rotations = [
            list(range(shift, len(fragments))) + list(range(shift))
            for shift in range(1, len(fragments))
        ]
        candidates = [order, *rotations]
        reordered = fragments
        for candidate in candidates:
            candidate_fragments = [fragments[index] for index in candidate]
            if candidate_fragments != fragments:
                reordered = candidate_fragments
                break
        replacement = prefix + "; ".join(reordered) + line_ending
        transformed.append(replacement)
        changed += replacement != line
        occurrence += 1
    return "".join(transformed), applicable, changed


def transform_training_pair(
    text_a: str,
    text_b: str,
    *,
    row_id: int | str,
    spec: AttributePermutationSpec,
) -> PairTextTransformResult:
    """Apply the pair-level attribute permutation without inspecting any labels."""
    if not _is_selected(row_id, spec):
        return PairTextTransformResult(text_a, text_b, False, 0, 0)
    transformed_a, applicable_a, changed_a = _permuted_attribute_text(
        text_a, row_id=row_id, prefix=_ATTRIBUTE_PREFIXES[0], seed=spec.seed
    )
    transformed_b, applicable_b, changed_b = _permuted_attribute_text(
        text_b, row_id=row_id, prefix=_ATTRIBUTE_PREFIXES[1], seed=spec.seed
    )
    return PairTextTransformResult(
        text_a=transformed_a,
        text_b=transformed_b,
        selected=True,
        applicable_fields=applicable_a + applicable_b,
        changed_fields=changed_a + changed_b,
    )


def raw_text_pairs(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Return unmodified tokenizer inputs for validation and serving paths."""
    return [(str(row["text_a"]), str(row["text_b"])) for row in rows]


def training_text_pairs(
    rows: Sequence[Mapping[str, Any]], spec: AttributePermutationSpec | None
) -> list[tuple[str, str]]:
    """Produce tokenization inputs for training; ``None`` preserves raw texts exactly."""
    if spec is None:
        return raw_text_pairs(rows)
    pairs = []
    for row in rows:
        if "row_id" not in row:
            raise ValueError("schema-robustness transform requires a stable row_id")
        result = transform_training_pair(
            str(row["text_a"]), str(row["text_b"]), row_id=row["row_id"], spec=spec
        )
        pairs.append((result.text_a, result.text_b))
    return pairs


def summarize_training_pairs(
    pairs: Iterable[tuple[str, str, int | str]], spec: AttributePermutationSpec
) -> dict[str, int]:
    """Count the clean and augmented rows without looking at labels or categories."""
    summary = {
        "rows": 0,
        "selected_pairs": 0,
        "changed_pairs": 0,
        "applicable_attribute_fields": 0,
        "changed_attribute_fields": 0,
    }
    for text_a, text_b, row_id in pairs:
        result = transform_training_pair(text_a, text_b, row_id=row_id, spec=spec)
        summary["rows"] += 1
        summary["selected_pairs"] += int(result.selected)
        summary["changed_pairs"] += int(result.changed_fields > 0)
        summary["applicable_attribute_fields"] += result.applicable_fields
        summary["changed_attribute_fields"] += result.changed_fields
    summary["clean_pairs"] = summary["rows"] - summary["changed_pairs"]
    return summary

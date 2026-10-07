"""Deterministic Gold-only training augmentations.

The materializers write an explicit ``is_synthetic`` marker and retain a
``source_row_id``.  The cross-encoder's OOF path consumes only real rows, while
the normal fold predicate keeps synthetic descendants with their source fold.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.features import pair_category
from matchcup.folds import DisjointSet
from matchcup.serialize import serialize_pair

_ATTRIBUTE_PREFIXES = ("attributes_a: ", "attributes_b: ")
_REQUIRED_PAIR_COLUMNS = {
    "row_id",
    "id1",
    "id2",
    "target",
    "category",
    "fold",
    "text_a",
    "text_b",
}


@dataclass(frozen=True)
class _AttributeSource:
    row_id: int
    category: str
    ratio: float


def _stable_uint64(*parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def _attribute_fragments(text: str, prefix: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(prefix):
            content = line[len(prefix) :]
            return [fragment for fragment in content.split("; ") if fragment]
    return []


def _attribute_ratio(text_a: str, text_b: str) -> float:
    left = len(_attribute_fragments(text_a, _ATTRIBUTE_PREFIXES[0]))
    right = len(_attribute_fragments(text_b, _ATTRIBUTE_PREFIXES[1]))
    return min(left, right) / max(left, right, 1)


def _replace_attribute_line(text: str, prefix: str, fragments: list[str]) -> str:
    replacement = prefix + "; ".join(fragments)
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            ending = "\n" if line.endswith("\n") else ""
            lines[index] = replacement + ending
            return "".join(lines)
    return text


def _rounded_larger_count(smaller: int, larger: int, target_ratio: float) -> int:
    """Choose the nearest attainable non-zero ratio by rounding the larger side."""
    if smaller <= 0 or larger <= 0 or target_ratio <= 0.0:
        return larger
    desired = int(math.floor(smaller / target_ratio + 0.5))
    return min(larger, max(smaller, desired))


def _drop_attributes(
    text_a: str,
    text_b: str,
    *,
    row_id: int,
    target_ratio: float,
    seed: int,
) -> tuple[str, str]:
    left = _attribute_fragments(text_a, _ATTRIBUTE_PREFIXES[0])
    right = _attribute_fragments(text_b, _ATTRIBUTE_PREFIXES[1])
    if not left or not right or len(left) == len(right):
        return text_a, text_b
    if len(left) > len(right):
        fuller, other, prefix, source_text, side = left, right, _ATTRIBUTE_PREFIXES[0], text_a, "a"
        unchanged_text = text_b
    else:
        fuller, other, prefix, source_text, side = right, left, _ATTRIBUTE_PREFIXES[1], text_b, "b"
        unchanged_text = text_a
    retained = _rounded_larger_count(len(other), len(fuller), target_ratio)
    remove = len(fuller) - retained
    if remove <= 0:
        return text_a, text_b
    order = sorted(
        range(len(fuller)),
        key=lambda index: _stable_uint64(seed, row_id, "attribute-dropout", side, index),
    )
    remove_indices = set(order[:remove])
    kept = [fragment for index, fragment in enumerate(fuller) if index not in remove_indices]
    changed = _replace_attribute_line(source_text, prefix, kept)
    return (changed, unchanged_text) if side == "a" else (unchanged_text, changed)


def _output_schema(source: pa.Schema) -> pa.Schema:
    fields = list(source)
    names = set(source.names)
    if "is_synthetic" not in names:
        fields.append(pa.field("is_synthetic", pa.bool_(), nullable=False))
    if "source_row_id" not in names:
        fields.append(pa.field("source_row_id", pa.int64(), nullable=True))
    return pa.schema(fields)


def _source_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["is_synthetic"] = bool(result.get("is_synthetic", False))
    result.setdefault("source_row_id", None)
    return result


def _require_unaugmented_source(records: list[dict[str, Any]], *, augmentation: str) -> None:
    if any(bool(record.get("is_synthetic", False)) for record in records):
        raise ValueError(
            f"{augmentation} must start from real rows; combined augmentations are forbidden"
        )


def _select_donor(
    source: _AttributeSource, candidates: list[_AttributeSource], *, seed: int
) -> _AttributeSource:
    alternatives = [candidate for candidate in candidates if candidate.row_id != source.row_id]
    pool = alternatives or candidates
    return pool[_stable_uint64(seed, source.row_id, "attribute-ratio-donor") % len(pool)]


def materialize_attribute_dropout(
    source_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    compression: str = "zstd",
) -> dict[str, int]:
    """Append one deterministic positive synthetic row per real positive source row.

    Donor ratios are sampled only from other real positive rows in the same
    category.  Deletion touches the fuller serialized ``attributes_*`` side
    only, retains at least one fragment on either side, and copies the source
    fold exactly.
    """
    source_path = Path(source_path)
    schema = pq.ParquetFile(source_path).schema
    missing = _REQUIRED_PAIR_COLUMNS - set(schema.names)
    if missing:
        raise ValueError(f"attribute dropout requires columns: {sorted(missing)}")
    table = pq.read_table(source_path)
    records = table.to_pylist()
    _require_unaugmented_source(records, augmentation="attribute dropout")
    if any(record.get("row_id") is None for record in records):
        raise ValueError("attribute dropout requires non-null row_id values")
    if any(record.get("fold") is None for record in records):
        raise ValueError("attribute dropout requires non-null fold values")

    donors: dict[str, list[_AttributeSource]] = defaultdict(list)
    for record in records:
        if bool(record.get("is_synthetic", False)) or float(record["target"]) <= 0.5:
            continue
        ratio = _attribute_ratio(str(record["text_a"]), str(record["text_b"]))
        if ratio > 0.0:
            donors[str(record["category"])].append(
                _AttributeSource(int(record["row_id"]), str(record["category"]), ratio)
            )
    for candidates in donors.values():
        candidates.sort(key=lambda candidate: candidate.row_id)

    max_row_id = max(int(record["row_id"]) for record in records)
    synthetic: list[dict[str, Any]] = []
    changed = 0
    for record in records:
        if bool(record.get("is_synthetic", False)) or float(record["target"]) <= 0.5:
            continue
        source = _AttributeSource(
            int(record["row_id"]),
            str(record["category"]),
            _attribute_ratio(str(record["text_a"]), str(record["text_b"])),
        )
        candidates = donors.get(source.category, [])
        if candidates:
            donor = _select_donor(source, candidates, seed=seed)
            text_a, text_b = _drop_attributes(
                str(record["text_a"]),
                str(record["text_b"]),
                row_id=source.row_id,
                target_ratio=donor.ratio,
                seed=seed,
            )
        else:
            text_a, text_b = str(record["text_a"]), str(record["text_b"])
        result = _source_record(record)
        result["row_id"] = max_row_id + len(synthetic) + 1
        result["source_row_id"] = source.row_id
        result["is_synthetic"] = True
        result["text_a"] = text_a
        result["text_b"] = text_b
        synthetic.append(result)
        changed += int(text_a != str(record["text_a"]) or text_b != str(record["text_b"]))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_schema = _output_schema(table.schema)
    source_rows = [_source_record(record) for record in records]
    pq.write_table(
        pa.Table.from_pylist([*source_rows, *synthetic], schema=output_schema),
        output_path,
        compression=compression,
    )
    return {
        "source_rows": len(records),
        "source_positive_rows": len(synthetic),
        "synthetic_rows": len(synthetic),
        "changed_synthetic_rows": changed,
    }


def _identifier_tokens(value: Any) -> set[str]:
    try:
        return {str(token) for token in json.loads(value or "[]") if str(token)}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _normalized_idf(total_items: int, document_frequency: int) -> float:
    if total_items <= 1 or document_frequency <= 0:
        return 0.0
    return math.log(total_items / document_frequency) / math.log(total_items)


def _lcp_fraction(left: str, right: str) -> float:
    common = 0
    for first, second in zip(left, right, strict=False):
        if first != second:
            break
        common += 1
    return common / max(len(left), len(right), 1)


def _canonical_columns(schema: pa.Schema) -> list[str]:
    wanted = [
        "id",
        "category",
        "name_norm",
        "name_raw",
        "brand_norm",
        "model_norm",
        "article_norm",
        "type_norm",
        "size_norm",
        "quantity_norm",
        "color_norm",
        "remaining_attributes",
        "identifier_tokens",
    ]
    return [column for column in wanted if column in schema.names]


def materialize_hard_identifier_negatives(
    source_path: str | Path,
    catalogue_items_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    min_identifier_idf: float = 0.70,
    min_lcp_fraction: float = 0.50,
    max_rows: int = 4_066,
    compression: str = "zstd",
) -> dict[str, int]:
    """Append every eligible real-catalogue hard identifier near-miss.

    A synthetic negative pairs a positive-source endpoint with a different
    catalogue item whose rare identifier has a long common prefix.  Donors in
    the known positive component of that endpoint are excluded.  The expected
    catalogue scan has at most 4,066 eligible pairs; exceeding that predeclared
    count fails rather than silently sampling a different experiment.
    """
    if not 0.0 <= min_identifier_idf <= 1.0:
        raise ValueError("min_identifier_idf must be in [0, 1]")
    if not 0.0 < min_lcp_fraction <= 1.0:
        raise ValueError("min_lcp_fraction must be in (0, 1]")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    source_path = Path(source_path)
    pair_schema = pq.ParquetFile(source_path).schema
    missing = _REQUIRED_PAIR_COLUMNS - set(pair_schema.names)
    if missing:
        raise ValueError(f"hard identifiers require columns: {sorted(missing)}")
    pair_table = pq.read_table(source_path)
    pair_records = pair_table.to_pylist()
    _require_unaugmented_source(pair_records, augmentation="hard identifiers")
    if any(record.get("row_id") is None or record.get("fold") is None for record in pair_records):
        raise ValueError("hard identifiers require non-null row_id and fold values")

    components = DisjointSet()
    source_by_item: dict[int, dict[str, Any]] = {}
    for record in sorted(pair_records, key=lambda row: int(row["row_id"])):
        if float(record["target"]) <= 0.5:
            continue
        left, right = int(record["id1"]), int(record["id2"])
        components.union(left, right)
        for item_id in (left, right):
            source_by_item.setdefault(item_id, record)
    if not source_by_item:
        raise ValueError("hard identifiers require at least one positive source pair")

    catalogue_items_path = Path(catalogue_items_path)
    catalogue = pq.ParquetFile(catalogue_items_path)
    if {"id", "identifier_tokens"} - set(catalogue.schema.names):
        raise ValueError("catalogue items require id and identifier_tokens columns")
    identifier_df: Counter[str] = Counter()
    total_items = 0
    anchor_tokens: dict[int, set[str]] = {}
    for batch in catalogue.iter_batches(batch_size=8192, columns=["id", "identifier_tokens"]):
        for item in batch.to_pylist():
            item_id = int(item["id"])
            identifiers = _identifier_tokens(item.get("identifier_tokens"))
            identifier_df.update(identifiers)
            total_items += 1
            if item_id in source_by_item:
                anchor_tokens[item_id] = identifiers
    if total_items <= 1:
        raise ValueError("catalogue must contain at least two items")

    anchors_by_prefix: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for item_id, identifiers in anchor_tokens.items():
        for identifier in identifiers:
            if _normalized_idf(total_items, identifier_df[identifier]) < min_identifier_idf:
                continue
            prefix_size = math.ceil(len(identifier) * min_lcp_fraction)
            anchors_by_prefix[identifier[:prefix_size]].add((item_id, identifier))
    if not anchors_by_prefix:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_schema = _output_schema(pair_table.schema)
        pq.write_table(
            pa.Table.from_pylist(
                [_source_record(record) for record in pair_records], schema=output_schema
            ),
            output_path,
            compression=compression,
        )
        return {
            "source_rows": len(pair_records),
            "source_positive_rows": sum(float(record["target"]) > 0.5 for record in pair_records),
            "eligible_pairs": 0,
            "synthetic_rows": 0,
        }

    canonical_columns = _canonical_columns(catalogue.schema)
    if "id" not in canonical_columns or "identifier_tokens" not in canonical_columns:
        raise ValueError("catalogue items are missing required canonical identifier fields")
    anchor_items: dict[int, dict[str, Any]] = {}
    donor_items: dict[int, dict[str, Any]] = {}
    eligible: set[tuple[int, int]] = set()
    for batch in catalogue.iter_batches(batch_size=4096, columns=canonical_columns):
        for donor in batch.to_pylist():
            donor_id = int(donor["id"])
            if donor_id in source_by_item:
                anchor_items[donor_id] = donor
            # A synthetic row inherits its anchor's fold.  Reusing an item
            # from *any* labelled positive component would therefore attach
            # that already assigned component to the anchor fold and break
            # component-disjoint Gold folds, even when the components differ.
            if donor_id in components.parent:
                continue
            donor_identifiers = [
                identifier
                for identifier in _identifier_tokens(donor.get("identifier_tokens"))
                if _normalized_idf(total_items, identifier_df[identifier]) >= min_identifier_idf
            ]
            if not donor_identifiers:
                continue
            for donor_identifier in donor_identifiers:
                matches: set[tuple[int, str]] = set()
                for length in range(1, len(donor_identifier) + 1):
                    matches.update(anchors_by_prefix.get(donor_identifier[:length], set()))
                for anchor_id, anchor_identifier in matches:
                    if donor_id == anchor_id or donor_identifier == anchor_identifier:
                        continue
                    if _lcp_fraction(anchor_identifier, donor_identifier) < min_lcp_fraction:
                        continue
                    eligible.add((anchor_id, donor_id))
                    donor_items[donor_id] = donor
                    if len(eligible) > max_rows:
                        raise ValueError(
                            "eligible hard-identifier pairs exceed the predeclared maximum "
                            f"of {max_rows}"
                        )

    missing_anchors = set(source_by_item) - set(anchor_items)
    if missing_anchors:
        raise ValueError(f"catalogue is missing positive source ids: {len(missing_anchors)}")
    max_row_id = max(int(record["row_id"]) for record in pair_records)
    synthetic: list[dict[str, Any]] = []
    for anchor_id, donor_id in sorted(
        eligible,
        key=lambda pair: (int(source_by_item[pair[0]]["row_id"]), pair[0], pair[1]),
    ):
        source = source_by_item[anchor_id]
        text_a, text_b = serialize_pair(anchor_items[anchor_id], donor_items[donor_id])
        result = _source_record(source)
        result.update(
            {
                "row_id": max_row_id + len(synthetic) + 1,
                "id1": anchor_id,
                "id2": donor_id,
                "target": 0.0,
                "category": pair_category(anchor_items[anchor_id], donor_items[donor_id]),
                "fold": int(source["fold"]),
                "text_a": text_a,
                "text_b": text_b,
                "source_row_id": int(source["row_id"]),
                "is_synthetic": True,
            }
        )
        synthetic.append(result)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_schema = _output_schema(pair_table.schema)
    pq.write_table(
        pa.Table.from_pylist(
            [*[_source_record(record) for record in pair_records], *synthetic], schema=output_schema
        ),
        output_path,
        compression=compression,
    )
    return {
        "source_rows": len(pair_records),
        "source_positive_rows": sum(float(record["target"]) > 0.5 for record in pair_records),
        "eligible_pairs": len(eligible),
        "synthetic_rows": len(synthetic),
    }

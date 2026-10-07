from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

from matchcup.normalize import identifier_tokens, normalize_identifier, numeric_tokens, tokens

PAIR_FIELDS = ("brand", "model", "article", "type", "color", "size", "quantity")


def _safe_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _char_ngrams(value: str, n: int = 3) -> set[str]:
    compact = " ".join(str(value or "").split())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _remaining_attribute_map(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in str(value or "").split("; "):
        key, separator, attribute_value = part.partition("=")
        if separator and key and attribute_value:
            result[key] = attribute_value
    return result


def _aligned_attribute_features(item_a: dict[str, Any], item_b: dict[str, Any]) -> dict[str, float]:
    attributes_a = _remaining_attribute_map(item_a.get("remaining_attributes"))
    attributes_b = _remaining_attribute_map(item_b.get("remaining_attributes"))
    keys_a = set(attributes_a)
    keys_b = set(attributes_b)
    common_keys = sorted(keys_a & keys_b)
    union_keys = keys_a | keys_b
    exact = 0
    similarities: list[float] = []
    identifier_common = 0
    identifier_overlap = 0
    identifier_conflict = 0
    numeric_common = 0
    numeric_overlap = 0
    numeric_conflict = 0
    for key in common_keys:
        value_a = attributes_a[key]
        value_b = attributes_b[key]
        exact += int(value_a == value_b)
        similarities.append(_jaccard(tokens(value_a), tokens(value_b)))
        identifiers_a = set(identifier_tokens(value_a))
        identifiers_b = set(identifier_tokens(value_b))
        if identifiers_a and identifiers_b:
            identifier_common += 1
            identifier_overlap += int(bool(identifiers_a & identifiers_b))
            identifier_conflict += int(not bool(identifiers_a & identifiers_b))
        numbers_a = set(numeric_tokens(value_a))
        numbers_b = set(numeric_tokens(value_b))
        if numbers_a and numbers_b:
            numeric_common += 1
            numeric_overlap += int(bool(numbers_a & numbers_b))
            numeric_conflict += int(not bool(numbers_a & numbers_b))
    common_count = len(common_keys)
    conflict = common_count - exact
    return {
        "aligned_attribute_key_count_min": float(min(len(keys_a), len(keys_b))),
        "aligned_attribute_key_count_max": float(max(len(keys_a), len(keys_b))),
        "aligned_attribute_key_common": float(common_count),
        "aligned_attribute_key_jaccard": common_count / len(union_keys) if union_keys else 0.0,
        "aligned_attribute_value_exact": float(exact),
        "aligned_attribute_value_exact_rate": exact / common_count if common_count else 0.0,
        "aligned_attribute_value_conflict": float(conflict),
        "aligned_attribute_value_conflict_rate": conflict / common_count if common_count else 0.0,
        "aligned_attribute_token_jaccard_mean": (
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
        "aligned_attribute_token_jaccard_min": min(similarities, default=0.0),
        "aligned_attribute_token_jaccard_max": max(similarities, default=0.0),
        "aligned_attribute_identifier_common": float(identifier_common),
        "aligned_attribute_identifier_overlap": float(identifier_overlap),
        "aligned_attribute_identifier_conflict": float(identifier_conflict),
        "aligned_attribute_numeric_common": float(numeric_common),
        "aligned_attribute_numeric_overlap": float(numeric_overlap),
        "aligned_attribute_numeric_conflict": float(numeric_conflict),
    }


def _field_relation(a: str, b: str) -> tuple[float, float, float]:
    a = str(a or "")
    b = str(b or "")
    both = float(bool(a and b))
    equal = float(bool(both and a == b))
    conflict = float(bool(both and a != b))
    return both, equal, conflict


def _transductive_features(item_a: dict[str, Any], item_b: dict[str, Any]) -> dict[str, float]:
    """Aggregate item-only catalogue statistics symmetrically for a pair."""
    left_idf = dict(item_a.get("_transductive_name_idf") or {})
    right_idf = dict(item_b.get("_transductive_name_idf") or {})
    if not left_idf and not right_idf:
        return {}
    shared = set(left_idf) & set(right_idf)
    unshared = set(left_idf) ^ set(right_idf)
    shared_idf = [min(float(left_idf[token]), float(right_idf[token])) for token in shared]
    unshared_idf = [float((left_idf | right_idf)[token]) for token in unshared]
    left_identifier = dict(item_a.get("_transductive_identifier_df") or {})
    right_identifier = dict(item_b.get("_transductive_identifier_df") or {})
    common_identifiers = set(left_identifier) & set(right_identifier)
    identifier_df = [
        min(float(left_identifier[token]), float(right_identifier[token]))
        for token in common_identifiers
    ]
    brand_values = sorted(
        [
            float(item_a.get("_transductive_brand_percentile") or 0.0),
            float(item_b.get("_transductive_brand_percentile") or 0.0),
        ]
    )
    result = {
        "transductive_rarest_shared_token_idf": max(shared_idf, default=0.0),
        "transductive_mean_shared_token_idf": (
            sum(shared_idf) / len(shared_idf) if shared_idf else 0.0
        ),
        "transductive_max_unshared_token_idf": max(unshared_idf, default=0.0),
        "transductive_brand_frequency_percentile_min": brand_values[0],
        "transductive_brand_frequency_percentile_max": brand_values[1],
        "transductive_identifier_df": min(identifier_df, default=0.0),
    }
    left_category_idf = dict(item_a.get("_transductive_category_name_idf") or {})
    right_category_idf = dict(item_b.get("_transductive_category_name_idf") or {})
    if (
        "_transductive_category_name_idf" in item_a
        or "_transductive_category_name_idf" in item_b
    ):
        category_shared = set(left_category_idf) & set(right_category_idf)
        category_unshared = set(left_category_idf) ^ set(right_category_idf)
        shared_category_idf = [
            min(float(left_category_idf[token]), float(right_category_idf[token]))
            for token in category_shared
        ]
        unshared_category_idf = [
            float((left_category_idf | right_category_idf)[token])
            for token in category_unshared
        ]
        result.update(
            {
                "transductive_category_rarest_shared_token_idf": max(
                    shared_category_idf, default=0.0
                ),
                "transductive_category_mean_shared_token_idf": (
                    sum(shared_category_idf) / len(shared_category_idf)
                    if shared_category_idf
                    else 0.0
                ),
                "transductive_category_max_unshared_token_idf": max(
                    unshared_category_idf, default=0.0
                ),
            }
        )
    left_attribute_idf = dict(item_a.get("_transductive_attribute_key_idf") or {})
    right_attribute_idf = dict(item_b.get("_transductive_attribute_key_idf") or {})
    if "_transductive_attribute_key_idf" in item_a or "_transductive_attribute_key_idf" in item_b:
        common_attribute_keys = set(left_attribute_idf) & set(right_attribute_idf)
        union_attribute_keys = set(left_attribute_idf) | set(right_attribute_idf)
        common_attribute_idf = [
            min(float(left_attribute_idf[key]), float(right_attribute_idf[key]))
            for key in common_attribute_keys
        ]
        union_attribute_idf = [
            max(float(left_attribute_idf.get(key, 0.0)), float(right_attribute_idf.get(key, 0.0)))
            for key in union_attribute_keys
        ]
        result.update(
            {
                "aligned_attribute_common_key_idf_sum": sum(common_attribute_idf),
                "aligned_attribute_common_key_idf_max": max(common_attribute_idf, default=0.0),
                "aligned_attribute_key_idf_weighted_jaccard": (
                    sum(common_attribute_idf) / sum(union_attribute_idf)
                    if sum(union_attribute_idf) > 0.0
                    else 0.0
                ),
            }
        )
    return result


def _measurement_features(
    a: dict[str, list[float]], b: dict[str, list[float]], tolerance: float
) -> tuple[float, float, float, float]:
    common = set(a) & set(b)
    compatible = 0
    conflict = 0
    relative_diffs: list[float] = []
    for kind in common:
        left = a[kind]
        right = b[kind]
        kind_compatible = False
        best = math.inf
        for x in left:
            for y in right:
                diff = abs(x - y) / max(abs(x), abs(y), 1e-12)
                best = min(best, diff)
                kind_compatible |= diff <= tolerance
        compatible += int(kind_compatible)
        conflict += int(not kind_compatible)
        if math.isfinite(best):
            relative_diffs.append(best)
    return (
        float(len(common)),
        float(compatible),
        float(conflict),
        min(relative_diffs, default=0.0),
    )


def pair_category(item_a: dict[str, Any], item_b: dict[str, Any]) -> str:
    """Category label for a pair, as the metric groups it.

    Shared with the text-only pair path, which skips the 67 CatBoost features but
    still needs this column: the cross-encoder trainer reads it, and macro AP is
    grouped by it. Keeping one definition stops the two paths from drifting.
    """
    categories = sorted(
        {str(value) for value in (item_a.get("category"), item_b.get("category")) if value}
    )
    return " | ".join(categories)


def pair_features(
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    *,
    ngram_size: int = 3,
    measurement_tolerance: float = 0.015,
) -> dict[str, Any]:
    name_a = str(item_a.get("name_norm") or "")
    name_b = str(item_b.get("name_norm") or "")
    tokens_a = tokens(name_a)
    tokens_b = tokens(name_b)
    id_a = set(_safe_json(item_a.get("identifier_tokens"), []))
    id_b = set(_safe_json(item_b.get("identifier_tokens"), []))
    numeric_a = set(_safe_json(item_a.get("numeric_tokens"), []))
    numeric_b = set(_safe_json(item_b.get("numeric_tokens"), []))
    attribute_counts = sorted(
        [float(item_a.get("attribute_count") or 0), float(item_b.get("attribute_count") or 0)]
    )
    result: dict[str, Any] = {
        "category": pair_category(item_a, item_b),
        "name_token_jaccard": _jaccard(tokens_a, tokens_b),
        "name_token_containment": _containment(tokens_a, tokens_b),
        "name_char_jaccard": _jaccard(
            _char_ngrams(name_a, ngram_size), _char_ngrams(name_b, ngram_size)
        ),
        "name_length_ratio": min(len(name_a), len(name_b)) / max(len(name_a), len(name_b), 1),
        "name_exact": float(bool(name_a and name_a == name_b)),
        "identifier_jaccard": _jaccard(id_a, id_b),
        "identifier_exact_nonempty": float(bool(id_a and id_a == id_b)),
        "identifier_conflict": float(bool(id_a and id_b and id_a != id_b)),
        "numeric_jaccard": _jaccard(numeric_a, numeric_b),
        "numeric_exact_nonempty": float(bool(numeric_a and numeric_a == numeric_b)),
        "numeric_conflict": float(bool(numeric_a and numeric_b and numeric_a != numeric_b)),
        "attribute_count_min": attribute_counts[0],
        "attribute_count_max": attribute_counts[1],
        "attribute_count_difference": attribute_counts[1] - attribute_counts[0],
        "attribute_count_ratio": attribute_counts[0] / max(attribute_counts[1], 1.0),
    }
    for field in PAIR_FIELDS:
        a = str(item_a.get(f"{field}_norm") or "")
        b = str(item_b.get(f"{field}_norm") or "")
        both, equal, conflict = _field_relation(a, b)
        result[f"{field}_both"] = both
        result[f"{field}_equal"] = equal
        result[f"{field}_conflict"] = conflict
        result[f"{field}_token_jaccard"] = _jaccard(tokens(a), tokens(b))
        if field in {"model", "article"}:
            result[f"{field}_identifier_equal"] = float(
                bool(a and b and normalize_identifier(a) == normalize_identifier(b))
            )
        if field == "brand":
            brand_a = str(item_a.get("brand_translit") or "")
            brand_b = str(item_b.get("brand_translit") or "")
            result["brand_translit_equal"] = float(bool(brand_a and brand_a == brand_b))
    measurements_a = _safe_json(item_a.get("measurements"), {})
    measurements_b = _safe_json(item_b.get("measurements"), {})
    common, compatible, conflict, min_diff = _measurement_features(
        measurements_a, measurements_b, measurement_tolerance
    )
    result.update(
        {
            "measurement_common_types": common,
            "measurement_compatible_types": compatible,
            "measurement_conflict_types": conflict,
            "measurement_min_relative_diff": min_diff,
        }
    )
    result.update(_aligned_attribute_features(item_a, item_b))
    result.update(_transductive_features(item_a, item_b))
    return result


def numeric_feature_names(records: Iterable[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for record in records:
        names.update(
            key
            for key, value in record.items()
            if key != "category" and isinstance(value, int | float)
        )
    return sorted(names)

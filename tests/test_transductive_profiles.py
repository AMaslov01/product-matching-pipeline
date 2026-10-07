from __future__ import annotations

import pytest

from matchcup.transductive_profile import (
    C1_FEATURE_NAMES,
    C2_FEATURE_NAMES,
    H12_V1_FEATURE_NAMES,
    feature_schema_digest,
    resolve_transductive_profile,
    validate_fusion_manifest,
    validate_fusion_schema,
)


def test_explicit_profile_locks_the_exact_transductive_feature_family() -> None:
    feature_names = ["name_token_jaccard", *H12_V1_FEATURE_NAMES, *C1_FEATURE_NAMES]
    profile = resolve_transductive_profile("h12-v1-c1", feature_names)

    assert profile.name == "h12-v1-c1"
    with pytest.raises(ValueError, match="does not match"):
        resolve_transductive_profile("h12-v1", feature_names)


def test_new_feature_schema_requires_matching_profile_and_digest() -> None:
    feature_names = ["text_score_category_rank", *H12_V1_FEATURE_NAMES, *C2_FEATURE_NAMES]
    schema = {
        "feature_names": feature_names,
        "transductive_profile": "h12-v1-c2",
        "feature_schema_sha256": feature_schema_digest(feature_names),
    }

    profile = validate_fusion_schema(schema)

    assert profile.name == "h12-v1-c2"
    with pytest.raises(ValueError, match="digest"):
        validate_fusion_schema({**schema, "feature_schema_sha256": "wrong"})


def test_legacy_schema_is_h12_v1_but_c1_c2_cannot_sneak_in_without_a_profile() -> None:
    assert validate_fusion_schema({"feature_names": list(H12_V1_FEATURE_NAMES)}).name == "h12-v1"

    with pytest.raises(ValueError, match="explicit transductive profile"):
        validate_fusion_schema({"feature_names": [*H12_V1_FEATURE_NAMES, *C1_FEATURE_NAMES]})


def test_manifest_must_match_the_fusion_schema_lock() -> None:
    feature_names = [*H12_V1_FEATURE_NAMES, *C1_FEATURE_NAMES, *C2_FEATURE_NAMES]
    schema = {
        "feature_names": feature_names,
        "transductive_profile": "h12-v1-c1-c2",
        "feature_schema_sha256": feature_schema_digest(feature_names),
    }
    manifest = {
        "format_version": 1,
        "transductive_profile": "h12-v1-c1-c2",
        "feature_schema_sha256": feature_schema_digest(feature_names),
    }

    validate_fusion_manifest(manifest, schema)

    with pytest.raises(ValueError, match="profile"):
        validate_fusion_manifest({**manifest, "transductive_profile": "h12-v1-c1"}, schema)

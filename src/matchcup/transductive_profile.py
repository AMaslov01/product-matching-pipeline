"""Immutable feature contracts for item-only transductive statistics.

The profile is deliberately narrower than the complete CatBoost matrix: it
locks only the catalogue-derived feature family, while the schema digest locks
the entire ordered matrix used by fusion.  That separation lets inference know
which corpus counters it must build without letting a newly added statistic
silently enter an old fusion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

H12_V1_FEATURE_NAMES = (
    "transductive_rarest_shared_token_idf",
    "transductive_mean_shared_token_idf",
    "transductive_max_unshared_token_idf",
    "transductive_brand_frequency_percentile_min",
    "transductive_brand_frequency_percentile_max",
    "transductive_identifier_df",
)
C1_FEATURE_NAMES = (
    "transductive_category_rarest_shared_token_idf",
    "transductive_category_mean_shared_token_idf",
    "transductive_category_max_unshared_token_idf",
)
C2_FEATURE_NAMES = (
    "aligned_attribute_common_key_idf_sum",
    "aligned_attribute_common_key_idf_max",
    "aligned_attribute_key_idf_weighted_jaccard",
)


@dataclass(frozen=True)
class TransductiveProfile:
    """One immutable catalogue-statistics recipe.

    ``feature_names`` are exactly the pair-level outputs that the profile owns;
    unrelated handcrafted features and category indicators remain outside this
    interface.  ``statistics`` tells the catalogue pass which counters it may
    build, so a legacy profile never pays for C1/C2 maps.
    """

    name: str
    feature_names: tuple[str, ...]
    statistics: frozenset[str]

    @property
    def includes_category_token_idf(self) -> bool:
        return "category_name_token" in self.statistics

    @property
    def includes_attribute_key_idf(self) -> bool:
        return "attribute_key" in self.statistics


H12_V1 = TransductiveProfile(
    name="h12-v1",
    feature_names=H12_V1_FEATURE_NAMES,
    statistics=frozenset({"name_token", "brand", "identifier"}),
)
H12_V1_C1 = TransductiveProfile(
    name="h12-v1-c1",
    feature_names=(*H12_V1_FEATURE_NAMES, *C1_FEATURE_NAMES),
    statistics=frozenset({"name_token", "brand", "identifier", "category_name_token"}),
)
H12_V1_C2 = TransductiveProfile(
    name="h12-v1-c2",
    feature_names=(*H12_V1_FEATURE_NAMES, *C2_FEATURE_NAMES),
    statistics=frozenset({"name_token", "brand", "identifier", "attribute_key"}),
)
H12_V1_C1_C2 = TransductiveProfile(
    name="h12-v1-c1-c2",
    feature_names=(*H12_V1_FEATURE_NAMES, *C1_FEATURE_NAMES, *C2_FEATURE_NAMES),
    statistics=frozenset(
        {"name_token", "brand", "identifier", "category_name_token", "attribute_key"}
    ),
)

_PROFILES = {
    profile.name: profile
    for profile in (H12_V1, H12_V1_C1, H12_V1_C2, H12_V1_C1_C2)
}
_KNOWN_FEATURE_NAMES = frozenset(
    (*H12_V1_FEATURE_NAMES, *C1_FEATURE_NAMES, *C2_FEATURE_NAMES)
)
_NEW_FEATURE_NAMES = frozenset((*C1_FEATURE_NAMES, *C2_FEATURE_NAMES))


def feature_schema_digest(feature_names: Iterable[str]) -> str:
    """Return the order-independent lock for an exact numeric feature set."""
    names = [str(name) for name in feature_names]
    if len(names) != len(set(names)):
        raise ValueError("Feature schema contains duplicate feature names")
    payload = json.dumps(sorted(names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def available_transductive_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_transductive_profile(profile_name: str) -> TransductiveProfile:
    """Return one named immutable profile without inspecting a feature matrix."""
    return _profile_or_error(profile_name)


def _profile_or_error(name: str) -> TransductiveProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported transductive profile {name!r}; "
            f"expected one of {sorted(_PROFILES)}"
        ) from exc


def resolve_transductive_profile(
    profile_name: str | None,
    feature_names: Iterable[str],
) -> TransductiveProfile:
    """Resolve a profile and reject an incompatible feature family.

    Schemas written before profiles existed are read as legacy ``h12-v1``.  The
    compatibility branch is intentionally one way: C1/C2 names in an unprofiled
    schema are an error because silently treating them as old H12 would make
    serving data depend on source-code version.
    """
    names = {str(name) for name in feature_names}
    if profile_name is None:
        if names & _NEW_FEATURE_NAMES:
            raise ValueError(
                "C1/C2 features require an explicit transductive profile; "
                "legacy schemas are h12-v1 only"
            )
        return H12_V1
    profile = _profile_or_error(str(profile_name))
    observed = names & _KNOWN_FEATURE_NAMES
    expected = set(profile.feature_names)
    if observed != expected:
        raise ValueError(
            f"Transductive profile {profile.name!r} does not match feature names: "
            f"expected {sorted(expected)}, got {sorted(observed)}"
        )
    return profile


def validate_fusion_schema(schema: Mapping[str, Any]) -> TransductiveProfile:
    """Validate profile metadata embedded in a fusion feature schema."""
    feature_names = schema.get("feature_names")
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ValueError("Fusion feature schema must contain a string feature_names list")
    profile_name = schema.get("transductive_profile")
    if profile_name is not None and not isinstance(profile_name, str):
        raise ValueError("Fusion transductive_profile must be a string")
    profile = resolve_transductive_profile(profile_name, feature_names)
    if profile_name is not None:
        actual = feature_schema_digest(feature_names)
        expected = schema.get("feature_schema_sha256")
        if not isinstance(expected, str) or expected != actual:
            raise ValueError("Fusion feature schema digest does not match feature_names")
    return profile


def fusion_manifest(
    profile: TransductiveProfile, feature_names: Iterable[str]
) -> dict[str, str | int]:
    """Create the small immutable lock persisted beside a new fusion."""
    return {
        "format_version": 1,
        "transductive_profile": profile.name,
        "feature_schema_sha256": feature_schema_digest(feature_names),
    }


def validate_fusion_manifest(
    manifest: Mapping[str, Any], schema: Mapping[str, Any]
) -> TransductiveProfile:
    """Fail closed unless the persisted training lock matches the fusion schema."""
    profile = validate_fusion_schema(schema)
    profile_name = schema.get("transductive_profile")
    if profile_name is None:
        # Pre-profile fusions have no manifest contract and remain readable as
        # legacy h12-v1.  Packaging rejects C1/C2 long before reaching here.
        return profile
    if manifest.get("format_version") != 1:
        raise ValueError("Fusion manifest has an unsupported format version")
    if manifest.get("transductive_profile") != profile.name:
        raise ValueError("Fusion manifest profile does not match feature schema")
    if manifest.get("feature_schema_sha256") != schema.get("feature_schema_sha256"):
        raise ValueError("Fusion manifest digest does not match feature schema")
    return profile

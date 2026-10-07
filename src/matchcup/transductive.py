"""Scale-invariant, label-free product-catalogue statistics for pair fusion."""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.normalize import tokens
from matchcup.parser import canonicalize_record
from matchcup.transductive_profile import (
    H12_V1,
    TransductiveProfile,
    get_transductive_profile,
)


def _json_list(value: Any) -> set[str]:
    try:
        return {str(item) for item in json.loads(value or "[]")}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _normalized_idf(total_items: int, document_frequency: int) -> float:
    if total_items <= 1 or document_frequency <= 0:
        return 0.0
    return math.log(total_items / document_frequency) / math.log(total_items)


def _normalized_df(total_items: int, document_frequency: int) -> float:
    if total_items <= 0 or document_frequency <= 0:
        return 0.0
    return math.log1p(document_frequency) / math.log1p(total_items)


def _remaining_attribute_keys(value: Any) -> set[str]:
    """Read exactly the key domain used by aligned-attribute pair features."""
    result: set[str] = set()
    for part in str(value or "").split("; "):
        key, separator, attribute_value = part.partition("=")
        if separator and key and attribute_value:
            result.add(key)
    return result


def _resolve_profile(profile: str | TransductiveProfile) -> TransductiveProfile:
    if isinstance(profile, TransductiveProfile):
        return profile
    return get_transductive_profile(profile)


@dataclass(frozen=True)
class TransductiveStatistics:
    """Item-only corpus statistics; no labels, matches, or pair graph are used."""

    total_items: int
    name_token_document_frequency: dict[str, int]
    brand_document_frequency: dict[str, int]
    identifier_document_frequency: dict[str, int]
    sorted_brand_frequencies: tuple[int, ...]
    profile: TransductiveProfile = H12_V1
    category_item_count: dict[str, int] = field(default_factory=dict)
    category_name_token_document_frequency: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    attribute_key_document_frequency: dict[str, int] = field(default_factory=dict)

    def annotate(self, item: dict[str, Any]) -> dict[str, Any]:
        """Attach item-local values so pair preparation remains process-safe."""
        result = dict(item)
        name_tokens = tokens(str(item.get("name_norm") or ""))
        identifiers = _json_list(item.get("identifier_tokens"))
        brand = str(item.get("brand_norm") or "")
        result["_transductive_name_idf"] = {
            token: _normalized_idf(
                self.total_items, self.name_token_document_frequency.get(token, 0)
            )
            for token in name_tokens
        }
        result["_transductive_identifier_df"] = {
            identifier: _normalized_df(
                self.total_items, self.identifier_document_frequency.get(identifier, 0)
            )
            for identifier in identifiers
        }
        brand_frequency = self.brand_document_frequency.get(brand, 0) if brand else 0
        result["_transductive_brand_percentile"] = (
            bisect_right(self.sorted_brand_frequencies, brand_frequency)
            / len(self.sorted_brand_frequencies)
            if brand_frequency and self.sorted_brand_frequencies
            else 0.0
        )
        if self.profile.includes_category_token_idf:
            category = str(item.get("category") or "")
            total_in_category = self.category_item_count.get(category, 0)
            result["_transductive_category_name_idf"] = (
                {
                    token: _normalized_idf(
                        total_in_category,
                        self.category_name_token_document_frequency.get((category, token), 0),
                    )
                    for token in name_tokens
                }
                if category
                else {}
            )
        if self.profile.includes_attribute_key_idf:
            result["_transductive_attribute_key_idf"] = {
                key: _normalized_idf(
                    self.total_items, self.attribute_key_document_frequency.get(key, 0)
                )
                for key in _remaining_attribute_keys(item.get("remaining_attributes"))
            }
        return result


@dataclass(frozen=True)
class TransductiveCounts:
    """Document frequencies of one chunk of canonical items.

    Kept separate from :class:`TransductiveStatistics` because a chunk is only
    ever an addend: it carries raw counts, not the corpus-wide derived values.
    """

    total_items: int
    name_token_document_frequency: Counter[str]
    brand_document_frequency: Counter[str]
    identifier_document_frequency: Counter[str]
    profile: TransductiveProfile = H12_V1
    category_item_count: Counter[str] = field(default_factory=Counter)
    category_name_token_document_frequency: Counter[tuple[str, str]] = field(
        default_factory=Counter
    )
    attribute_key_document_frequency: Counter[str] = field(default_factory=Counter)


def count_canonical_records(
    records: Iterable[dict[str, Any]], *, profile: str | TransductiveProfile = H12_V1
) -> TransductiveCounts:
    """Count document frequencies for already canonical items.

    ``tokens`` and ``_json_list`` both return sets, so a record contributes at
    most one count per key.  That is what makes counting chunk-order independent
    and lets a parallel canonicalization pass produce the same totals as a single
    sequential sweep of the same file.
    """
    selected_profile = _resolve_profile(profile)
    token_df: Counter[str] = Counter()
    brand_df: Counter[str] = Counter()
    identifier_df: Counter[str] = Counter()
    category_item_count: Counter[str] = Counter()
    category_token_df: Counter[tuple[str, str]] = Counter()
    attribute_key_df: Counter[str] = Counter()
    total_items = 0
    for item in records:
        total_items += 1
        name_tokens = tokens(str(item.get("name_norm") or ""))
        token_df.update(name_tokens)
        brand = str(item.get("brand_norm") or "")
        if brand:
            brand_df[brand] += 1
        identifier_df.update(_json_list(item.get("identifier_tokens")))
        if selected_profile.includes_category_token_idf:
            category = str(item.get("category") or "")
            if category:
                category_item_count[category] += 1
                category_token_df.update((category, token) for token in name_tokens)
        if selected_profile.includes_attribute_key_idf:
            attribute_key_df.update(_remaining_attribute_keys(item.get("remaining_attributes")))
    return TransductiveCounts(
        total_items=total_items,
        name_token_document_frequency=token_df,
        brand_document_frequency=brand_df,
        identifier_document_frequency=identifier_df,
        profile=selected_profile,
        category_item_count=category_item_count,
        category_name_token_document_frequency=category_token_df,
        attribute_key_document_frequency=attribute_key_df,
    )


class TransductiveStatisticsAccumulator:
    """Merge per-chunk counts into one corpus-wide :class:`TransductiveStatistics`.

    Merging is integer addition, so the result does not depend on how the file
    was split into chunks nor on which process counted them.
    """

    def __init__(self, *, profile: str | TransductiveProfile = H12_V1) -> None:
        self.profile = _resolve_profile(profile)
        self.total_items = 0
        self._token_df: Counter[str] = Counter()
        self._brand_df: Counter[str] = Counter()
        self._identifier_df: Counter[str] = Counter()
        self._category_item_count: Counter[str] = Counter()
        self._category_token_df: Counter[tuple[str, str]] = Counter()
        self._attribute_key_df: Counter[str] = Counter()

    def add(self, counts: TransductiveCounts) -> None:
        if counts.profile != self.profile:
            raise ValueError("Cannot merge counts from a different transductive profile")
        self.total_items += counts.total_items
        self._token_df.update(counts.name_token_document_frequency)
        self._brand_df.update(counts.brand_document_frequency)
        self._identifier_df.update(counts.identifier_document_frequency)
        self._category_item_count.update(counts.category_item_count)
        self._category_token_df.update(counts.category_name_token_document_frequency)
        self._attribute_key_df.update(counts.attribute_key_document_frequency)

    def build(self, source: str | Path) -> TransductiveStatistics:
        if self.total_items == 0:
            raise ValueError(f"No item rows found in {source}")
        return TransductiveStatistics(
            total_items=self.total_items,
            name_token_document_frequency=dict(self._token_df),
            brand_document_frequency=dict(self._brand_df),
            identifier_document_frequency=dict(self._identifier_df),
            sorted_brand_frequencies=tuple(sorted(self._brand_df.values())),
            profile=self.profile,
            category_item_count=dict(self._category_item_count),
            category_name_token_document_frequency=dict(self._category_token_df),
            attribute_key_document_frequency=dict(self._attribute_key_df),
        )


class DuckDBTransductiveStatistics:
    """Disk-backed equivalent used only when a full Counter exceeds the RSS budget.

    The temporary DuckDB database contains only label-free document frequencies.
    Small bounded lookup caches keep pair preparation practical without recreating
    the full three Python dictionaries that caused the original RSS breach.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        total_items: int,
        brand_count: int,
        profile: str | TransductiveProfile = H12_V1,
    ):
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - covered by H12 VM dependency install
            raise RuntimeError("Disk-backed transductive statistics require duckdb") from exc
        self.database_path = Path(database_path)
        self.total_items = total_items
        self.brand_count = brand_count
        self.profile = _resolve_profile(profile)
        self._connection = duckdb.connect(str(self.database_path), read_only=True)
        self._frequency_cache: dict[tuple[str, str, str], int] = {}
        self._brand_cache: dict[str, float] = {}
        self._category_token_cache: dict[tuple[str, str], int] = {}
        self._category_count_cache: dict[str, int] = {}

    def _document_frequency(self, table: str, column: str, value: str) -> int:
        key = (table, column, value)
        if key in self._frequency_cache:
            return self._frequency_cache[key]
        row = self._connection.execute(
            f"SELECT document_frequency FROM {table} WHERE {column} = ?", [value]
        ).fetchone()
        frequency = int(row[0]) if row is not None else 0
        if len(self._frequency_cache) >= 500_000:
            self._frequency_cache.pop(next(iter(self._frequency_cache)))
        self._frequency_cache[key] = frequency
        return frequency

    def _brand_percentile(self, brand: str) -> float:
        if brand in self._brand_cache:
            return self._brand_cache[brand]
        row = self._connection.execute(
            "SELECT percentile FROM brand_percentile WHERE brand = ?", [brand]
        ).fetchone()
        percentile = float(row[0]) if row is not None else 0.0
        if len(self._brand_cache) >= 250_000:
            self._brand_cache.pop(next(iter(self._brand_cache)))
        self._brand_cache[brand] = percentile
        return percentile

    def _category_item_total(self, category: str) -> int:
        if category in self._category_count_cache:
            return self._category_count_cache[category]
        row = self._connection.execute(
            "SELECT item_count FROM category_item_count WHERE category = ?", [category]
        ).fetchone()
        total = int(row[0]) if row is not None else 0
        if len(self._category_count_cache) >= 250_000:
            self._category_count_cache.pop(next(iter(self._category_count_cache)))
        self._category_count_cache[category] = total
        return total

    def _category_token_frequency(self, category: str, token: str) -> int:
        key = (category, token)
        if key in self._category_token_cache:
            return self._category_token_cache[key]
        row = self._connection.execute(
            "SELECT document_frequency FROM category_name_token_df "
            "WHERE category = ? AND token = ?",
            [category, token],
        ).fetchone()
        frequency = int(row[0]) if row is not None else 0
        if len(self._category_token_cache) >= 500_000:
            self._category_token_cache.pop(next(iter(self._category_token_cache)))
        self._category_token_cache[key] = frequency
        return frequency

    def annotate(self, item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        name_tokens = tokens(str(item.get("name_norm") or ""))
        identifiers = _json_list(item.get("identifier_tokens"))
        brand = str(item.get("brand_norm") or "")
        result["_transductive_name_idf"] = {
            token: _normalized_idf(
                self.total_items,
                self._document_frequency("name_token_df", "token", token),
            )
            for token in name_tokens
        }
        result["_transductive_identifier_df"] = {
            identifier: _normalized_df(
                self.total_items,
                self._document_frequency("identifier_df", "identifier", identifier),
            )
            for identifier in identifiers
        }
        result["_transductive_brand_percentile"] = (
            self._brand_percentile(brand) if brand else 0.0
        )
        if self.profile.includes_category_token_idf:
            category = str(item.get("category") or "")
            total_in_category = self._category_item_total(category) if category else 0
            result["_transductive_category_name_idf"] = (
                {
                    token: _normalized_idf(
                        total_in_category, self._category_token_frequency(category, token)
                    )
                    for token in name_tokens
                }
                if category
                else {}
            )
        if self.profile.includes_attribute_key_idf:
            result["_transductive_attribute_key_idf"] = {
                key: _normalized_idf(
                    self.total_items,
                    self._document_frequency("attribute_key_df", "attribute_key", key),
                )
                for key in _remaining_attribute_keys(item.get("remaining_attributes"))
            }
        return result

    def close(self) -> None:
        self._connection.close()
        self._frequency_cache.clear()
        self._brand_cache.clear()
        self._category_token_cache.clear()
        self._category_count_cache.clear()


def build_disk_backed_transductive_statistics(
    items_path: str | Path,
    database_path: str | Path,
    *,
    batch_size: int = 65_536,
    profile: str | TransductiveProfile = H12_V1,
) -> DuckDBTransductiveStatistics:
    """Materialize label-free catalogue frequencies in DuckDB, not Python Counters.

    ``items_path`` must already be canonical.  Each input row contributes at most
    one count per token or identifier, precisely matching ``build_transductive_statistics``.
    Raw value tables are dropped after aggregation, so the retained database is
    proportional to vocabulary size rather than the catalogue row count.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Disk-backed transductive statistics require duckdb") from exc
    selected_profile = _resolve_profile(profile)
    items_path = Path(items_path)
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing transductive database: {database_path}"
        )
    parquet = pq.ParquetFile(items_path)
    required = {"name_norm", "brand_norm", "identifier_tokens"}
    if selected_profile.includes_category_token_idf:
        required.add("category")
    if selected_profile.includes_attribute_key_idf:
        required.add("remaining_attributes")
    if missing := required - set(parquet.schema.names):
        raise ValueError(
            "Disk-backed transductive statistics require canonical items; "
            f"missing {sorted(missing)}"
        )
    connection = duckdb.connect(str(database_path))
    total_items = 0
    try:
        connection.execute("CREATE TABLE name_token_values (token VARCHAR)")
        connection.execute("CREATE TABLE brand_values (brand VARCHAR)")
        connection.execute("CREATE TABLE identifier_values (identifier VARCHAR)")
        if selected_profile.includes_category_token_idf:
            connection.execute("CREATE TABLE category_values (category VARCHAR)")
            connection.execute(
                "CREATE TABLE category_name_token_values (category VARCHAR, token VARCHAR)"
            )
        if selected_profile.includes_attribute_key_idf:
            connection.execute("CREATE TABLE attribute_key_values (attribute_key VARCHAR)")
        columns = ["name_norm", "brand_norm", "identifier_tokens"]
        if selected_profile.includes_category_token_idf:
            columns.append("category")
        if selected_profile.includes_attribute_key_idf:
            columns.append("remaining_attributes")
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            records = batch.to_pylist()
            total_items += len(records)
            token_values = [
                token for item in records for token in tokens(str(item.get("name_norm") or ""))
            ]
            brand_values = [
                brand
                for item in records
                if (brand := str(item.get("brand_norm") or ""))
            ]
            identifier_values = [
                identifier
                for item in records
                for identifier in _json_list(item.get("identifier_tokens"))
            ]
            category_values = (
                [
                    category
                    for item in records
                    if (category := str(item.get("category") or ""))
                ]
                if selected_profile.includes_category_token_idf
                else []
            )
            category_token_values = (
                [
                    (category, token)
                    for item in records
                    if (category := str(item.get("category") or ""))
                    for token in tokens(str(item.get("name_norm") or ""))
                ]
                if selected_profile.includes_category_token_idf
                else []
            )
            attribute_key_values = (
                [
                    key
                    for item in records
                    for key in _remaining_attribute_keys(item.get("remaining_attributes"))
                ]
                if selected_profile.includes_attribute_key_idf
                else []
            )
            for table, column, values in (
                ("name_token_values", "token", token_values),
                ("brand_values", "brand", brand_values),
                ("identifier_values", "identifier", identifier_values),
            ):
                if values:
                    registered = f"batch_{column}"
                    connection.register(registered, pa.table({column: values}))
                    connection.execute(f"INSERT INTO {table} SELECT {column} FROM {registered}")
                    connection.unregister(registered)
            if selected_profile.includes_category_token_idf:
                if category_values:
                    connection.register("batch_category", pa.table({"category": category_values}))
                    connection.execute(
                        "INSERT INTO category_values SELECT category FROM batch_category"
                    )
                    connection.unregister("batch_category")
                if category_token_values:
                    connection.register(
                        "batch_category_token",
                        pa.table(
                            {
                                "category": [value[0] for value in category_token_values],
                                "token": [value[1] for value in category_token_values],
                            }
                        ),
                    )
                    connection.execute(
                        "INSERT INTO category_name_token_values "
                        "SELECT category, token FROM batch_category_token"
                    )
                    connection.unregister("batch_category_token")
            if selected_profile.includes_attribute_key_idf and attribute_key_values:
                connection.register(
                    "batch_attribute_key", pa.table({"attribute_key": attribute_key_values})
                )
                connection.execute(
                    "INSERT INTO attribute_key_values SELECT attribute_key FROM batch_attribute_key"
                )
                connection.unregister("batch_attribute_key")
        if total_items == 0:
            raise ValueError(f"No item rows found in {items_path}")
        connection.execute(
            "CREATE TABLE name_token_df AS "
            "SELECT token, COUNT(*)::BIGINT AS document_frequency "
            "FROM name_token_values GROUP BY token"
        )
        if selected_profile.includes_category_token_idf:
            connection.execute(
                "CREATE TABLE category_item_count AS "
                "SELECT category, COUNT(*)::BIGINT AS item_count "
                "FROM category_values GROUP BY category"
            )
            connection.execute(
                "CREATE TABLE category_name_token_df AS "
                "SELECT category, token, COUNT(*)::BIGINT AS document_frequency "
                "FROM category_name_token_values GROUP BY category, token"
            )
        if selected_profile.includes_attribute_key_idf:
            connection.execute(
                "CREATE TABLE attribute_key_df AS "
                "SELECT attribute_key, COUNT(*)::BIGINT AS document_frequency "
                "FROM attribute_key_values GROUP BY attribute_key"
            )
        connection.execute(
            "CREATE TABLE identifier_df AS "
            "SELECT identifier, COUNT(*)::BIGINT AS document_frequency "
            "FROM identifier_values GROUP BY identifier"
        )
        connection.execute(
            "CREATE TABLE brand_df AS "
            "SELECT brand, COUNT(*)::BIGINT AS document_frequency "
            "FROM brand_values GROUP BY brand"
        )
        brand_count = int(connection.execute("SELECT COUNT(*) FROM brand_df").fetchone()[0])
        connection.execute(
            "CREATE TABLE brand_percentile AS "
            "SELECT brand, "
            "CAST(COUNT(*) OVER (ORDER BY document_frequency "
            "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS DOUBLE) / ? AS percentile "
            "FROM brand_df",
            [max(brand_count, 1)],
        )
        scratch_tables = ["name_token_values", "brand_values", "identifier_values"]
        if selected_profile.includes_category_token_idf:
            scratch_tables.extend(["category_values", "category_name_token_values"])
        if selected_profile.includes_attribute_key_idf:
            scratch_tables.append("attribute_key_values")
        for table in scratch_tables:
            connection.execute(f"DROP TABLE {table}")
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    return DuckDBTransductiveStatistics(
        database_path,
        total_items=total_items,
        brand_count=brand_count,
        profile=selected_profile,
    )


def build_transductive_statistics(
    items_path: str | Path,
    *,
    batch_size: int = 4096,
    profile: str | TransductiveProfile = H12_V1,
) -> TransductiveStatistics:
    """Stream a raw or canonical items parquet and derive document frequencies."""
    items_path = Path(items_path)
    parquet = pq.ParquetFile(items_path)
    selected_profile = _resolve_profile(profile)
    canonical = {"name_norm", "brand_norm", "identifier_tokens"} <= set(parquet.schema.names)
    columns = (
        [
            "id",
            "name_norm",
            "brand_norm",
            "identifier_tokens",
            *( ["category"] if selected_profile.includes_category_token_idf else [] ),
            *( ["remaining_attributes"] if selected_profile.includes_attribute_key_idf else [] ),
        ]
        if canonical
        else ["id", "name", "attributes", "category"]
    )
    accumulator = TransductiveStatisticsAccumulator(profile=selected_profile)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        records = batch.to_pylist()
        if not canonical:
            records = [canonicalize_record(record) for record in records]
        accumulator.add(count_canonical_records(records, profile=selected_profile))
    return accumulator.build(items_path)

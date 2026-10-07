from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.normalize import (
    compact_remaining_attributes,
    extract_measurements,
    identifier_tokens,
    normalize_color,
    normalize_identifier,
    normalize_size,
    normalize_text,
    numeric_tokens,
    safe_attributes,
    select_fields,
    transliterate_ru,
)

CANONICAL_FIELDS = ("type", "brand", "model", "article", "color", "size", "quantity")


def canonicalize_record(record: dict[str, Any], remaining_chars: int = 1800) -> dict[str, Any]:
    attributes, parse_error = safe_attributes(record.get("attributes"))
    selected = select_fields(attributes)
    name_raw = str(record.get("name") or "")
    category = str(record.get("category") or "")
    result: dict[str, Any] = {
        "id": int(record["id"]),
        "category": category,
        "name_raw": name_raw,
        "name_norm": normalize_text(name_raw),
        "name_translit": transliterate_ru(name_raw),
        "attribute_count": len(attributes),
        "parse_error": parse_error,
    }
    selected_keys: list[str] = []
    selected_values: list[str] = []
    for field in CANONICAL_FIELDS:
        candidate = selected[field]
        raw = candidate.raw if candidate else ""
        key = candidate.key if candidate else ""
        if candidate:
            selected_keys.append(key)
            selected_values.append(raw)
        result[f"{field}_raw"] = raw
        if field == "article":
            result[f"{field}_norm"] = normalize_identifier(raw)
        elif field == "size":
            result[f"{field}_norm"] = normalize_size(raw, key)
        elif field == "color":
            result[f"{field}_norm"] = normalize_color(raw)
        else:
            result[f"{field}_norm"] = normalize_text(raw)
        if field == "brand":
            result["brand_translit"] = transliterate_ru(raw)
    searchable = " ".join([name_raw, *selected_values])
    result["numeric_tokens"] = json.dumps(numeric_tokens(searchable), ensure_ascii=False)
    result["identifier_tokens"] = json.dumps(identifier_tokens(searchable), ensure_ascii=False)
    measurements = extract_measurements(name_raw, *selected_values, *attributes.values())
    result["measurements"] = json.dumps(measurements, ensure_ascii=False, sort_keys=True)
    result["remaining_attributes"] = compact_remaining_attributes(
        attributes, selected_keys, max_chars=remaining_chars
    )
    return result


def _canonicalize_chunk(payload: tuple[list[dict[str, Any]], int]) -> list[dict[str, Any]]:
    records, remaining_chars = payload
    return [canonicalize_record(record, remaining_chars) for record in records]


def iter_item_batches(
    path: str | Path, batch_size: int, accepted_ids: set[int] | None = None
) -> Iterator[list[dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["id", "name", "attributes", "category"]
    ):
        if accepted_ids is not None:
            keep = pa.array([int(value) in accepted_ids for value in batch.column(0).to_numpy()])
            batch = batch.filter(keep)
        if batch.num_rows:
            yield batch.to_pylist()


def canonicalize_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 4096,
    workers: int = 1,
    remaining_chars: int = 1800,
    compression: str = "zstd",
    accepted_ids: set[int] | None = None,
) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    errors = 0

    def write(records: list[dict[str, Any]]) -> None:
        nonlocal writer, rows, errors
        table = pa.Table.from_pylist(records)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
        writer.write_table(table, row_group_size=batch_size)
        rows += len(records)
        errors += sum(int(record["parse_error"]) for record in records)

    try:
        batches = iter_item_batches(input_path, batch_size, accepted_ids)
        if workers <= 1:
            for records in batches:
                write(_canonicalize_chunk((records, remaining_chars)))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending: set[Future[list[dict[str, Any]]]] = set()
                max_pending = max(workers * 2, 1)
                for batch in batches:
                    pending.add(pool.submit(_canonicalize_chunk, (batch, remaining_chars)))
                    if len(pending) >= max_pending:
                        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            write(future.result())
                for future in pending:
                    write(future.result())
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise ValueError(f"No item rows found in {input_path}")
    return {"rows": rows, "parse_errors": errors}


def load_canonical_items(path: str | Path) -> dict[int, dict[str, Any]]:
    table = pq.read_table(path)
    return {int(record["id"]): record for record in table.to_pylist()}

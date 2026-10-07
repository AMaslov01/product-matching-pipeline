from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.features import pair_category, pair_features
from matchcup.parser import load_canonical_items
from matchcup.serialize import serialize_pair
from matchcup.transductive_profile import TransductiveProfile, get_transductive_profile


def _require_statistics_profile(
    statistics: Any, profile: TransductiveProfile
) -> None:
    """Reject statistics whose catalogue contract differs from this pair run.

    ``prepare_pairs`` is also used as a library seam with a pre-built in-memory
    or DuckDB statistics object.  Without this check, a caller could annotate
    C1/C2 values while recording the legacy profile in the experiment contract.
    The later fusion check would reject that data, but only after a costly pair
    preparation; reject it at the first boundary instead.
    """
    actual = getattr(statistics, "profile", None)
    if actual != profile:
        actual_name = getattr(actual, "name", None)
        raise ValueError(
            "Transductive statistics profile does not match pair preparation: "
            f"expected {profile.name!r}, got {actual_name!r}"
        )


def prepare_pairs(
    items_path: str | Path,
    matches_path: str | Path,
    output_path: str | Path,
    *,
    chunk_size: int = 8192,
    include_text: bool = True,
    include_target: bool = True,
    include_features: bool = True,
    ngram_size: int = 3,
    measurement_tolerance: float = 0.015,
    compression: str = "zstd",
    temp_dir: str | Path | None = None,
    transductive_items_path: str | Path | None = None,
    transductive_statistics: Any | None = None,
    transductive_profile: str = "h12-v1",
) -> dict[str, int]:
    selected_profile = get_transductive_profile(transductive_profile)
    if transductive_items_path is not None and transductive_statistics is not None:
        raise ValueError(
            "Pass either transductive_items_path or transductive_statistics, not both"
        )
    has_transductive_statistics = (
        transductive_items_path is not None or transductive_statistics is not None
    )
    if transductive_statistics is not None:
        _require_statistics_profile(transductive_statistics, selected_profile)
    if has_transductive_statistics and not include_features:
        raise ValueError("transductive features require include_features=True")
    item_count = pq.ParquetFile(items_path).metadata.num_rows
    if item_count > 1_000_000:
        return _prepare_pairs_duckdb(
            items_path,
            matches_path,
            output_path,
            chunk_size=chunk_size,
            include_text=include_text,
            include_target=include_target,
            include_features=include_features,
            ngram_size=ngram_size,
            measurement_tolerance=measurement_tolerance,
            compression=compression,
            temp_dir=temp_dir,
            transductive_items_path=transductive_items_path,
            transductive_statistics=transductive_statistics,
            transductive_profile=selected_profile.name,
        )
    items = load_canonical_items(items_path)
    statistics = transductive_statistics
    if transductive_items_path is not None:
        from matchcup.transductive import build_transductive_statistics

        statistics = build_transductive_statistics(
            transductive_items_path, profile=selected_profile
        )
    if statistics is not None:
        items = {item_id: statistics.annotate(item) for item_id, item in items.items()}
    matches_file = pq.ParquetFile(matches_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    missing = 0
    schema_names = set(matches_file.schema.names)
    columns = ["id1", "id2"] + (["target"] if include_target else [])
    passthrough = [name for name in ("fold", "row_id") if name in schema_names]
    columns.extend(passthrough)
    try:
        for batch in matches_file.iter_batches(batch_size=chunk_size, columns=columns):
            output: list[dict[str, Any]] = []
            for match in batch.to_pylist():
                item_a = items.get(int(match["id1"]))
                item_b = items.get(int(match["id2"]))
                if item_a is None or item_b is None:
                    missing += 1
                    continue
                record: dict[str, Any] = {"id1": match["id1"], "id2": match["id2"]}
                if include_target:
                    record["target"] = float(match["target"])
                for name in passthrough:
                    record[name] = int(match[name])
                if include_text:
                    text_a, text_b = serialize_pair(item_a, item_b)
                    record["text_a"] = text_a
                    record["text_b"] = text_b
                if include_features:
                    record.update(
                        pair_features(
                            item_a,
                            item_b,
                            ngram_size=ngram_size,
                            measurement_tolerance=measurement_tolerance,
                        )
                    )
                else:
                    record["category"] = pair_category(item_a, item_b)
                output.append(record)
            if output:
                table = pa.Table.from_pylist(output)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                writer.write_table(table, row_group_size=chunk_size)
                rows += len(output)
    finally:
        if writer is not None:
            writer.close()
    if missing:
        raise ValueError(f"{missing} pairs reference items missing from {items_path}")
    return {"rows": rows, "missing_items": missing}


def _prepare_pairs_duckdb(
    items_path: str | Path,
    matches_path: str | Path,
    output_path: str | Path,
    *,
    chunk_size: int,
    include_text: bool,
    include_target: bool,
    include_features: bool,
    ngram_size: int,
    measurement_tolerance: float,
    compression: str,
    temp_dir: str | Path | None,
    transductive_items_path: str | Path | None,
    transductive_statistics: Any | None,
    transductive_profile: str,
) -> dict[str, int]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Large pair preparation requires the 'local' extra (duckdb)") from exc
    items_path = Path(items_path).resolve()
    matches_path = Path(matches_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(temp_dir) if temp_dir else output_path.parent / "duckdb_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    selected_profile = get_transductive_profile(transductive_profile)
    statistics = transductive_statistics
    if statistics is not None:
        _require_statistics_profile(statistics, selected_profile)
    if transductive_items_path is not None:
        from matchcup.transductive import build_transductive_statistics

        try:
            statistics = build_transductive_statistics(
                transductive_items_path, profile=selected_profile
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not build transductive statistics from "
                f"{Path(transductive_items_path)}"
            ) from exc
    canonical_columns = pq.ParquetFile(items_path).schema.names
    match_columns = pq.ParquetFile(matches_path).schema.names
    passthrough = [name for name in ("target", "fold", "row_id") if name in match_columns]
    select = ["m.id1", "m.id2", *[f"m.{name}" for name in passthrough]]
    for prefix in ("a", "b"):
        select.extend(f'{prefix}."{name}" AS "{prefix}__{name}"' for name in canonical_columns)
    query = f"""
        SELECT {", ".join(select)}
        FROM read_parquet(?) m
        INNER JOIN read_parquet(?) a ON a.id = m.id1
        INNER JOIN read_parquet(?) b ON b.id = m.id2
    """
    connection = duckdb.connect(str(scratch / "join.duckdb"))
    connection.execute("SET memory_limit='10GB'")
    connection.execute("SET threads=8")
    scratch_sql = str(scratch).replace("'", "''")
    connection.execute(f"SET temp_directory='{scratch_sql}'")
    reader = connection.execute(
        query, [str(matches_path), str(items_path), str(items_path)]
    ).fetch_record_batch(chunk_size)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in reader:
            output: list[dict[str, Any]] = []
            for joined in batch.to_pylist():
                item_a = {name: joined[f"a__{name}"] for name in canonical_columns}
                item_b = {name: joined[f"b__{name}"] for name in canonical_columns}
                if statistics is not None:
                    item_a = statistics.annotate(item_a)
                    item_b = statistics.annotate(item_b)
                record: dict[str, Any] = {"id1": joined["id1"], "id2": joined["id2"]}
                if include_target:
                    record["target"] = float(joined["target"])
                for name in ("fold", "row_id"):
                    if name in passthrough:
                        record[name] = int(joined[name])
                if include_text:
                    record["text_a"], record["text_b"] = serialize_pair(item_a, item_b)
                if include_features:
                    record.update(
                        pair_features(
                            item_a,
                            item_b,
                            ngram_size=ngram_size,
                            measurement_tolerance=measurement_tolerance,
                        )
                    )
                else:
                    # Silver pretraining reads only the text columns and category;
                    # the 67 CatBoost features cost 109 us/pair against 5 us for
                    # serialization and are never read for that stage.
                    record["category"] = pair_category(item_a, item_b)
                output.append(record)
            table = pa.Table.from_pylist(output)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
            writer.write_table(table, row_group_size=chunk_size)
            rows += len(output)
    finally:
        if writer is not None:
            writer.close()
        connection.close()
    expected = pq.ParquetFile(matches_path).metadata.num_rows
    if rows != expected:
        raise ValueError(f"Prepared {rows} rows but expected {expected}; some items are missing")
    return {"rows": rows, "missing_items": 0}

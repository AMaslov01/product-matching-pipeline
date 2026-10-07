from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sample_silver(
    matches_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = 2_000_000,
    seed: int = 20260812,
) -> dict[str, int]:
    """Deterministically sample silver pairs by their ten target levels.

    Category balancing is performed during training after the pairs are joined with items.
    The target-level reservoir prevents the 0.0 bucket from consuming the sample budget.
    """
    table = pq.read_table(matches_path, columns=["id1", "id2", "target"])
    target = table.column("target").to_numpy()
    unique, counts = np.unique(target, return_counts=True)
    floor = min(20_000, max_rows // max(len(unique), 1))
    allocation = np.minimum(counts, floor)
    remaining = max_rows - int(allocation.sum())
    capacity = counts - allocation
    if remaining > 0 and capacity.sum() > 0:
        extra = np.floor(remaining * capacity / capacity.sum()).astype(np.int64)
        allocation += np.minimum(extra, capacity)
    while allocation.sum() < min(max_rows, len(target)):
        candidates = np.flatnonzero(allocation < counts)
        if not len(candidates):
            break
        allocation[candidates[: min(len(candidates), max_rows - int(allocation.sum()))]] += 1

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for value, count in zip(unique, allocation, strict=True):
        indices = np.flatnonzero(target == value)
        selected.append(rng.choice(indices, size=int(count), replace=False))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    sampled = table.take(pa.array(indices))
    sampled = sampled.append_column("row_id", pa.array(np.arange(len(indices), dtype=np.int64)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sampled, output_path, compression="zstd")
    return {"rows": len(indices), "target_levels": len(unique)}

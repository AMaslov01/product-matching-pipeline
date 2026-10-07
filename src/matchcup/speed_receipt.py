"""Build immutable source-row fixtures for controlled Silver speed receipts."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from matchcup.cross_encoder import CategoryUniformBatchSampler


def write_silver_microbatch_fixture(
    source: str | Path,
    destination: str | Path,
    *,
    optimizer_steps: int,
    microbatches_per_step: int = 4,
    microbatch_size: int = 16,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Persist S6-shaped source-row microbatches for exactly one receipt.

    Only the category column is scanned: the row indices emitted by the sampler
    address the same source Silver parquet later used by both timing cases.
    """
    if optimizer_steps < 1 or microbatches_per_step < 1 or microbatch_size < 1:
        raise ValueError("fixture sizes must be positive")
    source = Path(source)
    destination = Path(destination)
    categories = pq.read_table(source, columns=["category"], memory_map=True)
    sampler = CategoryUniformBatchSampler(categories, None, microbatch_size, seed)
    count = optimizer_steps * microbatches_per_step
    microbatches = list(itertools.islice(iter(sampler), count))
    if len(microbatches) != count:
        raise ValueError("Silver input is too short for the requested timing fixture")
    payload = {
        "format": "matchcup_t6_microbatch_fixture_v1",
        "source_rows": categories.num_rows,
        "optimizer_steps": optimizer_steps,
        "microbatches_per_step": microbatches_per_step,
        "microbatch_size": microbatch_size,
        "seed": seed,
        "microbatches": microbatches,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        **{key: value for key, value in payload.items() if key != "microbatches"},
        "fixture_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }

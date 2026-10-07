from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.speed_receipt import write_silver_microbatch_fixture


def test_silver_microbatch_fixture_has_a_stable_source_row_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "silver.parquet"
    destination = tmp_path / "fixture.json"
    pq.write_table(pa.table({"category": ["A", "B"] * 8}), source)

    result = write_silver_microbatch_fixture(
        source,
        destination,
        optimizer_steps=2,
        microbatches_per_step=2,
        microbatch_size=2,
        seed=7,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["format"] == "matchcup_t6_microbatch_fixture_v1"
    assert payload["source_rows"] == 16
    assert len(payload["microbatches"]) == 4
    assert all(len(batch) == 2 for batch in payload["microbatches"])
    assert result["fixture_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def mine_hard_negatives(
    gold_path: str | Path,
    oof_path: str | Path,
    output_path: str | Path,
    *,
    top_fraction: float = 0.15,
) -> dict[str, int]:
    gold = pq.read_table(gold_path, columns=["row_id", "target", "category"]).to_pandas()
    path = Path(oof_path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    scores = pd.concat([pq.read_table(file).to_pandas() for file in files], ignore_index=True)
    scores = scores[["row_index", "text_score"]].rename(columns={"row_index": "row_id"})
    frame = gold.merge(scores, on="row_id", how="left", validate="one_to_one")
    if frame.text_score.isna().any():
        raise ValueError("OOF scores do not cover gold rows")
    negatives = frame[frame.target <= 0.5].copy()
    selected = (
        negatives.sort_values(["category", "text_score"], ascending=[True, False])
        .groupby("category", group_keys=False)
        .head(max(1, int(len(negatives) * top_fraction / negatives.category.nunique())))
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(selected[["row_id"]], preserve_index=False), output_path)
    return {"rows": len(selected)}

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.size: dict[int, int] = {}

    def find(self, item: int) -> int:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1
            return item
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return left_root


def make_component_folds(
    matches_path: str | Path,
    output_path: str | Path,
    *,
    n_folds: int = 5,
    seed: int = 20260812,
    items_path: str | Path | None = None,
) -> dict[str, object]:
    table = pq.read_table(matches_path, columns=["id1", "id2", "target"])
    id1 = table.column("id1").to_numpy()
    id2 = table.column("id2").to_numpy()
    target = table.column("target").to_numpy().astype(np.float32)
    dsu = DisjointSet()
    for left, right in zip(id1, id2, strict=True):
        dsu.union(int(left), int(right))
    component_rows: dict[int, list[int]] = defaultdict(list)
    for index, left in enumerate(id1):
        component_rows[dsu.find(int(left))].append(index)

    if items_path is not None:
        item_table = pq.read_table(items_path, columns=["id", "category"])
        category_by_id = dict(
            zip(
                item_table.column("id").to_numpy().tolist(),
                item_table.column("category").to_pylist(),
                strict=True,
            )
        )
        row_categories = np.asarray([category_by_id[int(value)] for value in id1], dtype=object)
    else:
        row_categories = np.full(len(target), "__all__", dtype=object)

    groups = np.asarray([dsu.find(int(value)) for value in id1], dtype=np.int64)
    target_class = (target > 0.5).astype(np.int8).astype(str)
    strata = np.char.add(np.char.add(row_categories.astype(str), "__"), target_class)
    folds = np.full(len(target), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, validation_rows) in enumerate(
        splitter.split(np.zeros(len(target), dtype=np.int8), strata, groups)
    ):
        folds[validation_rows] = fold

    if np.any(folds < 0):
        raise RuntimeError("Some pair rows were not assigned to a fold")
    fold_rows = np.bincount(folds, minlength=n_folds)
    fold_positives = np.bincount(folds, weights=target, minlength=n_folds)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = table.append_column("fold", pa.array(folds))
    output = output.append_column("row_id", pa.array(np.arange(len(target), dtype=np.int64)))
    pq.write_table(output, output_path, compression="zstd")
    stats = {
        "rows": len(target),
        "components": len(component_rows),
        "fold_rows": fold_rows.tolist(),
        "fold_positive_rate": (fold_positives / np.maximum(fold_rows, 1)).tolist(),
    }
    return stats


def validate_component_folds(path: str | Path) -> None:
    table = pq.read_table(path, columns=["id1", "id2", "fold"])
    item_folds: dict[int, int] = {}
    counts: Counter[int] = Counter()
    for row in table.to_pylist():
        fold = int(row["fold"])
        counts[fold] += 1
        for key in ("id1", "id2"):
            item = int(row[key])
            if item in item_folds and item_folds[item] != fold:
                raise ValueError(f"Item {item} leaks between folds")
            item_folds[item] = fold
    if len(counts) < 2 or min(counts.values()) == 0:
        raise ValueError("Fold assignment is degenerate")

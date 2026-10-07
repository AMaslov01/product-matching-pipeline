"""Diagnostic-only covariate-shift proxy between Human Gold and new LLM pairs.

No pair target is read into this analysis.  A two-sample classifier can show
that the observable catalogue covariates differ, but cannot attribute any
Public AP loss to that difference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matchcup.features import pair_features
from matchcup.holdout import _require_duckdb
from matchcup.parser import CANONICAL_FIELDS, canonicalize_record


def _sample_pairs(
    human_items_path: Path,
    human_matches_path: Path,
    llm_items_path: Path,
    llm_matches_path: Path,
    silver_matches_path: Path,
    *,
    max_pairs_per_category: int,
    min_pairs_per_category: int,
    seed: int,
    temp_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    duckdb = _require_duckdb()
    temp_dir.mkdir(parents=True, exist_ok=True)
    database = temp_dir / "proxy_shift.duckdb"
    connection = duckdb.connect(str(database))
    scratch_sql = str(temp_dir.resolve()).replace("'", "''")
    connection.execute("SET memory_limit='10GB'")
    connection.execute("SET threads=8")
    connection.execute(f"SET temp_directory='{scratch_sql}'")
    try:
        connection.execute(
            """
            CREATE TEMP TABLE seen_ids AS
            SELECT DISTINCT id FROM (
                SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
                UNION ALL SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
            )
            """,
            [
                str(human_matches_path),
                str(human_matches_path),
                str(silver_matches_path),
                str(silver_matches_path),
            ],
        )
        connection.execute(
            """
            CREATE TEMP TABLE human_pairs AS
            WITH left_items AS MATERIALIZED (
                SELECT m.id1, m.id2, CAST(a.category AS VARCHAR) AS category
                FROM read_parquet(?) AS m
                INNER JOIN read_parquet(?) AS a ON a.id = m.id1
                WHERE COALESCE(a.category, '') <> ''
            )
            SELECT left_items.id1, left_items.id2, left_items.category
            FROM left_items
            INNER JOIN read_parquet(?) AS b
              ON b.id = left_items.id2 AND b.category = left_items.category
            """,
            [str(human_matches_path), str(human_items_path), str(human_items_path)],
        )
        connection.execute(
            """
            CREATE TEMP TABLE llm_pairs AS
            WITH candidates AS MATERIALIZED (
                SELECT m.id1, m.id2
                FROM read_parquet(?) AS m
                LEFT JOIN seen_ids AS left_id ON left_id.id = m.id1
                LEFT JOIN seen_ids AS right_id ON right_id.id = m.id2
                WHERE left_id.id IS NULL AND right_id.id IS NULL
            ), left_items AS MATERIALIZED (
                SELECT candidates.id1, candidates.id2,
                  CAST(a.category AS VARCHAR) AS category
                FROM candidates
                INNER JOIN read_parquet(?) AS a ON a.id = candidates.id1
                WHERE COALESCE(a.category, '') <> ''
            )
            SELECT left_items.id1, left_items.id2, left_items.category
            FROM left_items
            INNER JOIN read_parquet(?) AS b
              ON b.id = left_items.id2 AND b.category = left_items.category
            """,
            [str(llm_matches_path), str(llm_items_path), str(llm_items_path)],
        )
        capacities = connection.execute(
            """
            WITH counts AS (
                SELECT 'human' AS source, category, COUNT(*) AS rows
                FROM human_pairs GROUP BY category
                UNION ALL
                SELECT 'llm' AS source, category, COUNT(*) AS rows
                FROM llm_pairs GROUP BY category
            ), paired AS (
                SELECT category, MIN(rows) AS rows
                FROM counts
                GROUP BY category
                HAVING COUNT(*) = 2
            )
            SELECT category, LEAST(rows, ?) AS quota
            FROM paired
            WHERE rows >= ?
            ORDER BY category
            """,
            [max_pairs_per_category, min_pairs_per_category],
        ).fetchall()
        quotas = pd.DataFrame(capacities, columns=["category", "quota"])
        if quotas.empty:
            raise ValueError("No category has enough Human and fully-new LLM pairs")
        quotas["quota"] = quotas.quota.astype(int)
        connection.register("quotas", quotas)

        def sample(table: str, source: str, items_path: Path) -> pd.DataFrame:
            return connection.execute(
                f"""
                WITH sampled AS (
                    SELECT id1, id2, category
                    FROM (
                    SELECT p.*, q.quota,
                      ROW_NUMBER() OVER (
                        PARTITION BY p.category ORDER BY hash(p.id1, p.id2, {seed})
                      ) AS sample_rank
                    FROM {table} AS p
                    INNER JOIN quotas AS q ON q.category = p.category
                    )
                    WHERE sample_rank <= quota
                )
                SELECT
                    s.id1,
                    s.id2,
                    s.category,
                    CAST(a.name AS VARCHAR) AS name_a,
                    CAST(a.attributes AS VARCHAR) AS attributes_a,
                    CAST(b.name AS VARCHAR) AS name_b,
                    CAST(b.attributes AS VARCHAR) AS attributes_b,
                    '{source}' AS source
                FROM sampled AS s
                INNER JOIN read_parquet(?) AS a ON a.id = s.id1
                INNER JOIN read_parquet(?) AS b ON b.id = s.id2
                ORDER BY s.category, hash(s.id1, s.id2, {seed})
                """
                ,
                [str(items_path), str(items_path)],
            ).fetchdf()

        human = sample("human_pairs", "human", human_items_path)
        llm = sample("llm_pairs", "llm", llm_items_path)
    finally:
        connection.close()
    result = pd.concat([human, llm], ignore_index=True)
    if result.empty or result.duplicated(["source", "id1", "id2"]).any():
        raise ValueError("Proxy sample is empty or contains duplicate source pairs")
    return result, {str(category): int(quota) for category, quota in capacities}


def _featurize_pairs(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    profiles: list[str] = ["name_length_mean", "attribute_count_mean"]
    profiles.extend(f"{field}_endpoint_presence_rate" for field in CANONICAL_FIELDS)
    for pair in sample.itertuples(index=False):
        item_a = canonicalize_record(
            {
                "id": int(pair.id1),
                "name": pair.name_a,
                "attributes": pair.attributes_a,
                "category": pair.category,
            }
        )
        item_b = canonicalize_record(
            {
                "id": int(pair.id2),
                "name": pair.name_b,
                "attributes": pair.attributes_b,
                "category": pair.category,
            }
        )
        features = pair_features(item_a, item_b)
        values = {
            "source": str(pair.source),
            "category": str(pair.category),
            **{name: float(value) for name, value in features.items() if name != "category"},
            "name_length_mean": (len(item_a["name_norm"]) + len(item_b["name_norm"])) / 2.0,
            "attribute_count_mean": (item_a["attribute_count"] + item_b["attribute_count"]) / 2.0,
        }
        for field in CANONICAL_FIELDS:
            values[f"{field}_endpoint_presence_rate"] = (
                float(bool(item_a[f"{field}_norm"])) + float(bool(item_b[f"{field}_norm"]))
            ) / 2.0
        rows.append(values)
    frame = pd.DataFrame(rows)
    feature_names = sorted(name for name in frame.columns if name not in {"source", "category"})
    if not np.isfinite(frame[feature_names].to_numpy(dtype=float)).all():
        raise ValueError("Proxy feature construction produced a non-finite value")
    return frame, feature_names, profiles


def _histogram_overlap(left: np.ndarray, right: np.ndarray) -> float:
    values = np.concatenate([left, right])
    if values.size == 0 or np.all(values == values[0]):
        return 1.0
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 11)))
    if len(edges) < 2:
        return 1.0
    left_hist, _ = np.histogram(left, bins=edges)
    right_hist, _ = np.histogram(right, bins=edges)
    return float(np.minimum(left_hist / left_hist.sum(), right_hist / right_hist.sum()).sum())


def _support_summary(
    human: pd.DataFrame, llm: pd.DataFrame, feature_names: list[str]
) -> dict[str, Any]:
    details: list[dict[str, float | str]] = []
    for feature in feature_names:
        human_values = human[feature].to_numpy(dtype=float)
        llm_values = llm[feature].to_numpy(dtype=float)
        lower_human, upper_human = np.quantile(human_values, [0.01, 0.99])
        lower_llm, upper_llm = np.quantile(llm_values, [0.01, 0.99])
        details.append(
            {
                "feature": feature,
                "llm_within_human_q01_q99": float(
                    ((llm_values >= lower_human) & (llm_values <= upper_human)).mean()
                ),
                "human_within_llm_q01_q99": float(
                    ((human_values >= lower_llm) & (human_values <= upper_llm)).mean()
                ),
                "histogram_overlap": _histogram_overlap(human_values, llm_values),
            }
        )
    ordered = sorted(details, key=lambda row: float(row["histogram_overlap"]))
    return {
        "feature_count": len(details),
        "median_llm_within_human_q01_q99": float(
            np.median([row["llm_within_human_q01_q99"] for row in details])
        ),
        "median_human_within_llm_q01_q99": float(
            np.median([row["human_within_llm_q01_q99"] for row in details])
        ),
        "median_histogram_overlap": float(
            np.median([row["histogram_overlap"] for row in details])
        ),
        "lowest_overlap_features": ordered[:5],
    }


def _profile(frame: pd.DataFrame, profile_columns: list[str]) -> dict[str, float]:
    return {column: float(frame[column].mean()) for column in profile_columns}


def _two_sample_auc(frame: pd.DataFrame, feature_names: list[str], seed: int) -> float | None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Proxy two-sample diagnostic requires scikit-learn") from exc
    y = (frame.source == "llm").to_numpy(dtype=int)
    source_counts = np.bincount(y, minlength=2)
    if source_counts.min() < 2:
        return None
    folds = min(5, int(source_counts.min()))
    prediction = np.empty(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    matrix = frame[feature_names].to_numpy(dtype=float)
    for train, test in splitter.split(matrix, y):
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        )
        classifier.fit(matrix[train], y[train])
        prediction[test] = classifier.predict_proba(matrix[test])[:, 1]
    return float(roc_auc_score(y, prediction))


def diagnose_proxy_domain_shift(
    human_items_path: str | Path,
    human_matches_path: str | Path,
    llm_items_path: str | Path,
    llm_matches_path: str | Path,
    silver_matches_path: str | Path,
    output_path: str | Path,
    *,
    max_pairs_per_source_category: int = 1_000,
    min_pairs_per_source_category: int = 100,
    seed: int = 20260812,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Describe a labelled-source proxy, never a target-risk estimate."""
    if max_pairs_per_source_category < min_pairs_per_source_category:
        raise ValueError("max_pairs_per_source_category must be at least the minimum")
    paths = [
        Path(path).resolve()
        for path in (
            human_items_path,
            human_matches_path,
            llm_items_path,
            llm_matches_path,
            silver_matches_path,
        )
    ]
    if missing := [str(path) for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"Missing proxy diagnostic input: {missing}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(temp_dir) if temp_dir else output_path.parent / "proxy_shift_tmp"
    sample, quotas = _sample_pairs(*paths, max_pairs_per_category=max_pairs_per_source_category,
                                   min_pairs_per_category=min_pairs_per_source_category,
                                   seed=seed, temp_dir=scratch)
    features, feature_names, profile_columns = _featurize_pairs(sample)
    by_category: dict[str, Any] = {}
    for category, group in features.groupby("category", sort=True):
        human, llm = group[group.source == "human"], group[group.source == "llm"]
        by_category[str(category)] = {
            "human_pairs": int(len(human)),
            "fully_new_llm_pairs": int(len(llm)),
            "two_sample_logistic_cv_roc_auc": _two_sample_auc(group, feature_names, seed),
            "support_overlap": _support_summary(human, llm, feature_names),
            "human_profile": _profile(human, profile_columns),
            "fully_new_llm_profile": _profile(llm, profile_columns),
        }
    aucs = [record["two_sample_logistic_cv_roc_auc"] for record in by_category.values()]
    result: dict[str, Any] = {
        "purpose": "proxy_covariate_shift_diagnostic_only_not_model_selection",
        "target_labels_used": False,
        "llm_pair_labels_used": False,
        "source_definition": {
            "human": "Human Gold pairs with same-category endpoints",
            "llm": "LLM pairs with both ids absent from Human Gold and actual Silver sample",
        },
        "classifier": {
            "kind": "per-category 5-fold logistic two-sample classifier",
            "features": "pair structure, lengths, and field completeness only",
            "interpretation": (
                "AUC above 0.5 detects proxy covariate difference, not Public AP loss."
            ),
        },
        "sampling": {
            "seed": seed,
            "max_pairs_per_source_category": max_pairs_per_source_category,
            "quotas_per_category": quotas,
        },
        "rows": {"total": int(len(features)), "per_source": int(len(features) // 2)},
        "categories": by_category,
        "median_two_sample_auc": float(np.median(aucs)) if aucs else None,
        "limitations": [
            "The LLM source is a proxy, not the hidden Public catalogue.",
            "No hidden/Public labels are available, so AP degradation is not identifiable.",
            "This output must not select a model, fusion, or submission recipe.",
        ],
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.inference import FusionPredictor
from matchcup.metrics import macro_average_precision

PUBLIC_MACRO_AP = 0.3664957802653864
DEFAULT_SELECTION_MARGIN = 0.002
DEFAULT_NEAR_PUBLIC_MARGIN = 0.05


def _balanced_quotas(
    capacities: Mapping[tuple[str, float], int], max_rows: int
) -> dict[tuple[str, float], int]:
    """Allocate a fixed sample evenly across available category/label strata."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    available = {key: int(value) for key, value in capacities.items() if int(value) > 0}
    if not available:
        raise ValueError("No eligible holdout strata")
    requested = min(max_rows, sum(available.values()))
    quota = {key: 0 for key in sorted(available)}
    remaining = requested
    while remaining:
        active = [key for key in quota if quota[key] < available[key]]
        if not active:
            break
        share = max(1, remaining // len(active))
        spent = 0
        for key in active:
            increment = min(share, available[key] - quota[key], remaining - spent)
            quota[key] += increment
            spent += increment
        remaining -= spent
    return quota


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "Novel-item holdout generation requires the 'local' extra (duckdb)"
        ) from exc
    return duckdb


def _pair_category_sql() -> str:
    """Match the category construction used by ``pair_features`` exactly."""
    return """
        CASE
          WHEN COALESCE(a.category, '') = '' THEN COALESCE(b.category, '')
          WHEN COALESCE(b.category, '') = '' THEN COALESCE(a.category, '')
          WHEN a.category = b.category THEN a.category
          WHEN a.category < b.category THEN a.category || ' | ' || b.category
          ELSE b.category || ' | ' || a.category
        END
    """


def make_novel_item_holdout(
    items_path: str | Path,
    matches_llm_path: str | Path,
    gold_matches_path: str | Path,
    silver_matches_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = 100_000,
    seed: int = 20260812,
    temp_dir: str | Path | None = None,
) -> dict[str, int]:
    """Create a binary, item-disjoint validation set from the LLM-labelled pairs.

    Both endpoints are excluded when they occur in either human gold data or the
    *actual* sampled silver-pretraining file.  Categories are joined before
    stratification, and categories without both binary labels are deliberately
    excluded because category-macro AP is undefined for them.
    """
    duckdb = _require_duckdb()
    items_path = Path(items_path).resolve()
    matches_llm_path = Path(matches_llm_path).resolve()
    gold_matches_path = Path(gold_matches_path).resolve()
    silver_matches_path = Path(silver_matches_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(temp_dir) if temp_dir else output_path.parent / "novel_holdout_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    database = scratch / "novel_holdout.duckdb"
    connection = duckdb.connect(str(database))
    scratch_sql = str(scratch.resolve()).replace("'", "''")
    connection.execute("SET memory_limit='10GB'")
    connection.execute("SET threads=8")
    connection.execute(f"SET temp_directory='{scratch_sql}'")
    try:
        connection.execute(
            """
            CREATE TEMP TABLE train_ids AS
            SELECT DISTINCT id FROM (
                SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
                UNION ALL SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
            )
            """,
            [
                str(gold_matches_path),
                str(gold_matches_path),
                str(silver_matches_path),
                str(silver_matches_path),
            ],
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE eligible AS
            SELECT DISTINCT
                m.id1,
                m.id2,
                CAST(m.target AS DOUBLE) AS target,
                {_pair_category_sql()} AS category
            FROM read_parquet(?) AS m
            INNER JOIN read_parquet(?) AS a ON a.id = m.id1
            INNER JOIN read_parquet(?) AS b ON b.id = m.id2
            LEFT JOIN train_ids AS left_id ON left_id.id = m.id1
            LEFT JOIN train_ids AS right_id ON right_id.id = m.id2
            WHERE m.target IN (0.0, 1.0)
              AND left_id.id IS NULL
              AND right_id.id IS NULL
            """,
            [str(matches_llm_path), str(items_path), str(items_path)],
        )
        count_rows = connection.execute(
            """
            SELECT category, target, COUNT(*) AS rows
            FROM eligible
            GROUP BY category, target
            ORDER BY category, target
            """
        ).fetchall()
        labels_by_category: dict[str, set[float]] = {}
        raw_capacities: dict[tuple[str, float], int] = {}
        for category, target, rows in count_rows:
            key = (str(category), float(target))
            raw_capacities[key] = int(rows)
            labels_by_category.setdefault(key[0], set()).add(key[1])
        capacities = {
            key: rows
            for key, rows in raw_capacities.items()
            if labels_by_category[key[0]] == {0.0, 1.0}
        }
        quotas = _balanced_quotas(capacities, max_rows)
        quota_rows = [
            {"category": category, "target": target, "take": take}
            for (category, target), take in quotas.items()
            if take
        ]
        connection.register("holdout_quotas", pa.Table.from_pylist(quota_rows))
        selected = connection.execute(
            """
            WITH ranked AS (
              SELECT
                eligible.*,
                ROW_NUMBER() OVER (
                  PARTITION BY category, target
                  ORDER BY hash(id1, id2, ?) ASC, id1 ASC, id2 ASC
                ) AS sample_rank
              FROM eligible
            )
            SELECT ranked.id1, ranked.id2, ranked.target, ranked.category
            FROM ranked
            INNER JOIN holdout_quotas AS q
              ON q.category = ranked.category AND q.target = ranked.target
            WHERE ranked.sample_rank <= q.take
            ORDER BY ranked.category, ranked.target, ranked.sample_rank, ranked.id1, ranked.id2
            """,
            [int(seed)],
        ).to_arrow_table()
    finally:
        connection.close()
    if not selected.num_rows:
        raise ValueError("No binary, fully novel pairs are available for the holdout")
    selected = selected.append_column(
        "row_id", pa.array(np.arange(selected.num_rows, dtype=np.int64))
    )
    pq.write_table(selected, output_path, compression="zstd")
    retained_categories = {category for category, _ in capacities}
    return {
        "rows": selected.num_rows,
        "eligible_rows": sum(raw_capacities.values()),
        "categories": len(retained_categories),
        "excluded_incomplete_categories": len(labels_by_category) - len(retained_categories),
    }


def make_natural_rate_public_proxy(
    items_path: str | Path,
    matches_llm_path: str | Path,
    gold_matches_path: str | Path,
    silver_matches_path: str | Path,
    output_path: str | Path,
    *,
    max_rows_per_category: int = 20_000,
    seed: int = 20260820,
    temp_dir: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fully-new Silver proxy while preserving each category's natural rate.

    This is deliberately distinct from :func:`make_novel_item_holdout`: it does
    not discard intermediate LLM votes or balance classes.  Its binary label is
    exactly the pre-registered 9/9 vote, and its only sample cap is a deterministic
    per-category row cap so macro metrics remain well defined.
    """
    if max_rows_per_category < 1:
        raise ValueError("max_rows_per_category must be positive")
    duckdb = _require_duckdb()
    items_path = Path(items_path).resolve()
    matches_llm_path = Path(matches_llm_path).resolve()
    gold_matches_path = Path(gold_matches_path).resolve()
    silver_matches_path = Path(silver_matches_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path) if report_path else output_path.with_suffix(".report.json")
    scratch = Path(temp_dir) if temp_dir else output_path.parent / "public_proxy_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    database = scratch / "public_proxy.duckdb"
    connection = duckdb.connect(str(database))
    scratch_sql = str(scratch.resolve()).replace("'", "''")
    connection.execute("SET memory_limit='10GB'")
    connection.execute("SET threads=8")
    connection.execute(f"SET temp_directory='{scratch_sql}'")
    try:
        connection.execute(
            """
            CREATE TEMP TABLE train_ids AS
            SELECT DISTINCT id FROM (
                SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
                UNION ALL SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
            )
            """,
            [
                str(gold_matches_path),
                str(gold_matches_path),
                str(silver_matches_path),
                str(silver_matches_path),
            ],
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE eligible AS
            SELECT DISTINCT
                m.id1,
                m.id2,
                CAST(m.target = 1.0 AS DOUBLE) AS target,
                {_pair_category_sql()} AS category
            FROM read_parquet(?) AS m
            INNER JOIN read_parquet(?) AS a ON a.id = m.id1
            INNER JOIN read_parquet(?) AS b ON b.id = m.id2
            LEFT JOIN train_ids AS left_id ON left_id.id = m.id1
            LEFT JOIN train_ids AS right_id ON right_id.id = m.id2
            WHERE left_id.id IS NULL AND right_id.id IS NULL
            """,
            [str(matches_llm_path), str(items_path), str(items_path)],
        )
        count_rows = connection.execute(
            """
            SELECT category, COUNT(*) AS rows, SUM(target) AS positives
            FROM eligible
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
        capacities: dict[str, int] = {}
        eligible_prevalence: dict[str, float] = {}
        excluded_incomplete = 0
        for category, rows, positives in count_rows:
            count = int(rows)
            positive_count = int(positives)
            if positive_count == 0 or positive_count == count:
                excluded_incomplete += 1
                continue
            capacities[str(category)] = count
            eligible_prevalence[str(category)] = positive_count / count
        if not capacities:
            raise ValueError("No fully novel category has both Public Proxy labels")
        quota_rows = [
            {"category": category, "take": min(max_rows_per_category, rows)}
            for category, rows in sorted(capacities.items())
        ]
        connection.register("public_proxy_quotas", pa.Table.from_pylist(quota_rows))
        selected = connection.execute(
            """
            WITH ranked AS (
              SELECT
                eligible.*,
                ROW_NUMBER() OVER (
                  PARTITION BY category
                  ORDER BY hash(id1, id2, ?) ASC, id1 ASC, id2 ASC
                ) AS sample_rank
              FROM eligible
            )
            SELECT ranked.id1, ranked.id2, ranked.target, ranked.category
            FROM ranked
            INNER JOIN public_proxy_quotas AS q ON q.category = ranked.category
            WHERE ranked.sample_rank <= q.take
            ORDER BY ranked.category, ranked.sample_rank, ranked.id1, ranked.id2
            """,
            [int(seed)],
        ).to_arrow_table()
    finally:
        connection.close()
    if not selected.num_rows:
        raise ValueError("No fully novel rows selected for the Public Proxy")
    selected = selected.append_column(
        "row_id", pa.array(np.arange(selected.num_rows, dtype=np.int64))
    )
    pq.write_table(selected, output_path, compression="zstd")
    selected_frame = selected.select(["category", "target"]).to_pandas()
    sample_prevalence = {
        str(category): float(group.target.mean())
        for category, group in selected_frame.groupby("category", sort=True)
    }
    report: dict[str, Any] = {
        "purpose": "silver_public_proxy_natural_rate",
        "rows": int(selected.num_rows),
        "eligible_rows": int(sum(capacities.values())),
        "categories": len(capacities),
        "excluded_incomplete_categories": excluded_incomplete,
        "max_rows_per_category": max_rows_per_category,
        "seed": seed,
        "label": "target == 1.0",
        "eligible_macro_prevalence": float(np.mean(list(eligible_prevalence.values()))),
        "sample_macro_prevalence": float(np.mean(list(sample_prevalence.values()))),
        "eligible_prevalence_by_category": eligible_prevalence,
        "sample_prevalence_by_category": sample_prevalence,
        "model_selection_allowed": True,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def describe_new_item_llm_slices(
    items_path: str | Path,
    matches_llm_path: str | Path,
    gold_matches_path: str | Path,
    silver_matches_path: str | Path,
    output_path: str | Path,
    *,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Describe fully-new LLM pairs without presenting them as Human-Gold validation.

    ``natural_rate`` keeps the observed LLM confidence mix. ``hard_slice`` is
    limited to the adjacent 4/9 and 5/9 confidence labels. Both are catalogue
    diagnostics only: their weak labels and class rates cannot choose a model.
    """
    duckdb = _require_duckdb()
    items_path = Path(items_path).resolve()
    matches_llm_path = Path(matches_llm_path).resolve()
    gold_matches_path = Path(gold_matches_path).resolve()
    silver_matches_path = Path(silver_matches_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(temp_dir) if temp_dir else output_path.parent / "new_item_llm_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    database = scratch / "new_item_llm_diagnostics.duckdb"
    connection = duckdb.connect(str(database))
    scratch_sql = str(scratch.resolve()).replace("'", "''")
    connection.execute("SET memory_limit='10GB'")
    connection.execute("SET threads=8")
    connection.execute(f"SET temp_directory='{scratch_sql}'")
    try:
        connection.execute(
            """
            CREATE TEMP TABLE train_ids AS
            SELECT DISTINCT id FROM (
                SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
                UNION ALL SELECT id1 AS id FROM read_parquet(?)
                UNION ALL SELECT id2 AS id FROM read_parquet(?)
            )
            """,
            [
                str(gold_matches_path),
                str(gold_matches_path),
                str(silver_matches_path),
                str(silver_matches_path),
            ],
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE eligible AS
            SELECT
                CAST(m.target AS DOUBLE) AS target,
                {_pair_category_sql()} AS category
            FROM read_parquet(?) AS m
            INNER JOIN read_parquet(?) AS a ON a.id = m.id1
            INNER JOIN read_parquet(?) AS b ON b.id = m.id2
            LEFT JOIN train_ids AS left_id ON left_id.id = m.id1
            LEFT JOIN train_ids AS right_id ON right_id.id = m.id2
            WHERE left_id.id IS NULL AND right_id.id IS NULL
            """,
            [str(matches_llm_path), str(items_path), str(items_path)],
        )
        label_rows = connection.execute(
            """
            SELECT target, COUNT(*) AS rows
            FROM eligible
            GROUP BY target
            ORDER BY target
            """
        ).fetchall()
        natural_rows = connection.execute(
            """
            SELECT
                category,
                COUNT(*) AS rows,
                AVG(CASE WHEN target > 0.5 THEN 1.0 ELSE 0.0 END) AS binary_proxy_positive_rate
            FROM eligible
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
        hard_rows = connection.execute(
            """
            SELECT
                category,
                COUNT(*) AS rows,
                SUM(CASE WHEN target = 0.4444444444444444 THEN 1 ELSE 0 END) AS label_4_9,
                SUM(CASE WHEN target = 0.5555555555555556 THEN 1 ELSE 0 END) AS label_5_9
            FROM eligible
            WHERE target IN (0.4444444444444444, 0.5555555555555556)
            GROUP BY category
            ORDER BY category
            """
        ).fetchall()
    finally:
        connection.close()
    labels = {str(target): int(rows) for target, rows in label_rows}
    natural = {
        str(category): {"rows": int(rows), "binary_proxy_positive_rate": float(rate)}
        for category, rows, rate in natural_rows
    }
    hard = {
        str(category): {
            "rows": int(rows),
            "label_4_9": int(label_4_9),
            "label_5_9": int(label_5_9),
        }
        for category, rows, label_4_9, label_5_9 in hard_rows
    }
    result: dict[str, Any] = {
        "purpose": "catalogue_shift_diagnostic_only_not_model_selection",
        "model_selection_allowed": False,
        "exclusion": "both ids absent from Human Gold and the actual 2M Silver sample",
        "rows": sum(labels.values()),
        "categories": len(natural),
        "soft_label_counts": labels,
        "natural_rate": {
            "definition": "all fully-new LLM pairs; binary proxy is target > 0.5",
            "per_category": natural,
        },
        "hard_slice": {
            "definition": "only adjacent LLM confidence labels 4/9 and 5/9",
            "rows": sum(record["rows"] for record in hard.values()),
            "per_category": hard,
        },
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def select_score_mode(
    text_macro_ap: float, hybrid_macro_ap: float, *, min_gain: float = DEFAULT_SELECTION_MARGIN
) -> str:
    if min_gain < 0:
        raise ValueError("min_gain must be non-negative")
    return "hybrid" if hybrid_macro_ap - text_macro_ap >= min_gain else "text"


def evaluate_novel_item_holdout(
    pairs_path: str | Path,
    text_scores_path: str | Path,
    fusion_dir: str | Path,
    output_path: str | Path,
    *,
    min_hybrid_gain: float = DEFAULT_SELECTION_MARGIN,
    public_macro_ap: float = PUBLIC_MACRO_AP,
    near_public_margin: float = DEFAULT_NEAR_PUBLIC_MARGIN,
) -> dict[str, Any]:
    """Compare text and fusion scores and write the submission gate report."""
    pairs = pq.read_table(pairs_path).to_pandas().reset_index(names="row_index")
    scores = pq.read_table(text_scores_path).to_pandas()
    required_pair = {"row_index", "id1", "id2", "target", "category"}
    required_score = {"row_index", "id1", "id2", "text_score"}
    if missing := required_pair - set(pairs.columns):
        raise ValueError(f"Prepared holdout is missing columns: {sorted(missing)}")
    if missing := required_score - set(scores.columns):
        raise ValueError(f"Text scores are missing columns: {sorted(missing)}")
    if scores.duplicated("row_index").any():
        raise ValueError("Text score file has duplicate row_index values")
    frame = pairs.merge(
        scores[["row_index", "id1", "id2", "text_score"]],
        on=["row_index", "id1", "id2"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != len(pairs) or frame["text_score"].isna().any():
        raise ValueError("Text scores do not cover every prepared holdout pair")
    if not np.isfinite(frame["text_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Text scores contain non-finite values")
    text_macro_ap, text_per_category = macro_average_precision(
        frame["target"], frame["text_score"], frame["category"]
    )
    fusion = FusionPredictor(fusion_dir)
    hybrid_score = fusion.predict(frame.to_dict("records"))
    if not np.isfinite(hybrid_score).all():
        raise ValueError("Fusion scores contain non-finite values")
    hybrid_macro_ap, hybrid_per_category = macro_average_precision(
        frame["target"], hybrid_score, frame["category"]
    )
    score_mode = select_score_mode(text_macro_ap, hybrid_macro_ap, min_gain=min_hybrid_gain)
    selected_macro_ap = hybrid_macro_ap if score_mode == "hybrid" else text_macro_ap
    near_public_threshold = public_macro_ap + near_public_margin
    result: dict[str, Any] = {
        "rows": len(frame),
        "categories": int(frame["category"].nunique()),
        "text_macro_ap": text_macro_ap,
        "hybrid_macro_ap": hybrid_macro_ap,
        "hybrid_gain": hybrid_macro_ap - text_macro_ap,
        "required_hybrid_gain": min_hybrid_gain,
        "selected_score_mode": score_mode,
        "selected_macro_ap": selected_macro_ap,
        "public_macro_ap_reference": public_macro_ap,
        "near_public_margin": near_public_margin,
        "near_public_threshold": near_public_threshold,
        "submission_allowed": selected_macro_ap > near_public_threshold,
        "requires_model_adaptation": selected_macro_ap <= near_public_threshold,
        "per_category": {"text": text_per_category, "hybrid": hybrid_per_category},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

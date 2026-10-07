"""Strict, reusable gates for the attribute-permutation research candidate.

The module deliberately consumes predictions only.  It cannot train a model and
it validates every join key before calculating AP, so a monitor cannot accept a
partial or cross-run OOF file by accident.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.metrics import macro_average_precision

_GOLD_COLUMNS = {"row_id", "id1", "id2", "target", "category", "fold"}
_SCORE_COLUMNS = {"row_index", "id1", "id2", "text_score"}


def _read_parquet_paths(paths: Iterable[str | Path]) -> pd.DataFrame:
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("At least one score parquet is required")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing score files: {missing}")
    return pd.concat((pq.read_table(path).to_pandas() for path in resolved), ignore_index=True)


def write_combined_oof(
    score_paths: Iterable[str | Path], output_path: str | Path
) -> dict[str, int | str]:
    """Combine already validated fold score files without silently overwriting one."""
    scores = _read_parquet_paths(score_paths)
    if _SCORE_COLUMNS - set(scores.columns):
        raise ValueError("Score files do not expose the required OOF columns")
    if scores.duplicated("row_index").any():
        raise ValueError("Combined OOF contains duplicate row_index values")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = scores.sort_values("row_index", kind="stable").reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(ordered, preserve_index=False), output, compression="zstd")
    return {"path": str(output), "rows": int(len(ordered))}


def write_probability_ensemble_oof(
    control_paths: Iterable[str | Path],
    candidate_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    control_weight: float = 0.5,
) -> dict[str, int | str | float]:
    """Write a key-aligned, precommitted probability ensemble."""
    if not 0.0 <= control_weight <= 1.0:
        raise ValueError("control_weight must be in [0, 1]")
    control = _read_parquet_paths(control_paths)
    candidate = _read_parquet_paths(candidate_paths)
    for name, frame in (("control", control), ("candidate", candidate)):
        if missing := _SCORE_COLUMNS - set(frame.columns):
            raise ValueError(f"{name} scores are missing columns: {sorted(missing)}")
        if frame.duplicated("row_index").any():
            raise ValueError(f"{name} scores contain duplicate row_index values")
        if not np.isfinite(frame["text_score"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} scores contain non-finite values")
    keys = ["row_index", "id1", "id2"]
    control = control.sort_values("row_index", kind="stable").reset_index(drop=True)
    candidate = candidate.sort_values("row_index", kind="stable").reset_index(drop=True)
    if not control[keys].equals(candidate[keys]):
        raise ValueError("Control and candidate OOF keys differ")
    result = control[keys].copy()
    result["text_score"] = (
        control_weight * control["text_score"].to_numpy(dtype=np.float64)
        + (1.0 - control_weight) * candidate["text_score"].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    if "fold" in control.columns and "fold" in candidate.columns:
        if not np.array_equal(control["fold"].to_numpy(), candidate["fold"].to_numpy()):
            raise ValueError("Control and candidate OOF folds differ")
        result["fold"] = control["fold"].to_numpy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), output, compression="zstd")
    return {
        "path": str(output),
        "rows": int(len(result)),
        "control_weight": float(control_weight),
        "candidate_weight": float(1.0 - control_weight),
    }


def strict_oof_frame(gold_path: str | Path, score_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Return full Gold joined with one finite prediction per exact pair key."""
    gold = pq.read_table(gold_path).to_pandas()
    missing_gold = _GOLD_COLUMNS - set(gold.columns)
    if missing_gold:
        raise ValueError(f"Gold pairs are missing columns: {sorted(missing_gold)}")
    if gold.duplicated("row_id").any():
        raise ValueError("Gold pairs contain duplicate row_id values")
    scores = _read_parquet_paths(score_paths)
    missing_scores = _SCORE_COLUMNS - set(scores.columns)
    if missing_scores:
        raise ValueError(f"Scores are missing columns: {sorted(missing_scores)}")
    if scores.duplicated("row_index").any():
        raise ValueError("Scores contain duplicate row_index values")
    if not np.isfinite(scores["text_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Scores contain non-finite text_score values")
    frame = gold.merge(
        scores[["row_index", "id1", "id2", "text_score"]],
        left_on=["row_id", "id1", "id2"],
        right_on=["row_index", "id1", "id2"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != len(gold) or frame["text_score"].isna().any():
        raise ValueError("Scores do not cover every Gold row with matching row_index/id1/id2")
    unexpected = scores.merge(
        gold[["row_id", "id1", "id2"]],
        left_on=["row_index", "id1", "id2"],
        right_on=["row_id", "id1", "id2"],
        how="left",
        indicator=True,
    )
    if (unexpected["_merge"] != "both").any() or len(unexpected) != len(gold):
        raise ValueError("Scores include rows outside the exact Gold OOF contract")
    return frame.sort_values("row_id", kind="stable").reset_index(drop=True)


def _score_summary(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    return macro_average_precision(
        frame["target"], frame["text_score"], frame["category"].astype(str)
    )


def ordinary_oof_gate(
    gold_path: str | Path,
    control_paths: Iterable[str | Path],
    candidate_paths: Iterable[str | Path],
    *,
    maximum_category_drop: float = 0.02,
) -> dict[str, Any]:
    """Apply the precommitted all-fold ordinary OOF gate."""
    if maximum_category_drop < 0:
        raise ValueError("maximum_category_drop must be non-negative")
    control = strict_oof_frame(gold_path, control_paths)
    candidate = strict_oof_frame(gold_path, candidate_paths)
    if not control[["row_id", "id1", "id2"]].equals(candidate[["row_id", "id1", "id2"]]):
        raise ValueError("Control and candidate OOF keys differ")
    control_macro, control_categories = _score_summary(control)
    candidate_macro, candidate_categories = _score_summary(candidate)
    categories = {
        category: {
            "control_ap": float(control_categories[category]),
            "candidate_ap": float(candidate_categories[category]),
            "difference": float(candidate_categories[category] - control_categories[category]),
        }
        for category in sorted(control_categories)
    }
    failing = {
        category: record
        for category, record in categories.items()
        if record["difference"] < -maximum_category_drop
    }
    macro_gain = float(candidate_macro - control_macro)
    return {
        "purpose": "schema_robustness_precommitted_full_oof_gate",
        "rows": int(len(candidate)),
        "control_macro_ap": float(control_macro),
        "candidate_macro_ap": float(candidate_macro),
        "macro_gain": macro_gain,
        "maximum_category_drop": maximum_category_drop,
        "per_category": categories,
        "failing_categories": failing,
        "passed": macro_gain >= 0.0 and not failing,
    }


def paired_fusion_gate(
    gold_path: str | Path,
    baseline_paths: Iterable[str | Path],
    candidate_paths: Iterable[str | Path],
    *,
    minimum_macro_gain: float = 0.002,
    maximum_category_drop: float = 0.02,
    samples: int = 1_000,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Gate candidate fusion with a category-stratified paired bootstrap."""
    if minimum_macro_gain < 0.0:
        raise ValueError("minimum_macro_gain must be non-negative")
    if maximum_category_drop < 0.0:
        raise ValueError("maximum_category_drop must be non-negative")
    if samples < 10:
        raise ValueError("samples must be at least 10")
    baseline = strict_oof_frame(gold_path, baseline_paths)
    candidate = strict_oof_frame(gold_path, candidate_paths)
    keys = ["row_id", "id1", "id2"]
    if not baseline[keys].equals(candidate[keys]):
        raise ValueError("Baseline and candidate fusion OOF keys differ")
    frame = baseline[[*keys, "target", "category", "text_score"]].rename(
        columns={"text_score": "baseline_score"}
    )
    frame["candidate_score"] = candidate["text_score"].to_numpy(dtype=np.float64)
    baseline_macro, baseline_categories = macro_average_precision(
        frame["target"], frame["baseline_score"], frame["category"].astype(str)
    )
    candidate_macro, candidate_categories = macro_average_precision(
        frame["target"], frame["candidate_score"], frame["category"].astype(str)
    )
    categories = {
        category: {
            "baseline_ap": float(baseline_categories[category]),
            "candidate_ap": float(candidate_categories[category]),
            "difference": float(candidate_categories[category] - baseline_categories[category]),
        }
        for category in sorted(baseline_categories)
    }
    failing = {
        category: record
        for category, record in categories.items()
        if record["difference"] < -maximum_category_drop
    }
    groups = [group.index.to_numpy() for _, group in frame.groupby("category", sort=True)]
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        for _ in range(100):
            sampled = np.concatenate(
                [rng.choice(group, len(group), replace=True) for group in groups]
            )
            drawn = frame.loc[sampled]
            if all((group.target > 0.5).any() for _, group in drawn.groupby("category")):
                break
        else:
            raise ValueError("Bootstrap sample lost all positives in a category too often")
        drawn_baseline, _ = macro_average_precision(
            drawn["target"], drawn["baseline_score"], drawn["category"].astype(str)
        )
        drawn_candidate, _ = macro_average_precision(
            drawn["target"], drawn["candidate_score"], drawn["category"].astype(str)
        )
        differences[sample_index] = drawn_candidate - drawn_baseline
    macro_gain = float(candidate_macro - baseline_macro)
    ci95_low = float(np.quantile(differences, 0.025))
    ci95_high = float(np.quantile(differences, 0.975))
    return {
        "purpose": "paired_rank_fusion_release_gate",
        "rows": int(len(frame)),
        "minimum_macro_gain": minimum_macro_gain,
        "maximum_category_drop": maximum_category_drop,
        "samples": samples,
        "baseline_macro_ap": float(baseline_macro),
        "candidate_macro_ap": float(candidate_macro),
        "macro_gain": macro_gain,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "bootstrap_positive_rate": float(np.mean(differences > 0.0)),
        "per_category": categories,
        "failing_categories": failing,
        "macro_gain_passed": macro_gain >= minimum_macro_gain,
        "confidence_passed": ci95_low >= 0.0,
        "category_drop_passed": not failing,
        "passed": bool(
            macro_gain >= minimum_macro_gain and ci95_low >= 0.0 and not failing
        ),
    }


def frozen_stress_gate(
    stress_holdout_path: str | Path,
    baseline_report_path: str | Path,
    candidate_paths: Iterable[str | Path],
    *,
    maximum_category_drop: float = 0.02,
) -> dict[str, Any]:
    """Score the immutable #5 Human Gold stress split from strict candidate OOF."""
    if maximum_category_drop < 0:
        raise ValueError("maximum_category_drop must be non-negative")
    holdout = pq.read_table(stress_holdout_path).to_pandas()
    required = {"row_id", "id1", "id2", "target", "category"}
    if missing := required - set(holdout.columns):
        raise ValueError(f"Frozen stress holdout is missing columns: {sorted(missing)}")
    if holdout.duplicated("row_id").any():
        raise ValueError("Frozen stress holdout contains duplicate row_id values")
    candidate = _read_parquet_paths(candidate_paths)
    if missing := _SCORE_COLUMNS - set(candidate.columns):
        raise ValueError(f"Candidate scores are missing columns: {sorted(missing)}")
    if candidate.duplicated("row_index").any():
        raise ValueError("Candidate scores contain duplicate row_index values")
    if not np.isfinite(candidate["text_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Candidate scores contain non-finite text_score values")
    frame = holdout.merge(
        candidate[["row_index", "id1", "id2", "text_score"]],
        left_on=["row_id", "id1", "id2"],
        right_on=["row_index", "id1", "id2"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != len(holdout) or frame["text_score"].isna().any():
        raise ValueError("Candidate OOF does not exactly cover the frozen stress holdout")
    baseline = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    baseline_macro = baseline.get("stress_macro_ap")
    baseline_categories = baseline.get("stress_per_category_ap")
    if not isinstance(baseline_macro, (int, float)) or not isinstance(baseline_categories, dict):
        raise ValueError("Frozen baseline stress report has no valid text-score reference")
    candidate_macro, candidate_categories = _score_summary(frame)
    if set(candidate_categories) != set(baseline_categories):
        raise ValueError("Candidate and baseline stress category coverage differs")
    categories = {
        category: {
            "baseline_ap": float(baseline_categories[category]),
            "candidate_ap": float(candidate_categories[category]),
            "difference": float(candidate_categories[category] - baseline_categories[category]),
        }
        for category in sorted(candidate_categories)
    }
    failing = {
        category: record
        for category, record in categories.items()
        if record["difference"] < -maximum_category_drop
    }
    macro_gain = float(candidate_macro - float(baseline_macro))
    return {
        "purpose": "schema_robustness_frozen_human_gold_stress_gate",
        "rows": int(len(frame)),
        "baseline_macro_ap": float(baseline_macro),
        "candidate_macro_ap": float(candidate_macro),
        "macro_gain": macro_gain,
        "maximum_category_drop": maximum_category_drop,
        "per_category": categories,
        "failing_categories": failing,
        "passed": macro_gain > 0.0 and not failing,
    }


def combined_gate(ordinary: dict[str, Any], stress: dict[str, Any]) -> dict[str, Any]:
    """Summarize the two gates without adding a model-selection metric."""
    return {
        "purpose": "schema_robustness_precommitted_candidate_gate",
        "ordinary_oof": ordinary,
        "frozen_stress": stress,
        "passed": ordinary.get("passed") is True and stress.get("passed") is True,
    }

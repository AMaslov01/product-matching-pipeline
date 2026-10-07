"""Reproducible diagnostics for the OOF-to-Public investigation.

The functions here are deliberately separate from production packaging. Their
outputs are research artefacts only; none rebuilds or submits a contest ZIP.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.metrics import macro_average_precision
from matchcup.score_transform import score_transform_from_feature_schema

VARIANTS = {"gold_only", "silver_control", "final_like"}


def _score_quantiles(values: np.ndarray) -> dict[str, float]:
    labels = ("min", "p01", "p10", "p50", "p90", "p99", "max", "mean", "std")
    numbers = [
        *np.quantile(values, [0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0]).tolist(),
        float(values.mean()),
        float(values.std()),
    ]
    return {name: float(value) for name, value in zip(labels, numbers, strict=True)}


def _read_scores(path: str | Path) -> pd.DataFrame:
    """Load one score parquet or the OOF directory produced by the fold jobs."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise ValueError(f"Score directory contains no parquet files: {path}")
        return pd.concat((pq.read_table(file).to_pandas() for file in files), ignore_index=True)
    return pq.read_table(path).to_pandas()


def _score_frame(
    gold_path: str | Path, score_path: str | Path, *, required_fold: int | None = None
) -> pd.DataFrame:
    gold = pq.read_table(gold_path).to_pandas()
    required_gold = {"row_id", "id1", "id2", "target", "category", "fold"}
    if missing := required_gold - set(gold.columns):
        raise ValueError(f"Gold pairs are missing columns: {sorted(missing)}")
    if required_fold is not None:
        gold = gold[gold.fold.astype(int) == required_fold].copy()
        if gold.empty:
            raise ValueError(f"No rows found for fold {required_fold}")
    scores = _read_scores(score_path)
    required_scores = {"row_index", "id1", "id2", "text_score"}
    if missing := required_scores - set(scores.columns):
        raise ValueError(f"Scores are missing columns: {sorted(missing)}")
    if scores.duplicated("row_index").any():
        raise ValueError("Scores contain duplicate row_index values")
    result = gold.merge(
        scores[["row_index", "id1", "id2", "text_score"]],
        left_on=["row_id", "id1", "id2"],
        right_on=["row_index", "id1", "id2"],
        how="left",
        validate="one_to_one",
    )
    if result.text_score.isna().any():
        raise ValueError("Scores do not cover every requested Gold pair")
    if not np.isfinite(result.text_score.to_numpy(dtype=np.float64)).all():
        raise ValueError("Scores contain non-finite values")
    return result


def evaluate_fold_scores(
    gold_path: str | Path, score_path: str | Path, *, fold: int
) -> dict[str, Any]:
    """Evaluate exactly one preselected validation fold with production AP."""
    frame = _score_frame(gold_path, score_path, required_fold=fold)
    macro_ap, per_category = macro_average_precision(
        frame.target, frame.text_score, frame.category.astype(str)
    )
    return {
        "fold": fold,
        "rows": int(len(frame)),
        "macro_ap": macro_ap,
        "per_category": per_category,
        "score_distribution": _score_quantiles(frame.text_score.to_numpy(dtype=np.float64)),
    }


def mine_training_hard_negatives(
    gold_path: str | Path,
    score_path: str | Path,
    output_path: str | Path,
    *,
    validation_fold: int,
    top_fraction: float = 0.15,
) -> dict[str, Any]:
    """Mine high-scoring negatives using only the selected fold's train rows."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    frame = _score_frame(gold_path, score_path)
    training = frame[frame.fold.astype(int) != validation_fold].copy()
    negatives = training[training.target <= 0.5].copy()
    if negatives.empty:
        raise ValueError("No training negatives are available for hard-negative mining")
    per_category = max(1, int(len(negatives) * top_fraction / negatives.category.nunique()))
    selected = (
        negatives.sort_values(["category", "text_score"], ascending=[True, False])
        .groupby("category", group_keys=False)
        .head(per_category)
        .sort_values("row_id")
    )
    if (selected.fold.astype(int) == validation_fold).any():
        raise AssertionError("Validation rows leaked into hard-negative mining")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(selected[["row_id"]], preserve_index=False),
        output_path,
        compression="zstd",
    )
    return {
        "rows": int(len(selected)),
        "training_rows": int(len(training)),
        "training_negatives": int(len(negatives)),
        "selected_negative_share": float(len(selected) / len(negatives)),
        "per_category_quota": per_category,
        "validation_fold": validation_fold,
        "mean_text_score": float(selected.text_score.mean()),
        "score_distribution": _score_quantiles(selected.text_score.to_numpy(dtype=np.float64)),
        "categories": {
            str(category): int(count)
            for category, count in selected.groupby("category").size().items()
        },
    }


def paired_bootstrap_difference(
    gold_path: str | Path,
    control_scores_path: str | Path,
    candidate_scores_path: str | Path,
    *,
    fold: int,
    samples: int = 1_000,
    seed: int = 20260812,
) -> dict[str, float | int]:
    """Category-stratified paired bootstrap for candidate minus control AP."""
    if samples < 10:
        raise ValueError("samples must be at least 10")
    control = _score_frame(gold_path, control_scores_path, required_fold=fold)
    candidate = _score_frame(gold_path, candidate_scores_path, required_fold=fold)
    columns = ["row_id", "fold", "target", "category", "text_score"]
    joined = control[columns].merge(
        candidate[["row_id", "text_score"]], on="row_id", suffixes=("_control", "_candidate")
    )
    expected_rows = len(control)
    if len(joined) != expected_rows:
        raise ValueError("Control and candidate do not cover the same validation rows")
    control_ap, _ = macro_average_precision(
        joined.target, joined.text_score_control, joined.category.astype(str)
    )
    candidate_ap, _ = macro_average_precision(
        joined.target, joined.text_score_candidate, joined.category.astype(str)
    )
    groups = [group.index.to_numpy() for _, group in joined.groupby("category", sort=True)]
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        for _ in range(100):
            sampled = np.concatenate(
                [rng.choice(group, len(group), replace=True) for group in groups]
            )
            drawn = joined.loc[sampled]
            if all((group.target > 0.5).any() for _, group in drawn.groupby("category")):
                break
        else:
            raise ValueError("Bootstrap sample lost all positives in a category too often")
        drawn_control, _ = macro_average_precision(
            drawn.target, drawn.text_score_control, drawn.category.astype(str)
        )
        drawn_candidate, _ = macro_average_precision(
            drawn.target, drawn.text_score_candidate, drawn.category.astype(str)
        )
        differences[index] = drawn_candidate - drawn_control
    return {
        "fold": fold,
        "samples": samples,
        "control_macro_ap": float(control_ap),
        "candidate_macro_ap": float(candidate_ap),
        "difference": float(candidate_ap - control_ap),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_positive_rate": float(np.mean(differences > 0.0)),
    }


def assess_candidate_gate(
    gold_path: str | Path,
    control_scores_path: str | Path,
    candidate_scores_path: str | Path,
    *,
    fold: int,
    minimum_macro_gain: float = 0.005,
    maximum_category_drop: float = 0.02,
    samples: int = 1_000,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Apply the precommitted Human-Gold-only gate for a full-fold follow-up."""
    if minimum_macro_gain < 0:
        raise ValueError("minimum_macro_gain must be non-negative")
    if maximum_category_drop < 0:
        raise ValueError("maximum_category_drop must be non-negative")
    control = _score_frame(gold_path, control_scores_path, required_fold=fold)
    candidate = _score_frame(gold_path, candidate_scores_path, required_fold=fold)
    frame = control[["row_id", "target", "category", "text_score"]].merge(
        candidate[["row_id", "text_score"]],
        on="row_id",
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    if len(frame) != len(control):
        raise ValueError("Control and candidate do not cover the same validation rows")
    _, control_category = macro_average_precision(
        frame.target, frame.text_score_control, frame.category.astype(str)
    )
    _, candidate_category = macro_average_precision(
        frame.target, frame.text_score_candidate, frame.category.astype(str)
    )
    per_category = {
        category: {
            "control_ap": control_category[category],
            "candidate_ap": candidate_category[category],
            "difference": candidate_category[category] - control_category[category],
        }
        for category in sorted(control_category)
    }
    dropped = {
        category: record
        for category, record in per_category.items()
        if record["difference"] < -maximum_category_drop
    }
    bootstrap = paired_bootstrap_difference(
        gold_path,
        control_scores_path,
        candidate_scores_path,
        fold=fold,
        samples=samples,
        seed=seed,
    )
    macro_passed = bootstrap["difference"] >= minimum_macro_gain
    return {
        "purpose": "human_gold_only_precommitted_scaling_gate",
        "fold": fold,
        "minimum_macro_gain": minimum_macro_gain,
        "maximum_category_drop": maximum_category_drop,
        "bootstrap": bootstrap,
        "per_category": per_category,
        "failing_categories": dropped,
        "macro_gain_passed": macro_passed,
        "category_drop_passed": not dropped,
        "passed": bool(macro_passed and not dropped),
    }


def compare_score_distributions(
    gold_path: str | Path,
    oof_scores_path: str | Path,
    final_scores_path: str | Path,
    *,
    top_fraction: float = 0.01,
) -> dict[str, Any]:
    """Describe final-vs-OOF score drift; this is not a validation metric."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    oof = _score_frame(gold_path, oof_scores_path)
    final = _score_frame(gold_path, final_scores_path)
    frame = oof[["row_id", "category", "text_score"]].merge(
        final[["row_id", "text_score"]],
        on="row_id",
        suffixes=("_oof", "_final"),
        validate="one_to_one",
    )
    if len(frame) != len(oof):
        raise ValueError("OOF and final scores do not cover the same Gold rows")
    categories: dict[str, Any] = {}
    for category, group in frame.groupby("category", sort=True):
        top_count = max(1, math.ceil(len(group) * top_fraction))
        oof_top = set(group.nlargest(top_count, "text_score_oof").row_id)
        final_top = set(group.nlargest(top_count, "text_score_final").row_id)
        categories[str(category)] = {
            "rows": int(len(group)),
            "rank_correlation": float(
                group.text_score_oof.rank(method="average").corr(
                    group.text_score_final.rank(method="average"), method="pearson"
                )
            ),
            "top_count": top_count,
            "top_overlap": float(len(oof_top & final_top) / len(oof_top | final_top)),
            "oof": _score_quantiles(group.text_score_oof.to_numpy(dtype=np.float64)),
            "final": _score_quantiles(group.text_score_final.to_numpy(dtype=np.float64)),
        }
    return {
        "purpose": "diagnostic_only_not_a_model_selection_metric",
        "rows": int(len(frame)),
        "top_fraction": top_fraction,
        "catboost_input_shift": {
            "changing_feature": "text_score",
            "unchanged_features": "Gold pair features are identical rows; only text_score changes.",
            "overall": {
                "oof": _score_quantiles(frame.text_score_oof.to_numpy(dtype=np.float64)),
                "final": _score_quantiles(
                    frame.text_score_final.to_numpy(dtype=np.float64)
                ),
            },
        },
        "categories": categories,
        "oof": _score_quantiles(frame.text_score_oof.to_numpy(dtype=np.float64)),
        "final": _score_quantiles(frame.text_score_final.to_numpy(dtype=np.float64)),
    }


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    correlation = left.rank(method="average").corr(right.rank(method="average"), method="pearson")
    if pd.isna(correlation):
        raise ValueError("Rank correlation is undefined for a diagnostic slice")
    return float(correlation)


def _top_overlap(
    frame: pd.DataFrame, left: str, right: str, top_fraction: float
) -> dict[str, float | int]:
    count = max(1, math.ceil(len(frame) * top_fraction))
    left_top = set(frame.nlargest(count, left).row_id)
    right_top = set(frame.nlargest(count, right).row_id)
    return {
        "top_count": count,
        "top_overlap": float(len(left_top & right_top) / len(left_top | right_top)),
    }


def _score_transform_from_fusion_dir(fusion_dir: str | Path):
    schema_path = Path(fusion_dir) / "feature_schema.json"
    if not schema_path.exists():
        return None
    return score_transform_from_feature_schema(json.loads(schema_path.read_text(encoding="utf-8")))


def describe_fusion_score_shift(
    gold_path: str | Path,
    oof_scores_path: str | Path,
    final_scores_path: str | Path,
    fusion_dir: str | Path,
    *,
    top_fraction: float = 0.01,
) -> dict[str, Any]:
    """Measure how an immutable OOF-trained fusion reacts to a new text-score input.

    The all-Gold refit score is necessarily in-sample on Gold.  This function
    therefore exposes only score and rank changes and labels the output as a
    diagnostic; it must never be used to select a model or a threshold.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    from matchcup.inference import FusionPredictor

    oof = _score_frame(gold_path, oof_scores_path)
    final = _score_frame(gold_path, final_scores_path)
    frame = oof.merge(
        final[["row_id", "text_score"]],
        on="row_id",
        suffixes=("_oof", "_final"),
        validate="one_to_one",
    )
    if len(frame) != len(oof):
        raise ValueError("OOF and final scores do not cover the same Gold rows")
    score_transform = _score_transform_from_fusion_dir(fusion_dir)
    if score_transform is not None:
        oof_features = score_transform.add_features(
            frame[["category", "text_score_oof"]].rename(
                columns={"text_score_oof": score_transform.source_feature}
            )
        )
        final_features = score_transform.add_features(
            frame[["category", "text_score_final"]].rename(
                columns={"text_score_final": score_transform.source_feature}
            )
        )
        frame["text_score_rank_oof"] = oof_features[score_transform.output_feature].to_numpy()
        frame["text_score_rank_final"] = final_features[score_transform.output_feature].to_numpy()
    predictor = FusionPredictor(fusion_dir)
    oof_records = frame.drop(columns=["text_score_final"]).rename(
        columns={"text_score_oof": "text_score"}
    )
    final_records = oof_records.copy()
    final_records["text_score"] = frame.text_score_final.to_numpy(dtype=np.float64)
    frame["hybrid_oof"] = predictor.predict(oof_records.to_dict("records"))
    frame["hybrid_final"] = predictor.predict(final_records.to_dict("records"))
    if not np.isfinite(frame[["hybrid_oof", "hybrid_final"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Fusion diagnostic produced non-finite predictions")

    def describe(group: pd.DataFrame) -> dict[str, Any]:
        input_shift = _top_overlap(group, "text_score_oof", "text_score_final", top_fraction)
        output_shift = _top_overlap(group, "hybrid_oof", "hybrid_final", top_fraction)
        result = {
            "rows": int(len(group)),
            "text_input_rank_correlation": _rank_correlation(
                group.text_score_oof, group.text_score_final
            ),
            "hybrid_output_rank_correlation": _rank_correlation(
                group.hybrid_oof, group.hybrid_final
            ),
            "oof_text_to_hybrid_rank_correlation": _rank_correlation(
                group.text_score_oof, group.hybrid_oof
            ),
            "final_text_to_hybrid_rank_correlation": _rank_correlation(
                group.text_score_final, group.hybrid_final
            ),
            "text_input_top_1pct": input_shift,
            "hybrid_output_top_1pct": output_shift,
            "oof_text": _score_quantiles(group.text_score_oof.to_numpy(dtype=np.float64)),
            "final_text": _score_quantiles(group.text_score_final.to_numpy(dtype=np.float64)),
            "oof_hybrid": _score_quantiles(group.hybrid_oof.to_numpy(dtype=np.float64)),
            "final_hybrid": _score_quantiles(group.hybrid_final.to_numpy(dtype=np.float64)),
        }
        if score_transform is None:
            result["rank_input"] = {"available": False, "reason": "legacy_raw_score_schema"}
        else:
            rank_input_shift = _top_overlap(
                group, "text_score_rank_oof", "text_score_rank_final", top_fraction
            )
            result["rank_input"] = {
                "available": True,
                "oof_rank_correlation": _rank_correlation(
                    group.text_score_oof, group.text_score_rank_oof
                ),
                "final_rank_correlation": _rank_correlation(
                    group.text_score_final, group.text_score_rank_final
                ),
                "oof_to_final_rank_correlation": _rank_correlation(
                    group.text_score_rank_oof, group.text_score_rank_final
                ),
                "top_1pct": rank_input_shift,
                "oof": _score_quantiles(group.text_score_rank_oof.to_numpy(dtype=np.float64)),
                "final": _score_quantiles(
                    group.text_score_rank_final.to_numpy(dtype=np.float64)
                ),
            }
        return result

    return {
        "purpose": "diagnostic_only_in_sample_final_score_no_model_selection",
        "rows": int(len(frame)),
        "top_fraction": top_fraction,
        "fusion_training_provenance": (
            "Fusion remains the immutable model trained on OOF text_score."
        ),
        "score_transform": (
            score_transform.report()
            if score_transform is not None
            else {"kind": "legacy_raw_score_schema", "available": False}
        ),
        "warning": (
            "Final-model text_score is in-sample on Gold; no AP in this report is a "
            "validation result."
        ),
        "overall": describe(frame),
        "categories": {
            str(category): describe(group)
            for category, group in frame.groupby("category", sort=True)
        },
    }


def graph_diagnostics(gold_path: str | Path, score_path: str | Path) -> dict[str, Any]:
    """Quantify the labelled Gold subset on which graph work could apply."""
    frame = _score_frame(gold_path, score_path)
    endpoints = np.concatenate([frame.id1.to_numpy(), frame.id2.to_numpy()])
    item_ids, degrees = np.unique(endpoints, return_counts=True)
    left_degree = degrees[np.searchsorted(item_ids, frame.id1.to_numpy())]
    right_degree = degrees[np.searchsorted(item_ids, frame.id2.to_numpy())]
    isolated = (left_degree == 1) & (right_degree == 1)

    def score_subset(mask: np.ndarray) -> dict[str, Any]:
        subset = frame.loc[mask]
        valid_categories = [
            category for category, group in subset.groupby("category") if (group.target > 0.5).any()
        ]
        subset = subset[subset.category.isin(valid_categories)]
        if subset.empty:
            return {"rows": 0, "macro_ap": None, "categories": 0}
        macro_ap, per_category = macro_average_precision(
            subset.target, subset.text_score, subset.category.astype(str)
        )
        return {
            "rows": int(len(subset)),
            "macro_ap": macro_ap,
            "categories": len(per_category),
            "per_category": per_category,
        }

    return {
        "pairs": int(len(frame)),
        "unique_items": int(len(item_ids)),
        "isolated_edges": int(isolated.sum()),
        "isolated_edge_share": float(isolated.mean()),
        "degree_one_item_share": float((degrees == 1).mean()),
        "max_degree": int(degrees.max()),
        "isolated": score_subset(isolated),
        "nonisolated": score_subset(~isolated),
        "note": "Gold topology does not infer hidden-test topology.",
    }


def run_fold_ablation(
    gold_path: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
    fold: int,
    base_model: str,
    silver_model: str | Path | None,
    model_revision: str | None,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    ranking_weight: float,
    mixed_precision: str,
    num_workers: int,
    seed: int,
    mining_model: str | Path | None = None,
    hard_negative_fraction: float = 0.15,
    schema_robustness_spec: str | Path | None = None,
) -> dict[str, Any]:
    """Train one strict fold variant and write research-only output artefacts."""
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported research variant: {variant!r}")
    if variant != "gold_only" and silver_model is None:
        raise ValueError(f"{variant} requires a Silver checkpoint")
    if variant == "final_like" and mining_model is None:
        raise ValueError("final_like requires a train-only mining model")
    from matchcup.cross_encoder import CrossEncoderConfig, score_cross_encoder, train_cross_encoder
    from matchcup.schema_robustness import load_attribute_permutation_spec

    transform_spec = None
    transform_provenance = None
    if schema_robustness_spec is not None:
        spec_path = Path(schema_robustness_spec).expanduser().resolve()
        transform_spec = load_attribute_permutation_spec(spec_path)
        if variant != transform_spec.variant:
            raise ValueError(
                "schema-robustness spec is only allowed with variant="
                f"{transform_spec.variant!r}"
            )
        transform_provenance = {
            "spec": transform_spec.normalized(),
            "spec_sha256": transform_spec.fingerprint(),
            "source_path": str(spec_path),
            "source_yaml_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_path: Path | None = None
    hard_info: dict[str, Any] | None = None
    if variant == "final_like":
        mining_scores = output_dir / "mining_scores.parquet"
        score_cross_encoder(
            gold_path,
            mining_model,
            mining_scores,
            max_length=max_length,
            batch_size=eval_batch_size,
            mixed_precision=mixed_precision,
            num_workers=num_workers,
        )
        hard_path = output_dir / "hard_negatives.parquet"
        hard_info = mine_training_hard_negatives(
            gold_path,
            mining_scores,
            hard_path,
            validation_fold=fold,
            top_fraction=hard_negative_fraction,
        )
    epochs = 2 if variant == "final_like" else 1
    config = CrossEncoderConfig(
        model_name=base_model if variant == "gold_only" else str(silver_model),
        model_revision=model_revision,
        max_length=max_length,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        epochs=epochs,
        # The historical OOF run used the balanced sampler whenever this is
        # non-zero, although the loss itself starts only at epoch 1. Keep that
        # sampler constant across controls; ``train_cross_encoder`` applies no
        # ranking loss during their sole epoch.
        ranking_weight=ranking_weight,
        mixed_precision=mixed_precision,
        num_workers=num_workers,
        seed=seed,
        save_best_checkpoint=False,
        save_final_checkpoint=True,
        train_text_transform=transform_spec,
    )
    training = train_cross_encoder(
        gold_path,
        output_dir / "model",
        config,
        validation_fold=fold,
        hard_negative_path=hard_path,
    )
    if transform_provenance is not None:
        transform_provenance["training_summary"] = training["train_text_transform_summary"]
    scores_path = output_dir / "validation_scores.parquet"
    scoring = score_cross_encoder(
        gold_path,
        output_dir / "model",
        scores_path,
        max_length=max_length,
        batch_size=eval_batch_size,
        mixed_precision=mixed_precision,
        fold=fold,
        num_workers=num_workers,
    )
    evaluation = evaluate_fold_scores(gold_path, scores_path, fold=fold)
    report = {
        "purpose": "research_only_not_a_submission_candidate",
        "variant": variant,
        "fold": fold,
        "recipe": {
            "initialization": "base" if variant == "gold_only" else "silver",
            "gold_epochs": epochs,
            "ranking_weight_epoch_0": 0.0,
            "ranking_weight_epoch_1": ranking_weight if variant == "final_like" else 0.0,
            "hard_negative_policy": "train_only" if variant == "final_like" else "none",
        },
        "training": training,
        "scoring": scoring,
        "evaluation": evaluation,
        "hard_negatives": hard_info,
        "schema_robustness": transform_provenance,
    }
    (output_dir / "variant_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

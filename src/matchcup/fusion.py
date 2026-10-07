from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.score_transform import (
    BatchCategoryPercentileScoreTransform,
    score_transform_from_feature_schema,
)
from matchcup.transductive_profile import (
    feature_schema_digest,
    fusion_manifest,
    resolve_transductive_profile,
    validate_fusion_schema,
)

EXCLUDED_COLUMNS = {
    "id1",
    "id2",
    "target",
    "fold",
    "row_id",
    "row_index",
    "text_a",
    "text_b",
    "category",
}


def _require_catboost() -> Any:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("Install matchcup with the 'local' extra") from exc
    return CatBoostRegressor


def _require_catboost_ranker() -> Any:
    try:
        from catboost import CatBoostRanker
    except ImportError as exc:
        raise RuntimeError("Install matchcup with the 'local' extra") from exc
    return CatBoostRanker


def _require_catboost_pool() -> Any:
    try:
        from catboost import Pool
    except ImportError as exc:
        raise RuntimeError("Install matchcup with the 'local' extra") from exc
    return Pool


RANKING_OBJECTIVES = frozenset({"YetiRank", "QueryRMSE"})


def _ranking_pool(
    Pool: Any,
    x: pd.DataFrame,
    category_codes: np.ndarray,
    row_indices: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> tuple[Any, np.ndarray]:
    """Build one CatBoost query pool, sorted without losing source row indices.

    CatBoost requires all members of a query to be adjacent.  Its prediction is
    therefore in query order, not the pair-feature/OOF row order.  Returning the
    sorted source positions makes every caller explicitly put those values back
    where their keys originated.
    """
    query_order = np.argsort(category_codes[row_indices], kind="stable")
    sorted_indices = row_indices[query_order]
    arguments: dict[str, Any] = {
        "group_id": category_codes[sorted_indices],
    }
    if labels is not None:
        arguments["label"] = labels[sorted_indices]
    return Pool(x.iloc[sorted_indices], **arguments), sorted_indices


def _verify_exported_fusion_round_trip(
    model: Any,
    x: pd.DataFrame,
    records: list[dict[str, Any]],
    model_path: Path,
    schema_path: Path,
) -> dict[str, float | bool]:
    """Fail closed unless the Python export is the model the archive would serve."""
    native = np.asarray(model.predict(x), dtype=np.float64)
    exported = ExportedFusionModel(model_path, schema_path).predict(records)
    if native.shape != exported.shape:
        raise RuntimeError(
            "Exported fusion prediction shape does not match native CatBoost model: "
            f"{exported.shape} != {native.shape}"
        )
    if not np.isfinite(native).all() or not np.isfinite(exported).all():
        raise RuntimeError("Fusion export round-trip produced non-finite predictions")
    max_abs_delta = float(np.max(np.abs(native - exported))) if len(native) else 0.0
    if max_abs_delta >= 1e-6:
        raise RuntimeError(
            "Exported fusion model does not reproduce native CatBoost predictions: "
            f"max_abs_delta={max_abs_delta:.12g}"
        )
    return {"passed": True, "max_abs_delta": max_abs_delta}


def _load_scores(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No OOF parquet files in {path}")
    return pd.concat([pq.read_table(file).to_pandas() for file in files], ignore_index=True)


def _matrix(
    frame: pd.DataFrame,
    *,
    categories: list[str] | None = None,
    feature_names: list[str] | None = None,
    score_transform: BatchCategoryPercentileScoreTransform | None = None,
    score_group_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    excluded = set(EXCLUDED_COLUMNS)
    if score_transform is not None:
        frame = score_transform.add_features(frame, group_columns=score_group_columns)
        excluded.add(score_transform.source_feature)
    if categories is None:
        categories = sorted(frame["category"].astype(str).unique().tolist())
    numeric = [
        name
        for name in frame.columns
        if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])
    ]
    result = frame[numeric].astype(np.float32).copy()
    category_values = frame["category"].astype(str)
    for category in categories:
        result[f"category__{category}"] = (category_values == category).astype(np.float32)
    if feature_names is not None:
        for missing in set(feature_names) - set(result.columns):
            result[missing] = 0.0
        result = result[feature_names]
    return result, list(result.columns), categories


def _strict_oof_frame(pair_features_path: str | Path, oof_path: str | Path) -> pd.DataFrame:
    frame = pq.read_table(pair_features_path).to_pandas()
    required_pairs = {"id1", "id2", "target", "category", "fold"}
    if missing := required_pairs - set(frame.columns):
        raise ValueError(f"Pair features are missing columns: {sorted(missing)}")
    pair_row_index = "row_id" if "row_id" in frame.columns else "row_index"
    if pair_row_index not in frame.columns:
        raise ValueError("Pair features are missing row_id or row_index")
    if frame.duplicated(pair_row_index).any():
        raise ValueError(f"Pair features contain duplicate {pair_row_index} values")
    scores = _load_scores(oof_path)
    required_scores = {"row_index", "id1", "id2", "text_score"}
    if missing := required_scores - set(scores.columns):
        raise ValueError(f"OOF scores are missing columns: {sorted(missing)}")
    if scores.duplicated("row_index").any():
        raise ValueError("OOF contains duplicate row_index values")
    if scores.duplicated(["id1", "id2"]).any():
        raise ValueError("OOF contains duplicate pairs")
    try:
        score_values = scores["text_score"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("OOF text_score must be numeric") from exc
    if not np.isfinite(score_values).all():
        raise ValueError("OOF text_score must be finite")
    if len(scores) != len(frame):
        raise ValueError("OOF must contain exactly one score for every feature row")
    score_names = ["row_index", "id1", "id2", "text_score"]
    rename = {"row_index": "_oof_row_index", "id1": "_oof_id1", "id2": "_oof_id2"}
    if "fold" in scores.columns:
        score_names.append("fold")
        rename["fold"] = "_oof_fold"
    score_columns = scores[score_names].rename(columns=rename)
    ordered = frame.assign(_pair_order=np.arange(len(frame), dtype=np.int64))
    joined = ordered.merge(
        score_columns,
        left_on=[pair_row_index, "id1", "id2"],
        right_on=["_oof_row_index", "_oof_id1", "_oof_id2"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined["text_score"].isna().any():
        raise ValueError("OOF scores do not cover all feature rows by row_index, id1, id2")
    if "_oof_fold" in joined and not np.array_equal(
        joined["fold"].to_numpy(dtype=int), joined["_oof_fold"].to_numpy(dtype=int)
    ):
        raise ValueError("OOF fold does not match pair-feature fold")
    return (
        joined.sort_values("_pair_order")
        .drop(
            columns=[
                "_pair_order",
                "_oof_row_index",
                "_oof_id1",
                "_oof_id2",
                *(["_oof_fold"] if "_oof_fold" in joined else []),
            ]
        )
        .reset_index(drop=True)
    )


def train_fusion(
    pair_features_path: str | Path,
    oof_path: str | Path,
    output_dir: str | Path,
    *,
    iterations: int = 1200,
    learning_rate: float = 0.04,
    depth_candidates: list[int] | None = None,
    l2_candidates: list[float] | None = None,
    seed: int = 20260812,
    score_source: str = "single_probability",
    # Defaults to the raw contract because the alternative is not merely a
    # different option: batch-category percentile ranks a row against whatever
    # else happens to be in the inference batch, which is why candidate A scored
    # 0.4227 against a raw control at 0.4401. Leaving it as the default cost that
    # same 0.0174 twice more on 2026-08-24, when a rebuild called the CLI without
    # the flag. Asking for it now has to be deliberate.
    score_contract: str = "raw",
    objective: str = "RMSE",
    transductive_profile: str | None = None,
    expected_feature_schema_sha256: str | None = None,
) -> dict[str, Any]:
    from matchcup.metrics import macro_average_precision

    frame = _strict_oof_frame(pair_features_path, oof_path)
    profile = resolve_transductive_profile(transductive_profile, frame.columns)
    y = frame["target"].astype(np.float32).to_numpy()
    folds = frame["fold"].astype(int).to_numpy()
    category = frame["category"].astype(str).to_numpy()
    categories = sorted(set(category.tolist()))
    category_codes = pd.Categorical(category, categories=categories).codes.astype(np.int64)
    if score_contract not in {"category_rank", "raw"}:
        raise ValueError(f"Unsupported fusion score contract: {score_contract!r}")
    transform = (
        BatchCategoryPercentileScoreTransform(score_source=score_source)
        if score_contract == "category_rank"
        else None
    )
    x, feature_names, _ = _matrix(
        frame,
        categories=categories,
        score_transform=transform,
        score_group_columns=("fold", "category") if transform is not None else None,
    )
    if transform is not None and transform.source_feature in feature_names:
        raise AssertionError("Raw text score leaked into rank-only fusion")
    schema_digest = feature_schema_digest(feature_names)
    if (
        transductive_profile is not None
        and expected_feature_schema_sha256 is not None
        and expected_feature_schema_sha256 != schema_digest
    ):
        raise ValueError("Predeclared feature schema digest does not match prepared pairs")
    if objective not in {"RMSE", *RANKING_OBJECTIVES}:
        raise ValueError(
            f"Unsupported fusion objective: {objective!r}; expected RMSE, YetiRank or QueryRMSE"
        )
    ranking_objective = objective in RANKING_OBJECTIVES
    CatBoostEstimator = _require_catboost_ranker() if ranking_objective else _require_catboost()
    Pool = _require_catboost_pool() if ranking_objective else None
    depth_candidates = depth_candidates or [6, 8]
    l2_candidates = l2_candidates or [5.0, 10.0]
    trials: list[dict[str, Any]] = []
    best_score = -1.0
    best_params: dict[str, Any] = {}
    best_predictions: np.ndarray | None = None
    for depth in depth_candidates:
        for l2 in l2_candidates:
            predictions = np.zeros(len(frame), dtype=np.float32)
            for fold in sorted(set(folds)):
                train_mask = folds != fold
                valid_mask = folds == fold
                model = CatBoostEstimator(
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=depth,
                    l2_leaf_reg=l2,
                    loss_function=objective,
                    random_seed=seed + int(fold),
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                )
                if ranking_objective:
                    assert Pool is not None
                    train_pool, _ = _ranking_pool(
                        Pool, x, category_codes, np.flatnonzero(train_mask), labels=y
                    )
                    valid_pool, valid_indices = _ranking_pool(
                        Pool, x, category_codes, np.flatnonzero(valid_mask)
                    )
                    model.fit(train_pool)
                    # CatBoost returns query-sorted predictions. Assigning by
                    # their original positions restores the strict OOF order.
                    predictions[valid_indices] = model.predict(valid_pool)
                else:
                    model.fit(x.loc[train_mask], y[train_mask])
                    predictions[valid_mask] = model.predict(x.loc[valid_mask])
            macro_ap, per_category = macro_average_precision(y, predictions, category)
            trial = {
                "depth": depth,
                "l2_leaf_reg": l2,
                "objective": objective,
                "macro_ap": macro_ap,
                "per_category": per_category,
            }
            trials.append(trial)
            if macro_ap > best_score:
                best_score = macro_ap
                best_params = {"depth": depth, "l2_leaf_reg": l2}
                best_predictions = predictions.copy()
    if best_predictions is None:
        raise RuntimeError("Fusion parameter search did not produce predictions")

    text_macro_ap, text_per_category = macro_average_precision(
        y, frame["text_score"].to_numpy(dtype=np.float64), category
    )
    score_feature = transform.output_feature if transform is not None else "text_score"
    contract_macro_ap, contract_per_category = macro_average_precision(
        y, x[score_feature].to_numpy(dtype=np.float64), category
    )
    feature_columns = [name for name in feature_names if name != score_feature]
    feature_predictions = np.zeros(len(frame), dtype=np.float32)
    for fold in sorted(set(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        feature_model = CatBoostEstimator(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=best_params["depth"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            loss_function=objective,
            random_seed=seed + int(fold),
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        if ranking_objective:
            assert Pool is not None
            train_pool, _ = _ranking_pool(
                Pool,
                x[feature_columns],
                category_codes,
                np.flatnonzero(train_mask),
                labels=y,
            )
            valid_pool, valid_indices = _ranking_pool(
                Pool, x[feature_columns], category_codes, np.flatnonzero(valid_mask)
            )
            feature_model.fit(train_pool)
            feature_predictions[valid_indices] = feature_model.predict(valid_pool)
        else:
            feature_model.fit(x.loc[train_mask, feature_columns], y[train_mask])
            feature_predictions[valid_mask] = feature_model.predict(
                x.loc[valid_mask, feature_columns]
            )
    feature_macro_ap, feature_per_category = macro_average_precision(
        y, feature_predictions, category
    )
    fusion_macro_ap, fusion_per_category = macro_average_precision(y, best_predictions, category)
    final_model = CatBoostEstimator(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        loss_function=objective,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    if ranking_objective:
        assert Pool is not None
        final_pool, _ = _ranking_pool(
            Pool, x, category_codes, np.arange(len(frame), dtype=np.int64), labels=y
        )
        final_model.fit(final_pool)
    else:
        final_model.fit(x, y)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_index_name = "row_id" if "row_id" in frame.columns else "row_index"
    fusion_oof = pd.DataFrame(
        {
            "row_index": frame[row_index_name].to_numpy(),
            "id1": frame["id1"].to_numpy(),
            "id2": frame["id2"].to_numpy(),
            "text_score": best_predictions,
            "fold": folds,
        }
    )
    pq.write_table(
        pa.Table.from_pandas(fusion_oof, preserve_index=False),
        output_dir / "fusion_oof.parquet",
        compression="zstd",
    )
    final_model.save_model(output_dir / "fusion.cbm")
    schema = {
        "feature_names": feature_names,
        "categories": categories,
        "score_source": score_source,
        "score_contract": (
            transform.to_schema()
            if transform is not None
            else {"version": 2, "kind": "raw_probability", "score_source": score_source}
        ),
        "fusion_objective": objective,
    }
    # The digest is a pure function of feature_names, unrelated to whether a
    # transductive profile is tracked, so it belongs in the schema
    # unconditionally: a profile-less two_score build (H39's bare 22-feature
    # fusion) still needs a way to prove two archives share the exact same
    # feature set, and "no profile" must not also mean "no digest to compare."
    schema["feature_schema_sha256"] = schema_digest
    if transductive_profile is not None:
        schema["transductive_profile"] = profile.name
    if transform is not None:
        schema["score_transform"] = transform.to_schema()
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if transductive_profile is not None:
        (output_dir / "fusion_manifest.json").write_text(
            json.dumps(fusion_manifest(profile, feature_names), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    exported_model_path = output_dir / "fusion_model.py"
    final_model.save_model(exported_model_path, format="python")
    export_round_trip = (
        _verify_exported_fusion_round_trip(
            final_model,
            x,
            frame.to_dict(orient="records"),
            exported_model_path,
            output_dir / "feature_schema.json",
        )
        if ranking_objective
        else None
    )
    report = {
        "best_macro_ap": best_score,
        "best_params": best_params,
        "objective": objective,
        "grouped_by_category": ranking_objective,
        "export_round_trip": export_round_trip,
        "ablations": {
            "features_only": {"macro_ap": feature_macro_ap, "per_category": feature_per_category},
            "cross_encoder_only": {"macro_ap": text_macro_ap, "per_category": text_per_category},
            "score_contract_only": {
                "macro_ap": contract_macro_ap,
                "per_category": contract_per_category,
            },
            "fusion": {"macro_ap": fusion_macro_ap, "per_category": fusion_per_category},
        },
        "trials": trials,
        "score_contract": schema["score_contract"],
        "transductive_profile": profile.name if transductive_profile is not None else None,
        "feature_schema_sha256": schema.get("feature_schema_sha256"),
        "fusion_oof": str(output_dir / "fusion_oof.parquet"),
    }
    (output_dir / "fusion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


class ExportedFusionModel:
    def __init__(self, model_path: str | Path, schema_path: str | Path) -> None:
        model_path = Path(model_path)
        spec = importlib.util.spec_from_file_location("matchcup_exported_fusion", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.apply = module.apply_catboost_model
        self.schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.transductive_profile = validate_fusion_schema(self.schema)

    def predict(self, records: list[dict[str, Any]]) -> np.ndarray:
        feature_names = self.schema["feature_names"]
        categories = self.schema["categories"]
        transform = score_transform_from_feature_schema(self.schema)
        transformed_records = transform.add_record_features(records) if transform else records
        matrix: list[list[float]] = []
        for record in transformed_records:
            values = dict(record)
            for category in categories:
                values[f"category__{category}"] = float(record.get("category") == category)
            matrix.append([float(values.get(name, 0.0)) for name in feature_names])
        return np.asarray([self.apply(row) for row in matrix], dtype=np.float64)

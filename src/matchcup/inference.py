from __future__ import annotations

import gc
import json
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from matchcup.features import pair_features
from matchcup.parser import _canonicalize_chunk, iter_item_batches
from matchcup.score_transform import score_transform_from_feature_schema
from matchcup.serialize import serialize_pair
from matchcup.transductive import (
    TransductiveCounts,
    TransductiveStatistics,
    TransductiveStatisticsAccumulator,
    count_canonical_records,
)
from matchcup.transductive_profile import validate_fusion_schema

# R3 shipped two models combined by a nominal 0.5/0.5 mean of raw probabilities
# and scored 0.5111099373, below the single-model R2 at 0.5136507571. The two
# score distributions had diverged (medians 0.0265 against 0.1180, standard
# deviations 0.2992 against 0.3725), so that mean correlated 0.9845 with the
# secondary model and 0.9305 with the primary: roughly 61% of the weaker model.
# Under this contract both scores reach the fusion as separate features and it
# learns the combination from OOF instead.
TWO_SCORE_SOURCE = "two_score"


class FusionPredictor:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
        self.transductive_profile = validate_fusion_schema(self.schema)
        self.score_transform = score_transform_from_feature_schema(self.schema)
        self.score_source = str(
            self.schema.get("score_source")
            or (self.schema.get("score_transform") or {}).get("score_source")
            or "single_probability"
        )
        try:
            from catboost import CatBoostError, CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError(
                "native CatBoost is required for submission inference; "
                "use the packaged MatchCup runtime image"
            ) from exc
        try:
            native = CatBoostRegressor()
            native.load_model(directory / "fusion.cbm")
        except (CatBoostError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"could not load native CatBoost model from {directory / 'fusion.cbm'}"
            ) from exc
        self.native = native

    def _matrix(self, records: list[dict[str, Any]]) -> np.ndarray:
        feature_names = self.schema["feature_names"]
        categories = self.schema["categories"]
        matrix = np.empty((len(records), len(feature_names)), dtype=np.float32)
        transformed_records = (
            self.score_transform.add_record_features(records) if self.score_transform else records
        )
        # The 0.0 default below is a silent failure mode for a two-column
        # fusion: a wiring slip that never sets ``second_score`` would build,
        # run, pass every phase, and only surface as a worse leaderboard number
        # days later, which is exactly how R3 lost 0.00254. Archives already
        # shipped on the single/mean contracts keep the permissive default. The
        # flag comes from the schema rather than the resolved attribute so it
        # holds for any predictor built around a schema.
        required = (
            frozenset(feature_names)
            if self.schema.get("score_source") == TWO_SCORE_SOURCE
            else None
        )
        for row_index, record in enumerate(transformed_records):
            values = dict(record)
            for category in categories:
                values[f"category__{category}"] = float(record.get("category") == category)
            if required is not None and not required <= values.keys():
                raise ValueError(
                    "Fusion record is missing features the schema names: "
                    f"{sorted(required - values.keys())}"
                )
            matrix[row_index] = [float(values.get(name, 0.0)) for name in feature_names]
        return matrix

    def predict(self, records: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(self.native.predict(self._matrix(records)), dtype=np.float64)


def _disable_modernbert_reference_compile(model: Any) -> None:
    """Use ModernBERT's eager embedding path in the contest runtime.

    Transformers 5.0 selects its compiled embedding helper automatically when
    Triton is present.  The public image deliberately has no C compiler for
    runtime compilation, so that path fails before the first inference.  This
    only chooses the model's equivalent eager implementation; it does not
    alter the checkpoint, tokenizer, or scoring computation.
    """
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "modernbert":
        config.reference_compile = False


def _cuda_precision_config(
    torch: Any, device_type: str, cuda_precision: str
) -> tuple[Any | None, Any]:
    """Return storage and autocast dtypes for a reproducible CUDA scorer."""
    if cuda_precision not in {"fp16_amp", "fp16_weights", "bf16_weights"}:
        raise ValueError(f"Unsupported CUDA precision: {cuda_precision!r}")
    model_dtype = None
    autocast_dtype = torch.float16
    if device_type == "cuda" and cuda_precision == "fp16_weights":
        model_dtype = torch.float16
    elif device_type == "cuda" and cuda_precision == "bf16_weights":
        model_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    return model_dtype, autocast_dtype


_ACCEPTED_ITEM_IDS: frozenset[int] | None = None


def _install_accepted_item_ids(accepted_ids: frozenset[int] | None) -> None:
    """Hand a canonicalization worker the pair id set once per process.

    Passing the set through the payload instead would re-pickle several megabytes
    for every chunk; a pool initializer pays that cost once per worker.
    """
    global _ACCEPTED_ITEM_IDS
    _ACCEPTED_ITEM_IDS = accepted_ids


def _canonicalize_serving_chunk(
    payload: tuple[list[dict[str, Any]], int, bool, str],
) -> tuple[list[dict[str, Any]], TransductiveCounts | None]:
    """Canonicalize one chunk and, when asked, count it before discarding rows.

    Transductive statistics describe every row of the submitted items file, while
    only items reachable from a requested pair need to travel back to the parent.
    Counting therefore happens before the accepted-id filter, never after.
    """
    records, remaining_chars, collect_statistics, transductive_profile = payload
    canonical = _canonicalize_chunk((records, remaining_chars))
    if not collect_statistics:
        return canonical, None
    counts = count_canonical_records(canonical, profile=transductive_profile)
    if _ACCEPTED_ITEM_IDS is not None:
        canonical = [item for item in canonical if int(item["id"]) in _ACCEPTED_ITEM_IDS]
    return canonical, counts


def _canonicalize_items(
    items_path: str | Path,
    workers: int,
    batch_size: int,
    *,
    accepted_ids: set[int] | None = None,
    collect_statistics: bool = False,
    transductive_profile: str = "h12-v1",
) -> tuple[dict[int, dict[str, Any]], TransductiveStatistics | None]:
    """Read the items file once, canonicalizing pair items and counting the corpus.

    Submission inference used to materialize the whole parquet table before
    filtering it implicitly through ``matches``.  The parser already exposes a
    bounded streaming reader, so preserve exactly the same record conversion
    while avoiding unrelated catalogue rows and a whole-table ``to_pylist``.

    ``collect_statistics`` folds the transductive document-frequency sweep into
    this pass.  It used to be a second, sequential, unfiltered sweep of the same
    file that re-canonicalized every row on one core; on ``items_human`` that
    sweep cost about four times the parallel pass it duplicated.  Statistics still
    cover the whole file, so the accepted-id filter moves out of the reader and
    into the worker, where it runs after that chunk has been counted.
    """
    accepted = frozenset(accepted_ids) if accepted_ids is not None else None
    # Exactly one of the two filters runs: the cheap reader-side one when only
    # pair items matter, the worker-side one when the corpus must be counted first.
    reader_filter = None if collect_statistics else accepted
    worker_filter = accepted if collect_statistics else None
    chunks = (
        (records, 1800, collect_statistics, transductive_profile)
        for records in iter_item_batches(items_path, batch_size, accepted_ids=reader_filter)
    )
    accumulator = (
        TransductiveStatisticsAccumulator(profile=transductive_profile)
        if collect_statistics
        else None
    )
    output: dict[int, dict[str, Any]] = {}

    def absorb(result: tuple[list[dict[str, Any]], TransductiveCounts | None]) -> None:
        canonical, counts = result
        if accumulator is not None and counts is not None:
            accumulator.add(counts)
        output.update((int(item["id"]), item) for item in canonical)

    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_install_accepted_item_ids,
            initargs=(worker_filter,),
        ) as pool:
            for result in pool.map(_canonicalize_serving_chunk, chunks, chunksize=1):
                absorb(result)
    else:
        _install_accepted_item_ids(worker_filter)
        try:
            for chunk in chunks:
                absorb(_canonicalize_serving_chunk(chunk))
        finally:
            _install_accepted_item_ids(None)
    return output, accumulator.build(items_path) if accumulator is not None else None


def _prepare_pair_chunk(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Build fusion records and symmetric tokenizer inputs in a worker process."""
    records: list[dict[str, Any]] = []
    texts: list[tuple[str, str]] = []
    for item_a, item_b in pairs:
        records.append(pair_features(item_a, item_b))
        texts.append(serialize_pair(item_a, item_b))
    return records, texts


def _tokenizers_compatible(left: Any, right: Any) -> bool:
    """Return whether one encoded batch is safe to feed to both models.

    The equality test is intentionally conservative: falling back to a second
    tokenization costs time but never changes the score contract.
    """
    return bool(
        left.get_vocab() == right.get_vocab()
        and left.special_tokens_map == right.special_tokens_map
        and left.model_max_length == right.model_max_length
        and left.padding_side == right.padding_side
        and left.truncation_side == right.truncation_side
    )


def _write_predictions(
    matches: pd.DataFrame, predictions: np.ndarray, output_path: str | Path
) -> pd.DataFrame:
    output = pd.DataFrame(
        {"id1": matches.id1.to_numpy(), "id2": matches.id2.to_numpy(), "predict": predictions}
    )
    if len(output) != len(matches) or not np.isfinite(output.predict.to_numpy()).all():
        raise RuntimeError("Inference did not produce a finite score for every pair")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def _combine_probability_scores(
    model_scores: list[np.ndarray], model_weights: tuple[float, ...] | None = None
) -> np.ndarray:
    if not model_scores:
        raise ValueError("At least one model score vector is required")
    lengths = {len(scores) for scores in model_scores}
    if len(lengths) != 1:
        raise ValueError("Model score vectors must have equal length")
    weights = model_weights or tuple(1.0 / len(model_scores) for _ in model_scores)
    if len(weights) != len(model_scores):
        raise ValueError("Model weights must match the model count")
    weight_values = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weight_values).all() or (weight_values < 0.0).any():
        raise ValueError("Model weights must be finite and non-negative")
    if not np.isclose(weight_values.sum(), 1.0, atol=1e-12):
        raise ValueError("Model weights must sum to one")
    stacked = np.vstack([np.asarray(scores, dtype=np.float64) for scores in model_scores])
    if not np.isfinite(stacked).all():
        raise ValueError("Model scores must be finite")
    return np.average(stacked, axis=0, weights=weight_values).astype(np.float32)


def _resolve_score_source(
    score_composition: str,
    model_count: int,
    score_mode: str,
    model_weights: tuple[float, ...] | None,
) -> str:
    """Name the score contract a run will satisfy, or refuse to serve at all.

    Packaging calls this too, so an archive can never be built in a shape the
    runtime would reject on the contest GPU, where the only signal is a failed
    phase.

    ``two_score`` is narrow on purpose. Without the fusion nothing combines the
    two columns, so the mode would be meaningless; with anything other than two
    models there is no second column to hand over; and it applies no weights at
    all, because a weighted mean in probability space is precisely what cost R3
    its 0.00254 against R2.
    """
    if score_composition == "mean_probability":
        return "mean_probability" if model_count > 1 else "single_probability"
    if score_composition != TWO_SCORE_SOURCE:
        raise ValueError(f"Unsupported score composition: {score_composition!r}")
    if model_count != 2:
        raise ValueError(f"two_score scoring needs exactly two models, got {model_count}")
    if score_mode != "hybrid":
        raise ValueError("two_score scoring needs the hybrid fusion to combine the two columns")
    if model_weights is not None:
        raise ValueError(
            "two_score scoring applies no probability-space weights; pass model_weights=None"
        )
    return TWO_SCORE_SOURCE


def _constant_category_mask(
    pair_records: list[dict[str, Any]], constant_names: frozenset[str]
) -> np.ndarray:
    """Flag pairs whose category is blanked to a constant for a leaderboard probe.

    Deliberately permissive about names that do not appear in the input: the
    hidden split may simply not carry one, and aborting there would forfeit a
    submission over a harmless mismatch. Typos are caught at packaging time
    instead, against the category vocabulary the fusion was trained on.

    Blanking everything is still fatal, because that produces a constant
    submission while claiming to be a model run.
    """
    mask = np.fromiter(
        (str(record.get("category")) in constant_names for record in pair_records),
        dtype=bool,
        count=len(pair_records),
    )
    if not constant_names:
        return mask
    if mask.all():
        raise ValueError("Probe would blank every category; nothing would be scored")
    if absent := constant_names - {str(record.get("category")) for record in pair_records}:
        print(
            json.dumps({"event": "probe_categories_absent", "categories": sorted(absent)}),
            file=sys.stderr,
        )
    return mask


def run_inference(
    items_path: str | Path,
    matches_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    fusion_path: str | Path,
    *,
    secondary_model_path: str | Path | None = None,
    model_weights: tuple[float, ...] | None = None,
    max_length: int = 384,
    model_batch_size: int = 256,
    pair_chunk_size: int = 8192,
    workers: int = 16,
    score_mode: str = "hybrid",
    score_composition: str = "mean_probability",
    cuda_precision: str = "fp16_amp",
    constant_categories: Iterable[str] | None = None,
    constant_value: float = 0.5,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if score_mode not in {"hybrid", "text"}:
        raise ValueError(f"Unsupported score mode: {score_mode!r}")
    model_paths = [Path(model_path)]
    if secondary_model_path is not None:
        model_paths.append(Path(secondary_model_path))
    # Resolved before anything expensive loads: a run that cannot honour its own
    # contract should cost a second, not a phase.
    score_source = _resolve_score_source(
        score_composition, len(model_paths), score_mode, model_weights
    )
    # Leaderboard probing: categories listed here receive one constant score, so
    # their average precision collapses to the category's positive rate. The
    # difference against a full submission isolates how much the model actually
    # lifts the remaining categories on the hidden split. Constant pairs skip the
    # cross-encoder entirely, so a half-and-half probe also runs in about half
    # the time.
    constant_names = frozenset(str(name) for name in constant_categories or ())
    if constant_names and not 0.0 < constant_value < 1.0:
        raise ValueError("constant_value must lie in (0, 1)")

    total_started = perf_counter()
    timings: dict[str, float] = {}

    stage_started = perf_counter()
    fusion = FusionPredictor(fusion_path) if score_mode == "hybrid" else None
    needs_transductive = bool(
        fusion
        and any(
            name in fusion.transductive_profile.feature_names
            for name in fusion.schema["feature_names"]
        )
    )
    timings["load_fusion"] = perf_counter() - stage_started

    stage_started = perf_counter()
    matches = pq.read_table(matches_path, columns=["id1", "id2"]).to_pandas()
    timings["read_matches"] = perf_counter() - stage_started

    stage_started = perf_counter()
    accepted_ids = {
        int(value) for column in ("id1", "id2") for value in matches[column].to_numpy()
    }
    timings["accepted_ids"] = perf_counter() - stage_started

    stage_started = perf_counter()
    items, transductive_statistics = _canonicalize_items(
        items_path,
        min(workers, os.cpu_count() or 1),
        4096,
        accepted_ids=accepted_ids,
        collect_statistics=needs_transductive,
        transductive_profile=fusion.transductive_profile.name if fusion else "h12-v1",
    )
    if transductive_statistics is not None:
        annotation_started = perf_counter()
        items = {
            item_id: transductive_statistics.annotate(item) for item_id, item in items.items()
        }
        # Reported separately but already inside canonicalize_items: the corpus
        # sweep no longer has a stage of its own.
        timings["transductive_annotation"] = perf_counter() - annotation_started
    timings["canonicalize_items"] = perf_counter() - stage_started
    missing = (set(matches.id1) | set(matches.id2)) - set(items)
    if missing:
        raise ValueError(f"{len(missing)} pair item IDs are missing")

    stage_started = perf_counter()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # Gold OOF text scores were generated by Accelerate's fp16 autocast with
    # fp32 model weights.  Selecting bf16 merely because the serving GPU can
    # support it changes rankings enough to invalidate a fusion trained on
    # those OOF scores.  Keep that scorer contract explicit here rather than
    # making the numeric path depend on whichever GPU runs the archive.
    model_dtype, autocast_dtype = _cuda_precision_config(torch, device.type, cuda_precision)
    timings["load_models"] = perf_counter() - stage_started

    pair_records: list[dict[str, Any]] = []
    text_pairs: list[tuple[str, str]] = []
    timings.update(
        pair_preparation=0.0,
        tokenization=0.0,
        model_loading=0.0,
        model_forward=0.0,
        fusion=0.0,
    )

    def pair_chunks() -> Iterable[list[tuple[dict[str, Any], dict[str, Any]]]]:
        for start in range(0, len(matches), pair_chunk_size):
            stop = min(start + pair_chunk_size, len(matches))
            yield [
                (items[int(row.id1)], items[int(row.id2)])
                for row in matches.iloc[start:stop].itertuples(index=False)
            ]

    stage_started = perf_counter()
    if workers > 1 and len(matches) >= 20_000:
        with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1)) as pool:
            prepared_chunks = pool.map(_prepare_pair_chunk, pair_chunks(), chunksize=1)
            for records, texts in prepared_chunks:
                pair_records.extend(records)
                text_pairs.extend(texts)
    else:
        for chunk in pair_chunks():
            records, texts = _prepare_pair_chunk(chunk)
            pair_records.extend(records)
            text_pairs.extend(texts)
    timings["pair_preparation"] = perf_counter() - stage_started

    constant_mask = _constant_category_mask(pair_records, constant_names)
    timings["scored_rows"] = float((~constant_mask).sum())
    timings["constant_rows"] = float(constant_mask.sum())
    order = sorted(
        np.flatnonzero(~constant_mask).tolist(),
        key=lambda index: len(text_pairs[index][0]) + len(text_pairs[index][1]),
    )
    stage_started = perf_counter()
    tokenizers = [
        AutoTokenizer.from_pretrained(current_model_path, use_fast=True, local_files_only=True)
        for current_model_path in model_paths
    ]
    timings["load_models"] += perf_counter() - stage_started
    shared_tokenizer = all(
        _tokenizers_compatible(tokenizers[0], tokenizer) for tokenizer in tokenizers[1:]
    )
    score_vectors: list[np.ndarray] = [
        np.zeros(len(text_pairs), dtype=np.float32) for _ in model_paths
    ]

    def load_model(current_model_path: Path) -> Any:
        nonlocal timings
        stage_started = perf_counter()
        model_kwargs: dict[str, Any] = {
            "num_labels": 1,
            "local_files_only": True,
        }
        if model_dtype is not None:
            model_kwargs["torch_dtype"] = model_dtype
        model = AutoModelForSequenceClassification.from_pretrained(
            current_model_path, **model_kwargs
        )
        _disable_modernbert_reference_compile(model)
        model = model.to(device)
        model.eval()
        timings["model_loading"] += perf_counter() - stage_started
        return model

    def encode(indices: list[int], tokenizer: Any) -> dict[str, Any]:
        stage_started = perf_counter()
        encoded = tokenizer(
            [text_pairs[index][0] for index in indices],
            [text_pairs[index][1] for index in indices],
            max_length=max_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        timings["tokenization"] += perf_counter() - stage_started
        return encoded

    def score_batch(
        model: Any, encoded: dict[str, Any], indices: list[int], target: np.ndarray
    ) -> None:
        stage_started = perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"
        ):
            score = torch.sigmoid(model(**encoded).logits.float().view(-1))
        target[indices] = score.cpu().numpy()
        timings["model_forward"] += perf_counter() - stage_started

    if shared_tokenizer:
        models = [load_model(current_model_path) for current_model_path in model_paths]
        with torch.inference_mode():
            for offset in range(0, len(order), model_batch_size):
                indices = order[offset : offset + model_batch_size]
                encoded = encode(indices, tokenizers[0])
                for model, text_scores in zip(models, score_vectors, strict=True):
                    score_batch(model, encoded, indices, text_scores)
        for model_index, model in enumerate(models):
            timings[f"model_{model_index}_rows"] = float(len(score_vectors[model_index]))
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        for model_index, (current_model_path, tokenizer, text_scores) in enumerate(
            zip(model_paths, tokenizers, score_vectors, strict=True)
        ):
            model = load_model(current_model_path)
            with torch.inference_mode():
                for offset in range(0, len(order), model_batch_size):
                    indices = order[offset : offset + model_batch_size]
                    score_batch(model, encode(indices, tokenizer), indices, text_scores)
            timings[f"model_{model_index}_rows"] = float(len(text_scores))
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    combined_scores = (
        None
        if score_source == TWO_SCORE_SOURCE
        else _combine_probability_scores(score_vectors, model_weights)
    )
    if fusion is None:
        predictions = combined_scores.astype(np.float64)
    else:
        if fusion.score_source != score_source:
            raise ValueError(
                "Fusion score source does not match runtime models: "
                f"expected {fusion.score_source}, got {score_source}"
            )
        stage_started = perf_counter()
        if combined_scores is None:
            primary_scores, secondary_scores = score_vectors
            for record, primary, secondary in zip(
                pair_records, primary_scores, secondary_scores, strict=True
            ):
                record["text_score"] = float(primary)
                record["second_score"] = float(secondary)
        else:
            for record, score in zip(pair_records, combined_scores, strict=True):
                record["text_score"] = float(score)
        # This call deliberately sees every pair at once. Batch-category rank
        # must not depend on the mechanical pair_chunk_size used above.
        predictions = fusion.predict(pair_records)
        timings["fusion"] += perf_counter() - stage_started

    if constant_mask.any():
        predictions = np.asarray(predictions, dtype=np.float64).copy()
        predictions[constant_mask] = constant_value

    stage_started = perf_counter()
    output = _write_predictions(matches, predictions, output_path)
    timings["write_output"] = perf_counter() - stage_started
    timings["total"] = perf_counter() - total_started
    result = {
        "rows": len(output),
        "output": str(output_path),
        "device": str(device),
        "cuda_precision": cuda_precision if device.type == "cuda" else "not_applicable",
        "score_mode": score_mode,
        "transductive_features": needs_transductive,
        "model_count": len(model_paths),
        "score_composition": score_composition,
        "score_source": score_source,
        "timings_seconds": {name: round(value, 3) for name, value in timings.items()},
    }
    print(json.dumps({"event": "inference_timing", **result}, ensure_ascii=False), file=sys.stderr)
    return result

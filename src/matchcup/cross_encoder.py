from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from matchcup.backbone_contract import apply_backbone_contract, verify_backbone_contract
from matchcup.checkpointing import (
    TrainingProgress,
    load_latest_progress,
    recipe_fingerprint,
    save_checkpoint,
)
from matchcup.schema_robustness import (
    AttributePermutationSpec,
    raw_text_pairs,
    summarize_training_pairs,
    training_text_pairs,
)


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str
    model_revision: str | None = None
    max_length: int = 384
    train_batch_size: int = 16
    train_categories_per_batch: int = 1
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 1
    ranking_weight: float = 0.0
    # The epoch index from which the ranking term switches on. 1 reproduces the
    # historical hardcode exactly. Setting it above `epochs` keeps the loss pure
    # BCE for the whole run, which is what lets the step budget be raised as a
    # single variable: `--epochs 2` alone would change the budget and the loss
    # at the same time.
    ranking_from_epoch: int = 1
    # Gold ranking batches stay category-local 16-row microgroups.  The
    # prevalence is an immutable recipe input and must be included in resume
    # identity rather than silently inheriting the historical 50/50 sampler.
    positive_prevalence: float = 0.5
    # Accelerate's own vocabulary ("no" | "fp16" | "bf16" | "fp8"); it rejects
    # anything else outright.  Every value here autocasts around fp32 master
    # weights, which the model load pins explicitly rather than inheriting from
    # the checkpoint's declared dtype.
    mixed_precision: str = "fp16"
    num_workers: int = 4
    seed: int = 20260812
    save_best_checkpoint: bool = True
    save_final_checkpoint: bool = False
    save_model_on_completion: bool = True
    train_text_transform: AttributePermutationSpec | None = None
    # A train-text transform normally belongs to an OOF fold, where the
    # unmodified validation collate is an additional guardrail.  The final
    # refit has no validation split, so it must opt in explicitly instead of
    # accidentally augmenting an arbitrary full-data training job.
    allow_full_training_transform: bool = False
    train_text_transform_provenance: dict[str, Any] | None = None


def _require_training_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from accelerate import Accelerator
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError("Install matchcup with the 'train' extra") from exc
    return (
        torch,
        Accelerator,
        DataLoader,
        Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )


class ArrowPairDataset:
    """Memory-mapped Arrow table wrapper suitable for gold and sampled silver."""

    def __init__(self, table: Any, indices: np.ndarray | None = None) -> None:
        self.table = table
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self.table.num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.indices[index]) if self.indices is not None else index
        result = {name: self.table[name][row].as_py() for name in self.table.column_names}
        result["_row_index"] = row
        return result


def _synthetic_mask(table: Any) -> np.ndarray:
    """Return the explicit augmentation marker, defaulting legacy tables to real rows."""
    if "is_synthetic" not in table.column_names:
        return np.zeros(table.num_rows, dtype=bool)
    values = table["is_synthetic"].to_pylist()
    if any(value is None for value in values):
        raise ValueError("is_synthetic must not contain nulls")
    return np.asarray(values, dtype=bool)


def _fold_values(table: Any) -> np.ndarray:
    if "fold" not in table.column_names:
        raise ValueError("fold selection requires a fold column")
    return table["fold"].to_numpy()


def training_indices_for_fold(table: Any, validation_fold: int) -> np.ndarray:
    """Select all non-validation rows, including synthetic rows from other folds.

    Augmentation materialization copies the source fold onto each synthetic
    row.  The ordinary component-disjoint train predicate therefore excludes a
    source component and all of its synthetic descendants together.
    """
    return np.flatnonzero(_fold_values(table) != validation_fold)


def oof_indices_for_fold(table: Any, fold: int) -> np.ndarray:
    """Select only real held-out rows for OOF scoring and validation."""
    return np.flatnonzero((_fold_values(table) == fold) & ~_synthetic_mask(table))


class CategoryBalancedBatchSampler:
    """Sample category-local 16-row microgroups at a fixed positive prevalence.

    The physical training batch can contain several microgroups (for example,
    four 16-row categories in a 64-row batch).  A deterministic floor/ceil
    schedule allocates their positive slots, so a non-integral operating point
    converges to the requested prevalence without random label-count noise.
    """

    microgroup_size = 16

    def __init__(
        self,
        table: Any,
        indices: np.ndarray | None,
        batch_size: int,
        seed: int,
        *,
        positive_prevalence: float = 0.5,
    ) -> None:
        if batch_size < self.microgroup_size or batch_size % self.microgroup_size:
            raise ValueError("CategoryBalancedBatchSampler requires 16-row microgroups")
        if not 0.0 < positive_prevalence < 1.0:
            raise ValueError("positive_prevalence must be in (0, 1)")
        expected_positives = self.microgroup_size * positive_prevalence
        if not 1.0 <= expected_positives <= self.microgroup_size - 1.0:
            raise ValueError(
                "positive_prevalence must leave at least one positive and one negative "
                "per 16-row microgroup"
            )
        self.batch_size = batch_size
        self.seed = seed
        self.positive_prevalence = positive_prevalence
        self.microgroups_per_batch = batch_size // self.microgroup_size
        self.epoch = 0
        # Batches already consumed in the current epoch, set on resume. The full
        # stream is recomputed (so the RNG advances identically) but the leading
        # ``skip_batches`` are not yielded, giving exact within-epoch resume
        # without depending on Accelerate's dataloader-state internals.
        self.skip_batches = 0
        rows = indices if indices is not None else np.arange(table.num_rows)
        categories = table["category"].to_numpy()[rows]
        targets = table["target"].to_numpy()[rows]
        self.groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for category in sorted(set(categories)):
            local = np.flatnonzero(categories == category)
            positive = local[targets[local] > 0.5]
            negative = local[targets[local] <= 0.5]
            if len(positive) and len(negative):
                self.groups[str(category)] = (positive, negative)
        if not self.groups:
            raise ValueError("No category contains both positive and negative rows")
        self.num_rows = len(rows)

    def __len__(self) -> int:
        return math.ceil(self.num_rows / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        skip = self.skip_batches
        self.skip_batches = 0
        categories = list(self.groups)
        rng.shuffle(categories)
        for batch_index in range(len(self)):
            microgroups: list[np.ndarray] = []
            for within_batch in range(self.microgroups_per_batch):
                microgroup_index = batch_index * self.microgroups_per_batch + within_batch
                category = categories[microgroup_index % len(categories)]
                positive, negative = self.groups[category]
                # The difference between cumulative floors is always either the
                # floor or the ceil of the requested count, and it is stable
                # across checkpoint/resume because it depends only on the
                # absolute microgroup position.
                positive_count = int(
                    math.floor(
                        (microgroup_index + 1)
                        * self.microgroup_size
                        * self.positive_prevalence
                    )
                    - math.floor(microgroup_index * self.microgroup_size * self.positive_prevalence)
                )
                negative_count = self.microgroup_size - positive_count
                microgroups.append(
                    np.concatenate(
                        [
                            rng.choice(
                                positive,
                                positive_count,
                                replace=len(positive) < positive_count,
                            ),
                            rng.choice(
                                negative,
                                negative_count,
                                replace=len(negative) < negative_count,
                            ),
                        ]
                    )
                )
            batch = np.concatenate(microgroups)
            rng.shuffle(batch)
            # Recompute the skipped batches to keep the RNG stream identical, but
            # do not hand them back to the training loop.
            if batch_index >= skip:
                yield batch.tolist()


class CategoryUniformBatchSampler:
    def __init__(
        self,
        table: Any,
        indices: np.ndarray | None,
        batch_size: int,
        seed: int,
        *,
        categories_per_batch: int = 1,
    ) -> None:
        if categories_per_batch < 1 or batch_size % categories_per_batch:
            raise ValueError("batch_size must be divisible by categories_per_batch")
        self.batch_size = batch_size
        self.categories_per_batch = categories_per_batch
        self.seed = seed
        self.epoch = 0
        self.skip_batches = 0
        rows = indices if indices is not None else np.arange(table.num_rows)
        categories = table["category"].to_numpy()[rows]
        self.groups = {
            str(category): np.flatnonzero(categories == category)
            for category in sorted(set(categories))
        }
        if categories_per_batch > len(self.groups):
            raise ValueError("categories_per_batch exceeds available categories")
        self.num_rows = len(rows)

    def __len__(self) -> int:
        return math.ceil(self.num_rows / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        skip = self.skip_batches
        self.skip_batches = 0
        categories = list(self.groups)
        rng.shuffle(categories)
        examples_per_category = self.batch_size // self.categories_per_batch
        for batch_index in range(len(self)):
            start = (batch_index * self.categories_per_batch) % len(categories)
            selected_categories = [
                categories[(start + offset) % len(categories)]
                for offset in range(self.categories_per_batch)
            ]
            batch = np.concatenate(
                [
                    rng.choice(
                        self.groups[category],
                        examples_per_category,
                        replace=len(self.groups[category]) < examples_per_category,
                    )
                    for category in selected_categories
                ]
            )
            rng.shuffle(batch)
            if batch_index >= skip:
                yield batch.tolist()


class FixedMicrobatchSampler:
    """Replay one immutable sequence of microbatches, optionally grouped.

    The T6 receipt consumes a single S6-shaped fixture both ways: the control
    receives its 16-row microbatches individually, while the candidate receives
    each four consecutive microbatches as one 64-row batch.  This keeps source
    rows fixed while measuring only the batching change.
    """

    def __init__(self, microbatches: list[list[int]], *, groups_per_batch: int) -> None:
        if groups_per_batch < 1:
            raise ValueError("groups_per_batch must be positive")
        if not microbatches:
            raise ValueError("microbatch fixture must not be empty")
        if len(microbatches) % groups_per_batch:
            raise ValueError("microbatch fixture must divide evenly into grouped batches")
        sizes = {len(batch) for batch in microbatches}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError("microbatch fixture must contain equally sized non-empty batches")
        self.microbatches = tuple(tuple(int(row) for row in batch) for batch in microbatches)
        self.groups_per_batch = groups_per_batch
        self.microbatch_size = sizes.pop()
        self.batch_size = self.microbatch_size * groups_per_batch
        self.epoch = 0
        self.skip_batches = 0

    def __len__(self) -> int:
        return len(self.microbatches) // self.groups_per_batch

    def __iter__(self):
        skip = self.skip_batches
        self.skip_batches = 0
        self.epoch += 1
        starts = range(0, len(self.microbatches), self.groups_per_batch)
        for batch_index, start in enumerate(starts):
            batch = [
                row
                for microbatch in self.microbatches[start : start + self.groups_per_batch]
                for row in microbatch
            ]
            if batch_index >= skip:
                yield batch


def _load_microbatch_fixture(path: str | Path) -> list[list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "matchcup_t6_microbatch_fixture_v1":
        raise ValueError("unsupported microbatch fixture format")
    batches = payload.get("microbatches")
    if not isinstance(batches, list) or not all(isinstance(batch, list) for batch in batches):
        raise ValueError("microbatch fixture is malformed")
    return batches


def _load_table(path: str | Path) -> pa.Table:
    columns = ["id1", "id2", "text_a", "text_b", "target", "category"]
    schema = set(pq.ParquetFile(path).schema.names)
    columns.extend(name for name in ("fold", "row_id", "is_synthetic") if name in schema)
    return pq.read_table(path, columns=columns, memory_map=True)


def _ranking_loss(torch: Any, logits: Any, targets: Any, categories: list[str]) -> Any:
    losses = []
    for category in sorted(set(categories)):
        mask = torch.tensor([value == category for value in categories], device=logits.device)
        positive = logits[mask & (targets > 0.5)]
        negative = logits[mask & (targets <= 0.5)]
        count = min(positive.numel(), negative.numel())
        if count:
            pos = positive[torch.randperm(positive.numel(), device=logits.device)[:count]]
            neg = negative[torch.randperm(negative.numel(), device=logits.device)[:count]]
            losses.append(torch.nn.functional.softplus(-(pos - neg)).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def applied_ranking_weight(config: CrossEncoderConfig, epoch: int) -> float:
    """Return the ranking coefficient selected for one zero-indexed epoch."""
    return config.ranking_weight if epoch >= config.ranking_from_epoch else 0.0


def train_cross_encoder(
    train_path: str | Path,
    output_dir: str | Path,
    config: CrossEncoderConfig,
    *,
    validation_fold: int | None = None,
    hard_negative_path: str | Path | None = None,
    hard_negative_repeats: int = 2,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval_seconds: float = 1200.0,
    checkpoint_sync: Callable[[Path], None] | None = None,
    resume: bool = True,
    max_optimizer_steps: int | None = None,
    batch_fixture_path: str | Path | None = None,
    fixture_groups_per_batch: int = 1,
) -> dict[str, Any]:
    (
        torch,
        Accelerator,
        DataLoader,
        Dataset,
        AutoModel,
        AutoTokenizer,
        get_scheduler,
    ) = _require_training_dependencies()
    del Dataset
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    table = _load_table(train_path)
    if validation_fold is not None:
        indices = training_indices_for_fold(table, validation_fold)
    else:
        indices = None
    if hard_negative_path is not None:
        hard_rows = (
            pq.read_table(hard_negative_path, columns=["row_id"]).column("row_id").to_numpy()
        )
        base = indices if indices is not None else np.arange(table.num_rows, dtype=np.int64)
        indices = np.concatenate([base, np.tile(hard_rows, hard_negative_repeats)])
    transform_summary: dict[str, int] | None = None
    if config.train_text_transform is not None:
        if validation_fold is None and not config.allow_full_training_transform:
            raise ValueError(
                "schema-robustness transform requires a strict validation fold or "
                "an explicitly permitted final refit"
            )
        if "row_id" not in table.column_names:
            raise ValueError("schema-robustness transform requires a stable row_id column")
        selected_indices = (
            indices if indices is not None else np.arange(table.num_rows, dtype=np.int64)
        )
        transform_summary = summarize_training_pairs(
            (
                (
                    table["text_a"][int(index)].as_py(),
                    table["text_b"][int(index)].as_py(),
                    table["row_id"][int(index)].as_py(),
                )
                for index in selected_indices
            ),
            config.train_text_transform,
        )
    dataset = ArrowPairDataset(table, indices)
    revision_kwargs = (
        {"revision": config.model_revision}
        if config.model_revision and not Path(config.model_name).exists()
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        **revision_kwargs,
    )
    model = AutoModel.from_pretrained(
        config.model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        # ``from_pretrained`` defaults to ``dtype="auto"``, which adopts the
        # checkpoint's declared dtype.  mmBERT declares float32, so this pin is
        # a no-op for every recipe measured so far; Qwen3 declares bfloat16, and
        # adopting it would hand Accelerate bf16 master weights under bf16
        # autocast, leaving AdamW to accumulate updates in 8 mantissa bits.
        # ``mixed_precision`` only means what Accelerate means by it when the
        # weights it prepares are fp32.
        dtype=torch.float32,
        **revision_kwargs,
    )
    # Applied before the first forward pass, not only before the save: a decoder
    # backbone without ``config.pad_token_id`` cannot run a 64-row batch at all,
    # and one trained on pair text whose two sides fuse would be fitted to an
    # input shape serving never produces.
    backbone_contract = apply_backbone_contract(model, tokenizer)

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        text_pairs = training_text_pairs(rows, config.train_text_transform)
        encoded = tokenizer(
            [text_a for text_a, _ in text_pairs],
            [text_b for _, text_b in text_pairs],
            max_length=config.max_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
        encoded["categories"] = [row["category"] for row in rows]
        return encoded

    num_workers = min(config.num_workers, os.cpu_count() or 1)
    # Creating a DataLoader iterator draws one value from whatever RNG seeds its
    # ``base_seed``; left as the global torch RNG, that draw lands mid-stream on a
    # resume (a fresh iterator is created after the epoch already advanced the
    # RNG) and shifts every subsequent dropout mask, breaking bit-exact resume.
    # A dedicated generator keeps iterator setup off the global RNG entirely.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed)
    loader_kwargs = {
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "generator": loader_generator,
    }
    # Hold a direct reference to the sampler so resume can restore its epoch. The
    # sampler seeds its RNG from ``seed + epoch`` and self-increments, so the
    # batch stream for training epoch E is fully determined by setting the
    # sampler's epoch to E before iterating — which is exactly what a mid-run
    # restart must reproduce. ``accelerator.prepare`` wraps the loader but keeps
    # this instance, so the reference stays live.
    fixture_sha256 = None
    if batch_fixture_path is not None:
        if indices is not None:
            raise ValueError("batch fixture is only supported for an unfiltered training table")
        fixture_path = Path(batch_fixture_path)
        fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        train_sampler = FixedMicrobatchSampler(
            _load_microbatch_fixture(fixture_path), groups_per_batch=fixture_groups_per_batch
        )
        if train_sampler.batch_size != config.train_batch_size:
            raise ValueError("batch fixture size does not match train_batch_size")
    elif config.ranking_weight > 0:
        train_sampler = CategoryBalancedBatchSampler(
            table,
            indices,
            config.train_batch_size,
            config.seed,
            positive_prevalence=config.positive_prevalence,
        )
    else:
        train_sampler = CategoryUniformBatchSampler(
            table,
            indices,
            config.train_batch_size,
            config.seed,
            categories_per_batch=config.train_categories_per_batch,
        )
    loader = DataLoader(dataset, batch_sampler=train_sampler, **loader_kwargs)
    validation_loader = None
    category_names = sorted(set(table["category"].to_pylist()))
    category_to_id = {name: index for index, name in enumerate(category_names)}
    if validation_fold is not None:
        validation_indices = oof_indices_for_fold(table, validation_fold)
        validation_dataset = ArrowPairDataset(table, validation_indices)

        def validation_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
            text_pairs = raw_text_pairs(rows)
            encoded = tokenizer(
                [text_a for text_a, _ in text_pairs],
                [text_b for _, text_b in text_pairs],
                max_length=config.max_length,
                truncation=True,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            encoded["eval_target"] = torch.tensor(
                [row["target"] for row in rows], dtype=torch.float32
            )
            encoded["eval_category"] = torch.tensor(
                [category_to_id[row["category"]] for row in rows], dtype=torch.long
            )
            return encoded

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=validation_collate,
            num_workers=min(2, num_workers),
            pin_memory=True,
        )
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=config.learning_rate)
    if max_optimizer_steps is not None and max_optimizer_steps < 1:
        raise ValueError("max_optimizer_steps must be positive")
    steps_per_epoch = math.ceil(
        len(loader) / (config.gradient_accumulation_steps * accelerator.num_processes)
    )
    planned_total_steps = steps_per_epoch * config.epochs
    total_steps = min(planned_total_steps, max_optimizer_steps or planned_total_steps)
    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    if validation_loader is None:
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )
    else:
        model, optimizer, loader, validation_loader, scheduler = accelerator.prepare(
            model, optimizer, loader, validation_loader, scheduler
        )
    completed = 0
    best_macro_ap = -1.0
    validation_metrics: list[dict[str, float | int]] = []
    output_dir = Path(output_dir)

    def save_model() -> None:
        nonlocal backbone_contract
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            # Re-applied here rather than trusted from construction: this is the
            # only moment the contract can still reach the files an archive is
            # built from, and ``accelerator.prepare`` returns a wrapper whose
            # config is not guaranteed to be the object mutated above.
            backbone_contract = apply_backbone_contract(unwrapped, tokenizer)
            unwrapped.save_pretrained(
                output_dir, save_function=accelerator.save, safe_serialization=True
            )
            tokenizer.save_pretrained(output_dir)

    # Resume identity: any change here (data, batch/world size, step budget)
    # makes an old optimizer/scheduler state unsafe to continue, so the loader
    # refuses a checkpoint whose fingerprint disagrees.
    fingerprint = recipe_fingerprint(
        {
            "train_path": str(train_path),
            "validation_fold": validation_fold,
            "hard_negative_path": str(hard_negative_path) if hard_negative_path else None,
            "hard_negative_repeats": hard_negative_repeats,
            "total_steps": total_steps,
            "world_size": accelerator.num_processes,
            "batch_size": config.train_batch_size,
            "grad_accum": config.gradient_accumulation_steps,
            "epochs": config.epochs,
            "ranking_from_epoch": config.ranking_from_epoch,
            "positive_prevalence": config.positive_prevalence,
            "max_optimizer_steps": max_optimizer_steps,
            "batch_fixture_sha256": fixture_sha256,
            "fixture_groups_per_batch": fixture_groups_per_batch if batch_fixture_path else None,
            "seed": config.seed,
            "model_name": config.model_name,
            "max_length": config.max_length,
            # Optimizer state carries the numerics it was accumulated under, so a
            # preempted bf16 run that resumed from an fp16 checkpoint would keep
            # going with silently wrong moments. Over a 49-hour Spot epoch that is
            # a likely enough sequence to be worth one field.
            "mixed_precision": config.mixed_precision,
        }
    )
    start_epoch = 0
    resume_micro = 0
    if checkpoint_dir is not None and resume:
        # ``accelerator.load_state`` restores the prepared dataloader's position
        # within the epoch on its own, so within-epoch fast-forward is Accelerate's
        # job, not ours. What Accelerate cannot know is our custom sampler's epoch,
        # which seeds the batch RNG (``seed + epoch``); restore it by hand so the
        # resumed epoch replays the identical stream.
        restored = load_latest_progress(
            accelerator.load_state, checkpoint_dir, expected_fingerprint=fingerprint
        )
        if restored is not None:
            if restored.total_steps != total_steps:
                raise ValueError(
                    "Checkpoint step budget does not match this run; refusing to resume"
                )
            completed = restored.completed
            best_macro_ap = restored.best_macro_ap
            validation_metrics = list(restored.validation_metrics)
            start_epoch = restored.epoch
            resume_micro = restored.micro_in_epoch
            train_sampler.epoch = restored.epoch
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "event": "resume",
                            "from_step": completed,
                            "epoch": start_epoch,
                            "micro_in_epoch": resume_micro,
                        }
                    ),
                    flush=True,
                )

    last_checkpoint_at = time.monotonic()

    def maybe_checkpoint(epoch: int, micro_in_epoch: int, *, force: bool = False) -> None:
        nonlocal last_checkpoint_at
        if checkpoint_dir is None:
            return
        now = time.monotonic()
        if not force and now - last_checkpoint_at < checkpoint_interval_seconds:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                accelerator.save_state,
                checkpoint_dir,
                TrainingProgress(
                    completed=completed,
                    epoch=epoch,
                    micro_in_epoch=micro_in_epoch,
                    best_macro_ap=best_macro_ap,
                    total_steps=total_steps,
                    recipe_fingerprint=fingerprint,
                    validation_metrics=validation_metrics,
                ),
                sync=checkpoint_sync,
            )
            print(json.dumps({"event": "checkpoint", "step": completed}), flush=True)
        accelerator.wait_for_everyone()
        last_checkpoint_at = time.monotonic()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_started = time.monotonic()
    stop_training = False
    model.train()
    for epoch in range(start_epoch, config.epochs):
        ranking_weight = applied_ranking_weight(config, epoch)
        train_sampler.epoch = epoch
        # On the resumed epoch, replay the sampler stream but drop the batches
        # already consumed before the checkpoint, so training continues from the
        # exact position. ``micro_in_epoch`` starts from that count to keep our
        # checkpoint bookkeeping consistent.
        if epoch == start_epoch and resume_micro > 0:
            train_sampler.skip_batches = resume_micro
            micro_in_epoch = resume_micro
        else:
            micro_in_epoch = 0
        for batch in loader:
            categories = batch.pop("categories")
            targets = batch.pop("labels")
            with accelerator.accumulate(model):
                logits = model(**batch).logits.float().view(-1)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                ranking = _ranking_loss(torch, logits, targets, categories)
                loss = bce + ranking_weight * ranking
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            micro_in_epoch += 1
            if accelerator.sync_gradients:
                completed += 1
                maybe_checkpoint(epoch, micro_in_epoch)
            if (
                accelerator.is_main_process
                and accelerator.sync_gradients
                and completed % 100 == 0
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": completed,
                            "loss": float(loss),
                            "ranking_weight": ranking_weight,
                        }
                    ),
                    flush=True,
                )
            if completed >= total_steps:
                stop_training = True
                break
        resume_micro = 0
        if stop_training:
            break
        if validation_loader is not None:
            model.eval()
            scores: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            category_ids: list[np.ndarray] = []
            with torch.inference_mode():
                for batch in validation_loader:
                    target = batch.pop("eval_target")
                    category_id = batch.pop("eval_category")
                    probability = torch.sigmoid(model(**batch).logits.float().view(-1))
                    scores.append(accelerator.gather_for_metrics(probability).cpu().numpy())
                    targets.append(accelerator.gather_for_metrics(target).cpu().numpy())
                    category_ids.append(accelerator.gather_for_metrics(category_id).cpu().numpy())
            if accelerator.is_main_process:
                from matchcup.metrics import macro_average_precision

                all_category_ids = np.concatenate(category_ids)
                names = [category_names[int(value)] for value in all_category_ids]
                macro_ap, _ = macro_average_precision(
                    np.concatenate(targets), np.concatenate(scores), names
                )
                print(json.dumps({"epoch": epoch, "validation_macro_ap": macro_ap}), flush=True)
                validation_metrics.append({"epoch": epoch, "macro_ap": float(macro_ap)})
                if config.save_best_checkpoint and macro_ap > best_macro_ap:
                    best_macro_ap = macro_ap
                    save_model()
            accelerator.wait_for_everyone()
            model.train()
    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.monotonic() - training_started
    if validation_loader is None and config.save_model_on_completion:
        save_model()
    elif config.save_final_checkpoint:
        save_model()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "matchcup_training.json").write_text(
            json.dumps(
                {
                    **asdict(config),
                    "steps": completed,
                    "training_seconds": training_seconds,
                    "best_macro_ap": best_macro_ap,
                    "validation_metrics": validation_metrics,
                    "train_text_transform_summary": transform_summary,
                    "schema_robustness": config.train_text_transform_provenance,
                    # Carried alongside the recipe so a later gate can assert
                    # ``verify_backbone_contract(output_dir) == this`` without
                    # re-deriving what the backbone was supposed to promise.
                    "backbone_contract": backbone_contract,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return {
        "rows": len(dataset),
        "steps": completed,
        "training_seconds": training_seconds,
        "output_dir": str(output_dir),
        "validation_metrics": validation_metrics,
        "train_text_transform_summary": transform_summary,
    }


# The two halves of every precision inference.py can serve, keyed by the name the
# backbone contract records.  Autocast alone does not pin the weights, so both
# have to travel together or OOF and serving diverge silently.
_SCORING_PRECISION = {
    "fp16_amp": ("float32", "fp16"),
    "fp16_weights": ("float16", "fp16"),
    "bf16_weights": ("bfloat16", "bf16"),
}


def score_cross_encoder(
    data_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    max_length: int = 384,
    batch_size: int = 64,
    mixed_precision: str = "fp16",
    fold: int | None = None,
    num_workers: int = 2,
) -> dict[str, int]:
    torch, Accelerator, DataLoader, Dataset, AutoModel, AutoTokenizer, _ = (
        _require_training_dependencies()
    )
    del Dataset
    # OOF scores are the ranking the fusion is fitted on, and the archive has to
    # reproduce that ranking (invariant 10).  Serving derives its precision from
    # the backbone, so scoring must read the same contract instead of trusting
    # whatever the caller passed: the default would score a bf16-native Qwen as
    # fp16 autocast over bf16 weights, which is neither precision inference.py
    # knows how to serve, and the mismatch would be invisible until Public.
    contract = verify_backbone_contract(model_path)
    cuda_precision = str(contract["cuda_precision"])
    weight_dtype_name, mixed_precision = _SCORING_PRECISION[cuda_precision]
    accelerator = Accelerator(mixed_precision=mixed_precision)
    table = _load_table(data_path)
    indices = None
    if fold is not None:
        indices = oof_indices_for_fold(table, fold)
    dataset = ArrowPairDataset(table, indices)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(
        model_path, num_labels=1, dtype=getattr(torch, weight_dtype_name)
    )

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        text_pairs = raw_text_pairs(rows)
        encoded = tokenizer(
            [text_a for text_a, _ in text_pairs],
            [text_b for _, text_b in text_pairs],
            max_length=max_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded["meta_row"] = torch.tensor([row["_row_index"] for row in rows], dtype=torch.long)
        encoded["meta_id1"] = torch.tensor([row["id1"] for row in rows], dtype=torch.long)
        encoded["meta_id2"] = torch.tensor([row["id2"] for row in rows], dtype=torch.long)
        encoded["meta_target"] = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
        encoded["meta_fold"] = torch.tensor(
            [row.get("fold", -1) for row in rows], dtype=torch.long
        )
        return encoded

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )
    model, loader = accelerator.prepare(model, loader)
    model.eval()
    local: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            metadata = {key: batch.pop(key) for key in list(batch) if key.startswith("meta_")}
            logits = model(**batch).logits.float().view(-1)
            probabilities = torch.sigmoid(logits)
            gathered = {
                key: accelerator.gather_for_metrics(value).cpu().numpy()
                for key, value in {**metadata, "text_score": probabilities}.items()
            }
            if accelerator.is_main_process:
                for index in range(len(gathered["text_score"])):
                    local.append(
                        {
                            "row_index": int(gathered["meta_row"][index]),
                            "id1": int(gathered["meta_id1"][index]),
                            "id2": int(gathered["meta_id2"][index]),
                            "target": float(gathered["meta_target"][index]),
                            "fold": int(gathered["meta_fold"][index]),
                            "text_score": float(gathered["text_score"][index]),
                        }
                    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        local.sort(key=lambda row: row["row_index"])
        pq.write_table(pa.Table.from_pylist(local), output_path, compression="zstd")
    return {"rows": len(local), "cuda_precision": cuda_precision}

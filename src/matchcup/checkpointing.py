"""Preemption-durable training checkpoints.

A Spot A100 can be reclaimed at any moment, and the Silver pretrain is a single
19-hour epoch, so a job that only saves at epoch boundaries loses everything on
the first preemption. This module lets the training loop drop a resumable
checkpoint every few minutes and pick up from the exact optimizer step after a
restart.

It is deliberately cloud-agnostic and torch-agnostic: the caller passes bound
``accelerator.save_state`` / ``load_state`` callables and, optionally, a ``sync``
callback that mirrors a finished checkpoint to durable storage (GCS), since the
Spot VM's local disk dies with the instance. Keeping this layer free of torch and
gcloud is what makes the resume logic unit-testable without a GPU or a bucket.

A checkpoint is a directory ``step_<N>`` holding whatever ``save_state`` wrote
plus a ``training_state.json`` sidecar and a ``_CHECKPOINT_COMPLETE`` marker
written last. Only directories carrying that marker are ever loaded, so a restart
that interrupts a half-written checkpoint silently falls back to the previous
good one — which is why the last two are always retained.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

CHECKPOINT_PREFIX = "step_"
IN_PROGRESS_SUFFIX = ".inprogress"
COMPLETE_MARKER = "_CHECKPOINT_COMPLETE"
SIDECAR_NAME = "training_state.json"


@dataclass(frozen=True)
class TrainingProgress:
    """Everything the training loop needs to resume that ``save_state`` omits.

    ``save_state`` covers model, optimizer, scheduler, gradient scaler and RNG.
    It does not cover our own loop counters or where we were in the data stream,
    so those live here.
    """

    completed: int
    epoch: int
    micro_in_epoch: int
    best_macro_ap: float
    total_steps: int
    recipe_fingerprint: str
    validation_metrics: list[dict[str, float | int]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> TrainingProgress:
        payload = json.loads(text)
        return cls(
            completed=int(payload["completed"]),
            epoch=int(payload["epoch"]),
            micro_in_epoch=int(payload["micro_in_epoch"]),
            best_macro_ap=float(payload["best_macro_ap"]),
            total_steps=int(payload["total_steps"]),
            recipe_fingerprint=str(payload["recipe_fingerprint"]),
            validation_metrics=list(payload["validation_metrics"]),
        )


def recipe_fingerprint(identity: Mapping[str, object]) -> str:
    """Stable hash of the fields that must match for a resume to be sound.

    Resuming into a different recipe (other data, batch size, world size, …)
    would silently corrupt the run, so the loader refuses a checkpoint whose
    fingerprint disagrees. Ordering is normalised so the hash is deterministic.
    """
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _step_of(path: Path) -> int:
    return int(path.name[len(CHECKPOINT_PREFIX) :])


def _is_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.startswith(CHECKPOINT_PREFIX)
        and not path.name.endswith(IN_PROGRESS_SUFFIX)
        and (path / COMPLETE_MARKER).is_file()
        and (path / SIDECAR_NAME).is_file()
    )


def complete_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """Complete checkpoints under ``checkpoint_dir``, oldest step first."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return []
    found = []
    for child in root.iterdir():
        if _is_complete(child):
            try:
                _step_of(child)
            except ValueError:
                continue
            found.append(child)
    return sorted(found, key=_step_of)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Newest complete checkpoint, or ``None`` if there is nothing to resume."""
    checkpoints = complete_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    save_state: Callable[[str], object],
    checkpoint_dir: str | Path,
    progress: TrainingProgress,
    *,
    keep_last: int = 2,
    sync: Callable[[Path], None] | None = None,
) -> Path:
    """Write one resumable checkpoint atomically and prune old ones.

    ``save_state`` is ``accelerator.save_state``; it is called with a temporary
    directory so a preemption mid-write never leaves a half-finished checkpoint
    where the loader can see it. The ``_CHECKPOINT_COMPLETE`` marker is the last
    file written, and only marked directories are ever loaded or synced.
    """
    if keep_last < 1:
        raise ValueError("keep_last must retain at least one checkpoint")
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"{CHECKPOINT_PREFIX}{progress.completed}"
    staging = root / f"{CHECKPOINT_PREFIX}{progress.completed}{IN_PROGRESS_SUFFIX}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    save_state(str(staging))
    (staging / SIDECAR_NAME).write_text(progress.to_json(), encoding="utf-8")
    # Marker last: its presence is the definition of "complete".
    (staging / COMPLETE_MARKER).write_text("", encoding="utf-8")

    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)

    _prune(root, keep_last)
    if sync is not None:
        sync(final)
    return final


def _prune(root: Path, keep_last: int) -> None:
    # Drop stale staging dirs from earlier interrupted writes, then keep only the
    # newest ``keep_last`` complete checkpoints so durable storage stays bounded.
    for child in root.iterdir():
        if child.is_dir() and child.name.endswith(IN_PROGRESS_SUFFIX):
            shutil.rmtree(child, ignore_errors=True)
    checkpoints = complete_checkpoints(root)
    for stale in checkpoints[:-keep_last]:
        shutil.rmtree(stale, ignore_errors=True)


def load_latest_progress(
    load_state: Callable[[str], object],
    checkpoint_dir: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> TrainingProgress | None:
    """Restore state from the newest complete checkpoint, or return ``None``.

    Refuses to resume across a recipe change: if ``expected_fingerprint`` is
    given and disagrees with the checkpoint, this raises rather than silently
    continuing a different run on top of stale optimizer state.
    """
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        return None
    progress = TrainingProgress.from_json((latest / SIDECAR_NAME).read_text(encoding="utf-8"))
    if expected_fingerprint is not None and progress.recipe_fingerprint != expected_fingerprint:
        raise ValueError(
            "Refusing to resume: checkpoint recipe fingerprint "
            f"{progress.recipe_fingerprint[:12]} does not match the current recipe "
            f"{expected_fingerprint[:12]}. Point checkpoint_dir at a fresh location "
            "or clear it to start over."
        )
    load_state(str(latest))
    return progress

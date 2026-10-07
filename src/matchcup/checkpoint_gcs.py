"""Mirror training checkpoints to GCS so they survive a Spot preemption.

A Spot VM's local disk dies with the instance, so a resumable checkpoint is only
durable once it is in the bucket. This wires ``cross_encoder``'s
``checkpoint_sync`` callback to ``gcloud storage`` and pulls the latest
checkpoint back on startup.

Two properties matter and neither is free:

*Completeness must survive the transfer.* Locally the ``_CHECKPOINT_COMPLETE``
marker is written last, so its presence proves the checkpoint is whole. A plain
recursive copy does not preserve that ordering - the small marker lands in the
bucket well before a 2.5 GB optimizer state - so this module uploads the payload
first and the marker afterwards, as a separate call. Otherwise a preemption
mid-upload leaves a remote checkpoint that advertises itself as complete while
missing the weights, and every restart crash-loops on it.

*The bucket must not grow without bound.* A 19-hour run at one checkpoint every
20 minutes is ~57 checkpoints of ~3.4 GB. Keeping them all would cost ~196 GB and
a resume would have to download more than the boot disk holds, so old remote
checkpoints are pruned and a resume fetches only the newest one.

The gcloud calls go through an injectable ``runner`` so the command shapes are
unit-testable without a bucket.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from matchcup.checkpointing import CHECKPOINT_PREFIX, COMPLETE_MARKER

Runner = Callable[[Sequence[str]], None]
Lister = Callable[[str], list[str]]


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _list(uri: str) -> list[str]:
    result = subprocess.run(
        ["gcloud", "storage", "ls", uri], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _normalise_prefix(gcs_prefix: str) -> str:
    if not gcs_prefix.startswith("gs://"):
        raise ValueError(f"Checkpoint GCS prefix must be a gs:// URI, got {gcs_prefix!r}")
    return gcs_prefix.rstrip("/")


def _step_of(uri: str) -> int:
    name = uri.rstrip("/").rsplit("/", 1)[-1]
    return int(name[len(CHECKPOINT_PREFIX) :])


def remote_checkpoints(gcs_prefix: str, *, lister: Lister = _list) -> list[str]:
    """Remote checkpoint directories, oldest step first."""
    found = []
    for uri in lister(f"{_normalise_prefix(gcs_prefix)}/"):
        name = uri.rstrip("/").rsplit("/", 1)[-1]
        if not name.startswith(CHECKPOINT_PREFIX):
            continue
        try:
            _step_of(uri)
        except ValueError:
            continue
        found.append(uri.rstrip("/"))
    return sorted(found, key=_step_of)


def upload_checkpoint(
    local_dir: str | Path, gcs_prefix: str, *, runner: Runner = _run
) -> str:
    """Mirror one checkpoint, uploading the completion marker strictly last.

    Ordering is the whole point: a remote checkpoint carrying the marker must be
    known to carry the weights too, because that is the only thing a restart can
    check before trusting it.
    """
    local_dir = Path(local_dir)
    destination = f"{_normalise_prefix(gcs_prefix)}/{local_dir.name}"
    payload = sorted(
        path for path in local_dir.iterdir() if path.is_file() and path.name != COMPLETE_MARKER
    )
    if not payload:
        raise ValueError(f"Refusing to upload an empty checkpoint: {local_dir}")
    for path in payload:
        runner(["gcloud", "storage", "cp", str(path), f"{destination}/{path.name}"])
    marker = local_dir / COMPLETE_MARKER
    if marker.is_file():
        runner(["gcloud", "storage", "cp", str(marker), f"{destination}/{COMPLETE_MARKER}"])
    return destination


def prune_remote(
    gcs_prefix: str, *, keep_last: int = 2, runner: Runner = _run, lister: Lister = _list
) -> list[str]:
    """Delete all but the newest ``keep_last`` remote checkpoints.

    Without this the bucket accumulates every checkpoint of the run and a resume
    would have to download far more than the boot disk holds.
    """
    if keep_last < 1:
        raise ValueError("keep_last must retain at least one checkpoint")
    checkpoints = remote_checkpoints(gcs_prefix, lister=lister)
    stale = checkpoints[:-keep_last]
    for uri in stale:
        runner(["gcloud", "storage", "rm", "--recursive", uri])
    return stale


def make_sync(
    gcs_prefix: str, *, keep_last: int = 2, runner: Runner = _run, lister: Lister = _list
) -> Callable[[Path], None]:
    """Return a ``checkpoint_sync`` callback that mirrors then prunes."""
    prefix = _normalise_prefix(gcs_prefix)

    def sync(path: Path) -> None:
        upload_checkpoint(path, prefix, runner=runner)
        prune_remote(prefix, keep_last=keep_last, runner=runner, lister=lister)

    return sync


def pull_latest(
    gcs_prefix: str,
    local_dir: str | Path,
    *,
    runner: Runner = _run,
    lister: Lister = _list,
) -> str | None:
    """Download only the newest remote checkpoint into ``local_dir``.

    Returns the URI fetched, or ``None`` on a first run with an empty prefix -
    which is not an error, just nothing to resume from.
    """
    prefix = _normalise_prefix(gcs_prefix)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = remote_checkpoints(prefix, lister=lister)
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    try:
        runner(["gcloud", "storage", "cp", "--recursive", latest, str(local_dir)])
    except subprocess.CalledProcessError:
        return None
    return latest

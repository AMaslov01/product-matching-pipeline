from __future__ import annotations

from pathlib import Path

import pytest

from matchcup.checkpointing import (
    COMPLETE_MARKER,
    SIDECAR_NAME,
    TrainingProgress,
    find_latest_checkpoint,
    load_latest_progress,
    recipe_fingerprint,
    save_checkpoint,
)


def _progress(step: int, fingerprint: str = "fp") -> TrainingProgress:
    return TrainingProgress(
        completed=step,
        epoch=0,
        micro_in_epoch=step * 4,
        best_macro_ap=-1.0,
        total_steps=1000,
        recipe_fingerprint=fingerprint,
        validation_metrics=[],
    )


def _fake_save_state(payload: str):
    """Mimic accelerator.save_state: drop an opaque file into the given dir."""

    def save(directory: str) -> None:
        Path(directory, "model.bin").write_text(payload, encoding="utf-8")

    return save


def test_progress_round_trips_through_json():
    progress = TrainingProgress(
        completed=17, epoch=1, micro_in_epoch=68, best_macro_ap=0.42,
        total_steps=100, recipe_fingerprint="abc",
        validation_metrics=[{"epoch": 0, "macro_ap": 0.4}],
    )
    assert TrainingProgress.from_json(progress.to_json()) == progress


def test_fingerprint_is_order_independent_and_sensitive():
    a = recipe_fingerprint({"lr": 8e-6, "batch": 16, "data": "silver"})
    b = recipe_fingerprint({"batch": 16, "data": "silver", "lr": 8e-6})
    assert a == b
    assert a != recipe_fingerprint({"lr": 8e-6, "batch": 32, "data": "silver"})


def test_save_and_load_round_trip(tmp_path: Path):
    save_checkpoint(_fake_save_state("weights@10"), tmp_path, _progress(10))
    loaded_payload = {}

    def load(directory: str) -> None:
        loaded_payload["text"] = Path(directory, "model.bin").read_text(encoding="utf-8")

    progress = load_latest_progress(load, tmp_path)
    assert progress is not None
    assert progress.completed == 10
    assert loaded_payload["text"] == "weights@10"


def test_latest_checkpoint_wins(tmp_path: Path):
    for step in (5, 10, 15):
        save_checkpoint(_fake_save_state(f"w{step}"), tmp_path, _progress(step))
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "step_15"


def test_keep_last_prunes_old_checkpoints(tmp_path: Path):
    for step in (1, 2, 3, 4):
        save_checkpoint(_fake_save_state(f"w{step}"), tmp_path, _progress(step), keep_last=2)
    remaining = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("step_"))
    assert remaining == ["step_3", "step_4"]


def test_incomplete_checkpoint_is_never_loaded(tmp_path: Path):
    """A checkpoint missing its completion marker must be ignored, not resumed."""
    save_checkpoint(_fake_save_state("good"), tmp_path, _progress(10))
    # Simulate a preemption during the next write: staged files exist, no marker.
    broken = tmp_path / "step_20"
    broken.mkdir()
    (broken / "model.bin").write_text("torn", encoding="utf-8")
    (broken / SIDECAR_NAME).write_text(_progress(20).to_json(), encoding="utf-8")

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "step_10"


def test_marker_is_written_last(tmp_path: Path):
    """If save_state raises, no complete checkpoint may appear."""

    def failing_save(_directory: str) -> None:
        raise RuntimeError("preempted mid-save")

    with pytest.raises(RuntimeError):
        save_checkpoint(failing_save, tmp_path, _progress(10))
    assert find_latest_checkpoint(tmp_path) is None


def test_sync_receives_completed_checkpoint(tmp_path: Path):
    synced: list[Path] = []

    def sync(path: Path) -> None:
        # By the time sync runs, the marker must already be present.
        assert (path / COMPLETE_MARKER).is_file()
        synced.append(path)

    save_checkpoint(_fake_save_state("w"), tmp_path, _progress(10), sync=sync)
    assert [p.name for p in synced] == ["step_10"]


def test_load_refuses_recipe_mismatch(tmp_path: Path):
    save_checkpoint(_fake_save_state("w"), tmp_path, _progress(10, fingerprint="recipe-a"))

    def load(_directory: str) -> None:
        raise AssertionError("must not load state on a fingerprint mismatch")

    with pytest.raises(ValueError, match="Refusing to resume"):
        load_latest_progress(load, tmp_path, expected_fingerprint="recipe-b")


def test_load_returns_none_when_nothing_to_resume(tmp_path: Path):
    assert load_latest_progress(lambda _d: None, tmp_path) is None
    assert find_latest_checkpoint(tmp_path) is None

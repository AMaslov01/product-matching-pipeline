from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from matchcup.oof_gate import assess_oof_gate, load_oof

CATEGORIES = ["a", "b", "c"]
PER_CATEGORY = 400


def _gold(tmp_path: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = {"id1": [], "id2": [], "target": [], "category": []}
    for index, category in enumerate(CATEGORIES):
        for row in range(PER_CATEGORY):
            rows["id1"].append(index * 10_000 + row)
            rows["id2"].append(index * 10_000 + row + 500_000)
            rows["target"].append(float(rng.random() < 0.25))
            rows["category"].append(category)
    path = tmp_path / "gold.parquet"
    pq.write_table(pa.table(rows), path)
    return path, rows


def _oof(tmp_path: Path, name: str, rows, signal: float, seed: int):
    """Scores correlated with the label; higher ``signal`` ranks better."""
    rng = np.random.default_rng(seed)
    target = np.asarray(rows["target"])
    score = signal * target + rng.normal(0, 1.0, len(target))
    path = tmp_path / f"{name}.parquet"
    pq.write_table(
        pa.table({"id1": rows["id1"], "id2": rows["id2"], "text_score": score.tolist()}), path
    )
    return path


def test_a_clearly_better_candidate_passes(tmp_path: Path):
    gold, rows = _gold(tmp_path)
    control = _oof(tmp_path, "control", rows, signal=0.6, seed=1)
    candidate = _oof(tmp_path, "candidate", rows, signal=2.2, seed=1)

    result = assess_oof_gate(gold, control, candidate, samples=40)

    assert result.passed, result.reasons
    assert result.verdict == "improvement"
    assert result.difference > 0.002
    assert result.ci95_low > 0
    assert result.bootstrap_positive_rate == pytest.approx(1.0)


def test_a_flat_candidate_is_inconclusive_not_blocked(tmp_path: Path):
    """Gold OOF cannot see the item coverage full Silver buys, and the new-item
    diagnostic is now inside training, so a flat Gold delta must not veto a
    submission - Public is the only instrument left that can answer it."""
    gold, rows = _gold(tmp_path)
    control = _oof(tmp_path, "control", rows, signal=1.0, seed=7)
    candidate = _oof(tmp_path, "candidate", rows, signal=1.0, seed=7)

    result = assess_oof_gate(gold, control, candidate, samples=40)

    assert result.verdict == "inconclusive"
    assert result.passed, "a flat result must still be packageable"
    assert not result.reasons
    assert "submit" in result.recommendation.lower()
    assert result.difference == pytest.approx(0.0, abs=1e-12)


def test_a_real_regression_is_blocked(tmp_path: Path):
    gold, rows = _gold(tmp_path)
    control = _oof(tmp_path, "control", rows, signal=2.4, seed=4)
    candidate = _oof(tmp_path, "candidate", rows, signal=0.2, seed=4)

    result = assess_oof_gate(gold, control, candidate, samples=40)

    assert result.verdict == "regression"
    assert not result.passed
    assert result.reasons
    assert "not package" in result.recommendation.lower()


def test_operating_point_is_reported_but_does_not_gate(tmp_path: Path):
    """H15: the operating-point metric informs, it does not decide."""
    gold, rows = _gold(tmp_path)
    control = _oof(tmp_path, "control", rows, signal=0.6, seed=3)
    candidate = _oof(tmp_path, "candidate", rows, signal=2.2, seed=3)

    result = assess_oof_gate(gold, control, candidate, samples=40)

    # Reported, and on a different scale from the Gold-prevalence number.
    assert result.control_at_operating_point < result.control_macro_ap
    assert result.candidate_at_operating_point < result.candidate_macro_ap
    # The verdict is explained only by the Gold-prevalence criteria.
    assert result.passed
    assert all("operating" not in reason for reason in result.reasons)
    assert result.verdict in {"improvement", "inconclusive"}


def test_per_category_worst_drop_is_surfaced(tmp_path: Path):
    gold, rows = _gold(tmp_path)
    control = _oof(tmp_path, "control", rows, signal=1.2, seed=5)
    candidate = _oof(tmp_path, "candidate", rows, signal=1.2, seed=99)

    result = assess_oof_gate(gold, control, candidate, samples=40)

    assert set(result.per_category) == set(CATEGORIES)
    assert result.worst_category in CATEGORIES
    assert result.worst_category_difference == min(
        record["difference"] for record in result.per_category.values()
    )


def test_partial_oof_coverage_is_rejected(tmp_path: Path):
    """A recipe compared on a subset of Gold is not a paired comparison."""
    gold, rows = _gold(tmp_path)
    control = _oof(tmp_path, "control", rows, signal=1.0, seed=1)
    short = {key: value[:-10] for key, value in rows.items()}
    candidate = _oof(tmp_path, "candidate", short, signal=1.0, seed=1)

    with pytest.raises(ValueError, match="every Gold row"):
        assess_oof_gate(gold, control, candidate, samples=20)


def test_non_finite_scores_are_rejected(tmp_path: Path):
    path = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table({"id1": [1, 2], "id2": [3, 4], "text_score": [0.5, float("nan")]}), path
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_oof(path)


def test_load_oof_concatenates_a_fold_directory(tmp_path: Path):
    directory = tmp_path / "oof"
    directory.mkdir()
    for fold in range(5):
        pq.write_table(
            pa.table({"id1": [fold], "id2": [fold + 100], "text_score": [0.5]}),
            directory / f"oof_fold_{fold}.parquet",
        )
    assert len(load_oof(directory)) == 5


def test_gate_is_reproducible_for_a_seed(tmp_path: Path):
    gold, rows = _gold(tmp_path)
    control = _oof(tmp_path, "control", rows, signal=0.8, seed=2)
    candidate = _oof(tmp_path, "candidate", rows, signal=1.6, seed=2)

    first = assess_oof_gate(gold, control, candidate, samples=30, seed=11)
    second = assess_oof_gate(gold, control, candidate, samples=30, seed=11)

    assert first.ci95_low == pytest.approx(second.ci95_low)
    assert first.difference_at_operating_point == pytest.approx(
        second.difference_at_operating_point
    )

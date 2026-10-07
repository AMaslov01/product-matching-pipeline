import numpy as np
import pytest

from matchcup.cascade import SCORE_SOURCE, CascadeGate, calibrate_threshold


def test_selection_depends_only_on_a_row_own_score() -> None:
    """The gate must not be a quantile: candidate A died on batch-relative ranking."""
    gate = CascadeGate(threshold=0.5, max_fraction=1.0)
    scores = np.array([0.9, 0.4, 0.6, 0.51, 0.49])

    selected = gate.select(scores)
    padded = gate.select(np.concatenate([scores, np.full(1000, 0.99)]))

    assert sorted(selected.tolist()) == [0, 2, 3]
    # The same five rows keep the same verdict even when the batch around them
    # is replaced by a thousand higher-scoring ones.
    assert sorted(index for index in padded.tolist() if index < scores.size) == [0, 2, 3]


def test_cap_is_a_pure_function_of_pair_count() -> None:
    gate = CascadeGate(threshold=0.0, max_fraction=0.30)

    assert gate.n_max(275_000) == 82_500
    assert gate.n_max(115_000) == 34_500
    assert gate.n_max(1_000) == 300
    assert gate.n_max(0) == 0


def test_cap_keeps_the_highest_cheap_scores_and_breaks_ties_by_position() -> None:
    gate = CascadeGate(threshold=0.0, max_fraction=0.5)
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.3, 0.7])

    selected = gate.select(scores)

    assert selected.tolist() == [1, 3, 5]
    assert np.array_equal(selected, gate.select(scores))


def test_compose_fills_only_the_rows_the_gate_skipped() -> None:
    cheap = np.array([0.9, 0.2, 0.7, 0.1])
    expensive = np.array([0.95, 0.60])
    selected = np.array([0, 2])

    substitute = CascadeGate(0.5, 1.0, "substitute").compose(cheap, expensive, selected)
    zero = CascadeGate(0.5, 1.0, "zero").compose(cheap, expensive, selected)
    missing = CascadeGate(0.5, 1.0, "nan").compose(cheap, expensive, selected)

    assert substitute.tolist() == pytest.approx([0.95, 0.2, 0.60, 0.1])
    assert zero.tolist() == pytest.approx([0.95, 0.0, 0.60, 0.0])
    assert missing[[0, 2]].tolist() == pytest.approx([0.95, 0.60])
    assert np.isnan(missing[[1, 3]]).all()


def test_calibration_on_gold_underspends_at_a_sparser_operating_point() -> None:
    """Gold is 2.35x denser in positives, so the same threshold selects less on the test."""
    rng = np.random.default_rng(20260824)
    positives = rng.beta(6.0, 2.0, size=20_000)
    negatives = rng.beta(2.0, 6.0, size=60_000)
    gold = np.concatenate([positives, negatives])
    sparse = np.concatenate([positives[: positives.size // 3], negatives])

    threshold = calibrate_threshold(gold, 0.30)
    gate = CascadeGate(threshold=threshold, max_fraction=1.0)

    assert gate.select(gold).size / gold.size == pytest.approx(0.30, abs=0.01)
    assert gate.select(sparse).size / sparse.size < 0.30


def test_schema_round_trip_and_rejection() -> None:
    gate = CascadeGate(threshold=0.487, max_fraction=0.30, fill="substitute")
    schema = gate.to_schema()

    assert schema["kind"] == SCORE_SOURCE
    assert CascadeGate.from_schema(schema) == gate
    with pytest.raises(ValueError, match="Not a cascade schema"):
        CascadeGate.from_schema({"kind": "mean_probability"})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"threshold": float("nan"), "max_fraction": 0.3}, "finite"),
        ({"threshold": 0.5, "max_fraction": 0.0}, "max_fraction"),
        ({"threshold": 0.5, "max_fraction": 1.5}, "max_fraction"),
        ({"threshold": 0.5, "max_fraction": 0.3, "fill": "average"}, "fill"),
    ],
)
def test_invalid_contracts_fail_closed(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CascadeGate(**kwargs)

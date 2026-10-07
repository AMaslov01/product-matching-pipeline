"""Two-stage scoring: a cheap model ranks everything, an expensive one reranks the top.

A single pass of a 4x-cost backbone does not fit the contest budget. Replaying
the measured receipt through ``research/cascade-simulation-20260823`` puts one
full expensive pass at roughly 1660 A100-equivalent seconds against a ceiling of
about 1335, so the expensive model can only be afforded on a slice. The same
simulation measured what that slice buys: at 30% of rows the cascade returns
88.5% of what a full expensive pass would.

Two properties are load-bearing and both are about *determinism*, because a
timeout does not merely cost a submission -- it disqualifies the archive from
being chosen as a final solution.

Absolute threshold, never a quantile
    Candidate A ranked scores inside the inference batch and collapsed on the
    hidden split, because the batch it was calibrated on was not the batch it
    served. The gate here compares against a frozen score value, so a row's fate
    depends only on its own score.

A cap that is a pure function of the input size
    The threshold alone cannot bound runtime: a shifted score distribution could
    select far more rows than the calibration implied. The cap is expressed as a
    fraction of the pair count rather than as a wall-clock budget, so it yields
    the same answer on the same input on any machine and in any phase. It is
    calibrated against the private phase, which is the tightest per-pair limit
    (780 s / 275k pairs vs 360 s / 115k), so a cap that fits there fits the
    others too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCORE_SOURCE = "cascade_two_score"
FILL_MODES = ("substitute", "zero", "nan")


@dataclass(frozen=True)
class CascadeGate:
    """The frozen contract that decides which rows the expensive model sees."""

    threshold: float
    max_fraction: float
    fill: str = "substitute"

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold):
            raise ValueError("Cascade threshold must be finite")
        if not 0.0 < self.max_fraction <= 1.0:
            raise ValueError("Cascade max_fraction must lie in (0, 1]")
        if self.fill not in FILL_MODES:
            raise ValueError(f"Unsupported cascade fill: {self.fill!r}")

    def n_max(self, pair_count: int) -> int:
        """Largest number of rows the expensive stage may score for this input."""
        if pair_count < 0:
            raise ValueError("pair_count must be non-negative")
        return int(np.floor(self.max_fraction * pair_count))

    def select(self, cheap_scores: np.ndarray) -> np.ndarray:
        """Row indices for the expensive stage, ordered by descending cheap score.

        Ordering is part of the contract, not a convenience: when the cap binds,
        the rows that survive must be the ones the cheap model ranked highest,
        and ``np.lexsort`` on (row index, -score) keeps ties resolved by position
        so the same input always yields the same selection.
        """
        scores = np.asarray(cheap_scores, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError("cheap_scores must be one-dimensional")
        eligible = np.flatnonzero(scores >= self.threshold)
        order = np.lexsort((eligible, -scores[eligible]))
        return eligible[order][: self.n_max(scores.size)]

    def compose(
        self, cheap_scores: np.ndarray, expensive_scores: np.ndarray, selected: np.ndarray
    ) -> np.ndarray:
        """Build the expensive feature column, filling rows the gate skipped.

        The fill is a real degree of freedom -- it decides what the fusion learns
        about an unscored row -- so it is pinned by replaying the simulation
        rather than chosen by taste. See ``scripts/cascade_replay.py``.
        """
        cheap = np.asarray(cheap_scores, dtype=np.float64)
        column = np.empty_like(cheap)
        if self.fill == "substitute":
            column[:] = cheap
        elif self.fill == "zero":
            column[:] = 0.0
        else:
            column[:] = np.nan
        column[selected] = np.asarray(expensive_scores, dtype=np.float64)
        return column

    def to_schema(self) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": SCORE_SOURCE,
            "threshold": float(self.threshold),
            "max_fraction": float(self.max_fraction),
            "fill": self.fill,
        }

    @classmethod
    def from_schema(cls, schema: dict[str, Any]) -> CascadeGate:
        if str(schema.get("kind")) != SCORE_SOURCE:
            raise ValueError(f"Not a cascade schema: {schema.get('kind')!r}")
        return cls(
            threshold=float(schema["threshold"]),
            max_fraction=float(schema["max_fraction"]),
            fill=str(schema.get("fill", "substitute")),
        )


def calibrate_threshold(cheap_scores: np.ndarray, target_fraction: float) -> float:
    """Score value that selects ``target_fraction`` of the calibration rows.

    Calibration happens on Gold OOF, which is denser in positives than the hidden
    test, so the same threshold selects a smaller share at serving time -- the
    simulation measured 0.30 on Gold against 0.1948 at the test operating point.
    The error therefore runs toward spending less time than budgeted, which is
    the direction a hard deadline can absorb.
    """
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must lie in (0, 1]")
    scores = np.asarray(cheap_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot calibrate a threshold without scores")
    return float(np.quantile(scores, 1.0 - target_fraction))

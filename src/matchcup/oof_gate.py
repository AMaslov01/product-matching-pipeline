"""Paired gate over a complete strict OOF, for comparing two training recipes.

``research.assess_candidate_gate`` compares a single fold; a recipe change like
full Silver has to be judged on all five, so this is the whole-OOF counterpart.

What this gate can and cannot see is the point of its design.

Gold OOF measures ranking quality on held-out Human Gold *components*, but those
are still Human Gold items. Full Silver is bought for a different reason: item
coverage (3.13M -> 12.38M items, 92% of the catalogue) and therefore transfer to
products the model has never seen. Gold OOF is structurally blind to that, so a
recipe can be flat here and still transfer better - and the reverse.

The obvious instrument for it, the new-item LLM diagnostic, is unusable for this
particular comparison: 100% of its 188,873 items now sit inside full-Silver
training. It diagnosed catalogue shift for the 2M sample and full Silver
swallowed it whole.

So there is no local instrument that can see what full Silver buys, and a hard
"no gain on Gold, no submission" rule would throw away the very thing we trained
for. The verdict is therefore graded: block only a real regression, and treat a
flat result as a reason to spend one submission and let Public - the only
validated instrument we have - answer the question.

It reports two operating points and gates on only one of them, deliberately:

*Gold prevalence* (~0.261) is the project's established criterion and what the
+0.002 gate has always meant. It stays the decision.

*The leaderboard's operating point* (~0.111, measured by the constant probe) is
reported alongside because a Gold number cannot be read as a Public forecast. It
does not gate, because the hypothesis that it predicts Public deltas better was
tested and refuted - it missed the one clean delta by more than the old gate did
(see docs/experiments/05, H15). Reporting it is useful; trusting it is not.

The bootstrap resamples categories with replacement and evaluates both scorers on
the *same* draw, so the paired difference cancels the shared sampling noise. An
unpaired comparison of two OOF runs is what produced the misleading +0.006 in
issue #6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from matchcup.metrics import (
    DEFAULT_PUBLIC_PREVALENCE,
    macro_average_precision,
    prevalence_matched_macro_ap,
)

MINIMUM_MACRO_GAIN = 0.002
MAXIMUM_CATEGORY_DROP = 0.02
# Below this the recipe is genuinely worse and something is wrong; above it but
# under MINIMUM_MACRO_GAIN the result is flat, which Gold OOF cannot distinguish
# from "better where it counts".
REGRESSION_LIMIT = -0.005


@dataclass
class GateResult:
    passed: bool
    verdict: str
    recommendation: str
    reasons: list[str]
    control_macro_ap: float
    candidate_macro_ap: float
    difference: float
    ci95_low: float
    ci95_high: float
    bootstrap_positive_rate: float
    control_at_operating_point: float
    candidate_at_operating_point: float
    difference_at_operating_point: float
    worst_category: str
    worst_category_difference: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: getattr(self, key)
            for key in (
                "passed", "verdict", "recommendation", "reasons",
                "control_macro_ap", "candidate_macro_ap",
                "difference", "ci95_low", "ci95_high", "bootstrap_positive_rate",
                "control_at_operating_point", "candidate_at_operating_point",
                "difference_at_operating_point", "worst_category",
                "worst_category_difference", "per_category",
            )
        }
        payload["minimum_macro_gain"] = MINIMUM_MACRO_GAIN
        payload["maximum_category_drop"] = MAXIMUM_CATEGORY_DROP
        payload["operating_point"] = DEFAULT_PUBLIC_PREVALENCE
        return payload


def load_oof(path: str | Path) -> pd.DataFrame:
    """Read a full OOF as one frame, from a directory of folds or a single file."""
    path = Path(path)
    files = sorted(path.glob("oof_fold_*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No OOF parquet files in {path}")
    frame = pd.concat(
        [pq.read_table(file).to_pandas() for file in files], ignore_index=True
    )
    missing = {"id1", "id2", "text_score"} - set(frame.columns)
    if missing:
        raise ValueError(f"OOF is missing columns {sorted(missing)} in {path}")
    if not np.isfinite(frame["text_score"].to_numpy()).all():
        raise ValueError(f"OOF contains non-finite scores: {path}")
    return frame


def _joined(gold_path: str | Path, control: pd.DataFrame, candidate: pd.DataFrame):
    gold = pq.read_table(
        Path(gold_path), columns=["id1", "id2", "target", "category"]
    ).to_pandas()
    frame = gold.merge(
        control[["id1", "id2", "text_score"]].rename(columns={"text_score": "control"}),
        on=["id1", "id2"], validate="one_to_one",
    ).merge(
        candidate[["id1", "id2", "text_score"]].rename(columns={"text_score": "candidate"}),
        on=["id1", "id2"], validate="one_to_one",
    )
    if len(frame) != len(gold):
        raise ValueError(
            f"Control and candidate must cover every Gold row; got {len(frame)} of {len(gold)}"
        )
    return frame


def assess_oof_gate(
    gold_path: str | Path,
    control_oof: str | Path,
    candidate_oof: str | Path,
    *,
    samples: int = 300,
    seed: int = 20260812,
    operating_point: float = DEFAULT_PUBLIC_PREVALENCE,
    minimum_macro_gain: float = MINIMUM_MACRO_GAIN,
    maximum_category_drop: float = MAXIMUM_CATEGORY_DROP,
) -> GateResult:
    """Compare two complete OOF score sets under the precommitted gate."""
    if samples < 10:
        raise ValueError("samples must be at least 10")
    frame = _joined(gold_path, load_oof(control_oof), load_oof(candidate_oof))
    target = frame["target"].to_numpy()
    category = frame["category"].astype(str).to_numpy()
    control = frame["control"].to_numpy()
    candidate = frame["candidate"].to_numpy()

    control_ap, control_per = macro_average_precision(target, control, category)
    candidate_ap, candidate_per = macro_average_precision(target, candidate, category)
    control_op, _ = prevalence_matched_macro_ap(
        target, control, category, target_prevalence=operating_point, seed=seed
    )
    candidate_op, _ = prevalence_matched_macro_ap(
        target, candidate, category, target_prevalence=operating_point, seed=seed
    )

    per_category = {
        name: {
            "control_ap": control_per[name],
            "candidate_ap": candidate_per[name],
            "difference": candidate_per[name] - control_per[name],
        }
        for name in sorted(control_per)
    }
    worst = min(per_category.items(), key=lambda item: item[1]["difference"])

    groups = [
        indices.to_numpy()
        for _, indices in frame.groupby("category", sort=True).groups.items()
    ]
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        drawn = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        # Both scorers see the identical draw, so shared sampling noise cancels.
        drawn_control, _ = macro_average_precision(
            target[drawn], control[drawn], category[drawn]
        )
        drawn_candidate, _ = macro_average_precision(
            target[drawn], candidate[drawn], category[drawn]
        )
        differences[index] = drawn_candidate - drawn_control

    difference = candidate_ap - control_ap
    reasons: list[str] = []
    collapsed = worst[1]["difference"] < -maximum_category_drop
    if collapsed:
        reasons.append(
            f"category {worst[0]!r} drops {worst[1]['difference']:+.5f}, "
            f"beyond the {maximum_category_drop} limit"
        )
    if difference < REGRESSION_LIMIT:
        reasons.append(
            f"macro AP fell {difference:+.5f}, past the {REGRESSION_LIMIT} regression limit"
        )

    if reasons:
        verdict = "regression"
        recommendation = (
            "Do not package. This is a real loss on Gold, not a flat result, so "
            "investigate the recipe before spending a submission."
        )
    elif difference >= minimum_macro_gain:
        verdict = "improvement"
        recommendation = "Package and submit; the recipe clears the established gate."
    else:
        verdict = "inconclusive"
        recommendation = (
            f"Package and submit. The Gold delta {difference:+.5f} is flat, but Gold "
            "OOF cannot see the item coverage full Silver was trained for, and the "
            "new-item diagnostic is now inside training. One submission answers "
            "what no local metric can."
        )
    return GateResult(
        passed=verdict != "regression",
        verdict=verdict,
        recommendation=recommendation,
        reasons=reasons,
        control_macro_ap=float(control_ap),
        candidate_macro_ap=float(candidate_ap),
        difference=float(difference),
        ci95_low=float(np.quantile(differences, 0.025)),
        ci95_high=float(np.quantile(differences, 0.975)),
        bootstrap_positive_rate=float(np.mean(differences > 0.0)),
        control_at_operating_point=float(control_op),
        candidate_at_operating_point=float(candidate_op),
        difference_at_operating_point=float(candidate_op - control_op),
        worst_category=worst[0],
        worst_category_difference=float(worst[1]["difference"]),
        per_category=per_category,
    )

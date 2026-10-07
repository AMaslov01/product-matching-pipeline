"""Batch-local, label-free score contracts shared by OOF and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SCORE_TRANSFORM_VERSION = 2
BATCH_PERCENTILE_KIND = "batch_category_percentile"
DEFAULT_SOURCE_FEATURE = "text_score"
DEFAULT_OUTPUT_FEATURE = "text_score_category_rank"
DEFAULT_GROUPING = ("category",)
DEFAULT_RANK_METHOD = "average"


@dataclass(frozen=True)
class BatchCategoryPercentileScoreTransform:
    """Replace model-specific score scale with a rank inside each batch group.

    OOF callers rank by ``(fold, category)`` so scores from different fold
    models never calibrate one another. Runtime callers rank the complete
    inference batch by category. Pandas' percentile rank assigns ``1.0`` to a
    singleton group and uses the average percentile for ties.
    """

    source_feature: str = DEFAULT_SOURCE_FEATURE
    output_feature: str = DEFAULT_OUTPUT_FEATURE
    grouping: tuple[str, ...] = DEFAULT_GROUPING
    score_source: str = "single_probability"

    def __post_init__(self) -> None:
        if not self.source_feature or not self.output_feature:
            raise ValueError("Score-transform feature names must not be empty")
        if self.source_feature == self.output_feature:
            raise ValueError("Score-transform source and output features must differ")
        if not self.grouping or any(not name for name in self.grouping):
            raise ValueError("Score-transform grouping must not be empty")
        if self.score_source not in {"single_probability", "mean_probability"}:
            raise ValueError(f"Unsupported score source: {self.score_source!r}")

    def add_features(
        self,
        frame: pd.DataFrame,
        *,
        group_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        groups = tuple(group_columns or self.grouping)
        required = {self.source_feature, *groups}
        if missing := required - set(frame.columns):
            raise ValueError(f"Score transform is missing columns: {sorted(missing)}")
        result = frame.copy()
        if result.empty:
            result[self.output_feature] = pd.Series(dtype=np.float32, index=result.index)
            return result
        try:
            scores = result[self.source_feature].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Score-transform input must be numeric") from exc
        if not np.isfinite(scores).all():
            raise ValueError("Score-transform input must be finite")
        ranked = result.groupby(
            list(groups), sort=False, dropna=False
        )[self.source_feature].rank(method=DEFAULT_RANK_METHOD, pct=True)
        values = ranked.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Batch percentile produced non-finite values")
        result[self.output_feature] = values.astype(np.float32)
        return result

    def add_record_features(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        result = [dict(record) for record in records]
        if not result:
            return result
        return self.add_features(pd.DataFrame(result)).to_dict("records")

    def to_schema(self) -> dict[str, Any]:
        return {
            "version": SCORE_TRANSFORM_VERSION,
            "kind": BATCH_PERCENTILE_KIND,
            "source_feature": self.source_feature,
            "output_feature": self.output_feature,
            "grouping": list(self.grouping),
            "rank_method": DEFAULT_RANK_METHOD,
            "percentile": True,
            "score_source": self.score_source,
        }

    @classmethod
    def from_schema(
        cls, schema: Mapping[str, Any]
    ) -> BatchCategoryPercentileScoreTransform:
        if schema.get("version") != SCORE_TRANSFORM_VERSION:
            raise ValueError(f"Unsupported score-transform version: {schema.get('version')!r}")
        if schema.get("kind") != BATCH_PERCENTILE_KIND:
            raise ValueError(f"Unsupported score-transform kind: {schema.get('kind')!r}")
        if schema.get("rank_method") != DEFAULT_RANK_METHOD:
            raise ValueError(f"Unsupported rank method: {schema.get('rank_method')!r}")
        if schema.get("percentile") is not True:
            raise ValueError("Batch score transform must use percentile ranks")
        grouping = schema.get("grouping")
        if not isinstance(grouping, list) or not all(isinstance(name, str) for name in grouping):
            raise ValueError("Score-transform grouping must be a list of strings")
        return cls(
            source_feature=str(schema.get("source_feature", "")),
            output_feature=str(schema.get("output_feature", "")),
            grouping=tuple(grouping),
            score_source=str(schema.get("score_source", "")),
        )

    def report(self) -> dict[str, Any]:
        return self.to_schema()


def score_transform_from_feature_schema(
    feature_schema: Mapping[str, Any],
) -> BatchCategoryPercentileScoreTransform | None:
    """Load the optional transform while keeping baseline schemas compatible."""
    transform_schema = feature_schema.get("score_transform")
    if transform_schema is None:
        return None
    if not isinstance(transform_schema, Mapping):
        raise ValueError("score_transform must be an object")
    return BatchCategoryPercentileScoreTransform.from_schema(transform_schema)

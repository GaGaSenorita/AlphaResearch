"""AlphaLDM with the Stage-2 Method-1 worst-quarter Train reward."""

from __future__ import annotations

import csv
import json
import math
from typing import Any

import numpy as np

from alpha_research.methods.ldm.history import History
from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.methods.split_robust.reward import SplitScore, SplitSettings, split_score
from alpha_research.types import EvaluationResult


SPLIT_OBJECTIVE = "worst_rankic"
QUARTER_VALUE_PREFIX = "rankic_"
QUARTER_DAYS_PREFIX = "days_"
FIXED_COLUMNS = (
    "expression",
    "score",
    "train_rankic",
    "worst_rankic",
    "worst_quarter",
    "usable_days",
    "rejected",
)


class QuarterlyWorstHistory(History):
    """LDM history with the full quarter-level provenance of every target."""

    def __init__(self, dim: int, schema_version: str, quarters: tuple[str, ...]) -> None:
        super().__init__(dim=dim, schema_version=schema_version)
        self.quarters = tuple(quarters)
        extra = ["train_rankic", "worst_rankic", "worst_quarter", "quarterly_rankic"]
        extra += [f"{QUARTER_VALUE_PREFIX}{quarter}" for quarter in self.quarters]
        self._df = self._df.reindex(columns=list(self._df.columns) + extra)

    def add_outcome(
        self,
        feature: np.ndarray,
        score: float,
        canonical: str,
        expression: str,
        round_id: int,
        outcome: SplitScore,
    ) -> None:
        row: dict[str, Any] = {
            column: float(value)
            for column, value in zip(
                self.feature_cols, np.asarray(feature, dtype=float)
            )
        }
        quarterly = dict(zip(outcome.quarters, outcome.quarterly_rankic))
        row.update(
            {
                "score": float(score),
                "canonical": canonical,
                "expression": expression,
                "round_id": int(round_id),
                "train_rankic": outcome.train_rankic,
                "worst_rankic": outcome.worst_rankic,
                "worst_quarter": outcome.worst_quarter,
                "quarterly_rankic": json.dumps(quarterly, separators=(",", ":")),
            }
        )
        row.update(
            {
                f"{QUARTER_VALUE_PREFIX}{quarter}": quarterly.get(quarter)
                for quarter in self.quarters
            }
        )
        self._df.loc[len(self._df)] = row


class SplitRobustLDM(AlphaLDM):
    name = "ldm_split_robust_reward"
    candidate_source = "ldm_standard"

    def __init__(self, *, split_settings: SplitSettings | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.search_objective != SPLIT_OBJECTIVE:
            raise ValueError(
                f"{self.name} requires search_objective={SPLIT_OBJECTIVE!r}, got "
                f"{self.search_objective!r}"
            )
        self.split_settings = split_settings or SplitSettings()
        self._split_records: list[dict[str, Any]] = []

    def _score(self, result: Any) -> float:
        """Compute the scalar target without triggering another evaluation."""
        return split_score(result, self.split_settings).score

    def _new_history(self, *, dim: int, schema_version: str) -> History:
        return QuarterlyWorstHistory(dim, schema_version, self.split_settings.quarters)

    def _record_observation(
        self,
        history: History,
        *,
        feature: np.ndarray,
        score: float,
        result: EvaluationResult,
        canonical: str,
        expression: str,
        round_id: int,
    ) -> None:
        outcome = split_score(result, self.split_settings)
        if not math.isfinite(outcome.score):
            raise ValueError(outcome.rejected or "non-finite worst-quarter RankIC")
        if not isinstance(history, QuarterlyWorstHistory):
            raise TypeError("worst-quarter method requires QuarterlyWorstHistory")
        history.add_outcome(feature, score, canonical, expression, round_id, outcome)
        row: dict[str, Any] = {
            "expression": expression,
            "score": outcome.score,
            "train_rankic": outcome.train_rankic,
            "worst_rankic": outcome.worst_rankic,
            "worst_quarter": outcome.worst_quarter,
            "usable_days": outcome.usable_days,
            "rejected": outcome.rejected or "",
        }
        for quarter, value, days in zip(
            outcome.quarters, outcome.quarterly_rankic, outcome.quarter_days
        ):
            row[f"{QUARTER_VALUE_PREFIX}{quarter}"] = value
            row[f"{QUARTER_DAYS_PREFIX}{quarter}"] = days
        self._split_records.append(row)

    def search(self, **kwargs: Any) -> Any:
        self._split_records = []
        try:
            return super().search(**kwargs)
        finally:
            self._write_split_history()

    def _write_split_history(self) -> None:
        if not self._split_records:
            return
        quarters = self.split_settings.quarters
        columns = list(FIXED_COLUMNS)
        columns += [f"{QUARTER_VALUE_PREFIX}{quarter}" for quarter in quarters]
        columns += [f"{QUARTER_DAYS_PREFIX}{quarter}" for quarter in quarters]
        path = self.output_dir / "split_reward_history.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._split_records)

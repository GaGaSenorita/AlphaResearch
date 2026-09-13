"""AlphaLDM with the Stage-2 Method-1 worst-year Train reward."""

from __future__ import annotations

import csv
import json
import math
from typing import Any

import numpy as np

from alpha_research.methods.ldm.history import History
from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.methods.split_robust.reward import (
    YearlySplitScore,
    YearlySplitSettings,
    yearly_split_score,
)
from alpha_research.methods.split_robust.contract import (
    validate_annual_output,
    validate_train_years,
)
from alpha_research.types import EvaluationResult


SPLIT_OBJECTIVE = "worst_rankic"
YEAR_VALUE_PREFIX = "rankic_"
YEAR_DAYS_PREFIX = "days_"
YEAR_FIXED_COLUMNS = (
    "expression",
    "score",
    "train_rankic",
    "worst_rankic",
    "worst_year",
    "usable_days",
    "rejected",
)


class YearlyWorstHistory(History):
    """LDM history with the full year-level provenance of every target."""

    def __init__(self, dim: int, schema_version: str, years: tuple[str, ...]) -> None:
        super().__init__(dim=dim, schema_version=schema_version)
        self.years = tuple(years)
        extra = ["train_rankic", "worst_rankic", "worst_year", "yearly_rankic"]
        extra += [f"{YEAR_VALUE_PREFIX}{year}" for year in self.years]
        extra += [f"{YEAR_DAYS_PREFIX}{year}" for year in self.years]
        self._df = self._df.reindex(columns=list(self._df.columns) + extra)

    def add_outcome(
        self,
        feature: np.ndarray,
        score: float,
        canonical: str,
        expression: str,
        round_id: int,
        outcome: YearlySplitScore,
    ) -> None:
        if outcome.years != self.years or outcome.rejected:
            raise ValueError("annual history requires a complete matching year contract")
        # Retain the base store's finite-fingerprint and canonical-identity checks.
        super().add(feature, score, canonical, expression, round_id)
        row: dict[str, Any] = {}
        yearly = dict(zip(outcome.years, outcome.yearly_rankic))
        year_days = dict(zip(outcome.years, outcome.year_days))
        row.update(
            {
                "score": float(score),
                "canonical": canonical,
                "expression": expression,
                "round_id": int(round_id),
                "train_rankic": outcome.train_rankic,
                "worst_rankic": outcome.worst_rankic,
                "worst_year": outcome.worst_year,
                "yearly_rankic": json.dumps(yearly, separators=(",", ":")),
            }
        )
        row.update(
            {
                f"{YEAR_VALUE_PREFIX}{year}": yearly.get(year)
                for year in self.years
            }
        )
        row.update(
            {
                f"{YEAR_DAYS_PREFIX}{year}": year_days.get(year)
                for year in self.years
            }
        )
        row_index = len(self._df) - 1
        # The initial base row has missing provenance; keep these columns textual.
        for column in ("worst_year", "yearly_rankic"):
            if self._df[column].dtype != object:
                self._df[column] = self._df[column].astype(object)
        for column, value in row.items():
            self._df.at[row_index, column] = value


class SplitRobustLDM(AlphaLDM):
    name = "ldm_split_robust_reward"
    candidate_source = "ldm_standard"

    def __init__(
        self, *, split_settings: YearlySplitSettings | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if self.search_objective != SPLIT_OBJECTIVE:
            raise ValueError(
                f"{self.name} requires search_objective={SPLIT_OBJECTIVE!r}, got "
                f"{self.search_objective!r}"
            )
        self.split_settings = split_settings or YearlySplitSettings()
        self._split_records: list[dict[str, Any]] = []

    def _score(self, result: Any) -> float:
        """Compute the scalar target without triggering another evaluation."""
        return yearly_split_score(result, self.split_settings).score

    def _generator_objective(self) -> str:
        return "worst calendar-year mean signed Train RankIC (2016--2020)"

    def _evaluated_record(
        self, *, expression: str, score: float, result: EvaluationResult,
    ) -> dict[str, Any]:
        outcome = yearly_split_score(result, self.split_settings)
        return {
            "expression": expression,
            "score": score,
            "worst_year": outcome.worst_year,
            "yearly_rankic": dict(zip(outcome.years, outcome.yearly_rankic)),
        }

    def _new_history(self, *, dim: int, schema_version: str) -> History:
        return YearlyWorstHistory(dim, schema_version, self.split_settings.years)

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
        outcome = yearly_split_score(result, self.split_settings)
        if not math.isfinite(outcome.score):
            raise ValueError(outcome.rejected or "non-finite worst-year RankIC")
        if not isinstance(history, YearlyWorstHistory):
            raise TypeError("worst-year method requires YearlyWorstHistory")
        history.add_outcome(feature, score, canonical, expression, round_id, outcome)
        row: dict[str, Any] = {
            "expression": expression,
            "score": outcome.score,
            "train_rankic": outcome.train_rankic,
            "worst_rankic": outcome.worst_rankic,
            "worst_year": outcome.worst_year,
            "usable_days": outcome.usable_days,
            "rejected": outcome.rejected or "",
        }
        for year, value, days in zip(
            outcome.years, outcome.yearly_rankic, outcome.year_days
        ):
            row[f"{YEAR_VALUE_PREFIX}{year}"] = value
            row[f"{YEAR_DAYS_PREFIX}{year}"] = days
        self._split_records.append(row)

    def search(self, **kwargs: Any) -> Any:
        validate_train_years(kwargs.get("train_period"), self.split_settings)
        validate_annual_output(self.output_dir)
        self._split_records = []
        try:
            return super().search(**kwargs)
        finally:
            self._write_split_history()

    def _write_split_history(self) -> None:
        if not self._split_records:
            return
        years = self.split_settings.years
        columns = list(YEAR_FIXED_COLUMNS)
        columns += [f"{YEAR_VALUE_PREFIX}{year}" for year in years]
        columns += [f"{YEAR_DAYS_PREFIX}{year}" for year in years]
        path = self.output_dir / "split_reward_history.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._split_records)

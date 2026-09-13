"""Annual Train rewards shared by Stage-2 Method 1 and Method 2.

Each factor is evaluated once over 2016--2020. Aggregate its signed daily
RankIC into five calendar-year means, then take their minimum. There is no
per-year sign adjustment or separate evaluation for each year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class YearlySplitSettings:
    """Frozen calendar-year Train split shared by both Stage-2 methods."""

    start_year: int = 2016
    end_year: int = 2020

    def __post_init__(self) -> None:
        if self.start_year != 2016 or self.end_year != 2020:
            raise ValueError(
                "Worst-Case Train-Split Objective is fixed to 2016--2020"
            )

    @property
    def years(self) -> tuple[str, ...]:
        return tuple(str(year) for year in range(self.start_year, self.end_year + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": "worst_rankic",
            "segmentation": "calendar_year",
            "train_years": [self.start_year, self.end_year],
            "years": list(self.years),
            "aggregation": "minimum_of_yearly_mean_signed_rankic",
            "factor_evaluations_per_score": 1,
            "per_year_sign_flip": False,
        }


@dataclass(frozen=True)
class YearlySplitScore:
    """One scalar GP target plus the year-level audit trail behind it."""

    score: float
    years: tuple[str, ...] = ()
    yearly_rankic: tuple[float, ...] = ()
    year_days: tuple[int, ...] = ()
    train_rankic: float | None = None
    worst_rankic: float | None = None
    worst_year: str | None = None
    usable_days: int = 0
    rejected: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score if math.isfinite(self.score) else None,
            "years": list(self.years),
            "yearly_rankic": {
                year: value for year, value in zip(self.years, self.yearly_rankic)
            },
            "year_days": {
                year: days for year, days in zip(self.years, self.year_days)
            },
            "train_rankic": self.train_rankic,
            "worst_rankic": self.worst_rankic,
            "worst_year": self.worst_year,
            "usable_days": self.usable_days,
            "rejected": self.rejected,
            "settings": dict(self.settings),
        }


def _finite_train_rankic(result: Any) -> float | None:
    metrics = getattr(result, "metrics", None)
    value = getattr(metrics, "rank_ic", None) if metrics is not None else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _daily_by_year(result: Any) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for record in getattr(result, "daily_metrics", ()) or ():
        if not isinstance(record, dict):
            continue
        raw_day, raw_value = record.get("date"), record.get("rank_ic")
        try:
            day = date.fromisoformat(str(raw_day)[:10])
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            grouped.setdefault(str(day.year), []).append(value)
    return grouped


def yearly_split_score(
    result: Any, settings: YearlySplitSettings | None = None
) -> YearlySplitScore:
    """Return ``min_y mean(daily signed RankIC in y)`` for 2016--2020.

    Aggregate an existing full-Train evaluation without another evaluator call
    or any per-year sign adjustment. Every Train year must have finite data.
    """

    settings = settings or YearlySplitSettings()
    expected = settings.years
    grouped = _daily_by_year(result)
    missing = [year for year in expected if not grouped.get(year)]
    usable_days = sum(len(grouped.get(year, ())) for year in expected)
    train_rankic = _finite_train_rankic(result)
    if missing:
        return YearlySplitScore(
            score=float("-inf"),
            train_rankic=train_rankic,
            usable_days=usable_days,
            rejected="missing finite daily rank_ic for: " + ", ".join(missing),
            settings=settings.to_dict(),
        )

    yearly = tuple(
        math.fsum(grouped[year]) / len(grouped[year]) for year in expected
    )
    nonfinite = [year for year, value in zip(expected, yearly) if not math.isfinite(value)]
    if nonfinite:
        return YearlySplitScore(
            score=float("-inf"),
            train_rankic=train_rankic,
            usable_days=usable_days,
            rejected="non-finite yearly mean rank_ic for: " + ", ".join(nonfinite),
            settings=settings.to_dict(),
        )

    counts = tuple(len(grouped[year]) for year in expected)
    worst_index = min(range(len(yearly)), key=yearly.__getitem__)
    worst_rankic = float(yearly[worst_index])
    return YearlySplitScore(
        score=worst_rankic,
        years=expected,
        yearly_rankic=yearly,
        year_days=counts,
        train_rankic=train_rankic,
        worst_rankic=worst_rankic,
        worst_year=expected[worst_index],
        usable_days=usable_days,
        settings=settings.to_dict(),
    )


# Public shorthand retained for callers of the shared scoring API.
SplitSettings = YearlySplitSettings
SplitScore = YearlySplitScore
split_score = yearly_split_score

"""Worst-case calendar-quarter reward from one full Train evaluation.

The evaluator is called once for the complete 2016--2020 Train window. This
module only aggregates the returned *signed* daily RankIC series: first one
mean per calendar quarter, then the minimum of those 20 means. It does not
evaluate quarters separately and never flips a factor's sign per quarter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any


def _quarter_labels(start_year: int, end_year: int) -> tuple[str, ...]:
    return tuple(
        f"{year}Q{quarter}"
        for year in range(start_year, end_year + 1)
        for quarter in range(1, 5)
    )


@dataclass(frozen=True)
class SplitSettings:
    """Frozen Stage-2 Method-1 split contract."""

    start_year: int = 2016
    end_year: int = 2020

    def __post_init__(self) -> None:
        if self.start_year != 2016 or self.end_year != 2020:
            raise ValueError(
                "Worst-Case Train-Split Objective is fixed to 2016Q1--2020Q4"
            )

    @property
    def quarters(self) -> tuple[str, ...]:
        return _quarter_labels(self.start_year, self.end_year)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": "worst_rankic",
            "segmentation": "calendar_quarter",
            "train_years": [self.start_year, self.end_year],
            "quarters": list(self.quarters),
            "aggregation": "minimum_of_quarterly_mean_signed_rankic",
            "factor_evaluations_per_score": 1,
            "per_quarter_sign_flip": False,
        }


@dataclass(frozen=True)
class SplitScore:
    """One scalar GP target plus the quarter-level audit trail behind it."""

    score: float
    quarters: tuple[str, ...] = ()
    quarterly_rankic: tuple[float, ...] = ()
    quarter_days: tuple[int, ...] = ()
    train_rankic: float | None = None
    worst_rankic: float | None = None
    worst_quarter: str | None = None
    usable_days: int = 0
    rejected: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score if math.isfinite(self.score) else None,
            "quarters": list(self.quarters),
            "quarterly_rankic": {
                quarter: value
                for quarter, value in zip(self.quarters, self.quarterly_rankic)
            },
            "quarter_days": {
                quarter: days for quarter, days in zip(self.quarters, self.quarter_days)
            },
            "train_rankic": self.train_rankic,
            "worst_rankic": self.worst_rankic,
            "worst_quarter": self.worst_quarter,
            "usable_days": self.usable_days,
            "rejected": self.rejected,
            "settings": dict(self.settings),
        }


def _quarter_label(day: date) -> str:
    return f"{day.year}Q{((day.month - 1) // 3) + 1}"


def _finite_train_rankic(result: Any) -> float | None:
    metrics = getattr(result, "metrics", None)
    value = getattr(metrics, "rank_ic", None) if metrics is not None else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _daily_by_quarter(result: Any) -> dict[str, list[float]]:
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
            grouped.setdefault(_quarter_label(day), []).append(value)
    return grouped


def split_score(result: Any, settings: SplitSettings | None = None) -> SplitScore:
    """Return ``min_q mean(daily signed RankIC in q)`` for 2016Q1--2020Q4."""

    settings = settings or SplitSettings()
    expected = settings.quarters
    grouped = _daily_by_quarter(result)
    missing = [quarter for quarter in expected if not grouped.get(quarter)]
    usable_days = sum(len(grouped.get(quarter, ())) for quarter in expected)
    train_rankic = _finite_train_rankic(result)
    if missing:
        return SplitScore(
            score=float("-inf"),
            train_rankic=train_rankic,
            usable_days=usable_days,
            rejected="missing finite daily rank_ic for: " + ", ".join(missing),
            settings=settings.to_dict(),
        )

    quarterly = tuple(
        math.fsum(grouped[quarter]) / len(grouped[quarter]) for quarter in expected
    )
    counts = tuple(len(grouped[quarter]) for quarter in expected)
    worst_index = min(range(len(quarterly)), key=quarterly.__getitem__)
    worst_rankic = float(quarterly[worst_index])
    return SplitScore(
        score=worst_rankic,
        quarters=expected,
        quarterly_rankic=quarterly,
        quarter_days=counts,
        train_rankic=train_rankic,
        worst_rankic=worst_rankic,
        worst_quarter=expected[worst_index],
        usable_days=usable_days,
        settings=settings.to_dict(),
    )

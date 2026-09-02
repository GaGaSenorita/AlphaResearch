"""Serializable records shared by AlphaResearch methods and environments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Iterable


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid ISO date {value!r}; expected YYYY-MM-DD") from exc


@dataclass(frozen=True)
class Period:
    """Inclusive date interval."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"period start {self.start} is after end {self.end}")

    @classmethod
    def from_strings(cls, start: str, end: str) -> "Period":
        return cls(parse_date(start), parse_date(end))

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class EvaluationMetrics:
    """AlphaBench-compatible IC metrics plus two diagnostic statistics."""

    ic: float | None = None
    rank_ic: float | None = None
    icir: float | None = None
    rank_icir: float | None = None
    quantile_spread: float | None = None
    turnover: float | None = None
    daily_count: int = 0
    observation_count: int = 0

    def value(self, objective: str) -> float:
        value = finite_or_none(getattr(self, objective, None))
        return value if value is not None else float("-inf")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    expression: str
    period: Period
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    error: str | None = None
    daily_metrics: tuple[dict[str, Any], ...] = ()
    cached: bool = False

    def to_dict(self, *, include_daily: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "success": self.success,
            "expression": self.expression,
            "period": self.period.to_dict(),
            "metrics": self.metrics.to_dict(),
            "error": self.error,
            "cached": self.cached,
        }
        if include_daily:
            data["daily_metrics"] = list(self.daily_metrics)
        return data


@dataclass(frozen=True)
class FactorCandidate:
    name: str
    expression: str
    reason: str = ""
    source: str = "llm"
    round_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    best: FactorCandidate
    best_train: EvaluationResult
    archive: tuple[tuple[FactorCandidate, EvaluationResult], ...]
    rounds_requested: int
    llm_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best.to_dict(),
            "best_train": self.best_train.to_dict(),
            "archive": [
                {"candidate": candidate.to_dict(), "train": result.to_dict()}
                for candidate, result in self.archive
            ],
            "rounds_requested": self.rounds_requested,
            "llm_calls": self.llm_calls,
        }


@dataclass(frozen=True)
class ValidationSelection:
    selected: tuple[FactorCandidate, ...]
    evaluations: tuple[tuple[FactorCandidate, EvaluationResult], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [candidate.to_dict() for candidate in self.selected],
            "evaluations": [
                {"candidate": candidate.to_dict(), "validation": result.to_dict()}
                for candidate, result in self.evaluations
            ],
        }


def aggregate_daily_metrics(rows: Iterable[dict[str, Any]]) -> EvaluationMetrics:
    """Aggregate already revealed daily IC rows without touching raw future data."""

    materialized = list(rows)
    ic_values = [float(row["ic"]) for row in materialized if finite_or_none(row.get("ic")) is not None]
    rank_values = [
        float(row["rank_ic"])
        for row in materialized
        if finite_or_none(row.get("rank_ic")) is not None
    ]
    spreads = [
        float(row["quantile_spread"])
        for row in materialized
        if finite_or_none(row.get("quantile_spread")) is not None
    ]
    turnovers = [
        float(row["turnover"])
        for row in materialized
        if finite_or_none(row.get("turnover")) is not None
    ]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def sample_std(values: list[float]) -> float | None:
        if len(values) < 2:
            return None
        center = sum(values) / len(values)
        return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))

    def ir(values: list[float]) -> float | None:
        center = mean(values)
        std = sample_std(values)
        if center is None or std is None or std <= 1e-12:
            return None
        return center * math.sqrt(len(values)) / std

    return EvaluationMetrics(
        ic=mean(ic_values),
        rank_ic=mean(rank_values),
        icir=ir(ic_values),
        rank_icir=ir(rank_values),
        quantile_spread=mean(spreads),
        turnover=mean(turnovers),
        daily_count=max(len(ic_values), len(rank_values)),
        observation_count=sum(int(row.get("observation_count", 0)) for row in materialized),
    )

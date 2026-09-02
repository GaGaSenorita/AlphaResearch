"""Cross-period stability reward computed from one evaluation's daily series."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


AGGREGATORS = ("mean_minus_lambda_std", "worst", "sign_weighted")


@dataclass(frozen=True)
class SplitSettings:
    embargo_days: int = 60
    penalty_lambda: float = 0.5
    aggregator: str = "mean_minus_lambda_std"
    min_segment_days: int = 120
    min_segments: int = 2

    def __post_init__(self) -> None:
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")
        if self.aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        if self.min_segment_days < 2 or self.min_segments < 2:
            raise ValueError("min_segment_days and min_segments must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "segmentation": "calendar_year",
            "embargo_days": self.embargo_days,
            "penalty_lambda": self.penalty_lambda,
            "aggregator": self.aggregator,
            "min_segment_days": self.min_segment_days,
            "min_segments": self.min_segments,
        }


@dataclass(frozen=True)
class SplitScore:
    score: float
    years: tuple[int, ...] = ()
    segment_means: tuple[float, ...] = ()
    segment_days: tuple[int, ...] = ()
    mean: float | None = None
    std: float | None = None
    worst: float | None = None
    sign_consistency: float | None = None
    noise_std: float | None = None
    dispersion_ratio: float | None = None
    usable_days: int = 0
    rejected: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score if math.isfinite(self.score) else None,
            "years": list(self.years),
            "segment_means": list(self.segment_means),
            "segment_days": list(self.segment_days),
            "mean": self.mean,
            "std": self.std,
            "worst": self.worst,
            "sign_consistency": self.sign_consistency,
            "noise_std": self.noise_std,
            "dispersion_ratio": self.dispersion_ratio,
            "usable_days": self.usable_days,
            "rejected": self.rejected,
            "settings": dict(self.settings),
        }


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    centre = sum(values) / len(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1))


def _daily_by_year(result: Any) -> dict[int, list[float]]:
    rows: list[tuple[str, int, float]] = []
    for record in getattr(result, "daily_metrics", ()) or ():
        if not isinstance(record, dict):
            continue
        day, value = record.get("date"), record.get("rank_ic")
        text = str(day).strip()
        if day is None or value is None or len(text) < 4 or not text[:4].isdigit():
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            rows.append((text, int(text[:4]), number))
    rows.sort(key=lambda item: item[0])
    grouped: dict[int, list[float]] = {}
    for _, year, number in rows:
        grouped.setdefault(year, []).append(number)
    return grouped


def split_score(result: Any, settings: SplitSettings) -> SplitScore:
    grouped = _daily_by_year(result)
    total = sum(len(values) for values in grouped.values())
    if not grouped:
        return SplitScore(
            score=float("-inf"),
            rejected="no finite daily rank_ic rows",
            settings=settings.to_dict(),
        )

    years: list[int] = []
    blocks: list[list[float]] = []
    for year in sorted(grouped):
        block = grouped[year][settings.embargo_days :]
        if len(block) >= settings.min_segment_days:
            years.append(year)
            blocks.append(block)
    if len(blocks) < settings.min_segments:
        return SplitScore(
            score=float("-inf"),
            usable_days=total,
            rejected=f"only {len(blocks)} usable calendar-year segments",
            settings=settings.to_dict(),
        )

    means = tuple(sum(block) / len(block) for block in blocks)
    counts = tuple(len(block) for block in blocks)
    mean = sum(means) / len(means)
    std = _sample_std(list(means))
    worst = min(means)
    sign_consistency = sum(value > 0 for value in means) / len(means)
    within_variances = [value ** 2 for block in blocks if (value := _sample_std(block)) is not None]
    noise_std = None
    dispersion_ratio = None
    if within_variances:
        pooled_daily_std = math.sqrt(sum(within_variances) / len(within_variances))
        noise_std = pooled_daily_std / math.sqrt(sum(counts) / len(counts))
        if noise_std > 1e-12 and std is not None:
            dispersion_ratio = std / noise_std

    if settings.aggregator == "worst":
        score = worst
    elif settings.aggregator == "sign_weighted":
        score = mean * sign_consistency if mean > 0 else mean
    else:
        score = mean - settings.penalty_lambda * (std or 0.0)
    return SplitScore(
        score=float(score),
        years=tuple(years),
        segment_means=means,
        segment_days=counts,
        mean=mean,
        std=std,
        worst=worst,
        sign_consistency=sign_consistency,
        noise_std=noise_std,
        dispersion_ratio=dispersion_ratio,
        usable_days=total,
        settings=settings.to_dict(),
    )

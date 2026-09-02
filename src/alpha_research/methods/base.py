"""Stable interface between alpha-research methods and environments."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

from alpha_research.types import FactorCandidate, Period, SearchResult


class ResearchMethod(Protocol):
    name: str

    def search(
        self,
        *,
        train_period: Period,
        rounds: int,
        seed_factors: Sequence[FactorCandidate] | None = None,
        prior_revealed_feedback: Sequence[dict[str, Any]] = (),
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        cycle_id: str = "static",
    ) -> SearchResult: ...

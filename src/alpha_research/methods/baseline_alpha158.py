"""Deterministic Alpha158 baseline with LDM-compatible de-correlation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from alpha_research.alphabench_runtime import load_alpha158_seeds, resolve_alphabench_root
from alpha_research.evaluator import FactorEvaluator, select_on_validation
from alpha_research.types import FactorCandidate, Period, SearchResult, ValidationSelection


EventSink = Callable[[dict[str, Any]], None]


class Alpha158Baseline:
    """Evaluate fixed non-VWAP Alpha158 factors with the LDM selection policy."""

    name = "baseline_alpha158"

    def __init__(
        self,
        *,
        evaluator: FactorEvaluator,
        alphabench_root: str | Path,
        objective: str = "rank_ic",
        seed_group: str = "all",
        random_seed: int = 42,
    ) -> None:
        self.evaluator = evaluator
        self.alphabench_root = resolve_alphabench_root(alphabench_root)
        self.objective = str(objective)
        self.seed_group = str(seed_group)
        self.random_seed = int(random_seed)

    def search(
        self,
        *,
        train_period: Period,
        rounds: int,
        seed_factors: Sequence[FactorCandidate] | None = None,
        prior_revealed_feedback: Sequence[dict[str, Any]] = (),
        event_sink: EventSink | None = None,
        cycle_id: str = "static",
    ) -> SearchResult:
        if prior_revealed_feedback:
            raise ValueError("baseline_alpha158 supports only the static protocol")
        emit = event_sink or (lambda _event: None)
        factors = (
            tuple(seed_factors)
            if seed_factors
            else load_alpha158_seeds(self.alphabench_root, seed_group=self.seed_group)
        )
        results = self.evaluator.evaluate_many(factors, train_period)
        archive = tuple(zip(factors, results))
        for candidate, result in archive:
            emit({
                "event": "train_evaluation",
                "cycle_id": cycle_id,
                "stage": "alpha158_fixed_library",
                "candidate": candidate.to_dict(),
                "result": result.to_dict(),
            })
        successful = [item for item in archive if item[1].success]
        if not successful:
            raise RuntimeError("none of the fixed Alpha158 factors evaluated successfully")
        best, best_train = max(
            successful,
            key=lambda item: item[1].metrics.value(self.objective),
        )
        emit({
            "event": "alpha158_fixed_library",
            "cycle_id": cycle_id,
            "factor_count": len(factors),
            "successful": len(successful),
            "exclude_var": "vwap",
            "seed_group": self.seed_group,
            "random_seed": self.random_seed,
            "deterministic": True,
        })
        return SearchResult(
            best=best,
            best_train=best_train,
            archive=archive,
            rounds_requested=max(0, int(rounds)),
            llm_calls=0,
        )

    def select_on_validation(
        self,
        *,
        archive,
        evaluator: FactorEvaluator,
        validation_period: Period,
        factor_budget: int,
        objective: str,
        event_sink: EventSink | None = None,
        cycle_id: str = "static",
        max_correlation: float | None = None,
    ) -> ValidationSelection:
        return select_on_validation(
            archive=archive,
            evaluator=evaluator,
            validation_period=validation_period,
            factor_budget=factor_budget,
            objective=objective,
            event_sink=event_sink,
            cycle_id=cycle_id,
            max_correlation=max_correlation,
            backfill_to_budget=True,
        )


# Historical name retained so old analysis notebooks keep importing cleanly.
Alpha158Corr07 = Alpha158Baseline

"""The LDM search, read out as a single validation-selected factor."""

from __future__ import annotations

from typing import Any, Sequence

from alpha_research.canonical import canonical_form
from alpha_research.evaluator import select_on_validation as select_validation_archive
from alpha_research.io import write_json
from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.types import EvaluationResult, FactorCandidate, Period, ValidationSelection


RUNNERS_UP_RECORDED = 10


def _safe_canonical(expression: str) -> str:
    try:
        return canonical_form(expression)
    except Exception:
        return expression.strip()


class SingleFactorLDM(AlphaLDM):
    """Keep the search intact and change only the final validation readout."""

    name = "ldm_single_factor"
    candidate_source = "ldm_standard"

    def select_on_validation(
        self,
        *,
        archive: Sequence[tuple[FactorCandidate, EvaluationResult]],
        evaluator: Any,
        validation_period: Period,
        factor_budget: int,
        objective: str,
        event_sink: Any | None = None,
        cycle_id: str = "static",
        max_correlation: float | None = None,
    ) -> ValidationSelection:
        if int(factor_budget) != 1:
            raise RuntimeError(
                f"{self.name} delivers one factor; configure factor-budget: 1"
            )
        if max_correlation is not None:
            raise RuntimeError(
                f"{self.name} has no correlation filter; configure max-correlation: null"
            )
        selection = select_validation_archive(
            archive=archive,
            evaluator=evaluator,
            validation_period=validation_period,
            factor_budget=1,
            objective=objective,
            event_sink=event_sink,
            cycle_id=cycle_id,
            max_correlation=None,
            backfill_to_budget=False,
        )
        if len(selection.selected) != 1:
            raise RuntimeError(f"expected one selected factor, got {len(selection.selected)}")
        self._write_selection_record(
            archive=archive,
            selection=selection,
            objective=objective,
            validation_period=validation_period,
            emit=event_sink or (lambda _event: None),
            cycle_id=cycle_id,
        )
        return selection

    def _write_selection_record(
        self,
        *,
        archive: Sequence[tuple[FactorCandidate, EvaluationResult]],
        selection: ValidationSelection,
        objective: str,
        validation_period: Period,
        emit: Any,
        cycle_id: str,
    ) -> None:
        train_by_expression: dict[str, EvaluationResult] = {}
        for candidate, result in archive:
            train_by_expression.setdefault(_safe_canonical(candidate.expression), result)
        ranked = sorted(
            (item for item in selection.evaluations if item[1].success),
            key=lambda item: item[1].metrics.value(objective),
            reverse=True,
        )

        def row(candidate: FactorCandidate, validation: EvaluationResult) -> dict[str, Any]:
            train = train_by_expression.get(_safe_canonical(candidate.expression))
            return {
                "candidate": candidate.to_dict(),
                "train": train.to_dict(include_daily=False) if train else None,
                "validation": validation.to_dict(include_daily=False),
            }

        winner_candidate, winner_validation = ranked[0]
        write_json(self.output_dir / "single_factor_selection.json", {
            "method": self.name,
            "objective": objective,
            "validation_period": validation_period.to_dict(),
            "selection_rule": f"highest validation {objective}; no correlation filter",
            "archive_candidates": len(archive),
            "unique_candidates_evaluated": len(selection.evaluations),
            "successful_on_validation": len(ranked),
            "winner": row(winner_candidate, winner_validation),
            "runners_up": [
                row(candidate, validation)
                for candidate, validation in ranked[1 : 1 + RUNNERS_UP_RECORDED]
            ],
        })
        winner_train = train_by_expression.get(_safe_canonical(winner_candidate.expression))
        emit({
            "event": "single_factor_selection",
            "cycle_id": cycle_id,
            "expression": winner_candidate.expression,
            "source": winner_candidate.source,
            "round_id": winner_candidate.round_id,
            "train": winner_train.metrics.value(objective) if winner_train else None,
            "validation": winner_validation.metrics.value(objective),
        })

"""Leakage-safe rolling auto-research environment."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from alpha_research.evaluator import FactorEvaluator, select_on_validation
from alpha_research.io import append_jsonl, write_json
from alpha_research.methods.base import ResearchMethod
from alpha_research.types import FactorCandidate, Period, aggregate_daily_metrics


def add_months(value: date, months: int) -> date:
    absolute = value.year * 12 + value.month - 1 + int(months)
    year, zero_month = divmod(absolute, 12)
    month = zero_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class RollingCycle:
    cycle_id: str
    train: Period
    validation: Period
    forward: Period

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "forward": self.forward.to_dict(),
        }


@dataclass(frozen=True)
class OnlineProtocol:
    history_start: date
    online_start: date
    online_end: date
    step_months: int
    validation_months: int
    train_lookback_months: int
    train_mode: str
    memory_mode: str
    factor_budget: int
    search_rounds_per_cycle: int
    objective: str = "rank_ic"

    def __post_init__(self) -> None:
        if not self.history_start < self.online_start <= self.online_end:
            raise ValueError("require history_start < online_start <= online_end")
        if self.step_months <= 0:
            raise ValueError("step_months must be positive")
        if self.validation_months <= 0:
            raise ValueError("validation_months must be positive")
        if self.train_lookback_months <= 0:
            raise ValueError("train_lookback_months must be positive")
        if self.train_mode not in {"rolling", "expanding"}:
            raise ValueError("train_mode must be rolling or expanding")
        if self.memory_mode not in {"continual", "reset"}:
            raise ValueError("memory_mode must be continual or reset")
        if self.factor_budget <= 0:
            raise ValueError("factor_budget must be positive")
        if self.search_rounds_per_cycle < 0:
            raise ValueError("search_rounds_per_cycle cannot be negative")
        first_validation = add_months(self.online_start, -self.validation_months)
        if first_validation <= self.history_start:
            raise ValueError("history_start leaves no training data before the first validation window")
        if self.train_mode == "rolling":
            first_train = add_months(first_validation, -self.train_lookback_months)
            if first_train < self.history_start:
                raise ValueError(
                    "history_start is too late for the requested first rolling train lookback"
                )

    def cycles(self) -> tuple[RollingCycle, ...]:
        cycles: list[RollingCycle] = []
        forward_start = self.online_start
        index = 0
        while forward_start <= self.online_end:
            next_start = add_months(forward_start, self.step_months)
            forward_end = min(self.online_end, next_start - timedelta(days=1))
            validation_start = add_months(forward_start, -self.validation_months)
            validation_end = forward_start - timedelta(days=1)
            train_end = validation_start - timedelta(days=1)
            if self.train_mode == "expanding":
                train_start = self.history_start
            else:
                train_start = add_months(validation_start, -self.train_lookback_months)
            cycles.append(RollingCycle(
                cycle_id=f"cycle_{index:03d}_{forward_start.isoformat()}",
                train=Period(train_start, train_end),
                validation=Period(validation_start, validation_end),
                forward=Period(forward_start, forward_end),
            ))
            forward_start = next_start
            index += 1
        return tuple(cycles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": "online",
            "history_start": self.history_start.isoformat(),
            "online_start": self.online_start.isoformat(),
            "online_end": self.online_end.isoformat(),
            "step_months": self.step_months,
            "validation_months": self.validation_months,
            "train_lookback_months": self.train_lookback_months,
            "train_mode": self.train_mode,
            "memory_mode": self.memory_mode,
            "factor_budget": self.factor_budget,
            "search_rounds_per_cycle": self.search_rounds_per_cycle,
            "objective": self.objective,
            "cycles": [cycle.to_dict() for cycle in self.cycles()],
            "leakage_policy": {
                "current_cycle_llm_visible": ["current_train_metrics", "past_revealed_forward_metrics"],
                "current_cycle_selection_only": ["current_validation_metrics"],
                "hidden_until_window_closes": ["current_forward_metrics"],
                "segment_tail_purge": "configured label_horizon_days in evaluator metadata",
            },
        }


class OnlineEnvironment:
    def __init__(
        self,
        *,
        protocol: OnlineProtocol,
        method: ResearchMethod,
        evaluator: FactorEvaluator,
        output_dir: str | Path,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.protocol = protocol
        self.method = method
        self.evaluator = evaluator
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.run_metadata = dict(run_metadata or {})

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.output_dir / "events.jsonl"
        events_path.write_text("", encoding="utf-8")
        emit = lambda event: append_jsonl(events_path, event)
        protocol_payload = {**self.protocol.to_dict(), "metadata": self.run_metadata}
        write_json(self.output_dir / "protocol.json", protocol_payload)

        previous_selected: tuple[FactorCandidate, ...] = ()
        revealed_feedback: list[dict[str, Any]] = []
        cycle_summaries: list[dict[str, Any]] = []
        stitched_daily: list[dict[str, Any]] = []

        for cycle in self.protocol.cycles():
            cycle_dir = self.output_dir / cycle.cycle_id
            cycle_dir.mkdir(parents=True, exist_ok=True)
            emit({"event": "cycle_started", **cycle.to_dict()})

            if self.protocol.memory_mode == "continual" and previous_selected:
                seeds = previous_selected
                feedback_for_agent = tuple(revealed_feedback)
            else:
                seeds = None
                feedback_for_agent = ()

            search = self.method.search(
                train_period=cycle.train,
                rounds=self.protocol.search_rounds_per_cycle,
                seed_factors=seeds,
                prior_revealed_feedback=feedback_for_agent,
                event_sink=emit,
                cycle_id=cycle.cycle_id,
            )
            selection = select_on_validation(
                archive=search.archive,
                evaluator=self.evaluator,
                validation_period=cycle.validation,
                factor_budget=self.protocol.factor_budget,
                objective=self.protocol.objective,
                event_sink=emit,
                cycle_id=cycle.cycle_id,
            )

            forward_individual = []
            individual_results = self.evaluator.evaluate_many(selection.selected, cycle.forward)
            for candidate, result in zip(selection.selected, individual_results):
                record = {"candidate": candidate.to_dict(), "forward": result.to_dict()}
                forward_individual.append(record)
                emit({"event": "forward_evaluation", "cycle_id": cycle.cycle_id, **record})
            forward_pool = self.evaluator.evaluate_pool(selection.selected, cycle.forward)
            emit({
                "event": "forward_pool_evaluation",
                "cycle_id": cycle.cycle_id,
                "selected": [candidate.to_dict() for candidate in selection.selected],
                "result": forward_pool.to_dict(),
            })
            stitched_daily.extend(forward_pool.daily_metrics)

            cycle_summary = {
                **cycle.to_dict(),
                "memory_available_at_cycle_start": list(feedback_for_agent),
                "selected_factors": [candidate.to_dict() for candidate in selection.selected],
                "forward": {
                    "individual": forward_individual,
                    "equal_weight_rank_pool": forward_pool.to_dict(),
                },
                "search_artifact": str(cycle_dir / "search.json"),
                "validation_artifact": str(cycle_dir / "validation_selection.json"),
            }
            write_json(cycle_dir / "search.json", search.to_dict())
            write_json(cycle_dir / "validation_selection.json", selection.to_dict())
            write_json(cycle_dir / "summary.json", cycle_summary)
            cycle_summaries.append(cycle_summary)

            # The simulation reaches this line only after the forward window has
            # closed. Its result is now legal historical feedback for cycle t+1.
            revealed = {
                "cycle_id": cycle.cycle_id,
                "forward_period": cycle.forward.to_dict(),
                "selected_factors": [candidate.to_dict() for candidate in selection.selected],
                "pool_metrics": forward_pool.metrics.to_dict(),
            }
            revealed_feedback.append(revealed)
            previous_selected = selection.selected
            emit({"event": "forward_feedback_revealed", **revealed})

        overall_metrics = aggregate_daily_metrics(stitched_daily)
        summary = {
            "status": "completed",
            "environment": "online",
            "method": self.method.name,
            "protocol": protocol_payload,
            "cycles_completed": len(cycle_summaries),
            "cycles": cycle_summaries,
            "stitched_forward": {
                "metrics": overall_metrics.to_dict(),
                "daily_metrics": stitched_daily,
            },
            "final_factor_pool": [candidate.to_dict() for candidate in previous_selected],
            "artifacts": {"events": str(events_path)},
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary

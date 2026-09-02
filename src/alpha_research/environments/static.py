"""Conventional fixed train/validation/test alpha-mining environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_research.evaluator import (
    FactorEvaluator,
    select_on_validation,
)
from alpha_research.io import append_jsonl, read_json, write_json
from alpha_research.methods.base import ResearchMethod
from alpha_research.types import Period


@dataclass(frozen=True)
class StaticProtocol:
    train: Period
    validation: Period
    test: Period
    factor_budget: int
    search_rounds: int
    objective: str = "rank_ic"
    max_correlation: float | None = None

    def __post_init__(self) -> None:
        if not self.train.end < self.validation.start:
            raise ValueError("static train period must end before validation starts")
        if not self.validation.end < self.test.start:
            raise ValueError("static validation period must end before test starts")
        if self.factor_budget <= 0:
            raise ValueError("factor_budget must be positive")
        if self.search_rounds < 0:
            raise ValueError("search_rounds cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": "static",
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "factor_budget": self.factor_budget,
            "search_rounds": self.search_rounds,
            "objective": self.objective,
            "max_correlation": self.max_correlation,
            "leakage_policy": {
                "llm_visible_during_search": ["train_metrics"],
                "selection_only": ["validation_metrics"],
                "report_only": ["test_metrics"],
                "segment_tail_purge": "configured label_horizon_days in evaluator metadata",
            },
        }


class StaticEnvironment:
    def __init__(
        self,
        *,
        protocol: StaticProtocol,
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
        preserve_events = bool(getattr(self.method, "preserve_existing_events", False))
        if not preserve_events or not events_path.exists():
            events_path.write_text("", encoding="utf-8")
        emit = lambda event: append_jsonl(events_path, event)

        protocol_payload = {
            **self.protocol.to_dict(),
            "metadata": self.run_metadata,
        }
        protocol_path = self.output_dir / "protocol.json"
        if preserve_events and protocol_path.is_file():
            previous = read_json(protocol_path)
            for key in (
                "environment",
                "train",
                "validation",
                "test",
                "factor_budget",
                "objective",
                "max_correlation",
                "metadata",
            ):
                if previous.get(key) != protocol_payload.get(key):
                    raise ValueError(
                        f"cannot resume with changed protocol field {key!r}"
                    )
        write_json(protocol_path, protocol_payload)
        emit({
            "event": "run_session_started",
            "cycle_id": "static",
            "resumed": preserve_events,
            "target_round": self.protocol.search_rounds,
        })

        search = self.method.search(
            train_period=self.protocol.train,
            rounds=self.protocol.search_rounds,
            event_sink=emit,
            cycle_id="static",
        )
        write_json(self.output_dir / "search.json", search.to_dict())

        selector = getattr(self.method, "select_on_validation", None)
        selection_fn = selector if callable(selector) else select_on_validation
        selection = selection_fn(
            archive=search.archive,
            evaluator=self.evaluator,
            validation_period=self.protocol.validation,
            factor_budget=self.protocol.factor_budget,
            objective=self.protocol.objective,
            event_sink=emit,
            cycle_id="static",
            max_correlation=self.protocol.max_correlation,
        )
        write_json(self.output_dir / "validation_selection.json", selection.to_dict())
        test_individual = []
        individual_results = self.evaluator.evaluate_many(selection.selected, self.protocol.test)
        for candidate, result in zip(selection.selected, individual_results):
            record = {"candidate": candidate.to_dict(), "test": result.to_dict()}
            test_individual.append(record)
            emit({"event": "test_evaluation", "cycle_id": "static", **record})
        test_pool = self.evaluator.evaluate_pool(selection.selected, self.protocol.test)
        emit({
            "event": "test_pool_evaluation",
            "cycle_id": "static",
            "selected": [candidate.to_dict() for candidate in selection.selected],
            "result": test_pool.to_dict(),
        })

        artifacts = {
            "events": str(events_path),
            "search": str(self.output_dir / "search.json"),
            "validation_selection": str(self.output_dir / "validation_selection.json"),
        }
        checkpoint_hook = getattr(self.method, "write_checkpoint_report", None)
        if callable(checkpoint_hook):
            report_path = checkpoint_hook(
                search=search,
                selection=selection,
                test_individual=test_individual,
                test_pool=test_pool,
                evaluator=self.evaluator,
                validation_period=self.protocol.validation,
                test_period=self.protocol.test,
                objective=self.protocol.objective,
                max_correlation=self.protocol.max_correlation,
                event_sink=emit,
                cycle_id="static",
            )
            artifacts.update({
                "factor_ledger": str(self.output_dir / "factor_ledger.jsonl"),
                "factor_sequence": str(self.output_dir / "factor_sequence.json"),
                "resume_state": str(self.output_dir / "resume_state.json"),
                "top5_combinations": str(report_path),
            })
        summary = {
            "status": "completed",
            "environment": "static",
            "protocol": protocol_payload,
            "method": self.method.name,
            "selected_factors": [candidate.to_dict() for candidate in selection.selected],
            "test": {
                "individual": test_individual,
                "equal_weight_rank_pool": test_pool.to_dict(),
            },
            "artifacts": artifacts,
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary

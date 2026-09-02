"""Continuous AlphaLDM search with committed rounds and staged Top-5 reports.

The search path is the pure scalar-UCB AlphaLDM path.  This variant changes
only durability and reporting: each complete round is committed to an
append-only ledger, a clean factor-order view is refreshed atomically, and a
later invocation can reconstruct the GP observations and RNG state exactly.

Checkpoint Validation/Test results are diagnostics.  They are never put into
the generator context, history, GP, or acquisition function.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alpha_research.evaluator import _daily_series, _series_correlation
from alpha_research.io import append_jsonl, read_json, write_json
from alpha_research.methods.ldm.history import History
from alpha_research.methods.ldm.method import (
    AlphaLDM,
    EventSink,
    _LDMSearchState,
    _safe_canonical,
)
from alpha_research.types import (
    EvaluationMetrics,
    EvaluationResult,
    FactorCandidate,
    Period,
    SearchResult,
    ValidationSelection,
)


LEDGER_SCHEMA = "alphaldm.continuous.factor-ledger.v1"
REPORT_SCHEMA = "alphaldm.continuous.top5-checkpoints.v1"


def _result_from_dict(data: dict[str, Any]) -> EvaluationResult:
    period = data.get("period") or {}
    metrics = data.get("metrics") or {}
    return EvaluationResult(
        success=bool(data.get("success")),
        expression=str(data.get("expression", "")),
        period=Period.from_strings(str(period["start"]), str(period["end"])),
        metrics=EvaluationMetrics(**{
            key: metrics.get(key)
            for key in EvaluationMetrics.__dataclass_fields__
        }),
        error=data.get("error"),
        daily_metrics=tuple(data.get("daily_metrics") or ()),
        cached=bool(data.get("cached", False)),
    )


def _candidate_from_dict(data: dict[str, Any]) -> FactorCandidate:
    return FactorCandidate(
        name=str(data["name"]),
        expression=str(data["expression"]),
        reason=str(data.get("reason", "")),
        source=str(data.get("source", "llm")),
        round_id=int(data.get("round_id", 0)),
    )


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"corrupt continuous ledger {path} at line {line_number}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _select_precomputed(
    evaluations: Sequence[tuple[FactorCandidate, EvaluationResult]],
    *,
    budget: int,
    objective: str,
    max_correlation: float | None,
) -> tuple[tuple[FactorCandidate, EvaluationResult], ...]:
    ranked = sorted(
        (item for item in evaluations if item[1].success),
        key=lambda item: item[1].metrics.value(objective),
        reverse=True,
    )
    daily_key = "rank_ic" if objective.startswith("rank") else "ic"
    selected: list[tuple[FactorCandidate, EvaluationResult]] = []
    selected_series: list[dict[str, float]] = []
    for candidate, result in ranked:
        if len(selected) >= int(budget):
            break
        series = _daily_series(result, daily_key)
        if max_correlation is not None:
            correlations = [
                _series_correlation(series, existing)
                for existing in selected_series
            ]
            finite = [value for value in correlations if value is not None]
            if finite and max(abs(value) for value in finite) > float(max_correlation):
                continue
        selected.append((candidate, result))
        selected_series.append(series)
    if len(selected) != int(budget):
        raise RuntimeError(
            f"checkpoint round cannot form Top-{budget} under the hard "
            f"correlation boundary {max_correlation!r}; found {len(selected)}"
        )
    return tuple(selected)


class ContinuousDiscoveryLDM(AlphaLDM):
    """Pure LDM with exact round resume and staged Top-5 diagnostics."""

    name = "ldm_continuous_discovery"
    candidate_source = "ldm_continuous_discovery"

    def __init__(
        self,
        *,
        resume: bool = False,
        legacy_source: str | Path | None = None,
        legacy_expected_protocol: dict[str, Any] | None = None,
        checkpoint_rounds: Sequence[int] = (),
        checkpoint_factor_budget: int = 5,
        checkpoint_rank_ic_threshold: float = 0.035,
        checkpoint_validation_period: Period,
        checkpoint_test_period: Period,
        checkpoint_max_correlation: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.resume = bool(resume)
        self.legacy_source = Path(legacy_source).expanduser().resolve() if legacy_source else None
        self.legacy_expected_protocol = legacy_expected_protocol
        if self.legacy_source is not None and not self.resume:
            raise ValueError("legacy import requires continuous resume enabled")
        requested_checkpoints = {int(value) for value in checkpoint_rounds}
        if any(value < 0 for value in requested_checkpoints):
            raise ValueError("continuous checkpoint rounds cannot be negative")
        # Round zero is the measured Alpha158 warm-start baseline. Keeping it as
        # a first-class checkpoint makes the later R10/R20/... trajectory start
        # from an actual Validation-selected Top-5 rather than an interpolated
        # or visually implied Test value.
        self.checkpoint_rounds = tuple(sorted(requested_checkpoints))
        self.checkpoint_factor_budget = max(1, int(checkpoint_factor_budget))
        self.checkpoint_rank_ic_threshold = float(checkpoint_rank_ic_threshold)
        if not math.isfinite(self.checkpoint_rank_ic_threshold):
            raise ValueError("checkpoint RankIC threshold must be finite")
        self.checkpoint_validation_period = checkpoint_validation_period
        self.checkpoint_test_period = checkpoint_test_period
        self.checkpoint_max_correlation = checkpoint_max_correlation

    @property
    def ledger_path(self) -> Path:
        return self.output_dir / "factor_ledger.jsonl"

    @property
    def preserve_existing_events(self) -> bool:
        return self.resume and self.ledger_path.is_file()

    def _signature(self, train_period: Period) -> dict[str, Any]:
        schema = self.profiler.schema
        references = list(getattr(self.profiler, "reference_expressions", ()))
        generator_client = getattr(self.generator, "client", None)
        reference_digest = hashlib.sha256(
            json.dumps(references, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "method": self.name,
            "train_period": train_period.to_dict(),
            "profile_schema_version": schema.version,
            "profile_feature_dim": len(schema.names),
            "reference_expressions_sha256": reference_digest,
            "objective": self.objective,
            "search_objective": self.search_objective,
            "seed_group": self.seed_group,
            "acquisition": self.acquisition,
            "acquisition_beta": self.acquisition_beta,
            "acquisition_xi": self.acquisition_xi,
            "softmax_temperature": self.softmax_temperature,
            "diversity_weight": self.diversity_weight,
            "evaluate_per_round": self.evaluate_per_round,
            "max_refill_batches": self.max_refill_batches,
            "generator": {
                "class": type(self.generator).__name__,
                "model": getattr(generator_client, "model_name", None),
                "candidates_per_round": getattr(
                    self.generator, "candidates_per_round", None
                ),
                "max_parallel": getattr(self.generator, "max_parallel", None),
                "history_shown": getattr(self.generator, "history_shown", None),
            },
            "gp": self.gp_kwargs,
            "random_seed": self.random_seed,
        }

    def _committed_rows(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = _read_jsonl(self.ledger_path)
        factors_by_attempt: dict[str, list[dict[str, Any]]] = {}
        commits: list[dict[str, Any]] = []
        for row in rows:
            attempt_id = str(row.get("attempt_id", ""))
            if row.get("record_type") == "factor_evaluation":
                factors_by_attempt.setdefault(attempt_id, []).append(row)
            elif row.get("record_type") == "round_commit":
                commits.append(row)

        accepted_commits: list[dict[str, Any]] = []
        committed_factors: list[dict[str, Any]] = []
        expected_round = 0
        for commit in commits:
            round_id = int(commit.get("round_id", -1))
            if round_id < expected_round:
                continue
            if round_id != expected_round:
                break
            attempt_id = str(commit.get("attempt_id", ""))
            attempt_rows = factors_by_attempt.get(attempt_id, [])
            expected_count = int(commit.get("observation_count", -1))
            if len(attempt_rows) != expected_count:
                break
            accepted_commits.append(commit)
            committed_factors.extend(attempt_rows)
            expected_round += 1
        return committed_factors, accepted_commits

    def _restore_search_state(
        self,
        *,
        train_period: Period,
        rounds: int,
        seed_factors: Sequence[FactorCandidate] | None,
        emit: EventSink,
        cycle_id: str,
    ) -> _LDMSearchState | None:
        del seed_factors
        if self.legacy_source is not None and not self.ledger_path.exists():
            from .legacy import import_legacy_run

            import_legacy_run(self, train_period=train_period, target_rounds=rounds)
        if not self.resume or not self.ledger_path.exists():
            return None
        factors, commits = self._committed_rows()
        if not commits:
            raise RuntimeError(
                f"{self.ledger_path} exists but contains no complete round commit"
            )
        latest = commits[-1]
        completed_round = int(latest["round_id"])
        if completed_round > int(rounds):
            raise ValueError(
                f"requested target {rounds} is below committed round {completed_round}"
            )
        signature = self._signature(train_period)
        if latest.get("signature") != signature:
            raise ValueError(
                "continuous resume signature mismatch; the data/schema/search settings "
                "changed, so resuming would reinterpret prior observations"
            )

        schema = self.profiler.schema
        history = self._new_history(dim=len(schema.names), schema_version=schema.version)
        archive: list[tuple[FactorCandidate, EvaluationResult]] = []
        evaluated_records: list[dict[str, Any]] = []
        best: FactorCandidate | None = None
        best_result: EvaluationResult | None = None
        best_score = float("-inf")
        for row in sorted(factors, key=lambda item: int(item["evaluation_index"])):
            candidate = _candidate_from_dict(row["candidate"])
            result = _result_from_dict(row["train"])
            score = float(row["search_score"])
            feature = np.asarray(row["feature"], dtype=float)
            self._record_observation(
                history,
                feature=feature,
                score=score,
                result=result,
                canonical=str(row["canonical"]),
                expression=candidate.expression,
                round_id=candidate.round_id,
            )
            archive.append((candidate, result))
            evaluated_records.append(self._evaluated_record(
                expression=candidate.expression,
                score=score,
                result=result,
            ))
            if score > best_score:
                best, best_result, best_score = candidate, result, score
        if history.n != int(latest["history_observations"]):
            raise RuntimeError(
                "continuous ledger/history mismatch: reconstructed "
                f"{history.n}, commit declares {latest['history_observations']}"
            )

        surrogate = self._new_surrogate(feature_dim=len(schema.names), history=history)
        if self.acquisition != "random" and history.n:
            surrogate.fit(history.features(), history.scores())
        rng = random.Random()
        rng.setstate(_as_tuple(latest["rng_state"]))
        state = _LDMSearchState(
            rng=rng,
            history=history,
            surrogate=surrogate,
            archive=archive,
            evaluated_records=evaluated_records,
            best=best,
            best_result=best_result,
            best_score=best_score,
            completed_round=completed_round,
            llm_calls_completed=int(latest.get("llm_calls_total", 0)),
        )
        restore_generator = getattr(self.generator, "restore_state", None)
        if callable(restore_generator):
            restore_generator(
                completed_round=completed_round,
                llm_calls_total=state.llm_calls_completed,
            )
        history.save(self.output_dir / "ldm_history.csv")
        self._write_factor_sequence(factors, commits)
        emit({
            "event": "ldm_continuous_resumed",
            "cycle_id": cycle_id,
            "completed_round": completed_round,
            "target_round": int(rounds),
            "gp_observations": history.n,
            "llm_calls_completed": state.llm_calls_completed,
        })
        # Reporting policy is deliberately absent from the resume signature:
        # adding a denser audit schedule must not change or invalidate the
        # already committed search trajectory. On any resume, fill missing
        # retrospective checkpoints that are already behind the search head.
        # The target round itself is written by StaticEnvironment's final
        # Validation/Test pass, so it is excluded here to avoid duplicate work.
        existing_checkpoint_rounds = {
            int(row.get("checkpoint_round", -1))
            for row in self._report_payload().get("checkpoints", [])
        }
        for checkpoint_round in self.checkpoint_rounds:
            if checkpoint_round in existing_checkpoint_rounds:
                continue
            if checkpoint_round > completed_round or checkpoint_round >= int(rounds):
                continue
            emit({
                "event": "ldm_continuous_backfill_checkpoint_started",
                "checkpoint_round": checkpoint_round,
                "completed_round": completed_round,
                "reporting_only": True,
            })
            self._evaluate_checkpoint(
                checkpoint_round=checkpoint_round, archive=archive,
                event_sink=emit, cycle_id=cycle_id,
            )
        return state

    def _persist_search_checkpoint(
        self,
        *,
        state: _LDMSearchState,
        train_period: Period,
        target_rounds: int,
        observations: Sequence[dict[str, Any]],
        llm_calls_total: int,
        emit: EventSink,
        cycle_id: str,
    ) -> None:
        round_id = int(state.completed_round)
        attempt_id = f"round-{round_id:06d}-{time.time_ns()}"
        if round_id == 0:
            features = state.history.features()
            scores = state.history.scores()
            materialized = [
                {
                    "candidate": candidate,
                    "result": result,
                    "feature": features[index],
                    "score": float(scores[index]),
                    "canonical": _safe_canonical(candidate.expression),
                    "refill_batch": None,
                    "selection_order": index + 1,
                }
                for index, (candidate, result) in enumerate(state.archive)
            ]
            first_index = 1
        else:
            materialized = list(observations)
            first_index = state.history.n - len(materialized) + 1
        for offset, observation in enumerate(materialized):
            candidate = observation["candidate"]
            result = observation["result"]
            append_jsonl(self.ledger_path, {
                "schema_version": LEDGER_SCHEMA,
                "record_type": "factor_evaluation",
                "attempt_id": attempt_id,
                "round_id": round_id,
                "evaluation_index": first_index + offset,
                "selection_order_within_round": observation["selection_order"],
                "refill_batch": observation["refill_batch"],
                "candidate": candidate.to_dict(),
                "canonical": observation["canonical"],
                "feature": np.asarray(observation["feature"], dtype=float).tolist(),
                "search_objective": self.search_objective,
                "search_score": float(observation["score"]),
                "train": result.to_dict(),
            })
        commit = {
            "schema_version": LEDGER_SCHEMA,
            "record_type": "round_commit",
            "attempt_id": attempt_id,
            "round_id": round_id,
            "target_round": int(target_rounds),
            "observation_count": len(materialized),
            "history_observations": state.history.n,
            "llm_calls_total": int(llm_calls_total),
            "rng_state": state.rng.getstate(),
            "signature": self._signature(train_period),
            "committed_at_unix": time.time(),
        }
        append_jsonl(self.ledger_path, commit)
        write_json(self.output_dir / "resume_state.json", commit)
        state.history.save(self.output_dir / "ldm_history.csv")
        factors, commits = self._committed_rows()
        self._write_factor_sequence(factors, commits)
        emit({
            "event": "ldm_continuous_round_committed",
            "cycle_id": cycle_id,
            "round_id": round_id,
            "target_round": int(target_rounds),
            "gp_observations": state.history.n,
            "factor_sequence_count": len(factors),
            "llm_calls_total": int(llm_calls_total),
        })

        if round_id in self.checkpoint_rounds and round_id < int(target_rounds):
            try:
                self._evaluate_checkpoint(
                    checkpoint_round=round_id,
                    archive=state.archive,
                    event_sink=emit,
                    cycle_id=cycle_id,
                )
            except Exception as exc:
                emit({
                    "event": "ldm_continuous_reporting_failed",
                    "cycle_id": cycle_id,
                    "checkpoint_round": round_id,
                    "error": str(exc),
                    "search_continues": True,
                })

    def _write_factor_sequence(
        self,
        factors: Sequence[dict[str, Any]],
        commits: Sequence[dict[str, Any]],
    ) -> None:
        clean = []
        for row in sorted(factors, key=lambda item: int(item["evaluation_index"])):
            train = row["train"]
            clean.append({
                "evaluation_index": int(row["evaluation_index"]),
                "round_id": int(row["round_id"]),
                "selection_order_within_round": row["selection_order_within_round"],
                "refill_batch": row["refill_batch"],
                "candidate": row["candidate"],
                "canonical": row["canonical"],
                "search_objective": row["search_objective"],
                "search_score": row["search_score"],
                "train_metrics": train.get("metrics", {}),
            })
        write_json(self.output_dir / "factor_sequence.json", {
            "schema_version": LEDGER_SCHEMA,
            "ordering": "successful real Train evaluations in committed search order",
            "completed_round": int(commits[-1]["round_id"]) if commits else None,
            "factor_count": len(clean),
            "factors": clean,
        })

    def _validation_evaluations(
        self,
        *,
        archive: Sequence[tuple[FactorCandidate, EvaluationResult]],
        event_sink: EventSink,
        cycle_id: str,
        checkpoint_round: int,
    ) -> tuple[tuple[FactorCandidate, EvaluationResult], ...]:
        ledger_path = self.output_dir / "checkpoint_validation_ledger.jsonl"
        cached_rows = _read_jsonl(ledger_path)
        cache: dict[str, EvaluationResult] = {}
        for row in cached_rows:
            if row.get("period") != self.checkpoint_validation_period.to_dict():
                continue
            result = _result_from_dict(row["validation"])
            if result.success:
                cache[str(row["canonical"])] = result

        candidates: list[FactorCandidate] = []
        seen: set[str] = set()
        for candidate, _train in archive:
            if candidate.round_id > checkpoint_round:
                continue
            canonical = _safe_canonical(candidate.expression)
            if canonical in seen:
                continue
            seen.add(canonical)
            candidates.append(candidate)
        missing = [
            candidate for candidate in candidates
            if _safe_canonical(candidate.expression) not in cache
        ]
        if missing:
            results = self.evaluator.evaluate_many(missing, self.checkpoint_validation_period)
            for candidate, result in zip(missing, results):
                canonical = _safe_canonical(candidate.expression)
                append_jsonl(ledger_path, {
                    "schema_version": REPORT_SCHEMA,
                    "checkpoint_round_first_requested": checkpoint_round,
                    "period": self.checkpoint_validation_period.to_dict(),
                    "canonical": canonical,
                    "candidate": candidate.to_dict(),
                    "validation": result.to_dict(),
                })
                event_sink({
                    "event": "continuous_validation_evaluation",
                    "cycle_id": cycle_id,
                    "checkpoint_round": checkpoint_round,
                    "candidate": candidate.to_dict(),
                    "result": result.to_dict(),
                })
                if result.success:
                    cache[canonical] = result
        return tuple(
            (candidate, cache.get(_safe_canonical(candidate.expression), EvaluationResult(
                False,
                candidate.expression,
                self.checkpoint_validation_period,
                error="no successful checkpoint validation result",
            )))
            for candidate in candidates
        )

    def _report_payload(self) -> dict[str, Any]:
        path = self.output_dir / "top5_combinations.json"
        if path.is_file():
            payload = read_json(path)
            if payload.get("schema_version") == REPORT_SCHEMA:
                return payload
        return {
            "schema_version": REPORT_SCHEMA,
            "selection": {
                "source": "Validation",
                "objective": self.objective,
                "top_k": self.checkpoint_factor_budget,
                "max_abs_daily_metric_correlation": self.checkpoint_max_correlation,
            },
            "qualification": {
                "source": "Test equal-weight rank pool",
                "metric": "rank_ic",
                "threshold": self.checkpoint_rank_ic_threshold,
                "diagnostic_only_not_returned_to_search": True,
            },
            "checkpoints": [],
            "qualifying_combinations": [],
        }

    def _sequence_index(self) -> dict[str, int]:
        path = self.output_dir / "factor_sequence.json"
        if not path.is_file():
            return {}
        rows = read_json(path).get("factors", [])
        return {
            str(row["canonical"]): int(row["evaluation_index"])
            for row in rows
        }

    def _store_checkpoint(
        self,
        *,
        checkpoint_round: int,
        selected_items: Sequence[tuple[FactorCandidate, EvaluationResult]],
        validation_pool: EvaluationResult,
        test_individual: Sequence[tuple[FactorCandidate, EvaluationResult]],
        test_pool: EvaluationResult,
    ) -> dict[str, Any]:
        sequence = self._sequence_index()
        ordered = []
        for validation_rank, (candidate, result) in enumerate(selected_items, start=1):
            canonical = _safe_canonical(candidate.expression)
            ordered.append({
                "validation_rank": validation_rank,
                "evaluation_index": sequence.get(canonical),
                "candidate": candidate.to_dict(),
                "validation": result.to_dict(),
            })
        test_by_expression = {
            candidate.expression: result for candidate, result in test_individual
        }
        for row in ordered:
            result = test_by_expression.get(row["candidate"]["expression"])
            row["test"] = result.to_dict() if result is not None else None
        rank_ic = test_pool.metrics.value("rank_ic") if test_pool.success else float("-inf")
        return {
            "checkpoint_round": int(checkpoint_round),
            "available_factor_count": sum(
                1 for row in read_json(self.output_dir / "factor_sequence.json")["factors"]
                if int(row["round_id"]) <= int(checkpoint_round)
            ),
            "selected_top5_validation_order": ordered,
            "selected_generation_order": sorted(
                ({
                    "evaluation_index": row["evaluation_index"],
                    "candidate": row["candidate"],
                } for row in ordered),
                key=lambda row: row["evaluation_index"] or 10**18,
            ),
            "validation_equal_weight_rank_pool": validation_pool.to_dict(),
            "test_equal_weight_rank_pool": test_pool.to_dict(),
            "qualifies": bool(test_pool.success and rank_ic >= self.checkpoint_rank_ic_threshold),
            "test_rank_ic_threshold": self.checkpoint_rank_ic_threshold,
        }

    def _merge_checkpoint(self, record: dict[str, Any]) -> Path:
        payload = self._report_payload()
        rows = [
            row for row in payload.get("checkpoints", [])
            if int(row.get("checkpoint_round", -1)) != int(record["checkpoint_round"])
        ]
        rows.append(record)
        rows.sort(key=lambda row: int(row["checkpoint_round"]))
        payload["checkpoints"] = rows
        payload["qualifying_combinations"] = [row for row in rows if row.get("qualifies")]
        path = self.output_dir / "top5_combinations.json"
        write_json(path, payload)
        return path

    def backfill_checkpoint_reports_from_sequence(
        self,
        *,
        event_sink: EventSink | None = None,
        cycle_id: str = "retrospective-checkpoint-backfill",
    ) -> Path:
        """Backfill reporting-only checkpoints from a saved factor sequence.

        Unlike search resume, this path does not need ``factor_ledger.jsonl`` or
        GP features. It is intended for recovered result snapshots: candidates
        are reconstructed in committed evaluation order, then each requested
        round selects Top-5 strictly on Validation and measures Test once. No
        value produced here is returned to generation, GP fitting, acquisition,
        or checkpoint selection.
        """
        emit = event_sink or (lambda _event: None)
        sequence_path = self.output_dir / "factor_sequence.json"
        state_path = self.output_dir / "resume_state.json"
        if not sequence_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(
                "checkpoint backfill requires factor_sequence.json and "
                "resume_state.json"
            )
        sequence_payload = read_json(sequence_path)
        state = read_json(state_path)
        completed_round = int(sequence_payload.get("completed_round", state["round_id"]))
        if completed_round != int(state["round_id"]):
            raise RuntimeError(
                "factor sequence and resume state disagree on the completed round"
            )

        train_period_data = (state.get("signature") or {}).get("train_period")
        train_period = (
            Period.from_strings(
                str(train_period_data["start"]), str(train_period_data["end"])
            )
            if isinstance(train_period_data, dict)
            else self.checkpoint_validation_period
        )
        archive: list[tuple[FactorCandidate, EvaluationResult]] = []
        rows = sorted(
            sequence_payload.get("factors", []),
            key=lambda row: int(row["evaluation_index"]),
        )
        for row in rows:
            candidate = _candidate_from_dict(row["candidate"])
            metrics_data = row.get("train_metrics") or {}
            metrics = EvaluationMetrics(**{
                key: metrics_data.get(key)
                for key in EvaluationMetrics.__dataclass_fields__
            })
            archive.append((candidate, EvaluationResult(
                success=True,
                expression=candidate.expression,
                period=train_period,
                metrics=metrics,
            )))

        existing_rounds = {
            int(row.get("checkpoint_round", -1))
            for row in self._report_payload().get("checkpoints", [])
        }
        requested = [
            checkpoint_round
            for checkpoint_round in self.checkpoint_rounds
            if checkpoint_round <= completed_round
        ]
        emit({
            "event": "ldm_continuous_reporting_backfill_started",
            "cycle_id": cycle_id,
            "completed_round": completed_round,
            "requested_checkpoint_rounds": requested,
            "existing_checkpoint_rounds": sorted(existing_rounds),
            "reporting_only": True,
        })
        for checkpoint_round in requested:
            if checkpoint_round in existing_rounds:
                continue
            self._evaluate_checkpoint(
                checkpoint_round=checkpoint_round,
                archive=archive,
                event_sink=emit,
                cycle_id=cycle_id,
            )
        path = self.output_dir / "top5_combinations.json"
        final_rounds = [
            int(row["checkpoint_round"])
            for row in self._report_payload().get("checkpoints", [])
        ]
        emit({
            "event": "ldm_continuous_reporting_backfill_completed",
            "cycle_id": cycle_id,
            "checkpoint_rounds": sorted(final_rounds),
            "report": str(path),
            "reporting_only": True,
        })
        return path

    def _evaluate_checkpoint(
        self,
        *,
        checkpoint_round: int,
        archive: Sequence[tuple[FactorCandidate, EvaluationResult]],
        event_sink: EventSink,
        cycle_id: str,
    ) -> Path:
        existing = self._report_payload().get("checkpoints", [])
        if any(int(row.get("checkpoint_round", -1)) == checkpoint_round for row in existing):
            return self.output_dir / "top5_combinations.json"
        evaluations = self._validation_evaluations(
            archive=archive,
            event_sink=event_sink,
            cycle_id=cycle_id,
            checkpoint_round=checkpoint_round,
        )
        selected_items = _select_precomputed(
            evaluations,
            budget=self.checkpoint_factor_budget,
            objective=self.objective,
            max_correlation=self.checkpoint_max_correlation,
        )
        selected = tuple(candidate for candidate, _result in selected_items)
        validation_pool = self.evaluator.evaluate_pool(
            selected, self.checkpoint_validation_period
        )
        individual_results = self.evaluator.evaluate_many(selected, self.checkpoint_test_period)
        test_individual = tuple(zip(selected, individual_results))
        test_pool = self.evaluator.evaluate_pool(selected, self.checkpoint_test_period)
        record = self._store_checkpoint(
            checkpoint_round=checkpoint_round,
            selected_items=selected_items,
            validation_pool=validation_pool,
            test_individual=test_individual,
            test_pool=test_pool,
        )
        path = self._merge_checkpoint(record)
        event_sink({
            "event": "ldm_continuous_top5_checkpoint",
            "cycle_id": cycle_id,
            "checkpoint_round": checkpoint_round,
            "selected": [candidate.to_dict() for candidate in selected],
            "validation_pool": validation_pool.to_dict(),
            "test_pool": test_pool.to_dict(),
            "qualifies": record["qualifies"],
            "report": str(path),
        })
        return path

    def write_checkpoint_report(
        self,
        *,
        search: SearchResult,
        selection: ValidationSelection,
        test_individual: Sequence[dict[str, Any]],
        test_pool: EvaluationResult,
        evaluator: Any,
        validation_period: Period,
        test_period: Period,
        objective: str,
        max_correlation: float | None,
        event_sink: EventSink,
        cycle_id: str,
    ) -> Path:
        del evaluator, validation_period, test_period, max_correlation
        if objective != self.objective:
            raise ValueError("checkpoint objective differs from method objective")
        selected_validation = {
            _safe_canonical(candidate.expression): (candidate, result)
            for candidate, result in selection.evaluations
        }
        selected_items = tuple(
            selected_validation[_safe_canonical(candidate.expression)]
            for candidate in selection.selected
        )
        validation_pool = self.evaluator.evaluate_pool(
            selection.selected, self.checkpoint_validation_period
        )
        individual = tuple(
            (
                _candidate_from_dict(row["candidate"]),
                _result_from_dict(row["test"]),
            )
            for row in test_individual
        )
        record = self._store_checkpoint(
            checkpoint_round=search.rounds_requested,
            selected_items=selected_items,
            validation_pool=validation_pool,
            test_individual=individual,
            test_pool=test_pool,
        )
        path = self._merge_checkpoint(record)
        event_sink({
            "event": "ldm_continuous_top5_checkpoint",
            "cycle_id": cycle_id,
            "checkpoint_round": search.rounds_requested,
            "selected": [candidate.to_dict() for candidate in selection.selected],
            "validation_pool": validation_pool.to_dict(),
            "test_pool": test_pool.to_dict(),
            "qualifies": record["qualifies"],
            "report": str(path),
        })
        return path

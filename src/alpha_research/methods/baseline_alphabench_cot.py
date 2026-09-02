"""Adapter from the official AlphaBench CoT algorithm to ResearchMethod."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Sequence

from alpha_research.alphabench_runtime import (
    build_qlib_search_fn,
    load_alpha158_seeds,
    load_cot_algo,
    load_ea_algo,
    resolve_alphabench_root,
)
from alpha_research.canonical import canonical_form
from alpha_research.evaluator import FactorEvaluator
from alpha_research.types import (
    EvaluationMetrics,
    EvaluationResult,
    FactorCandidate,
    Period,
    SearchResult,
)


EventSink = Callable[[dict[str, Any]], None]
SearchFn = Callable[..., dict[str, Any]]


class AlphaBenchCoT:
    """Official AlphaBench searchers, exposed to StaticEnvironment as one ResearchMethod.

    ``algorithm`` picks which of AlphaBench's searchers runs:

    ``coe``
        ``searcher.algo.cot.CoTAlgo`` -- the paper calls it Chain-of-Experience.
        One chain per seed, refining a single lineage for ``rounds`` steps.
    ``ea``
        ``searcher.algo.ea.EAAlgo`` -- population search with LLM mutation and
        crossover. AlphaBench reports this as the stronger of the two.

    Both consume the same evaluator and LLM generator, so a run of each under one
    protocol differs only in how candidates are proposed.
    """

    name = "alphabench_cot"

    def __init__(
        self,
        *,
        evaluator: FactorEvaluator,
        alphabench_root: str | Path,
        output_dir: str | Path,
        model: str = "deepseek-chat",
        temperature: float = 1.75,
        enable_reason: bool = True,
        accept_threshold: float = 0.0,
        workers: int = 1,
        chain_batch: int = 8,
        llm_base_url: str = "",
        llm_api_key_env: str = "ALPHARESEARCH_LLM_API_KEY",
        search_fn: SearchFn | None = None,
        algorithm: str = "coe",
        seed_group: str = "all",
        ea_candidates_per_round: int = 30,
        ea_mutation_rate: float = 0.4,
        ea_crossover_rate: float = 0.6,
        ea_pool_size: int = 30,
        ea_seeds_top_k: int = 12,
        random_seed: int = 42,
    ) -> None:
        self.evaluator = evaluator
        self.alphabench_root = resolve_alphabench_root(alphabench_root)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.model = model
        self.temperature = float(temperature)
        self.enable_reason = bool(enable_reason)
        self.accept_threshold = float(accept_threshold)
        self.workers = max(1, int(workers))
        self.chain_batch = max(1, int(chain_batch))
        self.llm_base_url = llm_base_url
        self.llm_api_key_env = llm_api_key_env
        self.search_fn = search_fn
        algorithm = str(algorithm).lower()
        if algorithm not in {"coe", "ea"}:
            raise ValueError(f"algorithm must be 'coe' or 'ea', got {algorithm!r}")
        self.algorithm = algorithm
        self.seed_group = str(seed_group)
        self.ea_candidates_per_round = max(1, int(ea_candidates_per_round))
        self.ea_mutation_rate = float(ea_mutation_rate)
        self.ea_crossover_rate = float(ea_crossover_rate)
        self.ea_pool_size = max(1, int(ea_pool_size))
        self.ea_seeds_top_k = max(1, int(ea_seeds_top_k))
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
            raise ValueError(
                "AlphaBenchCoT is currently wired only for StaticEnvironment; "
                "rolling feedback adaptation is intentionally not implemented"
            )
        emit = event_sink or (lambda _event: None)
        random.seed(self.random_seed)
        seeds = (
            tuple(seed_factors)
            if seed_factors
            else load_alpha158_seeds(self.alphabench_root, seed_group=self.seed_group)
        )
        evaluated_seeds: list[dict[str, Any]] = []
        for candidate, result in zip(seeds, self.evaluator.evaluate_many(seeds, train_period)):
            emit({
                "event": "train_evaluation",
                "cycle_id": cycle_id,
                "stage": "alphabench_alpha158_seed",
                "candidate": candidate.to_dict(),
                "result": result.to_dict(),
            })
            if result.success:
                evaluated_seeds.append({
                    "name": candidate.name,
                    "expression": candidate.expression,
                    "metrics": _official_metrics(result.metrics),
                })
        if not evaluated_seeds:
            raise RuntimeError("none of the official Alpha158 seeds evaluated successfully")

        def evaluate_one(expression: str) -> dict[str, Any]:
            return _official_result(self.evaluator.evaluate(expression, train_period))

        def evaluate_batch(factors: list[dict[str, str]]) -> list[dict[str, Any]]:
            candidates = tuple(
                FactorCandidate(
                    name=str(item.get("name") or f"candidate_{index}"),
                    expression=str(item.get("expression") or ""),
                    source="alphabench_cot",
                )
                for index, item in enumerate(factors)
            )
            return [
                _official_result(result)
                for result in self.evaluator.evaluate_many(candidates, train_period)
            ]

        search_fn = self.search_fn or build_qlib_search_fn(
            self.alphabench_root,
            base_url=self.llm_base_url,
            api_key_env=self.llm_api_key_env,
            model=self.model,
        )
        save_dir = self.output_dir / "alphabench_official"

        if self.algorithm == "ea":
            algo = load_ea_algo(self.alphabench_root)(
                evaluate_fn=evaluate_one,
                batch_evaluate_fn=evaluate_batch,
                search_fn=search_fn,
                config={
                    "rounds": max(0, int(rounds)),
                    "N": self.ea_candidates_per_round,
                    "mutation_rate": self.ea_mutation_rate,
                    "crossover_rate": self.ea_crossover_rate,
                    "pool_size": self.ea_pool_size,
                    "seeds_top_k": self.ea_seeds_top_k,
                    "model": self.model,
                    "temperature": self.temperature,
                    "enable_reason": self.enable_reason,
                    "accept_threshold": self.accept_threshold,
                },
            )
            # EA evolves one population, so every seed goes in and pool_size caps it.
            raw = algo.run(evaluated_seeds, str(save_dir / "ea"))
            emit({
                "event": "alphabench_ea_done",
                "cycle_id": cycle_id,
                "pool_size": len(raw.get("final_pool") or []),
                "history": len(raw.get("history") or []),
            })
            archive = _archive_from_official(raw, train_period)
            if not archive:
                raise RuntimeError("official AlphaBench EA returned an empty factor pool")
            best_pair = _best_pair(raw, archive)
            llm_calls = len(raw.get("history") or [])
            return SearchResult(
                best=best_pair[0],
                best_train=best_pair[1],
                archive=tuple(archive),
                rounds_requested=max(0, int(rounds)),
                llm_calls=llm_calls,
            )

        algo_class = load_cot_algo(self.alphabench_root)

        # CoTAlgo keeps only the top `workers` seeds and runs that many chains at
        # once. One chain explores a single lineage, so the seed count is what
        # gives the pool its diversity -- but the FFO backend serialises requests
        # and stops answering when more than a handful arrive together. Run every
        # seed we want, in batches small enough for the backend to survive.
        ranked_seeds = sorted(
            evaluated_seeds,
            key=lambda seed: seed.get("metrics", {}).get("ic", float("-inf")),
            reverse=True,
        )
        chain_seeds = ranked_seeds[: self.workers]
        batches = [
            chain_seeds[start : start + self.chain_batch]
            for start in range(0, len(chain_seeds), self.chain_batch)
        ]

        merged: dict[str, Any] = {"best": {}, "history": [], "final_pool": []}
        for index, batch in enumerate(batches):
            algo = algo_class(
                evaluate_fn=evaluate_one,
                batch_evaluate_fn=evaluate_batch,
                search_fn=search_fn,
                config={
                    "rounds": max(0, int(rounds)),
                    "workers": len(batch),
                    "model": self.model,
                    "temperature": self.temperature,
                    "enable_reason": self.enable_reason,
                    "accept_threshold": self.accept_threshold,
                },
            )
            raw_batch = algo.run(batch, str(save_dir / f"batch_{index:02d}"))
            merged["history"].extend(raw_batch.get("history") or [])
            merged["final_pool"].extend(raw_batch.get("final_pool") or [])
            candidate = raw_batch.get("best") or {}
            if _is_better_record(merged["best"], candidate):
                merged["best"] = candidate
            emit({
                "event": "alphabench_cot_batch",
                "cycle_id": cycle_id,
                "batch": index,
                "seeds": [seed.get("name") for seed in batch],
                "pool_size": len(raw_batch.get("final_pool") or []),
            })

        raw = merged
        archive = _archive_from_official(raw, train_period)
        if not archive:
            raise RuntimeError("official AlphaBench CoT returned an empty factor pool")
        best_pair = _best_pair(raw, archive)
        for record in raw.get("history") or []:
            emit({
                "event": "alphabench_cot_round",
                "cycle_id": cycle_id,
                "record": record,
            })
        llm_calls = sum(
            1 for record in (raw.get("history") or []) if int(record.get("round") or 0) > 0
        )
        return SearchResult(
            best=best_pair[0],
            best_train=best_pair[1],
            archive=tuple(archive),
            rounds_requested=max(0, int(rounds)),
            llm_calls=llm_calls,
        )


class MockAlphaBenchSearch:
    """No-network generator with the payload contract expected by official CoT."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        index = len(self.calls)
        factor = {
            "name": f"MockAlphaBenchRound{index}",
            "expression": f"Mean(Delta($close, {index + 2}), {index + 3})",
            "reason": "Deterministic no-network AlphaBench adapter smoke candidate.",
        }
        return {"success": True, "factors": [factor], "results": {factor["name"]: factor["expression"]}}


def _best_pair(
    raw: dict[str, Any],
    archive: list[tuple[FactorCandidate, EvaluationResult]],
) -> tuple[FactorCandidate, EvaluationResult]:
    """Locate the searcher's declared best in the archive, else fall back to IC."""

    best_expression = str((raw.get("best") or {}).get("expression") or "")
    return next(
        (pair for pair in archive if pair[0].expression == best_expression),
        max(archive, key=lambda pair: pair[1].metrics.value("ic")),
    )


def _is_better_record(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Mirror CoTAlgo._is_better so merged batches agree with a single run."""

    if not candidate:
        return False
    if not current:
        return True
    current_metrics = current.get("metrics") or {}
    candidate_metrics = candidate.get("metrics") or {}
    current_ic = current_metrics.get("ic", float("-inf"))
    candidate_ic = candidate_metrics.get("ic", float("-inf"))
    if candidate_ic != current_ic:
        return candidate_ic > current_ic
    return candidate_metrics.get("rank_ic", float("-inf")) > current_metrics.get(
        "rank_ic", float("-inf")
    )


def _official_metrics(metrics: EvaluationMetrics) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in ("ic", "rank_ic", "icir", "rank_icir", "quantile_spread", "turnover"):
        value = getattr(metrics, key)
        if value is not None and math.isfinite(float(value)):
            output[key] = float(value)
    return output


def _official_result(result: EvaluationResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "expression": result.expression,
        "metrics": _official_metrics(result.metrics),
        "error": result.error,
    }


def _archive_from_official(
    raw: dict[str, Any], period: Period
) -> list[tuple[FactorCandidate, EvaluationResult]]:
    records = list(raw.get("final_pool") or [])
    for history_record in raw.get("history") or []:
        round_id = history_record.get("round", 0)
        # CoE records the single candidate it proposed under "generated"; EA keeps
        # the whole generation under "candidates". Without the second branch the
        # archive would hold only EA's surviving population, so selection would
        # have almost nothing to choose from.
        generated = history_record.get("generated") or {}
        if generated.get("expression"):
            records.append({**generated, "round_id": round_id})
        for candidate in history_record.get("candidates") or []:
            if isinstance(candidate, dict) and candidate.get("expression"):
                records.append({**candidate, "round_id": round_id})
    if raw.get("best"):
        records.append(raw["best"])
    archive: list[tuple[FactorCandidate, EvaluationResult]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        expression = str(record.get("expression") or "").strip()
        if not expression:
            continue
        # The LLM writes the same factor as a prefix call on one round and with
        # infix operators on another; comparing raw strings keeps both copies and
        # inflates the archive with duplicates.
        key = canonical_form(expression)
        if key in seen:
            continue
        seen.add(key)
        metrics = _metrics_from_mapping(record.get("metrics") or {})
        candidate = FactorCandidate(
            name=str(record.get("name") or f"AlphaBenchFactor{index}"),
            expression=expression,
            reason=str(record.get("reason") or ""),
            source="alphabench_cot",
            round_id=int(record.get("round_id") or index),
        )
        archive.append((candidate, EvaluationResult(True, expression, period, metrics=metrics)))
    return archive


def _metrics_from_mapping(raw: dict[str, Any]) -> EvaluationMetrics:
    def number(*keys: str) -> float | None:
        for key in keys:
            try:
                value = float(raw[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    return EvaluationMetrics(
        ic=number("ic"),
        rank_ic=number("rank_ic"),
        icir=number("icir", "ir"),
        rank_icir=number("rank_icir"),
        quantile_spread=number("quantile_spread"),
        turnover=number("turnover"),
        daily_count=int(raw.get("daily_count") or 0),
        observation_count=int(raw.get("observation_count") or 0),
    )

"""AlphaLDM: behavioural-fingerprint GP search over LLM-proposed formulas.

One round of the loop, matching the reference flowchart:

    C_r        context: evaluated formulas + train scores + incumbent + grammar
    x_1..x_N ~ P(x | C_r)                       N independent LLM proposals
    X_i      = profile(x_i)                     label-free behavioural fingerprint
    mu, s    = GP(X)                            posterior over the train score
    a        = acquisition(mu, s)               UCB or EI
    i        ~ softmax(a)                       Boltzmann pick, not argmax
    R_i      = evaluate(x_i, train)             the one real evaluation this round
    D_r      = D_{r-1} u {(X_i, R_i)}

Only the selected candidate is evaluated; the other N-1 are discarded on the
surrogate's word alone. That is the whole economic claim of the method, and it
only pays if a fingerprint is materially cheaper than an evaluation -- the
per-round timings emitted in ``ldm_round`` are what make that checkable.

Seeds and the search history are scored on the training window only. Validation
selection and test reporting stay with StaticEnvironment, unchanged and shared
with the CoE baseline, so the two methods differ in search and nothing else.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from alpha_research.alphabench_runtime import load_alpha158_seeds
from alpha_research.canonical import canonical_form
from alpha_research.evaluator import select_on_validation as select_validation_archive
from alpha_research.methods.ldm.acquire import (
    acquisition_values,
    novelty_bonus,
    softmax_sample,
)
from alpha_research.methods.ldm.gp import LdmSurrogate
from alpha_research.methods.ldm.history import History
from alpha_research.types import (
    EvaluationResult,
    FactorCandidate,
    Period,
    SearchResult,
    ValidationSelection,
)


EventSink = Callable[[dict[str, Any]], None]


@dataclass
class _LDMSearchState:
    """Mutable state shared by a fresh search and an exact resumed search."""

    rng: random.Random
    history: History
    surrogate: Any
    archive: list[tuple[FactorCandidate, EvaluationResult]]
    evaluated_records: list[dict[str, Any]]
    best: FactorCandidate | None
    best_result: EvaluationResult | None
    best_score: float
    completed_round: int = 0
    llm_calls_completed: int = 0


def _cross_validated_rank_correlation(
    features: np.ndarray,
    scores: np.ndarray,
    gp_kwargs: dict[str, Any],
    folds: int = 5,
    seed: int = 0,
) -> float | None:
    """Out-of-fold rank correlation between surrogate prediction and truth.

    Reported once after warm-up. Near zero means the fingerprints carry no
    information about realised score on this seed set, so the acquisition step
    will be picking at random for the rest of the run -- worth knowing before
    spending the LLM budget rather than after.
    """
    from scipy.stats import spearmanr

    count = len(scores)
    if count < folds * 2:
        return None
    order = np.random.default_rng(seed).permutation(count)
    predicted = np.empty(count, dtype=float)
    for fold in range(folds):
        test_idx = order[fold::folds]
        train_idx = np.setdiff1d(order, test_idx)
        surrogate = LdmSurrogate(feature_dim=features.shape[1], **gp_kwargs)
        surrogate.fit(features[train_idx], scores[train_idx])
        mean, _ = surrogate.predict(features[test_idx])
        predicted[test_idx] = mean
    if float(np.std(predicted)) < 1e-12:
        return 0.0
    return float(spearmanr(predicted, scores).statistic)


LATE_OBJECTIVE = "late_rank_ic"


def _late_window_score(result: Any, fraction: float = 0.25) -> float:
    """Mean daily rank IC over the closing slice of the evaluation window.

    Three repetitions showed the plain window mean is the wrong thing to
    maximise: LDM reached a training median of 0.054 against the baseline's
    0.000 and finished at 0.0129 +/- 0.0107 on test against 0.0220 +/- 0.0027 --
    a stronger optimiser producing a worse and far less stable pool. Candidates
    that only fit the window as a whole score well on the mean; ones that still
    work at the end of it score well here, which is the property that has to
    survive into the validation and test years.

    Read off the daily series the evaluator already returns, so it costs
    nothing beyond the evaluation that was happening anyway.
    """
    daily = getattr(result, "daily_metrics", ()) or ()
    values = [
        float(row["rank_ic"])
        for row in daily
        if row.get("rank_ic") is not None and np.isfinite(float(row["rank_ic"]))
    ]
    if not values:
        return float("-inf")
    tail = max(1, int(len(values) * fraction))
    return float(np.mean(values[-tail:]))


def _safe_canonical(expression: str) -> str:
    """Canonical form for dedup, falling back to the raw string when unparsable."""
    try:
        return canonical_form(expression)
    except Exception:
        return expression.strip()


class AlphaLDM:
    """Large Discovery Model search, exposed to StaticEnvironment."""

    name = "ldm_standard"
    candidate_source = "ldm_standard"
    supported_acquisitions = frozenset({"ucb", "ei", "random"})
    preserve_existing_events = False

    def select_on_validation(
        self,
        *,
        archive: Sequence[tuple[FactorCandidate, EvaluationResult]],
        evaluator: Any,
        validation_period: Period,
        factor_budget: int,
        objective: str,
        event_sink: EventSink | None = None,
        cycle_id: str = "static",
        max_correlation: float | None = None,
    ) -> ValidationSelection:
        """Apply the validation correlation boundary as a hard constraint.

        The generic selector historically backfilled candidates rejected by the
        correlation filter in order to reach the requested budget.  That makes
        ``max_correlation`` a preference rather than a limit.  Formal LDM and
        Harness+LDM runs must never silently re-introduce a rejected factor, so
        they disable backfill and fail closed if the archive cannot supply a
        complete pool under the configured boundary.
        """

        selection = select_validation_archive(
            archive=archive,
            evaluator=evaluator,
            validation_period=validation_period,
            factor_budget=factor_budget,
            objective=objective,
            event_sink=event_sink,
            cycle_id=cycle_id,
            max_correlation=max_correlation,
            backfill_to_budget=False,
        )
        if len(selection.selected) != int(factor_budget):
            raise RuntimeError(
                "validation correlation constraint left "
                f"{len(selection.selected)} factors, but the formal protocol requires "
                f"{int(factor_budget)} (max_correlation={max_correlation!r}); "
                "the run is invalid and will not continue to Test"
            )
        return selection

    def __init__(
        self,
        *,
        evaluator: Any,
        profiler: Any,
        generator: Any,
        output_dir: str | Path,
        alphabench_root: str | Path | None = None,
        objective: str = "rank_ic",
        # What the surrogate is trained to predict, which need not be what the
        # pool is later selected on. Measured on the first full run: pushing
        # train rank_ic from 0.043 to 0.059 moved validation the wrong way,
        # 0.029 down to 0.024. Past roughly 0.045 the level stops being a
        # signal and starts being a fit to the window, so a search that
        # maximises it well ends up selecting worse factors. rank_icir divides
        # by the dispersion of the daily series and does not reward that.
        search_objective: str | None = None,
        late_window_fraction: float = 0.25,
        seed_group: str = "all",
        acquisition: str = "ucb",
        acquisition_beta: float = 2.0,
        acquisition_xi: float = 0.01,
        softmax_temperature: float = 1.0,
        diversity_weight: float = 0.0,
        evaluate_per_round: int = 1,
        max_refill_batches: int = 8,
        gp_min_fit_data: int = 20,
        # None keeps the surrogate's median-distance heuristic.
        gp_lengthscale: float | None = None,
        gp_noise: float = 0.05,
        gp_scale: float = 0.25,
        gp_train_iters: int = 100,
        gp_lr: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self.evaluator = evaluator
        self.profiler = profiler
        self.generator = generator
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.alphabench_root = Path(alphabench_root).expanduser().resolve() if alphabench_root else None
        self.objective = str(objective)
        self.search_objective = str(search_objective or objective)
        self.late_window_fraction = float(late_window_fraction)
        self.seed_group = str(seed_group)
        self.acquisition = str(acquisition).lower()
        if self.acquisition not in self.supported_acquisitions:
            expected = ", ".join(sorted(self.supported_acquisitions))
            raise ValueError(f"acquisition must be one of {expected}, got {acquisition!r}")
        self.acquisition_beta = float(acquisition_beta)
        self.acquisition_xi = float(acquisition_xi)
        self.softmax_temperature = float(softmax_temperature)
        # Weight on the novelty term; 0 reproduces plain UCB.
        self.diversity_weight = float(diversity_weight)
        # Observations per round. The surrogate only becomes useful past
        # roughly 120 observations (measured: rank correlation 0.20 at 80,
        # 0.28 at 120, 0.31 at 200), and 42 seeds plus one per round lands
        # at the bottom of that curve.
        self.evaluate_per_round = max(1, int(evaluate_per_round))
        self.max_refill_batches = max(1, int(max_refill_batches))
        self.gp_kwargs = {
            "min_fit_data": int(gp_min_fit_data),
            "lengthscale": float(gp_lengthscale) if gp_lengthscale else None,
            "noise": float(gp_noise),
            "scale": float(gp_scale),
            "train_iters": int(gp_train_iters),
            "lr": float(gp_lr),
        }
        self.random_seed = int(random_seed)

    def _score(self, result: Any) -> float:
        """Surrogate target for one evaluation."""
        if self.search_objective == LATE_OBJECTIVE:
            return _late_window_score(result, self.late_window_fraction)
        return result.metrics.value(self.search_objective)

    def _new_history(self, *, dim: int, schema_version: str) -> History:
        """Create the append-only observation store used by this method."""
        return History(dim=dim, schema_version=schema_version)

    def _new_surrogate(self, *, feature_dim: int, history: History) -> Any:
        """Create the surrogate; variants may return an API-compatible wrapper."""
        del history
        return LdmSurrogate(feature_dim=feature_dim, **self.gp_kwargs)

    def _record_observation(
        self,
        history: History,
        *,
        feature: np.ndarray,
        score: float,
        result: EvaluationResult,
        canonical: str,
        expression: str,
        round_id: int,
    ) -> None:
        """Append one verified result without changing the base history schema."""
        del result
        history.add(feature, score, canonical, expression, round_id)

    def _evaluated_record(
        self,
        *,
        expression: str,
        score: float,
        result: EvaluationResult,
    ) -> dict[str, Any]:
        """Return the train-only record exposed to the proposal generator."""
        del result
        return {"expression": expression, "score": score}

    def _generator_objective(self) -> str:
        return self.search_objective

    def _generator_best(
        self,
        candidate: FactorCandidate | None,
        result: EvaluationResult | None,
        score: float,
    ) -> dict[str, Any] | None:
        del result
        if candidate is None:
            return None
        return {"expression": candidate.expression, "score": score}

    def _candidate_acquisition(
        self,
        *,
        surrogate: Any,
        history: History,
        features: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Fit, predict and score one finite proposal pool.

        The final mapping is reserved for method-specific posterior diagnostics.
        Keeping this operation behind one hook lets multi-objective variants reuse
        the verified-evaluation and refill loop without changing standard LDM.
        """
        if self.acquisition == "random":
            mean = np.zeros(len(features), dtype=float)
            std = np.zeros(len(features), dtype=float)
            values = np.asarray([rng.random() for _ in features], dtype=float)
        else:
            # An empty D_0 has nothing to fit; the surrogate then returns its
            # flat prior and the softmax step samples uniformly, which is the
            # honest cold-start behaviour.
            if history.n:
                surrogate.fit(history.features(), history.scores())
            mean, std = surrogate.predict(features)
            values = acquisition_values(
                mean,
                std,
                mode=self.acquisition,
                beta=self.acquisition_beta,
                xi=self.acquisition_xi,
            )
            if self.diversity_weight:
                spread = float(np.std(values))
                if spread > 1e-12:
                    values = (values - float(np.mean(values))) / spread
                values = values + self.diversity_weight * novelty_bonus(
                    features, history.features()
                )
        return mean, std, values, {}

    def _candidate_batch_acquisition(
        self,
        *,
        surrogate: Any,
        history: History,
        features: np.ndarray,
        batch_size: int,
        rng: random.Random,
    ) -> dict[str, Any] | None:
        """Optional joint-batch acquisition hook.

        Returning ``None`` preserves the original pointwise acquisition plus
        repeated softmax sampling path exactly.  A method that genuinely scores
        candidate *sets* (for example qEHVI) returns posterior diagnostics, one
        selected local-index batch, and an auditable event payload.
        """

        del surrogate, history, features, batch_size, rng
        return None

    def _round_event_fields(self, history: History, surrogate: Any) -> dict[str, Any]:
        """Optional method diagnostics computed after a complete real batch."""

        del history, surrogate
        return {}

    def _warmup_event_fields(self, history: History, surrogate: Any) -> dict[str, Any]:
        del history, surrogate
        return {}

    def _restore_search_state(
        self,
        *,
        train_period: Period,
        rounds: int,
        seed_factors: Sequence[FactorCandidate] | None,
        emit: EventSink,
        cycle_id: str,
    ) -> _LDMSearchState | None:
        """Optional variant hook for restoring an already committed search."""
        del train_period, rounds, seed_factors, emit, cycle_id
        return None

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
        """Optional variant hook called only after a complete warm-up/round."""
        del (
            state,
            train_period,
            target_rounds,
            observations,
            llm_calls_total,
            emit,
            cycle_id,
        )

    # ----------------------------------------------------------------- warm-up

    def _warm_up(
        self,
        *,
        history: History,
        surrogate: LdmSurrogate,
        archive: list[tuple[FactorCandidate, EvaluationResult]],
        evaluated_records: list[dict[str, Any]],
        train_period: Period,
        seed_factors: Sequence[FactorCandidate] | None,
        emit: EventSink,
        cycle_id: str,
    ) -> tuple[FactorCandidate | None, EvaluationResult | None, float]:
        """Build D_0 from the Alpha158 seed library and fit the surrogate on it.

        Split out of ``search`` so a variant can replace the initial dataset
        without touching the loop that follows -- ``AlphaLDMColdStart`` overrides
        this to start from an empty D_0. ``history``, ``archive`` and
        ``evaluated_records`` are filled in place; the incumbent is returned.
        """
        schema = self.profiler.schema
        best: FactorCandidate | None = None
        best_result: EvaluationResult | None = None
        best_score = float("-inf")

        # ---------------------------------------------------------- D_0: seeds
        seeds = (
            tuple(seed_factors)
            if seed_factors
            else load_alpha158_seeds(self.alphabench_root, seed_group=self.seed_group)
        )
        seed_results = self.evaluator.evaluate_many(seeds, train_period)
        seed_failures: list[str] = []
        for candidate, result in zip(seeds, seed_results):
            emit({
                "event": "train_evaluation",
                "cycle_id": cycle_id,
                "stage": "alphabench_alpha158_seed",
                "candidate": candidate.to_dict(),
                "result": result.to_dict(),
            })
            if not result.success:
                seed_failures.append(f"{candidate.name}: {result.error}")
                continue
            score = self._score(result)
            if not np.isfinite(score):
                seed_failures.append(f"{candidate.name}: non-finite {self.search_objective}")
                continue
            try:
                feature = self.profiler.profile(candidate.expression)
            except Exception as exc:
                emit({
                    "event": "ldm_profile_failed",
                    "cycle_id": cycle_id,
                    "stage": "seed",
                    "expression": candidate.expression,
                    "error": str(exc),
                })
                seed_failures.append(f"{candidate.name}: profile failed: {exc}")
                continue
            self._record_observation(
                history,
                feature=feature,
                score=score,
                result=result,
                canonical=_safe_canonical(candidate.expression),
                expression=candidate.expression,
                round_id=0,
            )
            archive.append((candidate, result))
            evaluated_records.append(
                self._evaluated_record(
                    expression=candidate.expression,
                    score=score,
                    result=result,
                )
            )
            # Optional Harness-only diagnostics hook.  The original LDM
            # generator has no hook, so its history, GP, and acquisition path
            # are byte-for-byte unchanged.
            feedback_hook = getattr(self.generator, "record_evaluation", None)
            if callable(feedback_hook):
                feedback_hook(
                    candidate=candidate.to_dict(),
                    result=result,
                    score=score,
                )
            if score > best_score:
                best, best_result, best_score = candidate, result, score

        if history.n != len(seeds):
            raise RuntimeError(
                f"Alpha158 warm-up is incomplete: {history.n}/{len(seeds)} seeds produced "
                "a verified train RankIC and fingerprint. The run is invalid and will not "
                "continue. Failures: " + " | ".join(seed_failures[:20])
            )

        # D_0 is complete: fit and report the warm-up diagnostic once.
        warmup_started = time.monotonic()
        warmup_rank_correlation = None
        if self.acquisition != "random":
            surrogate.fit(history.features(), history.scores())
            warmup_rank_correlation = _cross_validated_rank_correlation(
                history.features(), history.scores(), self.gp_kwargs
            )
        warmup_event = {
            "event": "ldm_warmup",
            "cycle_id": cycle_id,
            "search_objective": self.search_objective,
            "seeds_offered": len(seeds),
            "observations": history.n,
            "feature_dim": len(schema.names),
            "schema_version": schema.version,
            "acquisition": self.acquisition,
            "gp_trained": surrogate.trained,
            "gp_lengthscale": float(surrogate.fitted_lengthscale),
            "cv_rank_correlation": warmup_rank_correlation,
            "best_seed_expression": best.expression if best else None,
            "best_seed_score": best_score if np.isfinite(best_score) else None,
            "seconds": round(time.monotonic() - warmup_started, 3),
        }
        warmup_event.update(self._warmup_event_fields(history, surrogate))
        emit(warmup_event)

        return best, best_result, best_score

    # ------------------------------------------------------------------ search

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
                "AlphaLDM is currently wired only for StaticEnvironment; "
                "rolling feedback adaptation is intentionally not implemented"
            )
        emit = event_sink or (lambda _event: None)
        target_rounds = max(0, int(rounds))
        schema = self.profiler.schema
        state = self._restore_search_state(
            train_period=train_period,
            rounds=target_rounds,
            seed_factors=seed_factors,
            emit=emit,
            cycle_id=cycle_id,
        )
        if state is None:
            rng = random.Random(self.random_seed)
            history = self._new_history(dim=len(schema.names), schema_version=schema.version)
            surrogate = self._new_surrogate(feature_dim=len(schema.names), history=history)
            archive: list[tuple[FactorCandidate, EvaluationResult]] = []
            evaluated_records: list[dict[str, Any]] = []

            emit({
                "event": "ldm_profile_schema",
                "cycle_id": cycle_id,
                "schema": schema.to_dict(),
                "reference_expressions": list(
                    getattr(self.profiler, "reference_expressions", ())
                ),
            })

            # --------------------------------------------------------- D_0
            best, best_result, best_score = self._warm_up(
                history=history,
                surrogate=surrogate,
                archive=archive,
                evaluated_records=evaluated_records,
                train_period=train_period,
                seed_factors=seed_factors,
                emit=emit,
                cycle_id=cycle_id,
            )
            state = _LDMSearchState(
                rng=rng,
                history=history,
                surrogate=surrogate,
                archive=archive,
                evaluated_records=evaluated_records,
                best=best,
                best_result=best_result,
                best_score=best_score,
            )
            self._persist_search_checkpoint(
                state=state,
                train_period=train_period,
                target_rounds=target_rounds,
                observations=(),
                llm_calls_total=0,
                emit=emit,
                cycle_id=cycle_id,
            )
        else:
            rng = state.rng
            history = state.history
            surrogate = state.surrogate
            archive = state.archive
            evaluated_records = state.evaluated_records
            best = state.best
            best_result = state.best_result
            best_score = state.best_score
            if state.completed_round > target_rounds:
                raise ValueError(
                    f"resume target {target_rounds} is behind committed round "
                    f"{state.completed_round}"
                )

        # ------------------------------------------------------- LDM iterations
        for round_id in range(state.completed_round + 1, target_rounds + 1):
            seen = set(history.canonical_forms())
            all_profiled: list[dict[str, Any]] = []
            posterior_mean: list[float] = []
            posterior_std: list[float] = []
            acquisition: list[float] = []
            acquisition_diagnostics: dict[str, list[float]] = {}
            chosen_indices: list[int] = []
            proposed_count = 0
            fresh_count = 0
            round_scores: list[float | None] = []
            generation_seconds = 0.0
            profile_seconds = 0.0
            evaluate_seconds = 0.0
            successful_evaluations = 0
            generation_batches = 0
            verified_observations: list[dict[str, Any]] = []

            # Refill across independent LLM batches as well as within a batch.
            # This handles duplicate/invalid proposals and FFO failures without
            # ever reducing the number of verified observations in a GP round.
            while (
                successful_evaluations < self.evaluate_per_round
                and generation_batches < self.max_refill_batches
            ):
                generation_batches += 1
                context = self.generator.build_context(
                    evaluated=evaluated_records,
                    best=self._generator_best(best, best_result, best_score),
                    objective=self._generator_objective(),
                    round_id=round_id,
                )
                started = time.monotonic()
                proposals = self.generator.generate(context)
                generation_seconds += time.monotonic() - started
                proposed_count += len(proposals)

                fresh: list[dict[str, Any]] = []
                for record in proposals:
                    canonical = _safe_canonical(record["expression"])
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                    fresh.append({**record, "canonical": canonical})
                fresh_count += len(fresh)

                started = time.monotonic()
                profiled: list[dict[str, Any]] = []
                for record in fresh:
                    try:
                        record["feature"] = self.profiler.profile(record["expression"])
                    except Exception as exc:
                        emit({
                            "event": "ldm_profile_failed", "cycle_id": cycle_id,
                            "round_id": round_id, "expression": record["expression"],
                            "refill_batch": generation_batches,
                            "error": str(exc),
                        })
                        continue
                    profiled.append(record)
                profile_seconds += time.monotonic() - started

                if not profiled:
                    emit({
                        "event": "ldm_refill_batch",
                        "cycle_id": cycle_id,
                        "round_id": round_id,
                        "refill_batch": generation_batches,
                        "proposed": len(proposals),
                        "fresh": len(fresh),
                        "profiled": 0,
                        "successful_evaluations": successful_evaluations,
                        "status": "no_profiled_candidates",
                    })
                    continue

                features = np.vstack([record["feature"] for record in profiled])
                requested_batch_size = min(
                    self.evaluate_per_round - successful_evaluations,
                    len(profiled),
                )
                batch_decision = self._candidate_batch_acquisition(
                    surrogate=surrogate,
                    history=history,
                    features=features,
                    batch_size=requested_batch_size,
                    rng=rng,
                )
                selection_queue: list[int] | None = None
                batch_event: dict[str, Any] | None = None
                if batch_decision is None:
                    mean, std, values, batch_diagnostics = self._candidate_acquisition(
                        surrogate=surrogate,
                        history=history,
                        features=features,
                        rng=rng,
                    )
                else:
                    mean = np.asarray(batch_decision["mean"], dtype=float).ravel()
                    std = np.asarray(batch_decision["std"], dtype=float).ravel()
                    values = np.asarray(
                        batch_decision["candidate_acquisition"], dtype=float
                    ).ravel()
                    batch_diagnostics = dict(batch_decision.get("diagnostics") or {})
                    selection_queue = [
                        int(index) for index in batch_decision["selected_indices"]
                    ]
                    if len(mean) != len(profiled) or len(std) != len(profiled):
                        raise ValueError("batch posterior diagnostics do not match slate size")
                    if len(values) != len(profiled):
                        raise ValueError("batch candidate diagnostics do not match slate size")
                    if (
                        len(selection_queue) != requested_batch_size
                        or len(set(selection_queue)) != len(selection_queue)
                        or any(index < 0 or index >= len(profiled) for index in selection_queue)
                    ):
                        raise ValueError(
                            "batch acquisition returned invalid selected indices: "
                            f"{selection_queue!r} for slate size {len(profiled)} and "
                            f"batch size {requested_batch_size}"
                        )
                    batch_event = dict(batch_decision.get("event") or {})

                offset = len(all_profiled)
                all_profiled.extend(profiled)
                posterior_mean.extend(float(v) for v in mean)
                posterior_std.extend(float(v) for v in std)
                acquisition.extend(float(v) for v in values)
                for key, diagnostic_values in batch_diagnostics.items():
                    acquisition_diagnostics.setdefault(key, []).extend(
                        float(value) for value in diagnostic_values
                    )
                remaining = (
                    list(range(len(profiled)))
                    if selection_queue is None
                    else list(selection_queue)
                )

                if batch_event is not None:
                    emit({
                        "event": "ldm_batch_acquisition",
                        "cycle_id": cycle_id,
                        "round_id": round_id,
                        "refill_batch": generation_batches,
                        "candidate_expressions": [
                            record["expression"] for record in profiled
                        ],
                        "selected_local_indices": list(selection_queue or ()),
                        "selected_global_indices": [
                            offset + index for index in (selection_queue or ())
                        ],
                        **batch_event,
                    })

                while remaining and successful_evaluations < self.evaluate_per_round:
                    if selection_queue is None:
                        position = softmax_sample(
                            values[remaining], rng, self.softmax_temperature
                        )
                        chosen = remaining.pop(position)
                    else:
                        chosen = remaining.pop(0)
                    chosen_indices.append(offset + chosen)
                    selected = profiled[chosen]
                    candidate = FactorCandidate(
                        name=selected["name"],
                        expression=selected["expression"],
                        reason=selected.get("reason", ""),
                        source=self.candidate_source,
                        round_id=round_id,
                    )
                    started = time.monotonic()
                    result = self.evaluator.evaluate(candidate.expression, train_period)
                    evaluate_seconds += time.monotonic() - started

                    emit({
                        "event": "train_evaluation",
                        "cycle_id": cycle_id,
                        "stage": f"{self.candidate_source}_selected",
                        "refill_batch": generation_batches,
                        "candidate": candidate.to_dict(),
                        "result": result.to_dict(),
                    })

                    score = self._score(result) if result.success else float("-inf")
                    feedback_hook = getattr(self.generator, "record_evaluation", None)
                    if callable(feedback_hook):
                        feedback_hook(
                            candidate=candidate.to_dict(),
                            result=result,
                            score=score if np.isfinite(score) else None,
                        )
                    if result.success and np.isfinite(score):
                        self._record_observation(
                            history,
                            feature=selected["feature"],
                            score=score,
                            result=result,
                            canonical=selected["canonical"],
                            expression=candidate.expression,
                            round_id=round_id,
                        )
                        archive.append((candidate, result))
                        evaluated_records.append(
                            self._evaluated_record(
                                expression=candidate.expression,
                                score=score,
                                result=result,
                            )
                        )
                        round_scores.append(score)
                        successful_evaluations += 1
                        verified_observations.append({
                            "candidate": candidate,
                            "result": result,
                            "feature": np.asarray(selected["feature"], dtype=float),
                            "score": float(score),
                            "canonical": selected["canonical"],
                            "refill_batch": generation_batches,
                            "selection_order": successful_evaluations,
                        })
                        if score > best_score:
                            best, best_result, best_score = candidate, result, score
                    else:
                        round_scores.append(None)

                emit({
                    "event": "ldm_refill_batch",
                    "cycle_id": cycle_id,
                    "round_id": round_id,
                    "refill_batch": generation_batches,
                    "proposed": len(proposals),
                    "fresh": len(fresh),
                    "profiled": len(profiled),
                    "successful_evaluations": successful_evaluations,
                    "status": (
                        "round_complete"
                        if successful_evaluations == self.evaluate_per_round
                        else "refill_required"
                    ),
                })

            round_event = {
                "event": "ldm_round",
                "cycle_id": cycle_id,
                "round_id": round_id,
                "proposed": proposed_count,
                "fresh": fresh_count,
                "profiled": len(all_profiled),
                "generation_batches": generation_batches,
                "max_refill_batches": self.max_refill_batches,
                "status": (
                    "evaluated"
                    if successful_evaluations == self.evaluate_per_round
                    else "insufficient_valid_evaluations"
                ),
                "successful_evaluations": successful_evaluations,
                "required_successful_evaluations": self.evaluate_per_round,
                "selected_indices": chosen_indices,
                "selected_expressions": [all_profiled[i]["expression"] for i in chosen_indices],
                "selected_scores": round_scores,
                "gp_trained": surrogate.trained,
                "gp_lengthscale": float(surrogate.fitted_lengthscale),
                "gp_observations": history.n,
                "posterior_mean": posterior_mean,
                "posterior_std": posterior_std,
                "acquisition": acquisition,
                "diversity_weight": self.diversity_weight,
                "best_score": best_score if np.isfinite(best_score) else None,
                # These three make the surrogate's economic case auditable: the
                # method only pays off while profile_seconds stays well under
                # (fresh - 1) * evaluate_seconds.
                "generation_seconds": round(generation_seconds, 3),
                "profile_seconds": round(profile_seconds, 3),
                "evaluate_seconds": round(evaluate_seconds, 3),
            }
            round_event.update(acquisition_diagnostics)
            round_event.update(self._round_event_fields(history, surrogate))
            emit(round_event)

            if successful_evaluations != self.evaluate_per_round:
                raise RuntimeError(
                    f"LDM round {round_id} produced only {successful_evaluations}/"
                    f"{self.evaluate_per_round} verified RankIC observations after "
                    f"{generation_batches} generation batches and {len(all_profiled)} profiled "
                    "candidates; refusing to update GP with an incomplete round"
                )

            state.best = best
            state.best_result = best_result
            state.best_score = best_score
            state.completed_round = round_id
            self._persist_search_checkpoint(
                state=state,
                train_period=train_period,
                target_rounds=target_rounds,
                observations=verified_observations,
                llm_calls_total=(
                    state.llm_calls_completed + int(getattr(self.generator, "calls", 0))
                ),
                emit=emit,
                cycle_id=cycle_id,
            )

        history.save(self.output_dir / "ldm_history.csv")

        if best is None or best_result is None:
            raise RuntimeError("LDM search produced no successful evaluation")

        return SearchResult(
            best=best,
            best_train=best_result,
            archive=tuple(archive),
            rounds_requested=target_rounds,
            llm_calls=(
                state.llm_calls_completed + int(getattr(self.generator, "calls", 0))
            ),
        )

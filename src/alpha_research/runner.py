#!/usr/bin/env python3
"""Construct and run AlphaResearch methods under the static experimental protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alpha_research.arguments import parse_args
from alpha_research.alphabench_runtime import PINNED_COMMIT, load_alpha158_seeds
from alpha_research.environments import StaticEnvironment, StaticProtocol
from alpha_research.evaluator import AlphaBenchFFOEvaluator, MockFactorEvaluator
from alpha_research.methods.baseline_alphabench_cot import AlphaBenchCoT, MockAlphaBenchSearch
from alpha_research.naming import (
    BASELINE_ALPHA158,
    BASELINE_ALPHABENCH_COT,
    LDM_COLD_START,
    LDM_CONTINUOUS_DISCOVERY,
    LDM_METHODS,
    LDM_MINIMAL_PROMPT_HARNESS,
    LDM_PROMPT_HARNESS,
    LDM_RANKIC_RANKICIR_EHVI,
    LDM_RANKIC_TURNOVER_EHVI,
    LDM_RANKIC_WORST_QEHVI,
    LDM_SINGLE_FACTOR,
    LDM_SPLIT_ROBUST_REWARD,
    standard_output_dir,
    standard_run_name,
)
from alpha_research.types import Period


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def static_protocol(args: argparse.Namespace) -> StaticProtocol:
    return StaticProtocol(
        train=Period.from_strings(args.train_start, args.train_end),
        validation=Period.from_strings(args.val_start, args.val_end),
        test=Period.from_strings(args.test_start, args.test_end),
        factor_budget=args.factor_budget,
        search_rounds=args.search_rounds,
        objective=args.objective,
        max_correlation=args.max_correlation,
    )


def build_evaluator(args: argparse.Namespace, alphabench_root: Path):
    if args.mock or args.mock_evaluator:
        return MockFactorEvaluator(None, min_observations=args.min_observations)
    evaluator = AlphaBenchFFOEvaluator(
        alphabench_root=alphabench_root,
        base_url=args.ffo_url,
        market=args.market,
        label=args.ffo_label,
        use_cache=args.ffo_use_cache,
        fast=args.ffo_fast,
        topk=args.ffo_topk,
        n_drop=args.ffo_n_drop,
        timeout=args.ffo_timeout,
        forward_n=args.ffo_forward_n,
        max_parallel=args.eval_parallel,
        max_attempts=args.ffo_max_attempts,
    )
    if not evaluator.health_check():
        raise RuntimeError(
            f"AlphaBench FFO backend is not healthy at {args.ffo_url}; "
            "start it with `ppo start backend`"
        )
    return evaluator


def profile_period(args: argparse.Namespace) -> Period:
    """Window the fingerprints are measured on.

    Defaults to the last two years of training rather than the whole span: the
    fingerprint has to stay cheap relative to an evaluation, and it only needs
    to characterise behaviour, not estimate it precisely.
    """
    train = Period.from_strings(args.train_start, args.train_end)
    if bool(args.profile_start) != bool(args.profile_end):
        raise ValueError("--profile-start and --profile-end must be supplied together")
    if args.profile_start:
        profile = Period.from_strings(args.profile_start, args.profile_end)
        if profile.start < train.start or profile.end > train.end:
            raise ValueError("profiling period must be contained in Train")
        return profile
    try:
        start = train.end.replace(year=train.end.year - 2)
    except ValueError:
        # February 29 has no counterpart in the target non-leap year.
        start = train.end.replace(year=train.end.year - 2, day=28)
    return Period(start=max(start, train.start), end=train.end)


def build_ldm(args: argparse.Namespace, evaluator, alphabench_root: Path, output_dir: Path):
    from alpha_research.formula import FormulaValidator
    from alpha_research.llm import OpenAICompatibleChatClient
    from alpha_research.methods.ldm import AlphaLDM
    from alpha_research.methods.ldm.generator import CandidateGenerator, MockCandidateGenerator
    from alpha_research.profile import FactorProfiler, MockFactorProfiler

    validator = FormulaValidator()
    mock_mode = bool(args.mock or args.mock_evaluator)

    if mock_mode:
        profiler = MockFactorProfiler()
    else:
        # The reference basket is the Alpha158 seed set, frozen for the run:
        # correlating against a growing pool would make a candidate's
        # fingerprint depend on when it was proposed. It is a measurement
        # basis rather than search data, so a cold-start run may keep it; pass
        # --ldm-reference-basket none to drop it and read three constant zeros.
        reference_expressions = (
            []
            if args.ldm_reference_basket == "none"
            else [
                seed.expression
                for seed in load_alpha158_seeds(alphabench_root, seed_group=args.seed_group)
            ]
        )
        profiler = FactorProfiler(
            provider_uri=resolve_repo_path(args.qlib_provider_uri),
            market=args.market,
            period=profile_period(args),
            reference_expressions=reference_expressions,
            validator=validator,
            min_observations=args.min_observations,
        )

    if args.mock or args.mock_llm:
        generator = MockCandidateGenerator(candidates_per_round=args.candidates_per_round)
    else:
        generator_class = CandidateGenerator
        if args.method == LDM_PROMPT_HARNESS:
            from alpha_research.methods.ldm_prompt_harness import HarnessCandidateGenerator
            from alpha_research.methods.ldm_prompt_harness.diversity import (
                DiversityController,
                DiversitySettings,
            )

            generator_class = HarnessCandidateGenerator
            diversity_settings = DiversitySettings(
                structural_mode=args.structural_diversity_mode,
                behavioral_mode=args.behavioral_diversity_mode,
                structural_similarity_threshold=args.structural_similarity_threshold,
                behavioral_correlation_threshold=args.behavioral_correlation_threshold,
                penalty_weight=args.diversity_penalty_weight,
                feedback_clusters=args.diversity_feedback_clusters,
            )
            diversity_controller = (
                DiversityController(
                    validator=validator,
                    settings=diversity_settings,
                    output_dir=output_dir,
                )
                if diversity_settings.active
                else None
            )
        else:
            diversity_controller = None
        if args.method == LDM_MINIMAL_PROMPT_HARNESS:
            from alpha_research.methods.ldm_minimal_prompt_harness import (
                MinimalHarnessCandidateGenerator,
            )

            generator_class = MinimalHarnessCandidateGenerator
        generator = generator_class(
            client=OpenAICompatibleChatClient(
                base_url=args.llm_base_url,
                model_name=args.llm_model,
                api_key_env=args.llm_api_key_env,
                temperature=args.llm_temperature,
                timeout_seconds=args.llm_timeout_seconds,
                max_tokens=args.llm_max_tokens,
                reasoning_effort=(
                    None
                    if args.llm_reasoning_effort == "none"
                    else args.llm_reasoning_effort
                ),
                thinking_enabled=args.llm_thinking,
                json_mode=True,
            ),
            validator=validator,
            candidates_per_round=args.candidates_per_round,
            max_parallel=args.generator_parallel,
            history_shown=args.history_shown,
            **(
                {"diversity_controller": diversity_controller}
                if args.method == LDM_PROMPT_HARNESS
                else {}
            ),
        )

    method_class = AlphaLDM
    if args.method == LDM_PROMPT_HARNESS:
        from alpha_research.methods.ldm_prompt_harness import HarnessLDM

        method_class = HarnessLDM
    elif args.method == LDM_MINIMAL_PROMPT_HARNESS:
        from alpha_research.methods.ldm_minimal_prompt_harness import MinimalHarnessLDM

        method_class = MinimalHarnessLDM
    elif args.method == LDM_COLD_START:
        from alpha_research.methods.ldm_cold_start import AlphaLDMColdStart

        method_class = AlphaLDMColdStart
    elif args.method == LDM_SINGLE_FACTOR:
        from alpha_research.methods.ldm_single_factor import SingleFactorLDM

        method_class = SingleFactorLDM
    elif args.method == LDM_SPLIT_ROBUST_REWARD:
        from alpha_research.methods.ldm_split_robust_reward import SplitRobustLDM

        method_class = SplitRobustLDM
    elif args.method == LDM_RANKIC_WORST_QEHVI:
        from alpha_research.methods.ldm_rankic_worst_qehvi import (
            RankICWorstQEHVIAlphaLDM,
        )

        method_class = RankICWorstQEHVIAlphaLDM
    elif args.method == LDM_RANKIC_RANKICIR_EHVI:
        from alpha_research.methods.ldm_rankic_rankicir_ehvi import (
            RankICRankICIREHVILDM,
        )

        method_class = RankICRankICIREHVILDM
    elif args.method == LDM_RANKIC_TURNOVER_EHVI:
        from alpha_research.methods.ldm_rankic_turnover_ehvi import (
            RankICTurnoverEHVILDM,
        )

        method_class = RankICTurnoverEHVILDM
    elif args.method == LDM_CONTINUOUS_DISCOVERY:
        from alpha_research.methods.ldm_continuous_discovery import (
            ContinuousDiscoveryLDM,
        )

        method_class = ContinuousDiscoveryLDM
    return method_class(
        evaluator=evaluator,
        profiler=profiler,
        generator=generator,
        output_dir=output_dir,
        alphabench_root=alphabench_root,
        objective=args.objective,
        search_objective=args.search_objective,
        late_window_fraction=args.late_window_fraction,
        seed_group=args.seed_group,
        acquisition=args.acquisition,
        acquisition_beta=args.acquisition_beta,
        acquisition_xi=args.acquisition_xi,
        softmax_temperature=args.softmax_temperature,
        diversity_weight=args.diversity_weight,
        evaluate_per_round=args.evaluate_per_round,
        max_refill_batches=args.ldm_max_refill_batches,
        gp_min_fit_data=args.gp_min_fit_data,
        gp_lengthscale=args.gp_lengthscale,
        gp_noise=args.gp_noise,
        gp_scale=args.gp_scale,
        gp_train_iters=args.gp_train_iters,
        gp_lr=args.gp_lr,
        random_seed=args.ldm_random_seed,
        **(
            {"split_settings": split_settings(args)}
            if args.method in {LDM_SPLIT_ROBUST_REWARD, LDM_RANKIC_WORST_QEHVI}
            else {}
        ),
        **(
            {
                "qehvi_reference_margin": args.qehvi_reference_margin,
                "qehvi_n_samples": args.ehvi_n_samples,
            }
            if args.method == LDM_RANKIC_WORST_QEHVI
            else {}
        ),
        **(
            {
                "resume": args.continuous_resume,
                "legacy_source": args.continuous_legacy_source,
                "legacy_expected_protocol": (
                    {**static_protocol(args).to_dict(), "metadata": public_metadata(args)}
                    if args.continuous_legacy_source else None
                ),
                "checkpoint_rounds": args.continuous_checkpoint_round,
                "checkpoint_factor_budget": args.continuous_checkpoint_factor_budget,
                "checkpoint_rank_ic_threshold": args.continuous_rank_ic_threshold,
                "checkpoint_validation_period": Period.from_strings(
                    args.val_start, args.val_end
                ),
                "checkpoint_test_period": Period.from_strings(
                    args.test_start, args.test_end
                ),
                "checkpoint_max_correlation": args.max_correlation,
            }
            if args.method == LDM_CONTINUOUS_DISCOVERY
            else {}
        ),
        **(
            {
                "ehvi_reference_rank_ic": args.ehvi_reference_rank_ic,
                "ehvi_reference_rank_icir": args.ehvi_reference_rank_icir,
                "ehvi_n_samples": args.ehvi_n_samples,
            }
            if args.method == LDM_RANKIC_RANKICIR_EHVI
            else {}
        ),
        **(
            {
                "ehvi_reference_rank_ic": args.ehvi_reference_rank_ic,
                "ehvi_reference_turnover": args.ehvi_reference_turnover,
                "ehvi_n_samples": args.ehvi_n_samples,
            }
            if args.method == LDM_RANKIC_TURNOVER_EHVI
            else {}
        ),
    )


def split_settings(args: argparse.Namespace):
    from alpha_research.methods.split_robust import YearlySplitSettings

    if args.method not in {LDM_SPLIT_ROBUST_REWARD, LDM_RANKIC_WORST_QEHVI}:
        raise ValueError("annual split settings apply only to Stage-2 methods")
    return YearlySplitSettings()


def build_components(args: argparse.Namespace, output_dir: Path):
    alphabench_root = resolve_repo_path(args.alphabench_root)
    evaluator = build_evaluator(args, alphabench_root)

    if args.method in LDM_METHODS:
        return evaluator, build_ldm(args, evaluator, alphabench_root, output_dir)

    if args.method == BASELINE_ALPHA158:
        from alpha_research.methods.baseline_alpha158 import Alpha158Baseline

        return evaluator, Alpha158Baseline(
            evaluator=evaluator,
            alphabench_root=alphabench_root,
            objective=args.objective,
            seed_group=args.seed_group,
            random_seed=args.random_seed,
        )

    search_fn = MockAlphaBenchSearch() if args.mock or args.mock_llm else None
    method = AlphaBenchCoT(
        evaluator=evaluator,
        alphabench_root=alphabench_root,
        output_dir=output_dir,
        model=args.llm_model,
        temperature=args.llm_temperature,
        enable_reason=args.enable_reason,
        accept_threshold=args.accept_threshold,
        workers=args.alphabench_workers,
        chain_batch=args.chain_batch,
        algorithm=args.search_algorithm,
        seed_group=args.seed_group,
        ea_candidates_per_round=args.ea_candidates_per_round,
        ea_mutation_rate=args.ea_mutation_rate,
        ea_crossover_rate=args.ea_crossover_rate,
        ea_pool_size=args.ea_pool_size,
        ea_seeds_top_k=args.ea_seeds_top_k,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        search_fn=search_fn,
    )
    return evaluator, method


def public_metadata(args: argparse.Namespace) -> dict[str, Any]:
    from alpha_research.llm import effective_model_name
    from alpha_research.profile import SCHEMA_VERSION as PROFILE_SCHEMA_VERSION

    provider_path = resolve_repo_path(args.qlib_provider_uri)
    calendar_path = provider_path / "calendars" / "day.txt"
    calendar_start = None
    calendar_end = None
    if calendar_path.is_file():
        calendar_dates = [
            line.strip()
            for line in calendar_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if calendar_dates:
            calendar_start = calendar_dates[0]
            calendar_end = calendar_dates[-1]

    ldm = None
    if args.method in LDM_METHODS:
        ldm = {
            "search_objective": args.search_objective or args.objective,
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "profile_window": profile_period(args).to_dict(),
            "profiler": "mock" if args.mock or args.mock_evaluator else "qlib_behavioural",
            "warm_start": (
                {"d0_source": None, "seed_group": None}
                if args.method == LDM_COLD_START
                else {
                    "d0_source": "factors.lib.alpha158 (exclude_var=vwap)",
                    "seed_group": args.seed_group,
                }
            ),
            "reference_basket": (
                None
                if args.ldm_reference_basket == "none"
                else f"factors.lib.alpha158 (exclude_var=vwap, group={args.seed_group})"
            ),
            "candidates_per_round": args.candidates_per_round,
            "acquisition": {
                "mode": args.acquisition,
                "beta": args.acquisition_beta,
                "xi": args.acquisition_xi,
                "softmax_temperature": args.softmax_temperature,
                "diversity_weight": args.diversity_weight,
            },
            "evaluate_per_round": args.evaluate_per_round,
            "max_refill_batches": args.ldm_max_refill_batches,
            "gp": {
                "kernel": "ScaleKernel(Matern nu=2.5, ARD)",
                "fit_policy": "full_history_deterministic_refit_v2",
                "min_fit_data": args.gp_min_fit_data,
                "lengthscale": args.gp_lengthscale,
                "noise": args.gp_noise,
                "scale": args.gp_scale,
                "train_iters": args.gp_train_iters,
                "lr": args.gp_lr,
            },
            "random_seed": args.ldm_random_seed,
            "validation_policy": {
                "metric": args.objective,
                "max_abs_daily_metric_correlation": args.max_correlation,
                "hard_limit": True,
                "backfill_rejected_factors": False,
                "require_full_factor_budget": True,
                # The pool arms report a set-level number the search never
                # optimised. ldm_single_factor keeps the archive-wide
                # validation evaluation and drops everything the pool readout
                # added on top of it, so the reported quantity and the search
                # objective are the same kind of object.
                "deliverable": (
                    "single_factor"
                    if args.method == LDM_SINGLE_FACTOR
                    else "equal_weight_rank_pool"
                ),
                "correlation_filter_applied": args.method != LDM_SINGLE_FACTOR,
            },
        }
        if args.method == LDM_SPLIT_ROBUST_REWARD:
            ldm["split_reward"] = split_settings(args).to_dict()
        if args.method == LDM_RANKIC_WORST_QEHVI:
            ldm["search_objectives"] = ["train_rankic", "worst_rankic"]
            ldm["split_reward"] = split_settings(args).to_dict()
            ldm["multi_objective_bo"] = {
                "surrogate": "two_independent_gaussian_processes",
                "acquisition": "monte_carlo_qehvi",
                "batch_selection": "exact_enumeration_of_finite_proposal_slate",
                "objective_directions": ["maximize", "maximize"],
                "normalization": "z_score_frozen_from_initial_42_train_observations",
                "reference_point": "componentwise_normalized_initial_min_minus_margin",
                "reference_margin": args.qehvi_reference_margin,
                "posterior_samples_per_batch": args.ehvi_n_samples,
            }
        if args.method == LDM_RANKIC_RANKICIR_EHVI:
            ldm["search_objectives"] = ["rank_ic", "rank_icir"]
            ldm["multi_objective_bo"] = {
                "surrogate": "independent_gp_per_objective",
                "acquisition": "monte_carlo_expected_hypervolume_improvement",
                "objective_directions": ["maximize", "maximize"],
                "reference_point": [
                    args.ehvi_reference_rank_ic,
                    args.ehvi_reference_rank_icir,
                ],
                "posterior_samples_per_candidate": args.ehvi_n_samples,
            }
        if args.method == LDM_RANKIC_TURNOVER_EHVI:
            ldm["search_objectives"] = ["rank_ic", "turnover"]
            ldm["multi_objective_bo"] = {
                "surrogate": "independent_gp_per_objective",
                "acquisition": "monte_carlo_expected_hypervolume_improvement",
                "objective_directions": ["maximize", "minimize"],
                "reference_point": [
                    args.ehvi_reference_rank_ic,
                    args.ehvi_reference_turnover,
                ],
                "posterior_samples_per_candidate": args.ehvi_n_samples,
            }
        if args.method == LDM_CONTINUOUS_DISCOVERY:
            ldm["continuous_discovery"] = {
                "schema_version": "alphaldm.continuous.v1",
                "resume_enabled": args.continuous_resume,
                "commit_boundary": "each complete GP round",
                "checkpoint_rounds": args.continuous_checkpoint_round,
                "checkpoint_factor_budget": args.continuous_checkpoint_factor_budget,
                "test_rank_ic_record_threshold": args.continuous_rank_ic_threshold,
                "factor_order_artifact": "factor_sequence.json",
                "combination_artifact": "top5_combinations.json",
                "checkpoint_test_policy": (
                    "diagnostic_only; never returned to generator, GP, or acquisition"
                ),
            }
            if args.continuous_legacy_source:
                ldm["continuous_discovery"]["legacy_source"] = str(
                    resolve_repo_path(args.continuous_legacy_source)
                )
                ldm["continuous_discovery"]["legacy_backfill_checkpoints"] = True
        if args.method == LDM_PROMPT_HARNESS:
            ldm["prompt_harness"] = {
                "schema_version": "alphaldm.prompt_harness.v2",
                "generation_mode": "one_batch_call_per_round",
                "hypothesis_first": True,
                "train_only_feedback": "best_plus_top3_bottom3_exact_scores",
                "higher_is_better_orientation": True,
                "batch_policy": "approximately_75_percent_refinement_25_percent_exploration",
                "validation_or_test_in_prompt": False,
                "diversity_control": {
                    "schema_version": "alphaldm.diversity.v1",
                    "structural_mode": args.structural_diversity_mode,
                    "behavioral_mode": args.behavioral_diversity_mode,
                    "structural_similarity_threshold": args.structural_similarity_threshold,
                    "behavioral_correlation_threshold": args.behavioral_correlation_threshold,
                    "penalty_weight": args.diversity_penalty_weight,
                    "feedback_clusters": args.diversity_feedback_clusters,
                    "behavioral_data": "Training daily RankIC series only",
                    "numeric_ldm_history_modified": False,
                },
            }
        if args.method == LDM_MINIMAL_PROMPT_HARNESS:
            ldm["minimal_prompt_harness"] = {
                "schema_version": "alphaldm.minimal_prompt_harness.v1",
                "generation_mode": "independent_single_candidate_calls",
                "parallel_calls_per_batch": args.generator_parallel,
                "hypothesis_first": True,
                "candidate_fields": [
                    "name",
                    "hypothesis",
                    "implementation_rationale",
                    "expression",
                ],
                "context_policy": "identical_to_pure_ldm",
                "batch_coordination": False,
                "llm_exploration_exploitation_quota": None,
                "top_bottom_summary": False,
                "motif_feedback": False,
                "automatic_sign_flip": False,
                "diversity_controller": False,
                "validation_or_test_in_prompt": False,
            }
    baseline_alpha158 = None
    if args.method == BASELINE_ALPHA158:
        baseline_alpha158 = {
            "library": "factors.lib.alpha158",
            "exclude_var": "vwap",
            "candidate_count": 42,
            "selection": (
                "validation rank_ic ranking under the configured "
                f"{args.max_correlation} daily-RankIC correlation boundary"
            ),
            "backfill_to_budget": True,
            "backfill_note": (
                "The fixed 42-factor library cannot fill the budget under a hard "
                "boundary, so rejected candidates backfill here where LDM fails closed."
            ),
            "random_seed": args.random_seed,
            "deterministic": True,
        }
    return {
        "method": args.method,
        "ldm": ldm,
        "baseline_alpha158": baseline_alpha158,
        "market": args.market,
        "objective": args.objective,
        "max_correlation": args.max_correlation,
        "qlib_provider": {
            "uri": str(provider_path),
            "calendar_start": calendar_start,
            "calendar_end": calendar_end,
        },
        "eval_parallel": args.eval_parallel,
        "evaluator": "mock" if args.mock or args.mock_evaluator else "alphabench_ffo",
        "ffo": {
            "url": args.ffo_url,
            "label": args.ffo_label,
            "fast": args.ffo_fast,
            "topk": args.ffo_topk,
            "n_drop": args.ffo_n_drop,
            "forward_n": args.ffo_forward_n,
            "use_cache": args.ffo_use_cache,
            "max_attempts": args.ffo_max_attempts,
            "label_horizon_days": {
                "close_return": 1,
                "close_return_lag": 2,
            }.get(args.ffo_label),
        },
        "llm": None if args.method == BASELINE_ALPHA158 else {
            "backend": "mock" if args.mock or args.mock_llm else "alphabench_official",
            "base_url": args.llm_base_url,
            "model": args.llm_model,
            "effective_model": effective_model_name(
                args.llm_base_url, args.llm_model
            ),
            "api_key_env": args.llm_api_key_env,
            "temperature": args.llm_temperature,
            "timeout_seconds": args.llm_timeout_seconds,
            "max_tokens": args.llm_max_tokens,
            "context_window": args.llm_context_window,
            "reasoning_effort": args.llm_reasoning_effort,
            "thinking_enabled": args.llm_thinking,
        },
        "alphabench": {
            "repository": "https://github.com/CityU-MLO/AlphaBench",
            "commit": PINNED_COMMIT,
            "root": str(resolve_repo_path(args.alphabench_root)),
            # LDM borrows AlphaBench only for the Alpha158 seed library; the
            # search itself is ours, so naming an official searcher here would
            # misdescribe the run.
            "algorithm": (
                None
                if args.method in LDM_METHODS or args.method == BASELINE_ALPHA158
                else "searcher.algo.ea.EAAlgo"
                if args.search_algorithm == "ea"
                else "searcher.algo.cot.CoTAlgo"
            ),
            "ea": {
                "candidates_per_round": args.ea_candidates_per_round,
                "mutation_rate": args.ea_mutation_rate,
                "crossover_rate": args.ea_crossover_rate,
                "pool_size": args.ea_pool_size,
                "seeds_top_k": args.ea_seeds_top_k,
            } if args.method == BASELINE_ALPHABENCH_COT and args.search_algorithm == "ea" else None,
            # The searcher is AlphaBench's, but the validation gate is LDM's, so
            # the two methods differ only in how candidates are proposed.
            "validation_policy": {
                "metric": args.objective,
                "max_abs_daily_metric_correlation": args.max_correlation,
                "hard_limit": True,
                "backfill_rejected_factors": False,
                "require_full_factor_budget": True,
            } if args.method == BASELINE_ALPHABENCH_COT else None,
            "seed_library": (
                f"profiler reference basket only (exclude_var=vwap, group={args.seed_group})"
                if args.method == LDM_COLD_START and args.ldm_reference_basket == "alpha158"
                else "not used"
                if args.method == LDM_COLD_START
                else f"factors.lib.alpha158 (exclude_var=vwap, group={args.seed_group})"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.factor_budget <= 0:
        raise ValueError("--factor-budget must be positive")
    if args.alphabench_workers <= 0:
        raise ValueError("--alphabench-workers must be positive")
    seed = args.ldm_random_seed if args.method in LDM_METHODS else args.random_seed
    expected_run_name = standard_run_name(
        method=args.method,
        model=args.llm_model,
        train_start=args.train_start,
        test_end=args.test_end,
        seed=seed,
    )
    if args.out_dir is None:
        output_dir = standard_output_dir(
            REPO_ROOT,
            method=args.method,
            model=args.llm_model,
            train_start=args.train_start,
            test_end=args.test_end,
            seed=seed,
        )
    else:
        output_dir = resolve_repo_path(args.out_dir)
        if output_dir.name != expected_run_name:
            raise ValueError(
                "--out-dir must end with the canonical run name "
                f"{expected_run_name!r}; got {output_dir.name!r}"
            )
    if args.out_dir is None and args.method in {
        LDM_SPLIT_ROBUST_REWARD, LDM_RANKIC_WORST_QEHVI,
    }:
        output_dir = (
            REPO_ROOT.parent / "AlphaResearch_local_results" / "stage2"
            / args.method / expected_run_name
        )
    protocol = static_protocol(args)
    is_stage2 = args.method in {LDM_SPLIT_ROBUST_REWARD, LDM_RANKIC_WORST_QEHVI}
    if is_stage2:
        from alpha_research.methods.split_robust.contract import (
            validate_annual_output, validate_train_years,
        )

        validate_train_years(protocol.train, split_settings(args))
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "protocol": protocol.to_dict(),
            "metadata": public_metadata(args),
            "output_dir": str(output_dir),
        }, indent=2, sort_keys=True))
        return 0

    if is_stage2:
        validate_annual_output(output_dir)
    evaluator, method = build_components(args, output_dir)
    summary = StaticEnvironment(
        protocol=protocol,
        method=method,
        evaluator=evaluator,
        output_dir=output_dir,
        run_metadata=public_metadata(args),
    ).run()
    print(json.dumps({
        "status": summary["status"],
        "environment": summary["environment"],
        "summary": str(output_dir / "summary.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

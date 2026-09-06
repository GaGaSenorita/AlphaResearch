"""Experiment argument contract; defaults are kept separate from orchestration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .naming import BASELINE_ALPHABENCH_COT, CANONICAL_METHODS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an AlphaResearch method under the static experimental protocol."
    )
    parser.add_argument("--environment", choices=["static"], default="static")
    parser.add_argument(
        "--method",
        choices=CANONICAL_METHODS,
        default=BASELINE_ALPHABENCH_COT,
        help=(
            "Which search method StaticEnvironment drives. ldm_cold_start is "
            "ldm_standard "
            "with an empty D_0: no Alpha158 warm-up, everything else identical. "
            "ldm_single_factor is ldm_standard with the pool readout removed: the same "
            "search, delivering the single best factor on validation. "
            "ldm_split_robust_reward is ldm_standard with the reward replaced by the "
            "worst of 20 calendar-quarter mean signed RankIC values on Train. "
            "ldm_rankic_worst_qehvi uses independent GPs and true batch qEHVI to "
            "maximise Train RankIC and worst-quarter RankIC in the Pareto sense. "
            "ldm_rankic_rankicir_ehvi fits independent RankIC and RankICIR GPs and "
            "acquires candidates by two-objective expected hypervolume improvement. "
            "ldm_rankic_turnover_ehvi applies the same EHVI search to maximise RankIC "
            "while minimising turnover. "
            "ldm_continuous_discovery keeps pure LDM search but commits every complete "
            "round for exact resume and produces staged Validation Top-5 reports."
        ),
    )
    parser.add_argument("--alphabench-root", default="external/AlphaBench")
    parser.add_argument("--ffo-url", default="http://127.0.0.1:19777")
    parser.add_argument("--ffo-label", default="close_return")
    parser.add_argument("--ffo-timeout", type=int, default=600)
    parser.add_argument("--ffo-topk", type=int, default=50)
    parser.add_argument("--ffo-n-drop", type=int, default=5)
    parser.add_argument("--ffo-forward-n", type=int, default=1)
    parser.add_argument(
        "--ffo-fast", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ffo-use-cache", action=argparse.BooleanOptionalAction, default=False,
        help="Use the persistent FFO factor cache. Off by default: the cache-hit "
             "path returns stored daily IC without re-checking trading-day "
             "coverage, so only a fresh evaluation is verified end to end.",
    )
    parser.add_argument(
        "--ffo-max-attempts", type=int, default=3,
        help="Maximum fresh evaluation attempts before a factor is explicitly rejected.",
    )
    parser.add_argument("--market", choices=["csi300"], default="csi300")
    parser.add_argument(
        "--objective",
        choices=["ic", "rank_ic", "icir", "rank_icir"],
        default="rank_ic",
    )
    parser.add_argument("--factor-budget", type=int, default=10)
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=None,
        help=(
            "Skip a validation candidate whose daily-IC series correlates above this "
            "with an already selected factor. Unset keeps plain top-k selection."
        ),
    )
    parser.add_argument(
        "--eval-parallel",
        type=int,
        default=1,
        help="Concurrent FFO evaluations per evaluate_many call.",
    )
    parser.add_argument("--search-rounds", type=int, default=20)
    parser.add_argument("--accept-threshold", type=float, default=0.0)
    parser.add_argument("--alphabench-workers", type=int, default=1)
    parser.add_argument(
        "--seed-group",
        choices=["all", "kbar_price"],
        default="all",
        help=(
            "Which Alpha158 seeds to start from. AlphaBench's own searching "
            "benchmark hands CoT only the kbar and price factors."
        ),
    )
    parser.add_argument(
        "--search-algorithm",
        choices=["coe", "ea"],
        default="coe",
        help="Which official AlphaBench searcher to run.",
    )
    parser.add_argument("--ea-candidates-per-round", type=int, default=30)
    parser.add_argument("--ea-mutation-rate", type=float, default=0.4)
    parser.add_argument("--ea-crossover-rate", type=float, default=0.6)
    parser.add_argument("--ea-pool-size", type=int, default=30)
    parser.add_argument("--ea-seeds-top-k", type=int, default=12)
    parser.add_argument(
        "--chain-batch",
        type=int,
        default=8,
        help="Chains run concurrently per batch; the FFO backend stalls above ~8.",
    )

    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2020-12-29")
    parser.add_argument("--val-start", default="2021-01-01")
    parser.add_argument("--val-end", default="2021-12-29")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2025-12-26")

    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("ALPHARESEARCH_LLM_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("ALPHARESEARCH_LLM_MODEL") or "deepseek-v4-pro",
    )
    parser.add_argument("--llm-api-key-env", default="ALPHARESEARCH_LLM_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=320000)
    parser.add_argument(
        "--llm-context-window",
        type=int,
        default=1000000,
        help="Declared model context window recorded in run metadata.",
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        choices=["none", "high", "max"],
        default="high",
    )
    parser.add_argument(
        "--llm-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Explicitly enable or disable provider-native thinking mode. Unset sends "
            "no provider-specific thinking field."
        ),
    )
    parser.add_argument(
        "--enable-reason", action=argparse.BooleanOptionalAction, default=True
    )

    # --- LDM ------------------------------------------------------------
    # The profiling window is deliberately a tail slice of the training window.
    # A fingerprint is only worth computing while it stays materially cheaper
    # than the evaluation it replaces, and cost scales with the window.
    parser.add_argument(
        "--qlib-provider-uri",
        default=str(REPO_ROOT.parent / "quantaalpha_qlib_csi300" / "cn_data"),
    )
    parser.add_argument("--profile-start", default=None)
    parser.add_argument("--profile-end", default=None)
    parser.add_argument(
        "--search-objective",
        choices=[
            "ic", "rank_ic", "icir", "rank_icir", "late_rank_ic", "worst_rankic",
        ],
        default=None,
        help=(
            "Metric the surrogate is trained on. Defaults to --objective. The "
            "pool is always selected on --objective, so the two can differ."
        ),
    )
    parser.add_argument(
        "--late-window-fraction",
        type=float,
        default=0.25,
        help="Closing share of the training window used by late_rank_ic.",
    )
    parser.add_argument("--candidates-per-round", type=int, default=8)
    parser.add_argument("--generator-parallel", type=int, default=8)
    parser.add_argument("--history-shown", type=int, default=24)
    parser.add_argument(
        "--structural-diversity-mode",
        choices=["off", "diagnostics", "feedback", "reject", "penalty"],
        default="off",
        help="Harness-only AST control. 'off' preserves the existing method exactly.",
    )
    parser.add_argument(
        "--behavioral-diversity-mode",
        choices=["off", "diagnostics", "feedback", "reject", "penalty"],
        default="off",
        help="Harness-only Train daily-RankIC control; Validation/Test are never read.",
    )
    parser.add_argument(
        "--structural-similarity-threshold",
        type=float,
        default=None,
        help="Optional archive clustering/reject threshold; unset uses exact parameter-normalized families.",
    )
    parser.add_argument(
        "--behavioral-correlation-threshold",
        type=float,
        default=None,
        help="Optional absolute Train daily-RankIC correlation threshold; unset is diagnostics-only.",
    )
    parser.add_argument(
        "--diversity-penalty-weight",
        type=float,
        default=0.0,
        help="Prompt-example priority penalty only; never changes BO/GP/acquisition or Train scores.",
    )
    parser.add_argument("--diversity-feedback-clusters", type=int, default=6)
    parser.add_argument(
        "--acquisition",
        choices=["ucb", "ei", "random", "ehvi", "qehvi"],
        default="ucb",
        help=(
            "random drops the surrogate from the decision; the ablation arm. "
            "ehvi is reserved for the two ldm_rankic_*_ehvi methods; qehvi is "
            "reserved for ldm_rankic_worst_qehvi."
        ),
    )
    parser.add_argument("--acquisition-beta", type=float, default=2.0)
    parser.add_argument("--acquisition-xi", type=float, default=0.01)
    parser.add_argument(
        "--ehvi-reference-rank-ic",
        type=float,
        default=-0.10,
        help="RankIC coordinate of the fixed, dominated EHVI reference point.",
    )
    parser.add_argument(
        "--ehvi-reference-rank-icir",
        type=float,
        default=-0.50,
        help="RankICIR coordinate of the fixed, dominated EHVI reference point.",
    )
    parser.add_argument(
        "--ehvi-reference-turnover",
        type=float,
        default=1.50,
        help=(
            "Turnover coordinate of the fixed EHVI reference point. Because turnover "
            "is minimised, this must be strictly above every warm-up Pareto value."
        ),
    )
    parser.add_argument(
        "--ehvi-n-samples",
        "--qehvi-n-samples",
        dest="ehvi_n_samples",
        type=int,
        default=128,
        help="Posterior draws for Monte Carlo EHVI/qEHVI.",
    )
    parser.add_argument(
        "--qehvi-reference-margin",
        type=float,
        default=0.1,
        help=(
            "Positive margin below each normalized initial-observation minimum "
            "used to freeze the RankIC-Worst qEHVI reference point."
        ),
    )
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on distance to the nearest observed fingerprint, added to "
            "standardised acquisition values. 0 is plain UCB."
        ),
    )
    parser.add_argument(
        "--evaluate-per-round",
        type=int,
        default=1,
        help=(
            "Candidates actually sent to the evaluator each round. More "
            "observations make the surrogate usable sooner but cut the "
            "evaluations it is meant to save."
        ),
    )
    parser.add_argument(
        "--ldm-max-refill-batches",
        type=int,
        default=8,
        help=(
            "Maximum independent LLM proposal batches used to obtain the required "
            "number of verified RankIC observations in one GP round."
        ),
    )
    parser.add_argument(
        "--ldm-reference-basket",
        choices=["alpha158", "none"],
        default="alpha158",
        help=(
            "Frozen basket the *_to_ref fingerprint coordinates are measured "
            "against. This is the profiler coordinate system, not search data. "
            "'none' leaves those three coordinates at a constant zero and makes "
            "a cold-start run entirely free of Alpha158."
        ),
    )
    parser.add_argument("--gp-min-fit-data", type=int, default=20)
    parser.add_argument(
        "--gp-lengthscale",
        type=float,
        default=None,
        help=(
            "Unset uses the median pairwise distance of the standardised "
            "history. A fixed 0.75 leaves the kernel blind at this width: "
            "typical standardised distances are around 4, so every pair "
            "evaluates to zero and the posterior collapses to the prior."
        ),
    )
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--gp-scale", type=float, default=0.25)
    parser.add_argument("--gp-train-iters", type=int, default=100)
    parser.add_argument("--gp-lr", type=float, default=0.05)
    parser.add_argument("--ldm-random-seed", type=int, default=42)
    parser.add_argument(
        "--continuous-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume ldm_continuous_discovery from its last committed round.",
    )
    parser.add_argument(
        "--continuous-checkpoint-round",
        action="append",
        type=int,
        default=[],
        help="Repeatable round at which to write a staged Validation Top-5/Test report.",
    )
    parser.add_argument(
        "--continuous-legacy-source",
        default=None,
        help="Import a completed ldm_standard run into a NEW continuous run; preserve its history.",
    )
    parser.add_argument("--continuous-checkpoint-factor-budget", type=int, default=5)
    parser.add_argument("--continuous-rank-ic-threshold", type=float, default=0.035)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Recorded replicate seed for deterministic non-LDM methods.",
    )

    parser.add_argument("--min-observations", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Optional result directory. Its leaf must follow "
            "method_llm_YYYY-YYYY_seedN; omitted uses runs/<method>/<standard-name>."
        ),
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-evaluator", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)

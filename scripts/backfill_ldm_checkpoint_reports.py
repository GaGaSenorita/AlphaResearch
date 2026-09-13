#!/usr/bin/env python3
"""Backfill R0/R5/... reports for preserved continual-discovery runs.

The script reads the committed ``factor_sequence.json`` and never invokes an
LLM or changes search state. At each requested round it evaluates all then-
available factors on Validation, selects Top-5 there, and audits that frozen
pool on Test. Existing checkpoints are preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alpha_research.cli import (
    resolved_runner_args as resolve_config_args,
    resolve_config_path,
)
from alpha_research.io import append_jsonl, read_json
from alpha_research.methods.ldm_continuous_discovery import ContinuousDiscoveryLDM
from alpha_research.naming import standard_output_dir
from alpha_research.runner import (
    REPO_ROOT,
    build_evaluator,
    resolve_repo_path,
)
from alpha_research.types import Period


DEFAULT_CONFIG = (
    "configs/ldm_continuous_discovery/"
    "ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml"
)


def checkpoint_schedule(target_round: int, step: int) -> list[int]:
    if target_round < 0:
        raise ValueError("target round cannot be negative")
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    rounds = list(range(0, target_round + 1, step))
    if rounds[-1] != target_round:
        rounds.append(target_round)
    return rounds


def resolved_runner_args(
    config_path: Path,
    *,
    seed: int,
    target_round: int,
    checkpoints: list[int],
    eval_parallel: int | None = None,
) -> argparse.Namespace:
    overrides = {
        "ldm-random-seed": seed,
        "search-rounds": target_round,
        "continuous-checkpoint-round": checkpoints,
    }
    if eval_parallel is not None:
        overrides["eval-parallel"] = max(1, int(eval_parallel))
    return resolve_config_args(config_path, overrides=overrides)


def existing_checkpoint_rounds(run_dir: Path) -> list[int]:
    path = run_dir / "top5_combinations.json"
    if not path.is_file():
        return []
    return sorted(
        int(row["checkpoint_round"])
        for row in read_json(path).get("checkpoints", [])
    )


def backfill_one(
    config_path: Path,
    *,
    seed: int,
    target_round: int,
    checkpoints: list[int],
    dry_run: bool,
    render: bool,
    eval_parallel: int | None = None,
) -> dict[str, Any]:
    args = resolved_runner_args(
        config_path,
        seed=seed,
        target_round=target_round,
        checkpoints=checkpoints,
        eval_parallel=eval_parallel,
    )
    run_dir = standard_output_dir(
        REPO_ROOT,
        method=args.method,
        model=args.llm_model,
        train_start=args.train_start,
        test_end=args.test_end,
        seed=seed,
    )
    if not (run_dir / "factor_sequence.json").is_file():
        raise FileNotFoundError(f"missing preserved factor sequence: {run_dir}")
    completed_round = int(read_json(run_dir / "resume_state.json")["round_id"])
    requested = [round_id for round_id in checkpoints if round_id <= completed_round]
    before = existing_checkpoint_rounds(run_dir)
    missing = [round_id for round_id in requested if round_id not in before]
    if dry_run:
        return {
            "seed": seed,
            "run_dir": str(run_dir),
            "completed_round": completed_round,
            "existing_checkpoints": before,
            "missing_checkpoints": missing,
            "eval_parallel": args.eval_parallel,
            "dry_run": True,
        }

    alphabench_root = resolve_repo_path(args.alphabench_root)
    evaluator = build_evaluator(args, alphabench_root)
    method = ContinuousDiscoveryLDM(
        evaluator=evaluator,
        profiler=None,
        generator=None,
        output_dir=run_dir,
        objective=args.objective,
        acquisition="random",
        random_seed=seed,
        checkpoint_rounds=checkpoints,
        checkpoint_factor_budget=args.continuous_checkpoint_factor_budget,
        checkpoint_rank_ic_threshold=args.continuous_rank_ic_threshold,
        checkpoint_validation_period=Period.from_strings(
            args.val_start, args.val_end
        ),
        checkpoint_test_period=Period.from_strings(
            args.test_start, args.test_end
        ),
        checkpoint_max_correlation=args.max_correlation,
    )
    events_path = run_dir / "checkpoint_backfill_events.jsonl"
    method.backfill_checkpoint_reports_from_sequence(
        event_sink=lambda event: append_jsonl(events_path, event),
    )
    after = existing_checkpoint_rounds(run_dir)

    plot_metadata = None
    if render:
        from alpha_research.reporting.financial import draw_financial_trajectory

        plot_metadata = draw_financial_trajectory(
            run_dir,
            REPO_ROOT / "figures" / f"ldm_financial_mining_seed{seed}_round{completed_round}",
            seed=seed,
        )
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "completed_round": completed_round,
        "checkpoints_before": before,
        "checkpoints_after": after,
        "backfilled_checkpoints": [round_id for round_id in after if round_id not in before],
        "llm_called": False,
        "search_state_changed": False,
        "rendered": bool(plot_metadata),
        "eval_parallel": args.eval_parallel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--target-round", type=int, default=100)
    parser.add_argument("--checkpoint-step", type=int, default=5)
    parser.add_argument(
        "--eval-parallel", type=int,
        help="override evaluator request parallelism for reporting-only backfill",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    options = parser.parse_args()

    config_path = resolve_config_path(options.config)
    checkpoints = checkpoint_schedule(options.target_round, options.checkpoint_step)
    results = [
        backfill_one(
            config_path,
            seed=seed,
            target_round=options.target_round,
            checkpoints=checkpoints,
            dry_run=options.dry_run,
            render=not options.no_render,
            eval_parallel=options.eval_parallel,
        )
        for seed in options.seeds
    ]
    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

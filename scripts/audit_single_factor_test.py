#!/usr/bin/env python3
"""Retrospectively evaluate every Train-qualified factor on Test.

The resulting Test ledger is descriptive audit evidence only.  It is never
used to generate candidates, choose a Validation winner, or admit a factor to
the formal library.  Rows are appended after each completed batch so the job
can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha_research.cli import resolved_runner_args
from alpha_research.factor_library import (
    LIBRARY_SCHEMA,
    build_single_factor_library,
    discover_run_dirs,
    write_single_factor_library,
)
from alpha_research.io import append_jsonl
from alpha_research.runner import REPO_ROOT, build_evaluator, resolve_repo_path
from alpha_research.types import FactorCandidate, Period

DEFAULT_CONFIG = (
    "configs/ldm_continuous_discovery/"
    "ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml"
)


def _build(
    run_dirs: dict[int, Path],
    *,
    output_dir: Path,
    train_threshold: float,
    validation_threshold: float,
    test_threshold: float,
    include_seed_factors: bool,
) -> dict:
    validation_ledger = output_dir / "single_factor_validation_ledger.jsonl"
    test_ledger = output_dir / "single_factor_test_ledger.jsonl"
    return build_single_factor_library(
        run_dirs,
        train_rank_ic_threshold=train_threshold,
        validation_rank_ic_threshold=validation_threshold,
        test_rank_ic_audit_threshold=test_threshold,
        include_seed_factors=include_seed_factors,
        extra_validation_ledgers=(
            [validation_ledger] if validation_ledger.is_file() else []
        ),
        extra_test_ledgers=[test_ledger] if test_ledger.is_file() else [],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / DEFAULT_CONFIG)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs" / "ldm_continuous_discovery",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--train-rank-ic-threshold", type=float, default=0.04)
    parser.add_argument("--validation-rank-ic-threshold", type=float, default=0.04)
    parser.add_argument("--test-rank-ic-audit-threshold", type=float, default=0.04)
    parser.add_argument("--include-seed-factors", action="store_true")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    options = parser.parse_args()
    if options.parallel <= 0 or options.batch_size <= 0:
        parser.error("--parallel and --batch-size must be positive")
    if options.limit is not None and options.limit < 0:
        parser.error("--limit cannot be negative")

    runs_root = options.runs_root.resolve()
    output_dir = (options.output_dir or runs_root / "factor_library").resolve()
    run_dirs = discover_run_dirs(runs_root, seeds=options.seeds)
    payload = _build(
        run_dirs,
        output_dir=output_dir,
        train_threshold=options.train_rank_ic_threshold,
        validation_threshold=options.validation_rank_ic_threshold,
        test_threshold=options.test_rank_ic_audit_threshold,
        include_seed_factors=options.include_seed_factors,
    )
    pending = [row for row in payload["factors"] if row["test_rank_ic"] is None]
    if options.limit is not None:
        pending = pending[: options.limit]
    preview = {
        "total_train_qualified": len(payload["factors"]),
        "already_measured_on_test": payload["summary"][
            "with_retrospective_test_evidence"
        ],
        "pending_test": len(pending),
        "test_is_retrospective_audit_only": True,
        "test_used_for_admission": False,
        "llm_called": False,
        "search_state_changed": False,
        "output_dir": str(output_dir),
    }
    if options.dry_run:
        print(json.dumps({**preview, "dry_run": True}, indent=2, ensure_ascii=False))
        return 0
    if not pending:
        write_single_factor_library(output_dir, payload)
        print(json.dumps({**preview, "evaluated": 0}, indent=2, ensure_ascii=False))
        return 0

    runner_args = resolved_runner_args(
        options.config.resolve(), overrides={"ldm-random-seed": options.seeds[0]},
    )
    runner_args.eval_parallel = options.parallel
    test_period = Period.from_strings(runner_args.test_start, runner_args.test_end)
    evaluator = build_evaluator(
        runner_args,
        resolve_repo_path(runner_args.alphabench_root),
    )
    ledger_path = output_dir / "single_factor_test_ledger.jsonl"
    completed = 0
    for offset in range(0, len(pending), options.batch_size):
        factor_rows = pending[offset : offset + options.batch_size]
        candidates = [
            FactorCandidate(
                name=str(row["name"]),
                expression=str(row["expression"]),
                reason=str(row.get("reason", "")),
                source=str(row.get("source", "ldm_continuous_discovery")),
                round_id=int(
                    (row.get("best_train_occurrence") or {}).get("round_id") or 0
                ),
            )
            for row in factor_rows
        ]
        results = evaluator.evaluate_many(candidates, test_period)
        for factor, candidate, result in zip(factor_rows, candidates, results):
            append_jsonl(
                ledger_path,
                {
                    "schema_version": LIBRARY_SCHEMA,
                    "canonical": factor["canonical"],
                    "candidate": candidate.to_dict(),
                    "seed": (factor.get("best_train_occurrence") or {}).get("seed"),
                    "period": test_period.to_dict(),
                    "audit_basis": {
                        "split": "Train",
                        "metric": "rank_ic",
                        "threshold": options.train_rank_ic_threshold,
                    },
                    "test": result.to_dict(),
                    "retrospective_audit_only": True,
                    "used_for_admission": False,
                },
            )
            completed += 1
        print(
            json.dumps(
                {
                    "progress": f"{completed}/{len(pending)}",
                    "successful_in_batch": sum(result.success for result in results),
                    "ledger": str(ledger_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    final_payload = _build(
        run_dirs,
        output_dir=output_dir,
        train_threshold=options.train_rank_ic_threshold,
        validation_threshold=options.validation_rank_ic_threshold,
        test_threshold=options.test_rank_ic_audit_threshold,
        include_seed_factors=options.include_seed_factors,
    )
    write_single_factor_library(output_dir, final_payload)
    print(
        json.dumps(
            {
                **preview,
                "evaluated": completed,
                "summary": final_payload["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

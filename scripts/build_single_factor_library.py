#!/usr/bin/env python3
"""Build the cross-seed AlphaLDM single-factor candidate/admission library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha_research.factor_library import (
    build_single_factor_library,
    discover_run_dirs,
    write_single_factor_library,
)
from alpha_research.runner import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs" / "ldm_continuous_discovery",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--train-rank-ic-threshold", type=float, default=0.04)
    parser.add_argument("--validation-rank-ic-threshold", type=float, default=0.04)
    parser.add_argument("--test-rank-ic-audit-threshold", type=float, default=0.04)
    parser.add_argument(
        "--include-seed-factors",
        action="store_true",
        help="Include fixed Alpha158 warm-start factors as well as LDM-generated factors.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    options = parser.parse_args()

    runs_root = options.runs_root.resolve()
    output_dir = (options.output_dir or runs_root / "factor_library").resolve()
    run_dirs = discover_run_dirs(runs_root, seeds=options.seeds)
    extra_ledger = output_dir / "single_factor_validation_ledger.jsonl"
    test_ledger = output_dir / "single_factor_test_ledger.jsonl"
    payload = build_single_factor_library(
        run_dirs,
        train_rank_ic_threshold=options.train_rank_ic_threshold,
        validation_rank_ic_threshold=options.validation_rank_ic_threshold,
        test_rank_ic_audit_threshold=options.test_rank_ic_audit_threshold,
        include_seed_factors=options.include_seed_factors,
        extra_validation_ledgers=[extra_ledger] if extra_ledger.is_file() else [],
        extra_test_ledgers=[test_ledger] if test_ledger.is_file() else [],
    )
    write_single_factor_library(output_dir, payload)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": payload["summary"],
                "test_used_for_admission": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

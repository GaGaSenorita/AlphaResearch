#!/usr/bin/env python3
"""Render all three five-round figures and their comparison from saved reports.

No evaluator, market data, API key, or LLM is used. Each plot is saved once in
figures/, as a PDF by default, with a JSON sidecar preserving its measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha_research.reporting.financial import draw_financial_trajectory
from alpha_research.reporting.comparison import load_seed_report, draw_comparison
from alpha_research.reporting.output import FIGURE_FORMATS


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--format", choices=FIGURE_FORMATS, default="pdf",
                        help="one output image format (default: pdf)")
    args = parser.parse_args()
    seed_dirs = {
        seed: args.research_root / "runs/ldm_continuous_discovery" / (
            f"ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}"
        )
        for seed in (42, 123, 456)
    }
    # Validate all 21 measurements per seed before overwriting any figure.
    for seed, run_dir in seed_dirs.items():
        load_seed_report(seed, run_dir)
    for seed, run_dir in seed_dirs.items():
        draw_financial_trajectory(
            run_dir,
            args.output_dir / f"ldm_financial_mining_seed{seed}_round100",
            seed=seed,
            output_format=args.format,
        )
        print(f"Seed {seed}: 21 measured checkpoints; figure updated", flush=True)
    combined = draw_comparison(
        seed_dirs, args.output_dir / "ldm_financial_mining_three_seed_comparison",
        output_format=args.format,
    )
    print(json.dumps(combined["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

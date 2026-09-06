#!/usr/bin/env python3
"""Render all three five-round figures and their comparison from saved reports.

No evaluator, market data, API key, or LLM is used. The copies in figures/ and
the canonical run directories are generated together to prevent stale plots.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from alpha_research.reporting.financial import draw_financial_trajectory
from alpha_research.reporting.comparison import load_seed_report, draw_comparison


ROOT = Path(__file__).resolve().parents[1]


def clean_svg_whitespace(path: Path) -> None:
    """Normalise Matplotlib's line-ending spaces without changing SVG geometry."""
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main() -> int:
    seed_dirs = {
        seed: ROOT / "runs/ldm_continuous_discovery" / (
            f"ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}"
        )
        for seed in (42, 123, 456)
    }
    # Validate all 21 measurements per seed before overwriting any figure.
    for seed, run_dir in seed_dirs.items():
        load_seed_report(seed, run_dir)
    for seed, run_dir in seed_dirs.items():
        metadata = draw_financial_trajectory(
            run_dir,
            ROOT / "figures" / f"ldm_financial_mining_seed{seed}_round100",
            seed=seed,
        )
        clean_svg_whitespace(Path(metadata["outputs"]["svg"]))
        for kind in ("png", "svg", "pdf"):
            shutil.copy2(metadata["outputs"][kind], run_dir / f"figure.{kind}")
        metadata["outputs"] = {
            kind: str(run_dir / f"figure.{kind}") for kind in ("png", "svg", "pdf")
        }
        metadata["outputs"]["metadata"] = str(run_dir / "figure.json")
        (run_dir / "figure.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Seed {seed}: 21 measured checkpoints; figures/ and run copy updated", flush=True)
    combined = draw_comparison(
        seed_dirs, ROOT / "figures/ldm_financial_mining_three_seed_comparison"
    )
    clean_svg_whitespace(Path(combined["outputs"]["svg"]))
    print(json.dumps(combined["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check the pinned AlphaBench integration; --apply explicitly installs its patch."""

import argparse
import subprocess
from pathlib import Path

from alpha_research.alphabench_runtime import configure_verified_evaluator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "external/AlphaBench",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        state = configure_verified_evaluator(args.root, apply=args.apply)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"AlphaBench setup failed: {exc}\n")
    print(f"AlphaBench integration: {state}")
    if state == "upstream":
        print("Integrity patch is not installed. Use --apply before starting the FFO service.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

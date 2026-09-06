"""Paths and configuration for the demo; credentials never enter job JSON."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FLASH_CONFIG = "configs/ldm_continuous_discovery/ldm_continuous_discovery_neolink-deepseek-v4-flash_2016-2025.yaml"


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO_ROOT
    runtime_dir: Path | None = None
    replay_dir: Path | None = None

    @property
    def jobs(self) -> Path:
        return (self.runtime_dir or self.repo_root / "runtime/api") / "jobs"

    @property
    def replays(self) -> Path:
        return self.replay_dir or self.repo_root / "demo/replays"


def online_args(seed: int, target: int, output_dir: Path, *, repo_root: Path = REPO_ROOT):
    # Heavy research imports belong in workers, never the replay HTTP path.
    from alpha_research.cli import load_config, append_cli_arg, expand_value, build_expansion_context
    from alpha_research.runner import parse_args

    path = repo_root / FLASH_CONFIG
    config = load_config(path)
    raw = dict(config["args"])
    raw.update({"ldm-random-seed": seed, "search-rounds": target,
                "continuous-checkpoint-round": list(range(0, target + 1, 5)),
                "continuous-resume": True, "out-dir": str(output_dir)})
    if target % 5:
        raw["continuous-checkpoint-round"].append(target)
    if os.environ.get("ALPHARESEARCH_QLIB_PROVIDER_URI"):
        raw["qlib-provider-uri"] = os.environ["ALPHARESEARCH_QLIB_PROVIDER_URI"]
    if os.environ.get("ALPHARESEARCH_FFO_URL"):
        raw["ffo-url"] = os.environ["ALPHARESEARCH_FFO_URL"]
    context = build_expansion_context(config, raw, path)
    argv = ["--method", "ldm_continuous_discovery", "--environment", "static"]
    for key, value in raw.items():
        append_cli_arg(argv, key, expand_value(value, path, context=context))
    args = parse_args(argv)
    if args.mock or args.mock_llm or args.mock_evaluator:
        raise ValueError("Online mode requires the real proposer and evaluator")
    return args

"""Canonical public names for methods, configs, and run directories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


BASELINE_ALPHA158 = "baseline_alpha158"
BASELINE_ALPHABENCH_COT = "baseline_alphabench_cot"
LDM_STANDARD = "ldm_standard"
LDM_COLD_START = "ldm_cold_start"
LDM_PROMPT_HARNESS = "ldm_prompt_harness"
LDM_MINIMAL_PROMPT_HARNESS = "ldm_minimal_prompt_harness"
LDM_SINGLE_FACTOR = "ldm_single_factor"
LDM_SPLIT_ROBUST_REWARD = "ldm_split_robust_reward"
LDM_RANKIC_RANKICIR_EHVI = "ldm_rankic_rankicir_ehvi"
LDM_RANKIC_TURNOVER_EHVI = "ldm_rankic_turnover_ehvi"
LDM_CONTINUOUS_DISCOVERY = "ldm_continuous_discovery"

CANONICAL_METHODS = (
    BASELINE_ALPHA158,
    BASELINE_ALPHABENCH_COT,
    LDM_STANDARD,
    LDM_COLD_START,
    LDM_PROMPT_HARNESS,
    LDM_MINIMAL_PROMPT_HARNESS,
    LDM_SINGLE_FACTOR,
    LDM_SPLIT_ROBUST_REWARD,
    LDM_RANKIC_RANKICIR_EHVI,
    LDM_RANKIC_TURNOVER_EHVI,
    LDM_CONTINUOUS_DISCOVERY,
)
LDM_METHODS = frozenset(
    {
        LDM_STANDARD,
        LDM_COLD_START,
        LDM_PROMPT_HARNESS,
        LDM_MINIMAL_PROMPT_HARNESS,
        LDM_SINGLE_FACTOR,
        LDM_SPLIT_ROBUST_REWARD,
        LDM_RANKIC_RANKICIR_EHVI,
        LDM_RANKIC_TURNOVER_EHVI,
        LDM_CONTINUOUS_DISCOVERY,
    }
)
NO_LLM_METHODS = frozenset({BASELINE_ALPHA158})


def _method_slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if slug not in CANONICAL_METHODS:
        raise ValueError(f"Unknown canonical method name: {value!r}")
    return slug


def _model_slug(value: Any) -> str:
    """Keep provider model names readable while making them path-safe."""
    slug = re.sub(r"[^a-z0-9.-]+", "-", str(value).strip().lower()).strip("-.")
    return slug or "unknown-llm"


def _year(value: Any, field: str) -> str:
    match = re.match(r"^(\d{4})-\d{2}-\d{2}$", str(value).strip())
    if not match:
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}")
    return match.group(1)


def random_seed(args: dict[str, Any]) -> Any:
    return args.get(
        "random-seed",
        args.get("ldm-random-seed", args.get("seed", 42)),
    )


def standard_run_name(
    *,
    method: Any,
    model: Any,
    train_start: Any,
    test_end: Any,
    seed: Any,
) -> str:
    """Return ``method_llm_YYYY-YYYY_seedN`` for every official run."""
    method_name = _method_slug(method)
    model_name = "no-llm" if method_name in NO_LLM_METHODS else _model_slug(model)
    period = f"{_year(train_start, 'train-start')}-{_year(test_end, 'test-end')}"
    return f"{method_name}_{model_name}_{period}_seed{seed}"


def standard_output_dir(
    repo_root: Path,
    *,
    method: Any,
    model: Any,
    train_start: Any,
    test_end: Any,
    seed: Any,
) -> Path:
    method_name = _method_slug(method)
    return repo_root / "runs" / method_name / standard_run_name(
        method=method_name,
        model=model,
        train_start=train_start,
        test_end=test_end,
        seed=seed,
    )

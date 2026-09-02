"""Loading helpers for the pinned, official AlphaBench checkout."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .types import FactorCandidate


PINNED_COMMIT = "31bb94bbb7744177c51c9d011e07c31e3092b93e"


def resolve_alphabench_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    required = (
        path / "searcher" / "algo" / "cot.py",
        path / "factors" / "lib" / "alpha158" / "qlib_compile_product.json",
        path / "ffo" / "client" / "factor_eval_client.py",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"AlphaBench checkout is incomplete at {path}; missing: {', '.join(missing)}"
        )
    head_file = path / ".git" / "HEAD"
    if head_file.is_file():
        head = head_file.read_text(encoding="utf-8").strip()
        if len(head) == 40 and head != PINNED_COMMIT:
            raise RuntimeError(
                f"AlphaBench checkout is at {head}, expected pinned commit {PINNED_COMMIT}"
            )
    return path


def activate_alphabench(root: str | Path) -> Path:
    path = resolve_alphabench_root(root)
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


def load_cot_algo(root: str | Path):
    activate_alphabench(root)
    return importlib.import_module("searcher.algo.cot").CoTAlgo


def load_ea_algo(root: str | Path):
    activate_alphabench(root)
    return importlib.import_module("searcher.algo.ea").EAAlgo


def load_alpha158_seeds(
    root: str | Path,
    *,
    seed_group: str = "all",
) -> tuple[FactorCandidate, ...]:
    """Use AlphaBench's own Alpha158 loader and its vwap exclusion policy.

    ``seed_group`` mirrors what AlphaBench's own searching benchmark feeds each
    algorithm. It loads kbar, rolling and price, then hands **only kbar and
    price** to CoT (``benchmark_searching.py`` filters on
    ``kbar_names + price_names``), while EA gets the rolling factors. Pass
    ``"kbar_price"`` to reproduce the CoT benchmark's 13 seeds; ``"all"`` keeps
    every loaded factor.
    """

    path = activate_alphabench(root)
    module = importlib.import_module("factors.lib.alpha158")
    factor_dir = path / "factors" / "lib" / "alpha158"
    # AlphaBench stores these as cwd-relative globals. Make them checkout-relative
    # so the official loader behaves identically from any experiment directory.
    module.FACTOR_DIR = str(factor_dir)
    module.COMPILE_FILE = str(factor_dir / "qlib_compile_product.json")
    _definitions, compiled = module.load_factors_alpha158(
        exclude_var="vwap",
        collection=["kbar", "rolling", "price"],
    )
    group = str(seed_group).lower()
    if group not in {"all", "kbar_price"}:
        raise ValueError(f"seed_group must be 'all' or 'kbar_price', got {seed_group!r}")
    if group == "kbar_price":
        groups = module.load_factors_alpha158_names()
        keep = {factor["name"] for factor in groups["kbar"]}
        keep |= {factor["name"] for factor in groups["price"]}
        compiled = {name: record for name, record in compiled.items() if name in keep}
    seeds = []
    for name, record in compiled.items():
        expression = str(
            record.get("qlib_expression_default") or record.get("qlib_expression") or ""
        ).strip()
        if expression:
            seeds.append(
                FactorCandidate(
                    name=str(name),
                    expression=expression,
                    reason="Official AlphaBench Alpha158 seed.",
                    source="alphabench_alpha158",
                )
            )
    if not seeds:
        raise RuntimeError("official AlphaBench Alpha158 loader returned no usable seeds")
    return tuple(seeds)


def build_qlib_search_fn(
    root: str | Path,
    *,
    base_url: str,
    api_key_env: str,
    model: str,
) -> Callable[..., dict[str, Any]]:
    """Configure and return AlphaBench's official Qlib LLM generator."""

    activate_alphabench(root)
    if not os.environ.get(api_key_env):
        raise EnvironmentError(
            f"environment variable {api_key_env!r} is required for AlphaBench LLM calls"
        )
    llm_client = importlib.import_module("agent.llm_client")
    provider = "alpha_research_gateway"
    llm_client._CFG.setdefault("providers", {})[provider] = {
        "api_key": f"${{{api_key_env}}}",
        "base_url": str(base_url).rstrip("/"),
    }
    llm_client._CFG.setdefault("models", {})[model] = {"provider": provider}
    generator = importlib.import_module("agent.generator_qlib_search")
    return generator.call_qlib_search

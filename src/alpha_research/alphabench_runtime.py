"""Loading helpers for the pinned, official AlphaBench checkout."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .types import FactorCandidate


PINNED_COMMIT = "31bb94bbb7744177c51c9d011e07c31e3092b93e"
INTEGRATION_DIR = Path(__file__).resolve().parents[2] / "integrations" / "alphabench"
SOURCE_EXPORT_MARKER = ".upstream-source.json"


def _export_revision(root: Path) -> str:
    """Verify the checksum manifest supplied with the Git-free source export."""
    marker = root / SOURCE_EXPORT_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(
            "AlphaBench has no own .git metadata or verified source-export marker"
        )
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Invalid AlphaBench source-export marker") from exc
    if not isinstance(metadata, dict) or metadata.get("upstream_commit") != PINNED_COMMIT:
        raise RuntimeError("AlphaBench source export differs from the pinned upstream commit")
    files = metadata.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("AlphaBench source export has no file checksums")
    integration = json.loads((INTEGRATION_DIR / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "searcher/algo/cot.py",
        "factors/lib/alpha158/qlib_compile_product.json",
        "ffo/client/factor_eval_client.py",
        *integration["files"],
    }
    if not required.issubset(files):
        raise RuntimeError("AlphaBench source-export marker omits required upstream files")
    for name, expected in files.items():
        if not isinstance(name, str):
            raise RuntimeError("Invalid path in AlphaBench source-export marker")
        relative = PurePosixPath(name)
        if (
            not name or relative.is_absolute() or str(relative) != name
            or "\\" in name or ".." in relative.parts or ".git" in relative.parts
            or name == SOURCE_EXPORT_MARKER
        ):
            raise RuntimeError(f"Unsafe path in AlphaBench source-export marker: {name!r}")
        target = root
        for part in relative.parts:
            target /= part
            if target.is_symlink():
                raise RuntimeError(f"Symlinked AlphaBench source-export path: {name}")
        if not target.is_file() or not target.resolve().is_relative_to(root):
            raise RuntimeError(f"Missing or unsafe AlphaBench source-export file: {name}")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RuntimeError(f"Invalid AlphaBench source-export checksum: {name}")
        allowed = {expected}
        if name in integration["files"]:
            hashes = integration["files"][name]
            if expected != hashes["upstream_sha256"]:
                raise RuntimeError(f"AlphaBench source-export upstream checksum mismatch: {name}")
            allowed.add(hashes["patched_sha256"])
        if hashlib.sha256(target.read_bytes()).hexdigest() not in allowed:
            raise RuntimeError(f"AlphaBench source-export checksum mismatch: {name}")
    return PINNED_COMMIT


def checkout_revision(root: Path) -> str:
    """Resolve a Git checkout or verify its explicit, checksummed source export.

    Never discover a parent repository's revision for an exported directory.
    """
    root = Path(root).expanduser().resolve()
    if not (root / ".git").exists() and not (root / ".git").is_symlink():
        return _export_revision(root)
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def configure_verified_evaluator(
    root: Path, *, apply: bool = False, integration_dir: Path = INTEGRATION_DIR,
) -> str:
    """Check or explicitly apply the pinned FFO patch without overwriting edits.

    A pristine checkout and an exactly patched checkout are the only accepted
    states. All targets and the patch are checked before any file is changed.
    No imports of the upstream server, credentials or market data are needed.
    """
    root = root.expanduser().resolve()
    manifest = json.loads((integration_dir / "manifest.json").read_text())
    if checkout_revision(root) != manifest["upstream_commit"]:
        raise RuntimeError("AlphaBench revision differs from the pinned integration")
    patch = integration_dir / "verified-evaluation.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise RuntimeError("AlphaBench integration patch checksum mismatch")
    expected_files = {"ffo/routes/factors.py", "ffo/utils/utils.py"}
    if set(manifest["files"]) != expected_files:
        raise RuntimeError("Unexpected AlphaBench integration targets")
    states = set()
    for name, hashes in manifest["files"].items():
        target = root / name
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f"Missing or symlinked integration target: {name}")
        if not target.resolve().is_relative_to(root):
            raise RuntimeError(f"Integration target escapes checkout: {name}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == hashes["patched_sha256"]:
            states.add("patched")
        elif digest == hashes["upstream_sha256"]:
            states.add("upstream")
        else:
            raise RuntimeError(f"Local modifications in {name}; refusing to overwrite")
    if len(states) != 1:
        raise RuntimeError("Partially patched AlphaBench checkout; refusing to overwrite")
    state = states.pop()
    if state == "patched" or not apply:
        return state
    subprocess.run(["git", "-C", str(root), "apply", "--no-index", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(root), "apply", "--no-index", str(patch)], check=True)
    return configure_verified_evaluator(root, integration_dir=integration_dir)


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
    head = checkout_revision(path)
    if head != PINNED_COMMIT:
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

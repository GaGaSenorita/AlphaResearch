"""Configuration-first command line interface for AlphaResearch."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from alpha_research.naming import CANONICAL_METHODS, random_seed, standard_run_name
from alpha_research.arguments import parse_args as parse_runner_args
from alpha_research.runner import main as run_static_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"
SUPPORTED_METHODS = frozenset(CANONICAL_METHODS)
SUPPORTED_ENVIRONMENTS = {"static", "online"}


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.list:
        list_configs()
        return 0
    if not options.config:
        raise SystemExit("Provide a config path, or use --list.")

    path = resolve_config_path(options.config)
    config = load_config(path)
    for override in options.set:
        apply_override(config, override)

    failures: list[str] = []
    for experiment, experiment_path in expand_experiments(config, path):
        try:
            run_config(experiment, experiment_path, dry_run=options.dry_run)
        except Exception as exc:
            if not options.keep_going:
                raise
            failures.append(f"{experiment.get('name', experiment_path.stem)}: {exc}")
    if failures:
        print(json.dumps({"failures": failures}, indent=2), flush=True)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AlphaResearch experiments.")
    parser.add_argument("config", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    return parser.parse_args(argv)


def config_to_argv(config: dict[str, Any], config_path: Path) -> list[str]:
    """Expand one experiment using the same contract for search and reporting."""
    validate_config(config, config_path)
    method = str(config["method"])
    environment = str(config["environment"])
    args = dict(config.get("args") or {})
    expansion_context = build_expansion_context(config, args, config_path)

    argv = ["--environment", environment, "--method", method]
    for key, value in args.items():
        append_cli_arg(
            argv,
            str(key),
            expand_value(value, config_path, context=expansion_context),
        )
    return argv


def resolved_runner_args(
    config_path: Path, *, overrides: dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Resolve configuration without constructing an evaluator or starting a run."""
    config = load_config(config_path)
    config["args"] = {**(config.get("args") or {}), **(overrides or {})}
    return parse_runner_args(config_to_argv(config, config_path))


def run_config(config: dict[str, Any], config_path: Path, *, dry_run: bool = False) -> None:
    argv = config_to_argv(config, config_path)
    if dry_run:
        argv.append("--dry-run")
    print(f"[alpha-research] {config.get('name', config_path.stem)}")
    code = run_static_experiment(argv)
    if code:
        raise RuntimeError(f"experiment returned exit code {code}")


def validate_config(config: dict[str, Any], path: Path) -> None:
    method = str(config.get("method", "")).strip()
    environment = str(config.get("environment", "")).strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"{path}: method must be one of {sorted(SUPPORTED_METHODS)}")
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"{path}: environment must be one of {sorted(SUPPORTED_ENVIRONMENTS)}")
    if environment != "static":
        raise ValueError(
            f"{path}: {method} is currently enabled only for the static environment"
        )
    if not isinstance(config.get("args", {}), dict):
        raise ValueError(f"{path}: args must be a mapping")
    name = str(config.get("name", "")).strip()
    if name and name != path.stem:
        raise ValueError(
            f"{path}: config name {name!r} must match filename stem {path.stem!r}"
        )


def append_cli_arg(argv: list[str], key: str, value: Any) -> None:
    if value is None or value is False:
        return
    flag = key if key.startswith("--") else "--" + key.replace("_", "-")
    if value is True:
        argv.append(flag)
    elif isinstance(value, list):
        for item in value:
            argv.extend([flag, str(item)])
    else:
        argv.extend([flag, str(value)])


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def build_expansion_context(
    config: dict[str, Any],
    args: dict[str, Any],
    config_path: Path,
) -> dict[str, str]:
    """Build stable path placeholders after runtime overrides are applied."""
    method = _slug(config.get("method", "unknown"))
    model = str(args.get("llm-model", "no-llm"))
    seed = random_seed(args)
    run_name = standard_run_name(
        method=method,
        model=model,
        train_start=args.get("train-start", "unknown"),
        test_end=args.get("test-end", "unknown"),
        seed=seed,
    )
    return {
        "repo_root": str(REPO_ROOT),
        "config_dir": str(config_path.parent),
        "method": method,
        "model": _slug(model),
        "method_model": f"{method}_{model}",
        "seed": str(seed),
        "run_name": run_name,
        "period": run_name.rsplit("_seed", 1)[0].rsplit("_", 1)[-1],
        "train_start": str(args.get("train-start", "unknown")),
        "train_end": str(args.get("train-end", "unknown")),
    }


def expand_value(
    value: Any,
    config_path: Path,
    *,
    context: dict[str, str] | None = None,
) -> Any:
    if not isinstance(value, str):
        return value
    placeholders = context or {
        "repo_root": str(REPO_ROOT),
        "config_dir": str(config_path.parent),
    }
    return os.path.expandvars(value.format_map(placeholders))


def resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("YAML configs require PyYAML.") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def expand_experiments(
    config: dict[str, Any],
    config_path: Path,
) -> list[tuple[dict[str, Any], Path]]:
    entries = config.get("experiments")
    if entries is None:
        return [(config, config_path)]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Suite configs require a non-empty experiments list.")
    expanded: list[tuple[dict[str, Any], Path]] = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError("Suite entries must be relative config paths.")
        child_path = (config_path.parent / entry).resolve()
        expanded.append((load_config(child_path), child_path))
    return expanded


def apply_override(config: dict[str, Any], raw: str) -> None:
    if "=" not in raw:
        raise ValueError(f"Override must look like PATH=VALUE, got {raw!r}")
    path_text, value_text = raw.split("=", 1)
    keys = [item for item in path_text.split(".") if item]
    if not keys:
        raise ValueError("Override path cannot be empty")
    try:
        value = json.loads(value_text)
    except json.JSONDecodeError:
        value = value_text
    cursor = config
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {raw!r}; {key!r} is not a mapping")
        cursor = child
    cursor[keys[-1]] = value


def list_configs() -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        data = load_config(path)
        rows.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "name": str(data.get("name", path.stem)),
            "method": str(data.get("method", "suite" if "experiments" in data else "")),
            "environment": str(data.get("environment", "")),
            "mode": str(data.get("mode", "")),
            "status": str(data.get("status", "active")),
        })
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())

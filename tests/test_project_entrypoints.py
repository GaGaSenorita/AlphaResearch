"""Check configuration/CLI contracts without credentials, data or paid calls."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from alpha_research import cli, runner
from alpha_research.arguments import parse_args


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config", sorted((ROOT / "configs").rglob("*.yaml")))
def test_all_saved_configs_resolve_without_starting_experiments(config):
    options = cli.resolved_runner_args(config)
    assert options.method in cli.SUPPORTED_METHODS
    assert options.environment == "static"
    assert options.train_start.startswith("2016")
    assert options.train_end.startswith("2020")
    assert runner.parse_args is parse_args


def test_reporting_and_search_expand_the_same_configuration():
    config = ROOT / "configs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml"
    overrides = {"ldm-random-seed": 123, "search-rounds": 100, "continuous-checkpoint-round": list(range(0, 101, 5))}
    payload = cli.load_config(config)
    payload["args"].update(overrides)
    assert vars(cli.resolved_runner_args(config, overrides=overrides)) == vars(
        parse_args(cli.config_to_argv(payload, config))
    )


@pytest.mark.parametrize("name", [
    "run_experiment.py", "setup_alphabench.py", "backfill_ldm_checkpoint_reports.py",
    "backfill_single_factor_validation.py", "audit_single_factor_test.py",
    "plot_ldm_financial_mining.py", "plot_ldm_three_seed_comparison.py",
    "build_single_factor_library.py", "run_stage2_seed_matrix.py",
])
def test_script_help_is_importable_without_starting_work(name):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_backfill_uses_shared_config_resolution():
    spec = importlib.util.spec_from_file_location("checkpoint_backfill", ROOT / "scripts/backfill_ldm_checkpoint_reports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = ROOT / module.DEFAULT_CONFIG
    options = module.resolved_runner_args(config, seed=42, target_round=100, checkpoints=[0, 5, 100])
    assert options.continuous_checkpoint_round == [0, 5, 100]


def test_manual_projection_option_is_no_longer_accepted():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/plot_ldm_financial_mining.py"),
         "--data-dir", ".", "--output-prefix", "unused", "--illustrative-r100", "0.0411"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr

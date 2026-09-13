from pathlib import Path

import pytest
import yaml

from alpha_research.naming import CANONICAL_METHODS, standard_run_name


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_llm_run_name_preserves_readable_model_name() -> None:
    assert standard_run_name(
        method="ldm_standard",
        model="DeepSeek-V4-Pro",
        train_start="2016-01-01",
        test_end="2025-12-26",
        seed=42,
    ) == "ldm_standard_deepseek-v4-pro_2016-2025_seed42"


def test_non_llm_baseline_uses_explicit_no_llm_component() -> None:
    assert standard_run_name(
        method="baseline_alpha158",
        model="ignored",
        train_start="2016-01-01",
        test_end="2025-12-26",
        seed=123,
    ) == "baseline_alpha158_no-llm_2016-2025_seed123"


def test_two_objective_method_uses_the_same_canonical_run_name_contract() -> None:
    assert standard_run_name(
        method="ldm_rankic_rankicir_ehvi",
        model="DeepSeek-V4-Pro",
        train_start="2016-01-01",
        test_end="2025-12-26",
        seed=456,
    ) == "ldm_rankic_rankicir_ehvi_deepseek-v4-pro_2016-2025_seed456"


def test_unknown_method_cannot_create_a_run_name() -> None:
    with pytest.raises(ValueError, match="Unknown canonical method"):
        standard_run_name(
            method="harness_ldm",
            model="deepseek-v4-pro",
            train_start="2016-01-01",
            test_end="2025-12-26",
            seed=42,
        )


def test_active_configs_use_canonical_names_and_safe_output_paths() -> None:
    for path in sorted((REPO_ROOT / "configs").glob("*/*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["name"] == path.stem
        assert config["method"] in CANONICAL_METHODS
        output = config.get("args", {}).get("out-dir")
        if config["method"] in {"ldm_split_robust_reward", "ldm_rankic_worst_qehvi"}:
            assert output == "{repo_root}/../AlphaResearch_local_results/stage2/{method}/{run_name}"
        elif path.parent.name == "smoke":
            assert output == "{repo_root}/runtime/smoke/{run_name}"
        else:
            assert output is None

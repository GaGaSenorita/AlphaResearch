"""Annual boundaries, output provenance and configuration must agree."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_research import cli, runner
from alpha_research.methods.split_robust import SplitSettings
from alpha_research.methods.split_robust.contract import validate_annual_output, validate_train_years
from alpha_research.types import Period


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ldm_split_robust_reward", "ldm_rankic_worst_qehvi")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules when loading a script.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", METHODS)
def test_stage2_config_defaults_to_local_results_and_tracks_seed_override(method):
    config = ROOT / "configs" / method / f"{method}_neolink-deepseek-v4-flash_2016-2025.yaml"
    args = cli.resolved_runner_args(config, overrides={"ldm-random-seed": 123})
    path = Path(args.out_dir).resolve()
    assert path.parent == ROOT.parent / "AlphaResearch_local_results" / "stage2" / method
    assert path.name.endswith("_seed123")
    assert runner.public_metadata(args)["ldm"]["split_reward"] == SplitSettings().to_dict()


@pytest.mark.parametrize("name", ["protocol.json", "summary.json", "mobo_state.json", "pareto_archive.json"])
def test_legacy_quarterly_contract_cannot_be_overwritten(tmp_path, name):
    split = {"segmentation": "calendar_quarter"}
    if name in {"protocol.json", "summary.json"}:
        payload = {"metadata": {"ldm": {"split_reward": split}}}
        if name == "summary.json":
            payload = {"protocol": payload}
    else:
        payload = {"split_reward": split}
    path = tmp_path / name
    original = json.dumps(payload)
    path.write_text(original)
    with pytest.raises(ValueError, match="not the current calendar-year"):
        validate_annual_output(tmp_path)
    assert path.read_text() == original


def test_legacy_history_is_rejected_even_without_protocol(tmp_path):
    path = tmp_path / "ldm_history.csv"
    path.write_text("score,worst_quarter,rankic_2016Q1\n0.1,2016Q1,0.1\n")
    with pytest.raises(ValueError, match="saved history is not annual"):
        validate_annual_output(tmp_path)


def test_saved_annual_contract_is_recognized(tmp_path):
    payload = {"metadata": {"ldm": {"split_reward": SplitSettings().to_dict()}}}
    (tmp_path / "protocol.json").write_text(json.dumps(payload))
    validate_annual_output(tmp_path)


@pytest.mark.parametrize("start,end", [("2017-01-01", "2020-12-29"), ("2016-06-01", "2020-12-29"), ("2016-01-01", "2020-06-30")])
def test_train_split_requires_all_five_calendar_years(start, end):
    with pytest.raises(ValueError, match="January 2016 through December 2020"):
        validate_train_years(Period.from_strings(start, end), SplitSettings())


def test_seed_launcher_does_not_skip_quarterly_run_as_annual(tmp_path, monkeypatch):
    launcher = _script("run_stage2_seed_matrix")
    monkeypatch.setattr(launcher, "RESULTS_ROOT", tmp_path)
    job = launcher.Job(METHODS[1], Path("unused"), 42)
    job.run_dir.mkdir(parents=True)
    protocol = {"search_rounds": 38, "metadata": {"ldm": {"split_reward": {"segmentation": "calendar_quarter"}}}}
    path = job.run_dir / "summary.json"
    path.write_text(json.dumps({"status": "completed", "protocol": protocol}))
    assert not launcher.completed(job)
    protocol["metadata"]["ldm"]["split_reward"] = SplitSettings().to_dict()
    path.write_text(json.dumps({"status": "completed", "protocol": protocol}))
    assert launcher.completed(job)


def test_annual_reanalysis_matches_validation_mean_for_one_year():
    reanalysis = _script("reanalyse_validation_worst_year_topk")
    period = {"start": "2021-01-01", "end": "2021-12-29"}
    evaluations = []
    for index, values in enumerate(([0.1, 0.4], [0.3, 0.3])):
        evaluations.append({
            "candidate": {"expression": f"Mean($close,{index + 2})", "name": str(index)},
            "validation": {"success": True, "period": period,
                           "metrics": {"daily_count": 2, "rank_ic": sum(values) / 2},
                           "daily_metrics": [
                               {"date": f"2021-{month:02d}-10", "rank_ic": value, "ic": value}
                               for month, value in zip((1, 12), values)]},
        })
    mean = reanalysis.select_topk(evaluations, period, 1, 0.8, "mean")
    annual = reanalysis.select_topk(evaluations, period, 1, 0.8, "worst-year")
    assert annual == mean
    assert annual["selected"][0]["candidate"]["name"] == "1"
    assert reanalysis.year_means([
        {"date": "2020-12-31", "rank_ic": -0.2},
        {"date": "2021-01-01", "rank_ic": 0.1},
    ]) == {"2020": -0.2, "2021": 0.1}

from datetime import date
import math
import json
from pathlib import Path

import pytest
import requests

from alpha_research.evaluator import MockFactorEvaluator
from alpha_research.methods.split_robust import split_score
from alpha_research.types import Period


def test_mock_dates_cover_five_years_deterministically():
    evaluator = MockFactorEvaluator()
    period = Period.from_strings("2016-01-01", "2020-12-29")
    result = evaluator.evaluate("$close", period)
    assert result == evaluator.evaluate("$close", period)
    dates = [date.fromisoformat(row["date"]) for row in result.daily_metrics]
    assert dates == sorted(set(dates))
    assert all(period.start <= day <= period.end and day.weekday() < 5 for day in dates)
    score = split_score(result)
    assert score.rejected is None
    assert len(score.yearly_rankic) == 5
    assert math.isfinite(score.score)
    assert score.score == min(score.yearly_rankic)


def test_mock_short_period_does_not_fabricate_dates_outside_the_request():
    period = Period.from_strings("2020-12-25", "2020-12-28")
    rows = MockFactorEvaluator().evaluate("$close", period).daily_metrics
    assert [row["date"] for row in rows] == ["2020-12-25", "2020-12-28"]


@pytest.mark.parametrize("method", ["ldm_split_robust_reward", "ldm_rankic_worst_qehvi"])
def test_stage2_mock_loop_and_formal_top30_selection_complete(tmp_path, monkeypatch, method):
    from alpha_research import cli
    from alpha_research.naming import standard_run_name

    def no_network(*_args, **_kwargs):
        pytest.fail("A mock smoke test must never issue an HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / method / f"{method}_neolink-deepseek-v4-flash_2016-2025.yaml"
    config = cli.load_config(path)
    args = config["args"]
    args.update({"mock": True, "search-rounds": 2, "gp-train-iters": 2})
    destination = tmp_path / standard_run_name(
        method=method, model=args["llm-model"], train_start=args["train-start"],
        test_end=args["test-end"], seed=args["ldm-random-seed"],
    )
    args["out-dir"] = str(destination)
    cli.run_config(config, path)
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["protocol"]["factor_budget"] == 30
    expected_partition = "calendar_year"
    assert summary["protocol"]["metadata"]["ldm"]["split_reward"]["segmentation"] == expected_partition

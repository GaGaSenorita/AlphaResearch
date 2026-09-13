import math
from datetime import date
from types import SimpleNamespace

import pytest

from alpha_research.methods.baseline_alpha158 import Alpha158Baseline
from alpha_research.methods.ldm.generator import MockCandidateGenerator
from alpha_research.methods.ldm_cold_start import AlphaLDMColdStart
from alpha_research.methods.ldm_continuous_discovery import ContinuousDiscoveryLDM
from alpha_research.methods.ldm_minimal_prompt_harness import MinimalHarnessLDM
from alpha_research.methods.ldm_prompt_harness import HarnessLDM
from alpha_research.methods.ldm_rankic_rankicir_ehvi import RankICRankICIREHVILDM
from alpha_research.methods.ldm_rankic_turnover_ehvi import RankICTurnoverEHVILDM
from alpha_research.methods.ldm_rankic_worst_qehvi import RankICWorstQEHVIAlphaLDM
from alpha_research.methods.ldm_single_factor import SingleFactorLDM
from alpha_research.methods.ldm_split_robust_reward import SplitRobustLDM
from alpha_research.methods.split_robust import SplitSettings, split_score
from alpha_research.types import EvaluationMetrics, EvaluationResult, Period


def test_recovered_method_names_are_canonical() -> None:
    methods = {
        Alpha158Baseline.name,
        AlphaLDMColdStart.name,
        ContinuousDiscoveryLDM.name,
        MinimalHarnessLDM.name,
        HarnessLDM.name,
        RankICRankICIREHVILDM.name,
        RankICTurnoverEHVILDM.name,
        RankICWorstQEHVIAlphaLDM.name,
        SingleFactorLDM.name,
        SplitRobustLDM.name,
    }
    assert methods == {
        "baseline_alpha158",
        "ldm_cold_start",
        "ldm_continuous_discovery",
        "ldm_minimal_prompt_harness",
        "ldm_prompt_harness",
        "ldm_rankic_rankicir_ehvi",
        "ldm_rankic_turnover_ehvi",
        "ldm_rankic_worst_qehvi",
        "ldm_single_factor",
        "ldm_split_robust_reward",
    }


def test_mock_generator_resume_advances_proposal_sequence() -> None:
    generator = MockCandidateGenerator(candidates_per_round=2)
    first = generator.generate("unused")
    generator.restore_state(completed_round=2, llm_calls_total=4)
    resumed = generator.generate("unused")
    assert first[0]["name"].startswith("MOCK_LDM_R1")
    assert resumed[0]["name"].startswith("MOCK_LDM_R3")
    assert generator.calls == 2


def _five_year_result(
    overrides: dict[str, tuple[float, float]] | None = None,
) -> EvaluationResult:
    overrides = overrides or {}
    rows = []
    for year in range(2016, 2021):
        values = overrides.get(str(year), (0.20, 0.20))
        rows.extend(
            {"date": f"{year}-01-{day:02d}", "rank_ic": value}
            for day, value in enumerate(values, start=10)
        )
    return EvaluationResult(
        success=True,
        expression="$close",
        period=Period(date(2016, 1, 1), date(2020, 12, 29)),
        metrics=EvaluationMetrics(rank_ic=0.123),
        daily_metrics=tuple(rows),
    )


def test_split_reward_is_minimum_of_five_year_means_not_daily_minimum() -> None:
    result = _five_year_result({"2018": (-0.80, 1.00), "2019": (0.05, 0.05)})
    score = split_score(result, SplitSettings())
    assert len(score.years) == 5
    assert score.train_rankic == pytest.approx(0.123)
    assert dict(zip(score.years, score.yearly_rankic))["2018"] == pytest.approx(0.10)
    assert score.worst_rankic == pytest.approx(0.05)
    assert score.score == pytest.approx(0.05)
    assert score.worst_year == "2019"


def test_split_reward_keeps_signed_year_means_without_per_year_flip() -> None:
    score = split_score(_five_year_result({"2017": (-0.04, -0.02)}), SplitSettings())
    assert score.score == pytest.approx(-0.03)
    assert score.worst_year == "2017"


def test_split_reward_rejects_an_incomplete_five_year_series() -> None:
    result = _five_year_result()
    incomplete = SimpleNamespace(
        daily_metrics=tuple(
            row for row in result.daily_metrics if not str(row["date"]).startswith("2020")
        ),
        metrics=result.metrics,
    )
    score = split_score(incomplete, SplitSettings())
    assert score.rejected == "missing finite daily rank_ic for: 2020"
    assert math.isinf(score.score) and score.score < 0


def test_split_history_keeps_one_scalar_observation_and_all_years(tmp_path) -> None:
    method = SplitRobustLDM(
        evaluator=SimpleNamespace(),
        profiler=SimpleNamespace(),
        generator=SimpleNamespace(),
        output_dir=tmp_path,
        objective="rank_ic",
        search_objective="worst_rankic",
    )
    history = method._new_history(dim=12, schema_version="test")
    result = _five_year_result({"2019": (0.05, 0.05)})
    score = method._score(result)
    assert score == pytest.approx(0.05)
    method._record_observation(
        history,
        feature=[0.0] * 12,
        score=score,
        result=result,
        canonical="$close",
        expression="$close",
        round_id=0,
    )
    history.save(tmp_path / "ldm_history.csv")

    assert history.n == 1
    stored = (tmp_path / "ldm_history.csv").read_text(encoding="utf-8")
    assert "train_rankic,worst_rankic,worst_year,yearly_rankic" in stored
    assert "rankic_2016" in stored
    assert "rankic_2020" in stored
    assert "days_2016" in stored
    assert "quarter" not in stored
    method._write_split_history()
    audit = (tmp_path / "split_reward_history.csv").read_text(encoding="utf-8")
    assert "worst_year" in audit
    assert "rankic_2016,rankic_2017,rankic_2018,rankic_2019,rankic_2020" in audit


@pytest.mark.parametrize(
    "method,segmentation",
    [
        ("ldm_split_robust_reward", "calendar_year"),
        ("ldm_rankic_worst_qehvi", "calendar_year"),
    ],
)
def test_runner_routes_both_stage2_methods_to_annual_train_partition(method, segmentation):
    from alpha_research.runner import split_settings

    assert split_settings(SimpleNamespace(method=method)).to_dict()["segmentation"] == segmentation

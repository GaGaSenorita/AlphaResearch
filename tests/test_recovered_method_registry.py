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
from alpha_research.methods.ldm_split_robust_reward import SplitRobustLDM, SplitSettings
from alpha_research.methods.split_robust import split_score
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


def _twenty_quarter_result(
    overrides: dict[str, tuple[float, float]] | None = None,
) -> EvaluationResult:
    overrides = overrides or {}
    rows = []
    for year in range(2016, 2021):
        for quarter, month in enumerate((1, 4, 7, 10), start=1):
            label = f"{year}Q{quarter}"
            values = overrides.get(label, (0.20, 0.20))
            rows.extend(
                {"date": f"{year}-{month:02d}-{day:02d}", "rank_ic": value}
                for day, value in enumerate(values, start=10)
            )
    return EvaluationResult(
        success=True,
        expression="$close",
        period=Period(date(2016, 1, 1), date(2020, 12, 29)),
        metrics=EvaluationMetrics(rank_ic=0.123),
        daily_metrics=tuple(rows),
    )


def test_split_reward_is_minimum_of_twenty_quarter_means_not_daily_minimum() -> None:
    result = _twenty_quarter_result(
        {"2018Q2": (-0.80, 1.00), "2019Q3": (0.05, 0.05)}
    )
    score = split_score(result, SplitSettings())

    assert len(score.quarters) == 20
    assert score.train_rankic == pytest.approx(0.123)
    assert dict(zip(score.quarters, score.quarterly_rankic))["2018Q2"] == pytest.approx(0.10)
    assert score.worst_rankic == pytest.approx(0.05)
    assert score.score == pytest.approx(0.05)
    assert score.worst_quarter == "2019Q3"


def test_split_reward_keeps_signed_quarter_means_without_per_quarter_flip() -> None:
    score = split_score(
        _twenty_quarter_result({"2017Q4": (-0.04, -0.02)}), SplitSettings()
    )
    assert score.score == pytest.approx(-0.03)
    assert score.worst_quarter == "2017Q4"


def test_split_reward_rejects_an_incomplete_twenty_quarter_series() -> None:
    result = _twenty_quarter_result()
    incomplete = SimpleNamespace(
        daily_metrics=tuple(
            row for row in result.daily_metrics if not str(row["date"]).startswith("2020-10")
        ),
        metrics=result.metrics,
    )
    score = split_score(incomplete, SplitSettings())
    assert score.rejected == "missing finite daily rank_ic for: 2020Q4"
    assert math.isinf(score.score) and score.score < 0


def test_split_history_keeps_one_scalar_observation_and_all_quarters(tmp_path) -> None:
    method = SplitRobustLDM(
        evaluator=SimpleNamespace(),
        profiler=SimpleNamespace(),
        generator=SimpleNamespace(),
        output_dir=tmp_path,
        objective="rank_ic",
        search_objective="worst_rankic",
    )
    history = method._new_history(dim=12, schema_version="test")
    result = _twenty_quarter_result({"2019Q3": (0.05, 0.05)})
    score = method._score(result)
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
    assert "train_rankic,worst_rankic,worst_quarter,quarterly_rankic" in stored
    assert "rankic_2016Q1" in stored
    assert "rankic_2020Q4" in stored

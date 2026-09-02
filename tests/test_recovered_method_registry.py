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
from alpha_research.methods.ldm_single_factor import SingleFactorLDM
from alpha_research.methods.ldm_split_robust_reward import SplitRobustLDM, SplitSettings
from alpha_research.methods.split_robust import split_score


def test_recovered_method_names_are_canonical() -> None:
    methods = {
        Alpha158Baseline.name,
        AlphaLDMColdStart.name,
        ContinuousDiscoveryLDM.name,
        MinimalHarnessLDM.name,
        HarnessLDM.name,
        RankICRankICIREHVILDM.name,
        RankICTurnoverEHVILDM.name,
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


def test_split_reward_uses_train_daily_series_only() -> None:
    rows = []
    for year, value in ((2019, 0.02), (2020, 0.04)):
        rows.extend(
            {"date": f"{year}-06-{(index % 28) + 1:02d}", "rank_ic": value}
            for index in range(12)
        )
    result = SimpleNamespace(daily_metrics=tuple(rows))
    score = split_score(
        result,
        SplitSettings(embargo_days=0, min_segment_days=10, min_segments=2),
    )
    assert score.years == (2019, 2020)
    assert score.mean == pytest.approx(0.03)
    assert score.score < score.mean

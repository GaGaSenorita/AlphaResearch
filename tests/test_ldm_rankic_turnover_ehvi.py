from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from alpha_research.methods.ldm.multi_objective import (
    expected_hypervolume_improvement_2d,
    pareto_front,
)
from alpha_research.methods.ldm_rankic_turnover_ehvi import (
    RankICTurnoverEHVILDM,
)
from alpha_research.methods.ldm_rankic_turnover_ehvi.method import (
    RankICTurnoverHistory,
)
from alpha_research.naming import standard_run_name
from alpha_research.types import EvaluationMetrics, EvaluationResult, Period


def result(*, rank_ic: float | None, turnover: float | None) -> EvaluationResult:
    return EvaluationResult(
        success=True,
        expression="Mean($close, 5)",
        period=Period(date(2016, 1, 1), date(2020, 12, 29)),
        metrics=EvaluationMetrics(rank_ic=rank_ic, turnover=turnover),
    )


def method(tmp_path, **overrides) -> RankICTurnoverEHVILDM:
    kwargs = {
        "evaluator": None,
        "profiler": None,
        "generator": None,
        "output_dir": tmp_path,
        "objective": "rank_ic",
        "acquisition": "ehvi",
        "gp_train_iters": 0,
    }
    kwargs.update(overrides)
    return RankICTurnoverEHVILDM(**kwargs)


def test_pareto_front_keeps_rankic_turnover_tradeoffs() -> None:
    points = [(0.04, 0.40), (0.03, 0.30), (0.02, 0.50)]
    assert pareto_front(points, maximize=(True, False)) == points[:2]


def test_ehvi_maximises_rankic_and_minimises_turnover() -> None:
    values = expected_hypervolume_improvement_2d(
        means=[np.array([0.05, 0.01]), np.array([0.30, 0.80])],
        standard_deviations=[np.zeros(2), np.zeros(2)],
        pareto_points=[(0.04, 0.40)],
        reference=(-0.10, 1.50),
        maximize=(True, False),
        n_samples=16,
        rng=np.random.default_rng(7),
    )
    assert values.shape == (2,)
    assert values[0] > 0.0
    assert values[1] == pytest.approx(0.0, abs=1e-12)


def test_history_persists_both_objectives(tmp_path) -> None:
    history = RankICTurnoverHistory(dim=2, schema_version="schema")
    history.add_objectives(
        np.array([1.0, 2.0]),
        rank_ic=0.04,
        turnover=0.30,
        canonical="Mean($close,5)",
        expression="Mean($close, 5)",
        round_id=1,
    )
    path = tmp_path / "ldm_history.csv"
    history.save(path)
    frame = pd.read_csv(path)
    assert frame.loc[0, "score"] == pytest.approx(0.04)
    assert frame.loc[0, "rank_ic"] == pytest.approx(0.04)
    assert frame.loc[0, "turnover"] == pytest.approx(0.30)


def test_method_rejects_observation_without_either_objective(tmp_path) -> None:
    search = method(tmp_path)
    assert search._score(result(rank_ic=0.04, turnover=0.30)) == pytest.approx(0.04)
    assert search._score(result(rank_ic=0.04, turnover=None)) == float("-inf")


def test_method_contract_rejects_wrong_acquisition_or_readout(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires acquisition='ehvi'"):
        method(tmp_path, acquisition="ucb")
    with pytest.raises(ValueError, match="validation/test objective"):
        method(tmp_path, objective="rank_icir")


def test_canonical_result_name_for_rankic_turnover_method() -> None:
    assert standard_run_name(
        method="ldm_rankic_turnover_ehvi",
        model="deepseek-v4-pro",
        train_start="2016-01-01",
        test_end="2025-12-26",
        seed=42,
    ) == "ldm_rankic_turnover_ehvi_deepseek-v4-pro_2016-2025_seed42"

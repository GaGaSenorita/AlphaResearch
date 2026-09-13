from __future__ import annotations

from datetime import date
import json
import re

import numpy as np
import pytest

from alpha_research.methods.ldm.multi_objective import (
    pareto_mask,
    q_expected_hypervolume_improvement_2d,
)
from alpha_research.methods.ldm_rankic_worst_qehvi import (
    FixedObjectiveNormalizer,
    RankICWorstHistory,
    RankICWorstQEHVIAlphaLDM,
    RankICWorstSurrogate,
    SplitSettings,
)
from alpha_research.methods.split_robust import SplitScore, split_score
from alpha_research.naming import standard_run_name
from alpha_research.types import EvaluationMetrics, EvaluationResult, FactorCandidate, Period


def _outcome(train_rankic: float, worst_rankic: float, index: int = 0) -> SplitScore:
    years = SplitSettings().years
    values = [worst_rankic + 0.01] * len(years)
    values[index % len(years)] = worst_rankic
    return SplitScore(
        score=worst_rankic,
        years=years,
        yearly_rankic=tuple(values),
        year_days=(50,) * len(years),
        train_rankic=train_rankic,
        worst_rankic=worst_rankic,
        worst_year=years[index % len(years)],
        usable_days=250,
        settings=SplitSettings().to_dict(),
    )


def _history() -> RankICWorstHistory:
    rng = np.random.default_rng(5)
    history = RankICWorstHistory(12, "test-schema", SplitSettings().years)
    for index in range(42):
        history.add_objectives(
            rng.normal(size=12),
            canonical=f"factor_{index}",
            expression=f"Mean($close,{index + 2})",
            round_id=0,
            outcome=_outcome(
                train_rankic=0.015 + index * 0.0007,
                worst_rankic=-0.035 + index * 0.0006,
                index=index,
            ),
        )
    return history


def test_pareto_mask_keeps_tradeoffs_and_rejects_dominated_points() -> None:
    points = [(0.04, -0.02), (0.03, -0.01), (0.02, -0.03)]
    assert pareto_mask(points, maximize=(True, True)).tolist() == [True, True, False]


def test_discrete_qehvi_scores_joint_batches_not_individual_candidates() -> None:
    batches, values = q_expected_hypervolume_improvement_2d(
        means=[np.array([1.0, 0.0, 0.6]), np.array([0.0, 1.0, 0.6])],
        covariance_matrices=[np.zeros((3, 3)), np.zeros((3, 3))],
        pareto_points=[(0.0, 0.0)],
        reference=(-1.0, -1.0),
        maximize=(True, True),
        batch_size=2,
        n_samples=8,
        rng=np.random.default_rng(9),
    )
    assert batches == [(0, 1), (0, 2), (1, 2)]
    assert batches[int(np.argmax(values))] == (0, 1)
    assert values[0] > values[1]
    assert values[0] > values[2]


def test_normalization_is_fitted_once_on_initial_42() -> None:
    initial = np.column_stack(
        [np.linspace(0.01, 0.05, 42), np.linspace(-0.04, 0.00, 42)]
    )
    normalizer = FixedObjectiveNormalizer.fit(initial)
    before = normalizer.to_dict()
    transformed = normalizer.transform(np.vstack([initial, [0.20, 0.10]]))
    after = normalizer.to_dict()
    assert transformed.shape == (43, 2)
    assert before == after
    assert before["observation_count"] == 42


def test_two_independent_gp_joint_posteriors_have_expected_shapes() -> None:
    history = _history()
    surrogate = RankICWorstSurrogate(
        feature_dim=12,
        history=history,
        gp_kwargs={
            "min_fit_data": 20,
            "lengthscale": None,
            "noise": 0.05,
            "scale": 0.25,
            "train_iters": 0,
            "lr": 0.05,
        },
        reference_margin=0.1,
    )
    surrogate.fit(history.features(), history.scores())
    candidate_features = np.random.default_rng(8).normal(size=(8, 12))
    means, covariances = surrogate.predict_objectives_joint(candidate_features)

    assert len(surrogate.models) == 2
    assert [values.shape for values in means] == [(8,), (8,)]
    assert [values.shape for values in covariances] == [(8, 8), (8, 8)]
    assert surrogate.normalizer is not None
    assert surrogate.normalizer.observation_count == 42
    assert surrogate.reference_point is not None


def test_method_selects_three_from_eight_with_true_batch_qehvi(tmp_path) -> None:
    history = _history()
    method = RankICWorstQEHVIAlphaLDM(
        evaluator=None,
        profiler=None,
        generator=None,
        output_dir=tmp_path,
        objective="rank_ic",
        acquisition="qehvi",
        evaluate_per_round=3,
        qehvi_n_samples=8,
        gp_train_iters=0,
        random_seed=42,
    )
    surrogate = method._new_surrogate(feature_dim=12, history=history)
    decision = method._candidate_batch_acquisition(
        surrogate=surrogate,
        history=history,
        features=np.random.default_rng(18).normal(size=(8, 12)),
        batch_size=3,
        rng=np.random.default_rng(0),  # ignored; method owns reproducible MC state
    )

    assert len(decision["selected_indices"]) == 3
    assert len(set(decision["selected_indices"])) == 3
    assert decision["event"]["batch_count"] == 56
    assert decision["event"]["batch_selection"].startswith("exact_enumeration")
    assert decision["event"]["acquisition"] == "monte_carlo_qehvi"
    assert np.asarray(decision["mean"]).shape == (8,)
    assert np.asarray(decision["diagnostics"]["posterior_mean_worst_rankic"]).shape == (8,)


def test_one_full_train_result_produces_two_outputs_and_five_years() -> None:
    rows = []
    values = []
    for year in range(2016, 2021):
        year_values = [-0.03, -0.018] if year == 2018 else [0.01] * (year - 2014)
        values.extend(year_values)
        rows.extend(
            {"date": f"{year}-01-{day:02d}", "rank_ic": value}
            for day, value in enumerate(year_values, start=10)
        )
    result = EvaluationResult(
        success=True,
        expression="Div(Delta($close,5),Add(Std($close,20),1e-12))",
        period=Period(date(2016, 1, 1), date(2020, 12, 29)),
        metrics=EvaluationMetrics(rank_ic=float(np.mean(values))),
        daily_metrics=tuple(rows),
    )
    outcome = split_score(result, SplitSettings())

    assert outcome.train_rankic == pytest.approx(np.mean(values))
    assert len(outcome.yearly_rankic) == 5
    assert outcome.worst_rankic == pytest.approx(-0.024)
    assert outcome.worst_year == "2018"

    method = RankICWorstQEHVIAlphaLDM(
        evaluator=None,
        profiler=None,
        generator=None,
        output_dir=".",
        objective="rank_ic",
        acquisition="qehvi",
        gp_train_iters=0,
    )
    mismatched_metric = EvaluationResult(
        success=result.success,
        expression=result.expression,
        period=result.period,
        metrics=EvaluationMetrics(rank_ic=0.999),
        daily_metrics=result.daily_metrics,
    )
    method_outcome = method._split_outcome(mismatched_metric)
    assert method_outcome.train_rankic == pytest.approx(np.mean(values))


def test_method_contract_and_canonical_run_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires acquisition='qehvi'"):
        RankICWorstQEHVIAlphaLDM(
            evaluator=None,
            profiler=None,
            generator=None,
            output_dir=tmp_path,
            acquisition="ehvi",
        )
    assert standard_run_name(
        method="ldm_rankic_worst_qehvi",
        model="neolink/technologies/deepseek-v4-flash",
        train_start="2016-01-01",
        test_end="2025-12-26",
        seed=42,
    ) == (
        "ldm_rankic_worst_qehvi_neolink-technologies-deepseek-v4-flash_"
        "2016-2025_seed42"
    )


class _Schema:
    names = tuple(f"feature_{index}" for index in range(12))
    version = "qehvi-test"

    def to_dict(self):
        return {"names": list(self.names), "version": self.version}


class _Profiler:
    schema = _Schema()
    reference_expressions = ()

    def profile(self, expression):
        number = float(re.findall(r"\d+", expression)[-1])
        return np.asarray(
            [np.sin(number / (index + 1)) for index in range(12)], dtype=float
        )


class _Generator:
    calls = 0

    def build_context(self, **_kwargs):
        return "test"

    def generate(self, _context):
        self.calls += 1
        return [
            {
                "name": f"CANDIDATE_{index}",
                "expression": f"Mean($open,{100 + index})",
                "reason": "qEHVI integration test",
            }
            for index in range(8)
        ]


def _evaluation(expression: str, period: Period) -> EvaluationResult:
    number = int(re.findall(r"\d+", expression)[-1])
    rows = []
    for year in range(2016, 2021):
        value = 0.005 + (number % 17) * 0.001 + (year - 2016) * 0.0002
        if year == 2016 + (number % 5):
            value -= 0.02
        rows.append({"date": f"{year}-01-10", "rank_ic": value})
    train_rankic = float(np.mean([row["rank_ic"] for row in rows]))
    return EvaluationResult(
        success=True,
        expression=expression,
        period=period,
        metrics=EvaluationMetrics(rank_ic=train_rankic),
        daily_metrics=tuple(rows),
    )


class _Evaluator:
    def __init__(self):
        self.generated_calls = 0

    def evaluate_many(self, factors, period):
        return tuple(_evaluation(factor.expression, period) for factor in factors)

    def evaluate(self, expression, period):
        self.generated_calls += 1
        return _evaluation(expression, period)


def test_full_one_round_loop_keeps_42_warmup_rows_and_adds_one_batch(tmp_path) -> None:
    evaluator = _Evaluator()
    method = RankICWorstQEHVIAlphaLDM(
        evaluator=evaluator,
        profiler=_Profiler(),
        generator=_Generator(),
        output_dir=tmp_path,
        objective="rank_ic",
        acquisition="qehvi",
        evaluate_per_round=3,
        qehvi_n_samples=8,
        gp_train_iters=0,
        random_seed=42,
    )
    seeds = tuple(
        FactorCandidate(
            name=f"SEED_{index}",
            expression=f"Mean($close,{index + 2})",
            source="seed",
        )
        for index in range(42)
    )
    events = []
    result = method.search(
        train_period=Period.from_strings("2016-01-01", "2020-12-29"),
        rounds=1,
        seed_factors=seeds,
        event_sink=events.append,
    )

    assert evaluator.generated_calls == 3
    assert len(result.archive) == 45
    history_lines = (tmp_path / "ldm_history.csv").read_text().splitlines()
    assert len(history_lines) == 46  # header + 42 warm-up + one q=3 batch
    state = json.loads((tmp_path / "mobo_state.json").read_text())
    assert state["split_reward"]["segmentation"] == "calendar_year"
    assert state["schema_version"] == "alphaldm.rankic_worst_qehvi.annual.v2"
    assert all(set(row["yearly_rankic"]) == set(SplitSettings().years)
               for row in state["all_observations"])
    assert all(set(row["year_days"]) == set(SplitSettings().years)
               for row in state["all_observations"])
    assert "quarter" not in (tmp_path / "ldm_history.csv").read_text()
    assert state["observation_count"] == 45
    assert state["normalization"]["observation_count"] == 42
    batch_event = next(event for event in events if event["event"] == "ldm_batch_acquisition")
    assert batch_event["batch_count"] == 56
    assert len(batch_event["selected_local_indices"]) == 3

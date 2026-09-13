"""Regression checks for the data boundary and numerical LDM search contract."""

from dataclasses import replace
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from alpha_research.methods.ldm.acquire import acquisition_values, ei, softmax_sample
from alpha_research.methods.ldm.generator import MockCandidateGenerator
from alpha_research.methods.ldm.gp import LdmSurrogate
from alpha_research.methods.ldm.history import History
from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.environments.static import StaticEnvironment, StaticProtocol
from alpha_research.formula import FormulaValidator
from alpha_research.profile import FactorProfiler, FEATURE_DIM
from alpha_research.types import FactorCandidate, Period, SearchResult, ValidationSelection

from test_ldm_verified_evaluations import _Evaluator, _Generator, _Profiler, _success


TRAIN = Period.from_strings("2016-01-01", "2020-12-31")


def test_gp_full_history_refit_is_independent_of_prior_fits():
    features = np.asarray([[0.0], [0.5], [1.5], [2.0]])
    scores = np.asarray([-0.03, 0.01, 0.02, -0.01])
    kwargs = dict(feature_dim=1, min_fit_data=2, train_iters=4)
    uninterrupted = LdmSurrogate(**kwargs)
    uninterrupted.fit(features[:2], scores[:2])
    assert uninterrupted.trained  # the minimum is inclusive
    uninterrupted.fit(features, scores)
    restored = LdmSurrogate(**kwargs)
    restored.fit(features, scores)
    for actual, expected in zip(uninterrupted.predict([[0.2], [1.2]]), restored.predict([[0.2], [1.2]])):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_ei_compares_with_the_real_incumbent():
    mean, std = np.asarray([0.02, 0.03]), np.asarray([0.002, 0.001])
    values = acquisition_values(mean, std, mode="ei", best=0.01, xi=0.0)
    np.testing.assert_allclose(values, ei(mean, std, best=0.01, xi=0.0))
    assert values[1] > 0.019
    with pytest.raises(ValueError, match="best observed"):
        acquisition_values(mean, std, mode="ei")


def test_softmax_retains_exploration_and_is_reproducible():
    values = np.asarray([0.01, 0.02, 0.03])
    rng1, rng2 = random.Random(7), random.Random(7)
    picks = [softmax_sample(values, rng1) for _ in range(100)]
    assert picks == [softmax_sample(values, rng2) for _ in range(100)]
    assert set(picks) == {0, 1, 2}


def test_history_refuses_bad_or_duplicate_verified_observations():
    history = History(2, "test")
    for feature, score in [([1.0], 0.02), ([1.0, float("nan")], 0.02), ([1.0, 2.0], float("inf"))]:
        with pytest.raises(ValueError):
            history.add(feature, score, "$close")
    history.add([1.0, 2.0], -0.02, "$close")
    with pytest.raises(ValueError, match="already contains"):
        history.add([1.0, 2.0], -0.02, "$close")
    assert history.scores().tolist() == [-0.02]


def test_search_uses_signed_training_scores_and_only_evaluates_selected_candidates(tmp_path):
    class Evaluator(_Evaluator):
        def __init__(self):
            self.periods = []

        def evaluate_many(self, factors, period):
            self.periods.append(period)
            return tuple(_success(f.expression, period, -0.02) for f in factors)

        def evaluate(self, expression, period):
            self.periods.append(period)
            return _success(expression, period, -0.01)

    class Generator(_Generator):
        def build_context(self, **kwargs):
            assert kwargs["objective"] == "rank_ic"
            assert [row["score"] for row in kwargs["evaluated"]] == [-0.02]
            assert set(kwargs["evaluated"][0]) == {"expression", "score"}
            return "train-only"

    evaluator = Evaluator()
    method = AlphaLDM(evaluator=evaluator, profiler=_Profiler(), generator=Generator(),
                      output_dir=tmp_path, acquisition="ucb", gp_train_iters=0)
    result = method.search(train_period=TRAIN, rounds=1,
                           seed_factors=(FactorCandidate("SEED", "$close"),))
    assert len(result.archive) == 2  # one seed and one of four proposals
    assert result.best_train.metrics.rank_ic == -0.01
    assert evaluator.periods == [TRAIN, TRAIN]


def test_search_rejects_a_profile_window_outside_training_before_evaluation(tmp_path):
    profiler = _Profiler()
    profiler.period = Period.from_strings("2020-01-01", "2021-01-01")
    method = AlphaLDM(evaluator=_Evaluator(), profiler=profiler, generator=_Generator(),
                      output_dir=tmp_path)
    with pytest.raises(ValueError, match="profiling window"):
        method.search(train_period=TRAIN, rounds=1)


@pytest.mark.parametrize("wrong", ["expression", "period", "count"])
def test_seed_archive_refuses_misaligned_evaluator_results(tmp_path, wrong):
    class Evaluator(_Evaluator):
        def evaluate_many(self, factors, period):
            result = _success(factors[0].expression, period, 0.02)
            if wrong == "expression":
                result = replace(result, expression="$open")
            elif wrong == "period":
                result = replace(result, period=Period.from_strings("2021-01-01", "2021-12-31"))
            return () if wrong == "count" else (result,)

    method = AlphaLDM(evaluator=Evaluator(), profiler=_Profiler(), generator=_Generator(),
                      output_dir=tmp_path, acquisition="random")
    with pytest.raises(RuntimeError, match="requested formula|different number"):
        method.search(train_period=TRAIN, rounds=1,
                      seed_factors=(FactorCandidate("SEED", "$close"),))


def test_mock_resume_restores_refill_batch_position():
    uninterrupted = MockCandidateGenerator(candidates_per_round=3)
    for _ in range(4):
        uninterrupted.generate("context")
    restored = MockCandidateGenerator(candidates_per_round=3)
    restored.restore_state(completed_round=2, llm_calls_total=12)
    assert restored.generate("context") == uninterrupted.generate("context")


def test_fingerprint_loads_only_factor_values_and_frozen_references(tmp_path, monkeypatch):
    import pandas as pd

    reads = []
    index = pd.MultiIndex.from_product(
        [["A", "B", "C"], pd.date_range("2020-01-01", periods=24)],
        names=["instrument", "datetime"],
    )

    def features(_instruments, expressions, **kwargs):
        reads.append((list(expressions), kwargs))
        return pd.DataFrame({expression: np.arange(len(index), dtype=float) + 1 for expression in expressions}, index=index)

    monkeypatch.setitem(sys.modules, "qlib.data", SimpleNamespace(D=SimpleNamespace(
        instruments=lambda market: market, features=features,
    )))
    profiler = FactorProfiler(provider_uri=tmp_path, market="all", period=TRAIN,
                              reference_expressions=("$open",), validator=FormulaValidator(),
                              min_observations=2)
    monkeypatch.setattr(profiler, "_ensure_initialized", lambda: None)
    first = profiler.profile("$close")
    second = profiler.profile("$close")
    assert first.shape == (FEATURE_DIM,) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert [expressions for expressions, _ in reads] == [["$close"], ["$open"]]
    assert all(kwargs["end_time"] == TRAIN.end.isoformat() for _, kwargs in reads)


@pytest.mark.parametrize("failed_stage", ["individual", "pool"])
def test_static_environment_does_not_mark_failed_test_reporting_completed(tmp_path, failed_stage):
    validation = Period.from_strings("2021-01-01", "2021-12-31")
    test = Period.from_strings("2022-01-01", "2022-12-31")
    candidate = FactorCandidate("CLOSE", "$close")
    train_result = _success(candidate.expression, TRAIN, 0.02)
    selection = ValidationSelection((candidate,), ((candidate, _success(candidate.expression, validation, 0.01)),))
    method = SimpleNamespace(
        name="ldm_standard",
        search=lambda **_: SearchResult(candidate, train_result, ((candidate, train_result),), 0, 0),
        select_on_validation=lambda **_: selection,
    )

    class Evaluator:
        def evaluate_many(self, candidates, period):
            result = _success(candidates[0].expression, period, 0.01)
            return (replace(result, success=False, error="test backend unavailable") if failed_stage == "individual" else result,)

        def evaluate_pool(self, candidates, period):
            result = _success("Pool($close)", period, 0.01)
            return replace(result, success=False, error="test pool unavailable") if failed_stage == "pool" else result

    environment = StaticEnvironment(
        protocol=StaticProtocol(TRAIN, validation, test, factor_budget=1, search_rounds=0),
        method=method, evaluator=Evaluator(), output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="test reporting is incomplete"):
        environment.run()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"

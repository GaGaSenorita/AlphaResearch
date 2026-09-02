from __future__ import annotations

import json

import numpy as np
import pytest

from alpha_research.methods.ldm_continuous_discovery import ContinuousDiscoveryLDM
from alpha_research.types import (
    EvaluationMetrics,
    EvaluationResult,
    FactorCandidate,
    Period,
)


class _Schema:
    names = ("length",)
    version = "continuous-test-v1"

    def to_dict(self):
        return {"names": list(self.names), "version": self.version}


class _Profiler:
    schema = _Schema()
    reference_expressions = ()

    def profile(self, expression):
        return np.asarray([float(len(expression))])


class _RoundGenerator:
    def __init__(self):
        self.calls = 0
        self.round_id = 0

    def build_context(self, **kwargs):
        self.round_id = int(kwargs["round_id"])
        return "context"

    def generate(self, _context):
        self.calls += 1
        return [
            {
                "name": f"R{self.round_id}C{index}",
                "expression": f"Mean($close,{self.round_id * 10 + index + 2})",
                "reason": "resume test",
            }
            for index in range(3)
        ]


class _Evaluator:
    def __init__(self):
        self.seed_batches = 0

    def _result(self, expression, period):
        score = 0.01 + (len(expression) % 11) / 1000
        daily = tuple(
            {
                "date": f"2021-01-{index + 1:02d}",
                "ic": score,
                "rank_ic": score,
            }
            for index in range(40)
        )
        return EvaluationResult(
            True,
            expression,
            period,
            metrics=EvaluationMetrics(ic=score, rank_ic=score, daily_count=40),
            daily_metrics=daily,
        )

    def evaluate(self, expression, period):
        return self._result(expression, period)

    def evaluate_many(self, factors, period):
        self.seed_batches += 1
        return tuple(self._result(factor.expression, period) for factor in factors)

    def evaluate_pool(self, factors, period):
        return self._result("Pool(" + ",".join(f.expression for f in factors) + ")", period)


def _method(tmp_path, evaluator, *, resume, checkpoint_rounds=()):
    return ContinuousDiscoveryLDM(
        evaluator=evaluator,
        profiler=_Profiler(),
        generator=_RoundGenerator(),
        output_dir=tmp_path,
        objective="rank_ic",
        acquisition="random",
        evaluate_per_round=1,
        random_seed=7,
        resume=resume,
        checkpoint_rounds=checkpoint_rounds,
        checkpoint_factor_budget=2,
        checkpoint_rank_ic_threshold=0.035,
        checkpoint_validation_period=Period.from_strings("2021-01-01", "2021-12-29"),
        checkpoint_test_period=Period.from_strings("2022-01-01", "2025-12-26"),
        checkpoint_max_correlation=None,
    )


def test_continuous_ldm_preserves_factor_order_and_resumes(tmp_path):
    train = Period.from_strings("2016-01-01", "2020-12-29")
    seed = FactorCandidate("SEED", "$close", source="seed")
    first_evaluator = _Evaluator()
    first = _method(tmp_path, first_evaluator, resume=True).search(
        train_period=train,
        rounds=2,
        seed_factors=(seed,),
    )
    assert len(first.archive) == 3

    sequence = json.loads((tmp_path / "factor_sequence.json").read_text())
    assert [row["round_id"] for row in sequence["factors"]] == [0, 1, 2]
    assert [row["evaluation_index"] for row in sequence["factors"]] == [1, 2, 3]

    second_evaluator = _Evaluator()
    events = []
    resumed = _method(tmp_path, second_evaluator, resume=True).search(
        train_period=train,
        rounds=4,
        seed_factors=(seed,),
        event_sink=events.append,
    )
    assert len(resumed.archive) == 5
    assert second_evaluator.seed_batches == 0
    assert [item[0].round_id for item in resumed.archive] == [0, 1, 2, 3, 4]
    assert any(event["event"] == "ldm_continuous_resumed" for event in events)

    state = json.loads((tmp_path / "resume_state.json").read_text())
    assert state["round_id"] == 4
    assert state["history_observations"] == 5


def test_continuous_ldm_refuses_to_resume_backwards(tmp_path):
    train = Period.from_strings("2016-01-01", "2020-12-29")
    seed = FactorCandidate("SEED", "$close", source="seed")
    _method(tmp_path, _Evaluator(), resume=True).search(
        train_period=train,
        rounds=2,
        seed_factors=(seed,),
    )
    with pytest.raises(ValueError, match="below committed round"):
        _method(tmp_path, _Evaluator(), resume=True).search(
            train_period=train,
            rounds=1,
            seed_factors=(seed,),
        )


def test_round_zero_and_dense_checkpoints_can_be_backfilled_from_sequence(tmp_path):
    train = Period.from_strings("2016-01-01", "2020-12-29")
    seeds = (
        FactorCandidate("SEED_CLOSE", "$close", source="seed"),
        FactorCandidate("SEED_OPEN", "$open", source="seed"),
    )
    method = _method(
        tmp_path,
        _Evaluator(),
        resume=True,
        checkpoint_rounds=(0, 1, 2),
    )
    method.search(train_period=train, rounds=2, seed_factors=seeds)

    # Direct search writes checkpoints strictly before the target; the static
    # environment normally writes the target itself during final reporting.
    report = json.loads((tmp_path / "top5_combinations.json").read_text())
    assert [row["checkpoint_round"] for row in report["checkpoints"]] == [0, 1]
    assert [row["available_factor_count"] for row in report["checkpoints"]] == [2, 3]

    events = []
    method.backfill_checkpoint_reports_from_sequence(event_sink=events.append)
    report = json.loads((tmp_path / "top5_combinations.json").read_text())
    assert [row["checkpoint_round"] for row in report["checkpoints"]] == [0, 1, 2]
    assert [row["available_factor_count"] for row in report["checkpoints"]] == [2, 3, 4]
    assert events[0]["reporting_only"] is True
    assert events[-1]["event"] == "ldm_continuous_reporting_backfill_completed"


def test_negative_checkpoint_round_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="cannot be negative"):
        _method(
            tmp_path,
            _Evaluator(),
            resume=True,
            checkpoint_rounds=(-1, 0, 10),
        )

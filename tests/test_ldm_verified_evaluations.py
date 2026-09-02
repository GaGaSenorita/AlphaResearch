import numpy as np

from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.types import (
    EvaluationMetrics,
    EvaluationResult,
    FactorCandidate,
    Period,
)


class _Schema:
    names = ("x",)
    version = "test"

    def to_dict(self):
        return {"names": list(self.names), "version": self.version}


class _Profiler:
    schema = _Schema()
    reference_expressions = ()

    def profile(self, expression):
        return np.asarray([float(len(expression))])


class _Generator:
    calls = 0

    def build_context(self, **_kwargs):
        return {}

    def generate(self, _context):
        self.calls += 1
        return [
            {"name": f"C{i}", "expression": f"Mean($close,{i + 2})", "reason": "test"}
            for i in range(4)
        ]


class _SparseFirstBatchGenerator(_Generator):
    def generate(self, _context):
        self.calls += 1
        count = 1 if self.calls == 1 else 3
        offset = 0 if self.calls == 1 else 10
        return [
            {
                "name": f"SPARSE_{self.calls}_{i}",
                "expression": f"Mean($open,{offset + i + 2})",
                "reason": "test",
            }
            for i in range(count)
        ]


def _success(expression, period, score):
    return EvaluationResult(
        True,
        expression,
        period,
        metrics=EvaluationMetrics(ic=score, rank_ic=score, daily_count=1),
        daily_metrics=({"date": period.start.isoformat(), "ic": score, "rank_ic": score},),
    )


class _Evaluator:
    def __init__(self):
        self.generated_calls = 0

    def evaluate_many(self, factors, period):
        return tuple(_success(f.expression, period, 0.01) for f in factors)

    def evaluate(self, expression, period):
        self.generated_calls += 1
        if self.generated_calls == 1:
            return EvaluationResult(False, expression, period, error="synthetic failure")
        return _success(expression, period, 0.02 + self.generated_calls / 1000)


def test_failed_acquisition_is_replaced_before_gp_round_completes(tmp_path):
    evaluator = _Evaluator()
    method = AlphaLDM(
        evaluator=evaluator,
        profiler=_Profiler(),
        generator=_Generator(),
        output_dir=tmp_path,
        acquisition="random",
        evaluate_per_round=2,
        random_seed=42,
    )
    seed = FactorCandidate("SEED", "$close", source="seed")
    result = method.search(
        train_period=Period.from_strings("2021-01-04", "2021-01-04"),
        rounds=1,
        seed_factors=(seed,),
    )

    assert evaluator.generated_calls == 3
    assert len(result.archive) == 3  # one seed + exactly two verified round observations


def test_insufficient_unique_proposals_trigger_another_generation_batch(tmp_path):
    evaluator = _Evaluator()
    generator = _SparseFirstBatchGenerator()
    method = AlphaLDM(
        evaluator=evaluator,
        profiler=_Profiler(),
        generator=generator,
        output_dir=tmp_path,
        acquisition="random",
        evaluate_per_round=2,
        max_refill_batches=3,
        random_seed=42,
    )
    seed = FactorCandidate("SEED", "$close", source="seed")
    events = []
    result = method.search(
        train_period=Period.from_strings("2021-01-04", "2021-01-04"),
        rounds=1,
        seed_factors=(seed,),
        event_sink=events.append,
    )

    assert generator.calls == 2
    assert evaluator.generated_calls == 3  # first failed, two verified replacements
    assert len(result.archive) == 3
    final_round = [event for event in events if event.get("event") == "ldm_round"][-1]
    assert final_round["generation_batches"] == 2
    assert final_round["successful_evaluations"] == 2
    assert final_round["status"] == "evaluated"

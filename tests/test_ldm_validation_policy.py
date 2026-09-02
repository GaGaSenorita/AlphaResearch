from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.types import (
    EvaluationMetrics,
    EvaluationResult,
    FactorCandidate,
    Period,
)


class _ValidationEvaluator:
    def evaluate_many(self, factors, period):
        return tuple(self.evaluate(factor.expression, period) for factor in factors)

    def evaluate(self, expression, period):
        if expression == "A":
            values = [float(index) for index in range(40)]
            score = 0.30
        elif expression == "B":
            values = [float(index * 2) for index in range(40)]
            score = 0.20
        else:
            values = [float((index % 4) - 2) for index in range(40)]
            score = 0.10
        daily = tuple(
            {"date": f"2021-03-{index + 1:02d}", "rank_ic": value}
            for index, value in enumerate(values)
        )
        return EvaluationResult(
            success=True,
            expression=expression,
            period=period,
            metrics=EvaluationMetrics(rank_ic=score),
            daily_metrics=daily,
        )


def _archive(period):
    evaluator = _ValidationEvaluator()
    candidates = tuple(
        FactorCandidate(name=expression, expression=expression)
        for expression in ("A", "B", "C")
    )
    return tuple(
        (candidate, evaluator.evaluate(candidate.expression, period))
        for candidate in candidates
    )


def test_ldm_correlation_limit_never_backfills_rejected_factor():
    period = Period.from_strings("2021-01-01", "2021-12-29")
    selection = AlphaLDM.select_on_validation(
        object.__new__(AlphaLDM),
        archive=_archive(period),
        evaluator=_ValidationEvaluator(),
        validation_period=period,
        factor_budget=2,
        objective="rank_ic",
        max_correlation=0.8,
    )
    assert [candidate.expression for candidate in selection.selected] == ["A", "C"]


def test_ldm_fails_closed_when_hard_limit_cannot_fill_budget():
    import pytest

    period = Period.from_strings("2021-01-01", "2021-12-29")
    with pytest.raises(RuntimeError, match="formal protocol requires 3"):
        AlphaLDM.select_on_validation(
            object.__new__(AlphaLDM),
            archive=_archive(period),
            evaluator=_ValidationEvaluator(),
            validation_period=period,
            factor_budget=3,
            objective="rank_ic",
            max_correlation=0.8,
        )

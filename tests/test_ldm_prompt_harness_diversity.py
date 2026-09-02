from types import SimpleNamespace

import pytest

from alpha_research.formula import FormulaValidator
from alpha_research.methods.ldm_prompt_harness.diversity import (
    DiversityController,
    DiversitySettings,
    StructuralAnalyzer,
    series_correlation,
)
from alpha_research.methods.ldm_prompt_harness.generator import HarnessCandidateGenerator


def _result(values):
    return SimpleNamespace(
        success=True,
        daily_metrics=tuple(
            {"date": f"2020-01-{index:02d}", "rank_ic": value}
            for index, value in enumerate(values, start=1)
        ),
    )


def _candidate(name, expression, round_id=1):
    return {
        "name": name,
        "expression": expression,
        "source": "ldm_prompt_harness",
        "round_id": round_id,
    }


def test_parameter_normalized_ast_detects_window_only_variants():
    analyzer = StructuralAnalyzer(FormulaValidator())
    five = analyzer.profile("Div(Delta($close,5),Mean($close,5))")
    twenty = analyzer.profile("Div(Delta($close,20),Mean($close,20))")
    assert five.canonical_ast != twenty.canonical_ast
    assert five.parameter_normalized_ast == twenty.parameter_normalized_ast
    assert five.window_count == 2
    assert five.parameter_count == 2


def test_subtree_similarity_finds_same_core_under_wrapper():
    analyzer = StructuralAnalyzer(FormulaValidator())
    core = analyzer.profile("Div(Delta($close,5),Mean($close,5))")
    wrapped = analyzer.profile("Abs(Div(Delta($close,10),Mean($close,10)))")
    similarity, common_nodes, signature = analyzer.similarity(core, wrapped)
    assert similarity == pytest.approx(1.0)
    assert common_nodes == core.node_count
    assert signature == core.parameter_normalized_ast


def test_train_only_behavioral_archive_and_compressed_feedback():
    controller = DiversityController(
        validator=FormulaValidator(),
        settings=DiversitySettings(
            structural_mode="feedback",
            behavioral_mode="feedback",
            behavioral_correlation_threshold=0.9,
        ),
    )
    controller.record_train_evaluation(
        candidate=_candidate("A", "Mean($close,5)"),
        result=_result([0.1, -0.2, 0.3, 0.4]),
        score=0.02,
    )
    event = controller.record_train_evaluation(
        candidate=_candidate("B", "Std($close,10)", 2),
        result=_result([0.2, -0.4, 0.6, 0.8]),
        score=0.03,
    )
    assert event["train_period_only"] is True
    assert event["max_abs_behavioral_correlation"] == pytest.approx(1.0)
    assert event["nearest_behavioral_neighbor"] == "Mean($close,5)"
    feedback = controller.compressed_feedback()
    assert len(feedback["crowded_train_only_behavioral_families"]) == 1
    assert "validation" not in str(feedback).lower()
    assert "test" not in str(feedback).lower()


def test_structural_reject_requires_no_arbitrary_similarity_threshold_for_window_variants():
    controller = DiversityController(
        validator=FormulaValidator(),
        settings=DiversitySettings(structural_mode="reject"),
    )
    controller.record_train_evaluation(
        candidate=_candidate("A", "Mean($close,5)"),
        result=_result([0.1, 0.2, 0.3]),
        score=0.01,
    )
    _profile, rejected = controller.assess_proposal(_candidate("B", "Mean($close,20)"))
    assert rejected is True


def test_penalty_changes_prompt_priority_only_not_stored_train_score():
    controller = DiversityController(
        validator=FormulaValidator(),
        settings=DiversitySettings(structural_mode="penalty", penalty_weight=0.01),
    )
    controller.record_train_evaluation(
        candidate=_candidate("A", "Mean($close,5)"),
        result=_result([0.1, 0.2, 0.3]),
        score=0.05,
    )
    controller.record_train_evaluation(
        candidate=_candidate("B", "Mean($close,20)", 2),
        result=_result([0.2, 0.1, 0.3]),
        score=0.04,
    )
    entry = controller.entries[controller.analyzer.profile("Mean($close,20)").canonical_ast]
    assert entry.train_score == 0.04
    assert controller.prompt_adjusted_score("Mean($close,20)", 0.04) < 0.04


def test_series_correlation_uses_date_intersection():
    assert series_correlation(
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": -2.0, "b": -4.0, "c": -6.0, "d": 9.0},
    ) == pytest.approx(-1.0)


def test_feedback_is_compressed_into_next_prompt_without_full_series():
    controller = DiversityController(
        validator=FormulaValidator(),
        settings=DiversitySettings(
            structural_mode="feedback",
            behavioral_mode="feedback",
            structural_similarity_threshold=0.9,
            behavioral_correlation_threshold=0.9,
        ),
    )
    rows = [
        (_candidate("A", "Mean($close,5)"), [0.1, -0.2, 0.3], 0.02),
        (_candidate("B", "Mean($close,20)", 2), [0.2, -0.4, 0.6], 0.03),
    ]
    for candidate, values, score in rows:
        controller.record_train_evaluation(
            candidate=candidate,
            result=_result(values),
            score=score,
        )
    generator = HarnessCandidateGenerator(
        client=object(),
        validator=FormulaValidator(),
        candidates_per_round=2,
        diversity_controller=controller,
    )
    context = generator.build_context(
        evaluated=[{"expression": row[0]["expression"], "score": row[2]} for row in rows],
        best={"expression": rows[1][0]["expression"], "score": rows[1][2]},
        objective="rank_ic",
        round_id=3,
    )
    assert "DETERMINISTIC DIVERSITY FEEDBACK" in context
    assert "crowded_structural_families" in context
    assert "crowded_train_only_behavioral_families" in context
    assert "2020-01-01" not in context

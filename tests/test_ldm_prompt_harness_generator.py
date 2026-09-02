import json

from alpha_research.formula import FormulaValidator
from alpha_research.methods.ldm_prompt_harness.generator import (
    SCHEMA_VERSION,
    HarnessCandidateGenerator,
)


def _candidate(name, expression, direction="positive"):
    return {
        "name": name,
        "expression": expression,
        "reason": "A concise mechanism and direction summary.",
        "hypothesis": "A concise and testable financial hypothesis.",
        "mechanism": "A plausible behavioral or market mechanism.",
        "expected_direction": direction,
        "implementation_note": "The fields and operators implement the hypothesis.",
        "novelty_note": "The mechanism is distinct from the other candidate.",
    }


class StubClient:
    model_name = "stub"

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        return json.dumps(self.payload)


def test_batch_prompt_and_strict_parse():
    payload = {
        "schema_version": SCHEMA_VERSION,
        "round_id": 3,
        "candidates": [
            _candidate("MOMENTUM", "Div(Delta($close,20),Add(Mean($close,20),1e-12))"),
            _candidate("REVERSAL", "Div(Sub($close,$open),Add(Mean(Sub($high,$low),10),1e-12))", "negative"),
        ],
    }
    client = StubClient(payload)
    generator = HarnessCandidateGenerator(
        client=client,
        validator=FormulaValidator(),
        candidates_per_round=2,
    )
    context = generator.build_context(
        evaluated=[{"expression": "Mean($close,5)", "score": 0.01}],
        best={"expression": "Mean($close,5)", "score": 0.01},
        objective="rank_ic",
        round_id=3,
    )
    result = generator.generate(context)
    assert [candidate["name"] for candidate in result] == ["MOMENTUM", "REVERSAL"]
    assert result[1]["expected_direction"] == "positive"
    assert result[1]["expression"].startswith("Mul(-1,")
    assert generator.calls == 1
    assert "Never use future observations" in client.calls[0][0]
    assert "COMPACT TRAIN-ONLY SUCCESS EVIDENCE" in context
    assert '"observed_train_score": 0.01' in context
    assert "BEST TRAIN-ONLY RESULT SO FAR" in context
    assert "higher-is-better" in context
    assert "validation results" not in context.lower()


def test_context_includes_bottom_train_scores_as_failure_evidence():
    generator = HarnessCandidateGenerator(
        client=StubClient({}),
        validator=FormulaValidator(),
        candidates_per_round=2,
    )
    context = generator.build_context(
        evaluated=[
            {"expression": "Mean($close,5)", "score": 0.04},
            {"expression": "Std($close,5)", "score": -0.03},
        ],
        best={"expression": "Mean($close,5)", "score": 0.04},
        objective="rank_ic",
        round_id=2,
    )
    failure_section = context.split("COMPACT TRAIN-ONLY FAILURE EVIDENCE", 1)[1]
    assert '"observed_train_score": -0.03' in failure_section


def test_schema_rejects_missing_alignment_fields():
    payload = {
        "schema_version": SCHEMA_VERSION,
        "round_id": 1,
        "candidates": [{"name": "BAD", "expression": "Mean($close,5)", "reason": "x"}],
    }
    generator = HarnessCandidateGenerator(
        client=StubClient(payload),
        validator=FormulaValidator(),
        candidates_per_round=1,
        max_attempts=1,
    )
    context = generator.build_context(evaluated=[], best=None, objective="rank_ic", round_id=1)
    assert generator.generate(context) == []

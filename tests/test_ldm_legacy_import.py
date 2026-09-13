from __future__ import annotations

import json
from copy import deepcopy

import pytest
import numpy as np

from alpha_research.io import append_jsonl, write_json
from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.methods.ldm_continuous_discovery import ContinuousDiscoveryLDM
from alpha_research.methods.ldm_continuous_discovery.legacy import (
    SOURCE_FILES, _digest, replay_sampler, validate_protocol,
)
from alpha_research.types import FactorCandidate, Period, EvaluationResult, EvaluationMetrics


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

    def build_context(self, **kwargs):
        self.round_id = kwargs["round_id"]
        return "context"

    def generate(self, context):
        self.calls += 1
        return [{"name": f"R{self.round_id}C{i}",
                 "expression": f"Mean($close,{self.round_id * 10 + i + 2})"}
                for i in range(3)]


class _Evaluator:
    def __init__(self):
        self.seed_batches = 0

    def evaluate(self, expression, period):
        score = 0.01 + len(expression) % 11 / 1000
        return EvaluationResult(True, expression, period,
                                metrics=EvaluationMetrics(rank_ic=score, ic=score, daily_count=0))

    def evaluate_many(self, factors, period):
        self.seed_batches += 1
        return tuple(self.evaluate(c.expression, period) for c in factors)

    def evaluate_pool(self, factors, period):
        return self.evaluate("Pool(" + ",".join(c.expression for c in factors) + ")", period)


TRAIN = Period.from_strings("2016-01-01", "2020-12-29")
VAL = Period.from_strings("2021-01-01", "2021-12-29")
TEST = Period.from_strings("2022-01-01", "2025-12-26")


def _kwargs(path, evaluator):
    return dict(evaluator=evaluator, profiler=_Profiler(), generator=_RoundGenerator(),
                output_dir=path, acquisition="ucb", evaluate_per_round=1,
                random_seed=123, gp_min_fit_data=1000)


def _legacy_run(tmp_path, **options):
    path = tmp_path / "old"
    path.mkdir()
    events = []
    arguments = _kwargs(path, _Evaluator())
    arguments.update(options)
    method = AlphaLDM(**arguments)
    result = method.search(
        train_period=TRAIN, rounds=3,
        seed_factors=(FactorCandidate("SEED", "$close", source="seed"),),
        event_sink=events.append,
    )
    protocol = dict(environment="static", train=TRAIN.to_dict(), validation=VAL.to_dict(),
                    test=TEST.to_dict(), objective="rank_ic", max_correlation=None,
                    factor_budget=30, search_rounds=3,
                    metadata={"method": "ldm", "ffo": {}, "ldm": {"gp": dict(method.gp_kwargs)},
                              "llm": {"base_url": "https://api.deepseek.com/v1",
                                      "model": "deepseek-v4-pro"}})
    write_json(path / "protocol.json", protocol)
    write_json(path / "search.json", result.to_dict())
    write_json(path / "summary.json", {"status": "completed"})
    write_json(path / "validation_selection.json", {"evaluations": [
        {"candidate": c.to_dict(), "validation": _Evaluator().evaluate(c.expression, VAL).to_dict()}
        for c, _ in result.archive
    ]})
    for event in events:
        append_jsonl(path / "events.jsonl", event)
    return path, protocol, result, events


@pytest.fixture
def legacy(tmp_path):
    return _legacy_run(tmp_path)


def _destination(tmp_path, legacy, **overrides):
    source, protocol, _, _ = legacy
    options = dict(
        **_kwargs(tmp_path / "new", _Evaluator()), resume=True,
        legacy_source=source, legacy_expected_protocol=protocol,
        checkpoint_rounds=(), checkpoint_factor_budget=2,
        checkpoint_validation_period=VAL, checkpoint_test_period=TEST,
    )
    options.update(overrides)
    return ContinuousDiscoveryLDM(**options)


def test_migration_preserves_every_factor_and_replays_rng(tmp_path, legacy):
    source, _, old, events = legacy
    before = {name: _digest(source / name) for name in SOURCE_FILES}
    method = _destination(tmp_path, legacy)
    resumed = method.search(train_period=TRAIN, rounds=5)
    assert resumed.archive[:4] == old.archive
    assert len(resumed.archive) == 6
    assert method.evaluator.seed_batches == 0  # no warm-up or repeated old Train evaluation
    assert before == {name: _digest(source / name) for name in SOURCE_FILES}
    state = json.loads((method.output_dir / "resume_state.json").read_text())
    assert state["round_id"] == 5
    manifest = json.loads((method.output_dir / "legacy_import.json").read_text())
    assert manifest["sampler_choices_verified"] == 3
    assert manifest["gp_compatibility"]["compatibility_basis"] == "no_hyperparameter_training_verified"
    full = AlphaLDM(**_kwargs(tmp_path / "full", _Evaluator())).search(
        train_period=TRAIN, rounds=5,
        seed_factors=(FactorCandidate("SEED", "$close", source="seed"),),
    )
    assert [c.expression for c, _ in resumed.archive] == [c.expression for c, _ in full.archive]
    sequence = json.loads((method.output_dir / "factor_sequence.json").read_text())["factors"]
    assert [r["evaluation_index"] for r in sequence] == list(range(1, 7))
    again = _destination(tmp_path, legacy).search(train_period=TRAIN, rounds=6)
    assert len(again.archive) == 7


def test_import_fails_closed_on_history_tampering(tmp_path, legacy):
    source, _, _, _ = legacy
    csv = source / "ldm_history.csv"
    csv.write_text(csv.read_text().replace("continuous-test-v1", "changed-schema"))
    method = _destination(tmp_path, legacy)
    with pytest.raises(ValueError, match="history schema"):
        method.search(train_period=TRAIN, rounds=5)
    assert not method.ledger_path.exists()


def test_import_rejects_changed_data_protocol(tmp_path, legacy):
    changed = deepcopy(legacy[1])
    changed["train"]["end"] = "2020-12-30"
    method = _destination(tmp_path, legacy, legacy_expected_protocol=changed)
    with pytest.raises(ValueError, match="train"):
        method.search(train_period=TRAIN, rounds=5)
    assert not method.ledger_path.exists()


def test_replay_refuses_mismatched_choice(legacy):
    events = deepcopy(legacy[3])
    row = next(e for e in events if e.get("event") == "ldm_round")
    row["selected_indices"][0] = (row["selected_indices"][0] + 1) % 3
    with pytest.raises(ValueError, match="sampler replay"):
        replay_sampler(events, rounds=3, seed=123, temperature=1.0)


def test_model_alias_is_allowed_but_thinking_change_is_not(legacy):
    source = legacy[1]
    expected = deepcopy(source)
    expected["metadata"]["llm"]["model"] = "openai/deepseek-v4-pro"
    validate_protocol(source, expected)
    expected["metadata"]["llm"]["thinking_enabled"] = True
    with pytest.raises(ValueError, match="thinking_enabled"):
        validate_protocol(source, expected)


def test_backfilled_checkpoint_uses_only_then_available_factors(tmp_path, legacy):
    method = _destination(tmp_path, legacy, checkpoint_rounds=(1, 3))
    method.search(train_period=TRAIN, rounds=4)
    report = json.loads((method.output_dir / "top5_combinations.json").read_text())
    assert [r["checkpoint_round"] for r in report["checkpoints"]] == [1, 3]
    for checkpoint in report["checkpoints"]:
        assert checkpoint["available_factor_count"] == checkpoint["checkpoint_round"] + 1
        assert all(r["candidate"]["round_id"] <= checkpoint["checkpoint_round"]
                   for r in checkpoint["selected_top5_validation_order"])


@pytest.mark.parametrize("source_policy", [None, "likelihood_only_warm_start_v1"])
def test_trained_legacy_import_rejects_missing_or_old_fit_policy_without_writing(tmp_path, source_policy):
    legacy = _legacy_run(tmp_path, gp_min_fit_data=1, gp_train_iters=3)
    source, protocol, _, events = legacy
    assert any(row.get("gp_trained") is True for row in events)
    if source_policy is not None:
        protocol["metadata"]["ldm"]["gp"]["fit_policy"] = source_policy
        write_json(source / "protocol.json", protocol)
    before = {name: _digest(source / name) for name in SOURCE_FILES}
    method = _destination(tmp_path, legacy, gp_min_fit_data=1, gp_train_iters=3)
    with pytest.raises(ValueError, match="legacy GP fit policy"):
        method.search(train_period=TRAIN, rounds=5)
    assert not method.ledger_path.exists()
    assert not (method.output_dir / "resume_state.json").exists()
    assert before == {name: _digest(source / name) for name in SOURCE_FILES}


def test_matching_trained_fit_policy_imports_with_explicit_provenance(tmp_path):
    legacy = _legacy_run(tmp_path, gp_min_fit_data=1, gp_train_iters=3)
    source, protocol, old, _ = legacy
    method = _destination(tmp_path, legacy, gp_min_fit_data=1, gp_train_iters=3)
    policy = method._signature(TRAIN)["gp_fit_policy"]
    protocol["metadata"]["ldm"]["gp"]["fit_policy"] = policy
    write_json(source / "protocol.json", protocol)
    resumed = method.search(train_period=TRAIN, rounds=5)
    assert resumed.archive[:len(old.archive)] == old.archive
    manifest = json.loads((method.output_dir / "legacy_import.json").read_text())
    assert manifest["gp_compatibility"] == {
        "source_fit_policy": policy,
        "destination_fit_policy": policy,
        "compatibility_basis": "matching_explicit_fit_policy",
    }
    commits = [json.loads(row) for row in method.ledger_path.read_text().splitlines()]
    assert all(row["source_gp_compatibility"] == manifest["gp_compatibility"]
               for row in commits if row.get("record_type") == "round_commit" and row.get("origin") == "legacy_import")


def test_missing_training_flags_cannot_bypass_legacy_policy_guard(tmp_path, legacy):
    source, _, _, events = legacy
    for row in events:
        row.pop("gp_trained", None)
    (source / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    with pytest.raises(ValueError, match="legacy GP fit policy"):
        _destination(tmp_path, legacy).search(train_period=TRAIN, rounds=5)


def test_untrained_import_allows_new_metadata_policy_without_changing_gp_settings(tmp_path, legacy):
    expected = deepcopy(legacy[1])
    method = _destination(tmp_path, legacy, legacy_expected_protocol=expected)
    expected["metadata"]["ldm"]["gp"]["fit_policy"] = method._signature(TRAIN)["gp_fit_policy"]
    assert len(method.search(train_period=TRAIN, rounds=5).archive) == 6

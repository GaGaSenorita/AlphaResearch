from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alpha_research.api.app import create_app
from alpha_research.api.jobs import JobManager
from alpha_research.api.records import checkpoint_view, snapshot
from alpha_research.api.settings import REPO_ROOT, Settings
from alpha_research.api.worker import PauseRequested, Reporter, online, redact
from alpha_research.io import write_json


def wait_for(client, job_id, predicate, seconds=12):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = client.get(f"/api/runs/{job_id}").json()
        if predicate(result):
            return result
        time.sleep(.1)
    raise AssertionError(result)


@pytest.fixture
def settings(tmp_path):
    return Settings(runtime_dir=tmp_path)


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_replay_is_measured_ordered_and_has_no_future_factors(seed):
    replay = json.loads((REPO_ROOT / f"demo/replays/seed{seed}.json").read_text())
    frames = replay["frames"]
    assert [f["committed_round"] for f in frames] == list(range(0, 101, 5))
    for f in frames:
        r = f["committed_round"]
        assert f["factor_count"] == 42 + 3*r
        assert f["gp_observations"] == f["factor_count"]
        assert [p["round"] for p in f["test"]] == list(range(0, r+1, 5))
        assert all(p["measured"] and p["rank_ic"] is not None for p in f["test"])
        assert all(factor["round"] <= r for factor in f["top5"] + f["leaderboard"] + f["recent_factors"])
        for i, p in enumerate(f["test_best_so_far"]):
            assert p["rank_ic"] == max(v["rank_ic"] for v in f["test"][:i+1])
    if seed in [123, 456]:
        assert any(b["rank_ic"] < a["rank_ic"] for a, b in zip(frames[-1]["test"], frames[-1]["test"][1:]))


def test_failed_test_is_missing_not_zero():
    row = {"checkpoint_round": 5, "test_equal_weight_rank_pool": {"success": False, "metrics": {"rank_ic": 0}}}
    point = checkpoint_view(row)
    assert point["rank_ic"] is None
    assert point["measured"] is False
    assert point["qualified"] is False


def test_train_snapshot_uses_only_committed_rounds():
    sequence = {"factors": [{"round_id": r, "evaluation_index": r+1, "candidate": {"name": f"f{r}", "expression": "$close", "round_id": r}, "train_metrics": {"rank_ic": r/100}} for r in range(11)]}
    value = snapshot(sequence, {"checkpoints": []}, committed_round=7, target=100)
    assert [p["round"] for p in value["train"]] == [0, 5]
    assert value["best_factor"]["round"] == 7
    assert value["train"][-1]["rank_ic"] == .05


def test_mock_process_pause_resume_and_service_reconnect(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"mode":"mock", "seed":42, "target_rounds":100, "interval_seconds":.2})
        assert response.status_code == 202
        job_id = response.json()["id"]
        wait_for(client, job_id, lambda r: "train" in r)
        assert client.post(f"/api/runs/{job_id}/pause").status_code == 202
        paused = wait_for(client, job_id, lambda r: r["status"] == "paused" and not r["alive"])
        assert all(p["round"] <= paused["committed_round"] for p in paused["test"])
    # New HTTP service attaches to the same durable state.
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/runs/{job_id}").json()["status"] == "paused"
        assert client.post(f"/api/runs/{job_id}/resume", json={}).status_code == 202
        completed = wait_for(client, job_id, lambda r: r["status"] == "completed")
        assert completed["committed_round"] == 100
        assert completed["factor_count"] == 342
        assert len(completed["train"]) == len(completed["test"]) == 21
        export = client.get(f"/api/runs/{job_id}/export")
        assert export.json()["latest_test"]["rank_ic"] == completed["latest_test"]["rank_ic"]
        assert len(export.json()["factors"]) == 342
        factors = client.get(f"/api/runs/{job_id}/factors?offset=42&limit=15").json()
        assert factors["total"] == 342
        assert [f["evaluation_index"] for f in factors["factors"]] == list(range(43, 58))


def test_invalid_inputs_origin_and_secret_exclusion(settings, monkeypatch):
    with TestClient(create_app(settings)) as client:
        for body in [{"mode":"other"}, {"mode":"mock", "seed":999}, {"mode":"mock", "target_rounds":105},
                     {"mode":"online", "target_rounds":7}, {"mode":"online", "mock":True},
                     {"mode":"online", "out_dir":"/tmp/elsewhere"}]:
            assert client.post("/api/runs", json=body).status_code == 422
        assert client.post("/api/runs", json={"mode":"mock"}, headers={"Origin":"https://unrelated.example"}).status_code == 403
        assert client.get("/api/runs/not-a-job").status_code == 404
        monkeypatch.setenv("ALPHARESEARCH_LLM_API_KEY", "test-key-do-not-expose")
        ready = client.get("/api/readiness")
        assert "test-key-do-not-expose" not in ready.text
        assert client.get("/runtime/").status_code == 404
        assert client.get("/api/archives").json()["archives"][0]["initial"]["committed_round"] == 0
        rejected = client.post("/api/runs", json={"mode":"mock", "api_key":"credential-must-not-echo"})
        assert rejected.status_code == 422
        assert "credential-must-not-echo" not in rejected.text


def test_online_capacity_and_backwards_target(settings, monkeypatch):
    manager = JobManager(settings)
    monkeypatch.setattr(manager, "list", lambda: [{"id":"a", "mode":"online", "alive":True, "status":"running"}])
    with pytest.raises(ValueError, match="online run is already active"):
        manager._check_capacity("online")


def test_live_adapter_exact_round_resume_with_test_components(tmp_path, monkeypatch):
    """Test the real adapter/commit path with deterministic, clearly synthetic dependencies."""
    import alpha_research.runner as runner
    import alpha_research.api.worker as worker
    original = runner.build_components
    built = []
    fail_target_report = [True]
    def components(args, output):
        args.mock = True
        args.gp_train_iters = 1
        evaluator, method = original(args, output)
        pool = evaluator.evaluate_pool
        def transient_pool(factors, period):
            state_path = output / "resume_state.json"
            committed = json.loads(state_path.read_text())["round_id"] if state_path.exists() else -1
            if period.start.year == 2022 and committed == 5 and fail_target_report[0]:
                from alpha_research.types import EvaluationResult
                fail_target_report[0] = False
                return EvaluationResult(False, "pool", period, error="simulated report failure")
            return pool(factors, period)
        evaluator.evaluate_pool = transient_pool
        built.append(method)
        return evaluator, method
    monkeypatch.setattr(runner, "build_components", components)
    monkeypatch.setattr(worker, "load_credential", lambda: None)
    spec = {"id":"synthetic-integration", "seed":42, "mode":"online", "target_rounds":5, "repo_root":str(REPO_ROOT)}
    reporter = Reporter(tmp_path, spec)
    # Safe pause occurs after warm-up commit. The next invocation restores it.
    write_json(tmp_path / "control.json", {"pause":True})
    with pytest.raises(PauseRequested):
        online(reporter)
    assert json.loads((tmp_path / "snapshot.json").read_text())["factor_count"] == 42
    write_json(tmp_path / "control.json", {"pause":False})
    with pytest.raises(RuntimeError, match="no verified test_equal_weight_rank_pool"):
        online(reporter)
    run_dir = next((tmp_path / "run").iterdir())
    saved_after_failure = json.loads((run_dir / "factor_sequence.json").read_text())
    assert saved_after_failure["completed_round"] == 5
    assert len(saved_after_failure["factors"]) == 57
    online(reporter)
    assert json.loads((run_dir / "factor_sequence.json").read_text()) == saved_after_failure
    result = json.loads((tmp_path / "snapshot.json").read_text())
    assert result["committed_round"] == 5
    assert result["gp_observations"] == 57
    assert [p["round"] for p in result["test"]] == [0,5]
    before = json.loads((run_dir / "factor_sequence.json").read_text())["factors"]
    spec["target_rounds"] = 10
    online(Reporter(tmp_path, spec))
    after = json.loads((run_dir / "factor_sequence.json").read_text())["factors"]
    assert after[:57] == before
    assert len(after) == 72


def test_error_redaction(monkeypatch):
    monkeypatch.setenv("ALPHARESEARCH_LLM_API_KEY", "specific-test-credential")
    assert "specific-test-credential" not in redact("failed with specific-test-credential")
    assert "sk-" not in redact("unauthorised sk-abcdefgh12345678")

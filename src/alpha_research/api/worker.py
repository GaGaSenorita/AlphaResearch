"""Isolated replay/live workers. HTTP requests never execute the research loop."""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
import sys
import threading
import time
from functools import wraps
from pathlib import Path

from alpha_research.api.records import read_json, snapshot, number
from alpha_research.api.settings import online_args
from alpha_research.io import write_json, append_jsonl


class PauseRequested(BaseException):
    """Exit only after a complete round and its checkpoint report are persisted."""


def redact(message: str) -> str:
    message = re.sub(r"(?i)(?:sk-[\w-]{8,}|gh[pousr]_[\w]+|Bearer\s+\S+)", "[redacted]", message)
    secret = os.environ.get("ALPHARESEARCH_LLM_API_KEY")
    if secret:
        message = message.replace(secret, "[redacted]")
    return message[:1200]


class Reporter:
    def __init__(self, directory: Path, spec: dict):
        self.directory, self.spec = directory, spec
        self.state = read_json(directory / "state.json", {})
        self.timeline = read_json(directory / "timeline.json", [])
        self.lock = threading.RLock()

    def update(self, *, stage=None, message=None, **values):
        with self.lock:
            if stage and (stage != self.state.get("stage") or message != self.state.get("message")):
                if stage != self.state.get("stage"):
                    self.state["stage_started_at"] = time.time()
                self.timeline.append({"stage": stage, "message": message or stage,
                                      "round": values.get("active_round", self.state.get("active_round", 0)),
                                      "at": time.time()})
                self.timeline = self.timeline[-60:]
                write_json(self.directory / "timeline.json", self.timeline)
            self.state.update(values)
            if stage:
                self.state["stage"] = stage
            if message:
                self.state["message"] = message
            self.state["updated_at"] = time.time()
            write_json(self.directory / "state.json", self.state)

    def pause_requested(self):
        return read_json(self.directory / "control.json", {}).get("pause", False)

    def refresh(self, run_dir: Path):
        sequence = read_json(run_dir / "factor_sequence.json", {})
        state = read_json(run_dir / "resume_state.json", {})
        # A commit can appear just before its factor view. Use the common safe head.
        committed = min(int(sequence.get("completed_round") or 0), int(state.get("round_id") or 0))
        payload = snapshot(sequence, read_json(run_dir / "top5_combinations.json", {}),
                           committed_round=committed, target=self.spec["target_rounds"])
        write_json(self.directory / "snapshot.json", payload)
        self.update(committed_round=committed, llm_calls=state.get("llm_calls_total", 0))


def replay(reporter: Reporter):
    spec = reporter.spec
    archive = read_json(Path(spec["replay_dir"]) / f"seed{spec['seed']}.json")
    if not archive or archive.get("checkpoint_step") != 5:
        raise ValueError("Measured five-round replay is unavailable")
    previous = read_json(reporter.directory / "snapshot.json", {})
    start = int(previous.get("committed_round", -5))
    reporter.update(status="running", stage="replay", message="Playing measured historical results", error=None)
    for frame in archive["frames"]:
        r = frame["committed_round"]
        if r <= start or r > spec["target_rounds"]:
            continue
        if r > 0:
            deadline = time.monotonic() + spec["interval_seconds"]
            while time.monotonic() < deadline:
                if reporter.pause_requested():
                    raise PauseRequested()
                time.sleep(min(.1, max(0, deadline - time.monotonic())))
        if reporter.pause_requested():
            raise PauseRequested()
        write_json(reporter.directory / "snapshot.json", {**frame, "target_rounds": spec["target_rounds"]})
        reporter.update(stage="replay", message=f"Historical checkpoint R{r} revealed",
                        committed_round=r, active_round=r)
    reporter.update(status="completed", stage="completed", message="Historical replay complete")


def load_credential():
    if not os.environ.get("ALPHARESEARCH_LLM_API_KEY") and sys.platform == "darwin":
        result = subprocess.run(["security", "find-generic-password", "-s", "AlphaResearch-LiteLLM",
                                 "-a", os.environ.get("USER", ""), "-w"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            os.environ["ALPHARESEARCH_LLM_API_KEY"] = result.stdout.strip()
    if not os.environ.get("ALPHARESEARCH_LLM_API_KEY"):
        raise RuntimeError("LLM credential unavailable. Activate the AlphaResearch Conda environment.")


def online(reporter: Reporter):
    from alpha_research.runner import build_components, static_protocol, public_metadata
    from alpha_research.naming import standard_run_name

    spec = reporter.spec
    load_credential()
    run_name = standard_run_name(method="ldm_continuous_discovery",
        model="neolink/technologies/deepseek-v4-flash", train_start="2016-01-01", test_end="2025-12-26", seed=spec["seed"])
    run_dir = reporter.directory / "run" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args = online_args(spec["seed"], spec["target_rounds"], run_dir, repo_root=Path(spec["repo_root"]))
    search_config = {k: v for k, v in vars(args).items()
                     if k not in {"search_rounds", "continuous_checkpoint_round", "out_dir", "continuous_resume"}}
    prior_config = read_json(run_dir / "api_search_config.json")
    if prior_config is not None and prior_config != search_config:
        raise ValueError("Search configuration changed; this saved run cannot resume with different settings")
    write_json(run_dir / "api_search_config.json", search_config)
    protocol = static_protocol(args)
    metadata = public_metadata(args)
    protocol_record = {**protocol.to_dict(), "metadata": metadata}
    previous = read_json(run_dir / "protocol.json")
    if previous:
        # The target and reporting schedule may extend; all search inputs stay fixed.
        for key in ("train", "validation", "test", "objective", "factor_budget", "max_correlation"):
            if previous[key] != protocol_record[key]:
                raise ValueError(f"Cannot resume with changed {key}")
    write_json(run_dir / "protocol.json", protocol_record)
    reporter.update(status="running", stage="initializing", message="Loading the evaluator and 12D factor profiler", error=None,
                    active_round=reporter.state.get("committed_round", 0))
    evaluator, method = build_components(args, run_dir)

    original_merge = method._merge_checkpoint
    def merge_verified(record):
        # Failed reports must remain retryable; never install them as completed checkpoints.
        for field in ("validation_equal_weight_rank_pool", "test_equal_weight_rank_pool"):
            pool = record.get(field) or {}
            if not pool.get("success") or number(pool.get("metrics", {}).get("rank_ic")) is None:
                raise RuntimeError(f"Checkpoint R{record['checkpoint_round']} has no verified {field}")
        return original_merge(record)
    method._merge_checkpoint = merge_verified

    def instrument(obj, name, stage, message):
        original = getattr(obj, name)
        @wraps(original)
        def wrapped(*pos, **kw):
            reporter.update(stage=stage, message=message, phase_completed=None, phase_total=None)
            return original(*pos, **kw)
        setattr(obj, name, wrapped)

    instrument(method.generator, "generate", "proposal", "LLM is proposing candidate formulas")
    build_context = method.generator.build_context
    @wraps(build_context)
    def observed_context(*pos, **kw):
        if "round_id" in kw:
            reporter.update(active_round=int(kw["round_id"]))
        return build_context(*pos, **kw)
    method.generator.build_context = observed_context
    chat = getattr(getattr(method.generator, "client", None), "chat", None)
    if chat is not None:
        @wraps(chat)
        def observed_chat(*pos, **kw):
            with reporter.lock:
                reporter.update(provider_requests=(reporter.state.get("provider_requests") or 0) + 1)
            try:
                result = chat(*pos, **kw)
            except Exception as exc:
                with reporter.lock:
                    reporter.update(provider_errors=(reporter.state.get("provider_errors") or 0) + 1,
                                    last_provider_error=redact(str(exc)))
                raise
            with reporter.lock:
                reporter.update(provider_responses=(reporter.state.get("provider_responses") or 0) + 1,
                                last_provider_error=None)
            return result
        method.generator.client.chat = observed_chat
    instrument(method.profiler, "profile", "representation", "Computing label-free 12D behavioural features")
    original_surrogate = method._new_surrogate
    def new_surrogate(*pos, **kw):
        surrogate = original_surrogate(*pos, **kw)
        instrument(surrogate, "fit", "surrogate", "Fitting the Gaussian-process surrogate")
        instrument(surrogate, "predict", "acquisition", "Scoring candidates with UCB")
        return surrogate
    method._new_surrogate = new_surrogate

    for name in ("evaluate", "evaluate_many", "evaluate_pool"):
        original = getattr(evaluator, name)
        def make_wrapper(fn, operation):
            @wraps(fn)
            def wrapped(items, period, *pos, **kw):
                stage = "train" if period == protocol.train else "validation" if period == protocol.validation else "test"
                message = {"train": "Measuring candidate Train RankIC", "validation": "Selecting the Validation Top-5",
                           "test": "Auditing the selected combination on Test"}[stage]
                values = {"phase_completed": 0, "phase_total": len(items)} if operation == "evaluate_many" else {}
                if operation == "evaluate_pool":
                    values = {"phase_completed": None, "phase_total": None}
                reporter.update(stage=stage, message=message, **values)
                result = fn(items, period, *pos, **kw)
                if operation == "evaluate":
                    with reporter.lock:
                        total = reporter.state.get("phase_total")
                        if total:
                            reporter.update(phase_completed=min(total, (reporter.state.get("phase_completed") or 0) + 1))
                return result
            return wrapped
        setattr(evaluator, name, make_wrapper(original, name))

    original_commit = method._persist_search_checkpoint
    def persist(*pos, **kw):
        original_commit(*pos, **kw)
        reporter.refresh(run_dir)
        committed = reporter.state.get("committed_round", 0)
        reporter.update(active_round=min(committed + 1, spec["target_rounds"]))
        if reporter.pause_requested():
            raise PauseRequested()
    method._persist_search_checkpoint = persist

    started_round = time.monotonic()
    def emit(event):
        nonlocal started_round
        append_jsonl(run_dir / "events.jsonl", event)
        name = event.get("event")
        if name == "ldm_continuous_round_committed":
            elapsed = time.monotonic() - started_round
            started_round = time.monotonic()
            reporter.refresh(run_dir)
            reporter.update(stage="commit", message=f"Round {event['round_id']} saved",
                            committed_round=event["round_id"], last_round_seconds=round(elapsed, 2),
                            llm_calls=event.get("llm_calls_total", 0))
        elif name == "ldm_continuous_top5_checkpoint":
            reporter.refresh(run_dir)
            reporter.update(stage="checkpoint", message=f"R{event['checkpoint_round']} Train / Test checkpoint ready")
        elif name == "ldm_continuous_reporting_failed":
            reporter.update(reporting_warning=redact(event.get("error", "Checkpoint failed")))
        elif name == "ldm_continuous_resumed":
            reporter.refresh(run_dir)
        elif name == "ldm_round":
            reporter.update(active_round=event["round_id"],
                            last_proposals=event.get("proposed"), last_verified=event.get("successful_evaluations"))

    result = method.search(train_period=protocol.train, rounds=spec["target_rounds"], event_sink=emit)
    write_json(run_dir / "search.json", result.to_dict())
    reporter.update(stage="validation", message="Finishing the final checkpoint and checking report completeness")
    # The baseline writes the final checkpoint after search. This also retries
    # any earlier report failure without changing the saved search or GP state.
    method.backfill_checkpoint_reports_from_sequence(event_sink=emit)
    reporter.refresh(run_dir)
    payload = read_json(reporter.directory / "snapshot.json")
    required = set(args.continuous_checkpoint_round)
    complete = {p["round"] for p in payload["test"] if p["measured"]}
    if not required <= complete:
        raise RuntimeError(f"Search saved, but measured reports are missing: {sorted(required - complete)}")
    write_json(run_dir / "summary.json", {"status": "completed", "method": "ldm_continuous_discovery",
               "protocol": protocol_record, "selected_factors": payload["top5"],
               "final_test_rank_ic": payload["latest_test"]["rank_ic"],
               "last_safe_round": payload["committed_round"], "api_job": spec["id"]})
    reporter.update(status="completed", stage="completed", message="Search and all measured checkpoints completed",
                    reporting_warning=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    directory = parser.parse_args().job_dir.resolve()
    spec = read_json(directory / "job.json")
    reporter = Reporter(directory, spec)
    lock = None
    awake = None
    try:
        if spec["mode"] == "online":
            lock = (directory.parent.parent / "online.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if sys.platform == "darwin":
                awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            online(reporter)
        else:
            replay(reporter)
    except PauseRequested:
        reporter.update(status="paused", stage="paused", message="Paused; saved progress is ready to resume")
    except BaseException as exc:
        reporter.update(status="failed", stage="failed", message="Run stopped; committed progress is preserved",
                        error=redact(str(exc)))
        print(f"{type(exc).__name__}: {redact(str(exc))}", flush=True)
        return 1
    finally:
        if lock:
            lock.close()
        if awake:
            awake.terminate()
            awake.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Durable jobs supervised by a single local FastAPI service."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import psutil

from alpha_research.api.records import read_json, factor_view
from alpha_research.api.settings import Settings
from alpha_research.io import write_json

TERMINAL = {"completed", "paused", "failed", "interrupted"}
JOB_ID = re.compile(r"^[a-f0-9]{16}$")


def worker_alive(job_dir: Path) -> bool:
    identity = read_json(job_dir / "process.json", {})
    try:
        process = psutil.Process(identity["pid"])
        return (abs(process.create_time() - identity["created"]) < .1 and
                "alpha_research.api.worker" in process.cmdline() and
                str(job_dir) in process.cmdline() and process.status() != psutil.STATUS_ZOMBIE)
    except (KeyError, psutil.Error):
        return False


def credential_available() -> bool:
    if os.environ.get("ALPHARESEARCH_LLM_API_KEY"):
        return True
    if sys.platform == "darwin":
        # Metadata-only lookup. Never return a secret to the HTTP client.
        try:
            result = subprocess.run(["security", "find-generic-password", "-s", "AlphaResearch-LiteLLM",
                                     "-a", os.environ.get("USER", "")],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
    return False


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.jobs.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.children: list[subprocess.Popen] = []

    def directory(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise KeyError("Unknown run")
        path = self.settings.jobs / job_id
        if not (path / "job.json").is_file():
            raise KeyError("Unknown run")
        return path

    def get(self, job_id: str) -> dict:
        path = self.directory(job_id)
        spec = read_json(path / "job.json")
        state = read_json(path / "state.json", {"status": "queued", "stage": "queued", "committed_round": 0})
        alive = worker_alive(path)
        if state["status"] not in TERMINAL and not alive and time.time() - spec.get("launched_at", 0) > 8:
            state = {**state, "status": "interrupted", "stage": "interrupted",
                     "message": "Worker stopped. Resume from the last committed round."}
        return {"id": job_id, "mode": spec["mode"], "seed": spec["seed"],
                "target_rounds": spec["target_rounds"], "interval_seconds": spec["interval_seconds"],
                "created_at": spec["created_at"], "alive": alive, **state}

    def list(self) -> list[dict]:
        for process in self.children:
            process.poll()  # Reap exited children without blocking the service.
        self.children = [p for p in self.children if p.returncode is None]
        ids = [p.parent.name for p in self.settings.jobs.glob("*/job.json")]
        return sorted((self.get(i) for i in ids), key=lambda j: j["created_at"], reverse=True)

    def _check_capacity(self, mode: str, exclude: str | None = None):
        jobs = self.list()
        active = [j for j in jobs if j["id"] != exclude and (j["alive"] or j["status"] not in TERMINAL)]
        if mode == "online" and any(j["mode"] == "online" for j in active):
            raise ValueError("An online run is already active. Pause it before starting another.")
        if len(active) >= 8:
            raise ValueError("Too many active sessions; pause an existing replay first.")

    def create(self, *, mode: str, seed: int, target_rounds: int, interval_seconds: float) -> dict:
        with self.lock:
            self._check_capacity(mode)
            if mode == "mock" and not (self.settings.replays / f"seed{seed}.json").is_file():
                raise ValueError("Measured replay is not prepared for this seed")
            job_id = uuid.uuid4().hex[:16]
            path = self.settings.jobs / job_id
            path.mkdir()
            spec = {"id": job_id, "mode": mode, "seed": seed, "target_rounds": target_rounds,
                    "interval_seconds": interval_seconds, "created_at": time.time(),
                    "repo_root": str(self.settings.repo_root), "replay_dir": str(self.settings.replays)}
            write_json(path / "job.json", spec)
            write_json(path / "state.json", {"status": "queued", "stage": "queued", "committed_round": 0,
                       "message": "Preparing measured replay" if mode == "mock" else "Preparing the research environment"})
            self._launch(path, spec)
            return self.get(job_id)

    def _launch(self, path: Path, spec: dict):
        spec["launched_at"] = time.time()
        write_json(path / "job.json", spec)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # Avoid oversubscribing the CPU while leaving all search parameters unchanged.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(name, "2")
        with (path / "worker.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-m", "alpha_research.api.worker", "--job-dir", str(path)],
                cwd=self.settings.repo_root, env=env, stdout=log, stderr=log, start_new_session=True)
        self.children.append(process)
        write_json(path / "process.json", {"pid": process.pid, "created": psutil.Process(process.pid).create_time()})

    def pause(self, job_id: str) -> dict:
        with self.lock:
            path = self.directory(job_id)
            state = self.get(job_id)
            if state["status"] in TERMINAL:
                return state
            write_json(path / "control.json", {"pause": True})
            # The worker owns state.json. Pause never races its atomic writer.
            return {**state, "pause_requested": True}

    def resume(self, job_id: str, *, target_rounds: int | None = None, interval_seconds: float | None = None) -> dict:
        with self.lock:
            path = self.directory(job_id)
            state = self.get(job_id)
            if state["alive"]:
                raise ValueError("This run still has a live worker")
            spec = read_json(path / "job.json")
            new_target = target_rounds if target_rounds is not None else spec["target_rounds"]
            if new_target < state.get("committed_round", 0):
                raise ValueError("Target cannot precede the last committed round")
            if spec["mode"] == "mock" and new_target > 100:
                raise ValueError("Historical replay ends at round 100")
            if state["status"] == "completed" and new_target <= spec["target_rounds"]:
                raise ValueError("Run is complete; create a new replay or extend an online target")
            self._check_capacity(spec["mode"], exclude=job_id)
            spec["target_rounds"] = new_target
            if interval_seconds is not None:
                spec["interval_seconds"] = interval_seconds
            write_json(path / "control.json", {"pause": False})
            write_json(path / "state.json", {**read_json(path / "state.json", {}),
                       "status": "queued", "stage": "resuming", "error": None, "message": "Resuming saved run"})
            self._launch(path, spec)
            return self.get(job_id)

    def snapshot(self, job_id: str) -> dict:
        path = self.directory(job_id)
        state = self.get(job_id)
        data = read_json(path / "snapshot.json", {})
        return {**data, **state, "timeline": read_json(path / "timeline.json", []),
                "pause_requested": read_json(path / "control.json", {}).get("pause", False)}

    def factors(self, job_id: str) -> list[dict]:
        path = self.directory(job_id)
        state = self.get(job_id)
        if state["mode"] == "mock":
            rows = read_json(self.settings.replays / f"seed{state['seed']}.json", {}).get("factors", [])
        else:
            sequences = list((path / "run").glob("*/factor_sequence.json"))
            rows = [factor_view(row) for row in read_json(sequences[0], {}).get("factors", [])] if sequences else []
        return sorted((row for row in rows if row["round"] <= state.get("committed_round", 0)),
                      key=lambda row: row["evaluation_index"])

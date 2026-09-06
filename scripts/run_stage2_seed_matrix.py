#!/usr/bin/env python3
"""Run the remaining Stage-2 seed matrix with durable status and logs.

The launcher runs the two seeds for one method as a pair, then advances to the
next method. Each experiment retains its configured internal LLM proposal
parallelism. Completed 38-round summaries are detected and skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "runtime" / "logs" / "stage2_seed_matrix"
STATUS_PATH = LOG_DIR / "status.json"
STATUS_LOCK = threading.RLock()


@dataclass(frozen=True)
class Job:
    method: str
    config: Path
    seed: int

    @property
    def run_dir(self) -> Path:
        stem = {
            "ldm_split_robust_reward": (
                "ldm_split_robust_reward_"
                "neolink-technologies-deepseek-v4-flash_2016-2025"
            ),
            "ldm_rankic_worst_qehvi": (
                "ldm_rankic_worst_qehvi_"
                "neolink-technologies-deepseek-v4-flash_2016-2025"
            ),
        }[self.method]
        return REPO_ROOT / "runs" / self.method / f"{stem}_seed{self.seed}"

    @property
    def label(self) -> str:
        return f"{self.method}:seed{self.seed}"


CONFIGS = {
    "ldm_split_robust_reward": REPO_ROOT / "configs" / "ldm_split_robust_reward"
    / "ldm_split_robust_reward_neolink-deepseek-v4-flash_2016-2025.yaml",
    "ldm_rankic_worst_qehvi": REPO_ROOT / "configs" / "ldm_rankic_worst_qehvi"
    / "ldm_rankic_worst_qehvi_neolink-deepseek-v4-flash_2016-2025.yaml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(payload: dict[str, Any]) -> None:
    with STATUS_LOCK:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        temporary = STATUS_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(STATUS_PATH)


def completed(job: Job) -> bool:
    path = job.run_dir / "summary.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = payload.get("protocol") or {}
    return payload.get("status") == "completed" and int(
        protocol.get("search_rounds", -1)
    ) == 38


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pids(pids: list[int], poll_seconds: float, state: dict[str, Any]) -> None:
    living = [pid for pid in pids if process_alive(pid)]
    if not living:
        return
    state.update({"phase": "waiting_for_existing_processes", "wait_pids": living})
    write_status(state)
    while any(process_alive(pid) for pid in living):
        time.sleep(poll_seconds)


def run_job(job: Job, state: dict[str, Any], max_attempts: int) -> bool:
    if completed(job):
        state["jobs"][job.label] = {
            "status": "completed",
            "attempts": 0,
            "run_dir": str(job.run_dir),
            "skipped_existing": True,
        }
        write_status(state)
        return True

    log_path = LOG_DIR / f"{job.method}_seed{job.seed}.log"
    for attempt in range(1, max_attempts + 1):
        state["phase"] = "running"
        state["current_job"] = job.label
        state["jobs"][job.label] = {
            "status": "running",
            "attempts": attempt,
            "run_dir": str(job.run_dir),
            "log": str(log_path),
            "started_at": utc_now(),
        }
        write_status(state)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_experiment.py"),
            str(job.config),
            "--set",
            f"args.ldm-random-seed={job.seed}",
        ]
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] attempt {attempt}: {' '.join(command)}\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0 and completed(job):
            state["jobs"][job.label].update({
                "status": "completed",
                "returncode": result.returncode,
                "completed_at": utc_now(),
            })
            write_status(state)
            return True
        state["jobs"][job.label].update({
            "status": "failed_attempt",
            "returncode": result.returncode,
            "failed_at": utc_now(),
        })
        write_status(state)
    state["jobs"][job.label]["status"] = "failed"
    write_status(state)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-concurrent", type=int, default=2)
    options = parser.parse_args()

    jobs = [
        Job("ldm_split_robust_reward", CONFIGS["ldm_split_robust_reward"], 123),
        Job("ldm_split_robust_reward", CONFIGS["ldm_split_robust_reward"], 456),
        Job("ldm_rankic_worst_qehvi", CONFIGS["ldm_rankic_worst_qehvi"], 123),
        Job("ldm_rankic_worst_qehvi", CONFIGS["ldm_rankic_worst_qehvi"], 456),
    ]
    state: dict[str, Any] = {
        "schema_version": 1,
        "started_at": utc_now(),
        "launcher_pid": os.getpid(),
        "phase": "starting",
        "current_job": None,
        "jobs": {},
    }
    write_status(state)
    wait_for_pids(options.wait_pid, max(1.0, options.poll_seconds), state)

    all_completed = True
    for method in CONFIGS:
        method_jobs = [job for job in jobs if job.method == method]
        with ThreadPoolExecutor(
            max_workers=min(max(1, options.max_concurrent), len(method_jobs))
        ) as pool:
            results = list(pool.map(
                lambda job: run_job(job, state, max(1, options.max_attempts)),
                method_jobs,
            ))
        all_completed = all(results) and all_completed
    state["phase"] = "completed" if all_completed else "completed_with_failures"
    state["current_job"] = None
    state["completed_at"] = utc_now()
    write_status(state)
    return 0 if all_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Migrate and run independent legacy seeds concurrently, then render real-data figures.

--prepare-only performs no LLM/evaluator calls. --launch detaches one locked worker
per seed. Repeating --launch resumes a failed seed; it never overwrites old runs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025.yaml"
LOG_ROOT = REPO / "runtime/logs/ldm_continuous_discovery_legacy_pair"


def paths(seed):
    source = REPO / f"runs/ldm_standard/ldm_standard_deepseek-v4-pro_2016-2025_seed{seed}"
    out = REPO / f"runs/ldm_continuous_discovery/ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}"
    return source, out


def overrides(seed, target):
    source, out = paths(seed)
    return {
        "search-rounds": target, "ldm-random-seed": seed,
        "continuous-legacy-source": str(source),
        # Preserve the original legacy search configuration; only reporting becomes Top-5.
        "llm-reasoning-effort": "high", "llm-thinking": True,
    }


def resolved_args(seed, target):
    from alpha_research.cli import load_config, build_expansion_context, expand_value, append_cli_arg
    from alpha_research.runner import parse_args
    config = load_config(CONFIG)
    config["args"].update(overrides(seed, target))
    context = build_expansion_context(config, config["args"], CONFIG)
    argv = ["--method", config["method"], "--environment", config["environment"]]
    for key, value in config["args"].items():
        append_cli_arg(argv, key, expand_value(value, CONFIG, context=context))
    return parse_args(argv)


def progress(seed, target, status, **details):
    from alpha_research.io import write_json
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    source, out = paths(seed)
    state_path = out / "resume_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    payload = {
        "updated_at": dt.datetime.now().astimezone().isoformat(), "seed": seed,
        "target_round": target, "status": status, "worker_pid": os.getpid(),
        "last_safe_round": state.get("round_id"),
        "gp_observations": state.get("history_observations"),
        "run_dir": str(out), **details,
    }
    write_json(LOG_ROOT / f"seed{seed}_status.json", payload)
    with (LOG_ROOT / f"seed{seed}_progress.log").open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def prepare(seed, target):
    from alpha_research.runner import build_components, static_protocol, public_metadata
    from alpha_research.io import write_json, append_jsonl
    from alpha_research.methods.ldm_continuous_discovery.legacy import import_legacy_run
    source, out = paths(seed)
    if not source.is_dir():
        raise ValueError(f"missing legacy seed {seed}")
    if (out / "factor_ledger.jsonl").exists():
        manifest = json.loads((out / "legacy_import.json").read_text())
        if manifest["source_dir"] != str(source):
            raise ValueError("destination belongs to another run")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to import into a nonempty destination without a ledger")
    args = resolved_args(seed, target)
    _evaluator, method = build_components(args, out)
    protocol = static_protocol(args)
    manifest = import_legacy_run(method, train_period=protocol.train, target_rounds=target)
    factors, commits = method._committed_rows()
    method._write_factor_sequence(factors, commits)
    write_json(out / "protocol.json", {**protocol.to_dict(), "metadata": public_metadata(args)})
    append_jsonl(out / "events.jsonl", {"event": "ldm_legacy_import_completed", **manifest})
    progress(seed, target, "prepared", sampler_choices_verified=manifest["sampler_choices_verified"])


def render(seed, target):
    from plot_ldm_financial_mining import draw_trajectory_with_single_test
    source, out = paths(seed)
    state = json.loads((out / "resume_state.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    if state["round_id"] != target or summary["status"] != "completed":
        raise RuntimeError("refusing to plot unfinished search as completed")
    if summary["protocol"]["search_rounds"] != target:
        raise RuntimeError("summary belongs to an earlier target; final evaluation is unfinished")
    if not summary["test"]["equal_weight_rank_pool"]["success"]:
        raise RuntimeError("final Test evaluation failed; no measured final figure")
    report = json.loads((out / "top5_combinations.json").read_text())
    expected_rounds = {r for r in (20, 38, 50, 75, 100) if r <= target} | {target}
    if not expected_rounds.issubset({r["checkpoint_round"] for r in report["checkpoints"]}):
        raise RuntimeError("missing checkpoint reports; figure is not complete")
    for row in report["checkpoints"]:
        if not row["test_equal_weight_rank_pool"]["success"]:
            raise RuntimeError("checkpoint Test evaluation failed; cannot plot as measured")
    prefix = REPO / f"figures/ldm_financial_mining_seed{seed}_round{target}"
    return draw_trajectory_with_single_test(out, prefix, show_single_test=False, seed=seed)


def worker(seed, target):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / f"seed{seed}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"seed {seed} already has a worker", flush=True)
            return 0
        source, out = paths(seed)
        phase = "preparing"
        try:
            prepare(seed, target)
            key = Path("/home/zsgpu/.alpharesearch_llm_key").read_text().strip()
            if not key:
                raise RuntimeError("protected API credential is empty")
            child_env = dict(os.environ, ALPHARESEARCH_LLM_API_KEY=key,
                             PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
            state = json.loads((out / "resume_state.json").read_text())
            if state["round_id"] > target:
                raise ValueError("target is behind the current committed round")
            summary_path = out / "summary.json"
            already_done = (state["round_id"] == target and summary_path.exists()
                            and json.loads(summary_path.read_text()).get("status") == "completed"
                            and json.loads(summary_path.read_text()).get("protocol", {}).get("search_rounds") == target)
            if not already_done:
                cmd = [sys.executable, "-m", "alpha_research.cli", str(CONFIG)]
                for key_name, value in overrides(seed, target).items():
                    cmd += ["--set", "args." + key_name + "=" + json.dumps(value)]
                phase = "running"
                with (out / "full_run.log").open("a") as log:
                    process = subprocess.Popen(cmd, cwd=REPO, env=child_env, stdout=log, stderr=subprocess.STDOUT)
                    progress(seed, target, phase, alpha_research_pid=process.pid)
                    returncode = process.wait()
                if returncode:
                    raise RuntimeError(f"alpha_research exited with code {returncode}; see full_run.log")
            phase = "rendering"
            progress(seed, target, phase)
            metadata = render(seed, target)
            progress(seed, target, "completed", outputs=metadata["outputs"],
                     test_checkpoints=metadata["measured_top5_test_checkpoints"])
            return 0
        except Exception as exc:
            safe = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(exc))
            progress(seed, target, "failed", failed_phase=phase, error=safe)
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456])
    parser.add_argument("--target-rounds", type=int, default=100)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--worker", type=int)
    mode.add_argument("--render-only", action="store_true")
    options = parser.parse_args()
    if options.target_rounds < 38:
        parser.error("target must be at least legacy round 38")
    if options.worker is not None:
        return worker(options.worker, options.target_rounds)
    for seed in options.seeds:
        if seed not in (123, 456):
            parser.error("this launcher is scoped to legacy seeds 123 and 456")
        if options.prepare_only:
            prepare(seed, options.target_rounds)
        elif options.render_only:
            print(json.dumps(render(seed, options.target_rounds), ensure_ascii=False))
        else:
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            with (LOG_ROOT / f"seed{seed}_worker.log").open("a") as log:
                # Each seed is independent; no global wait-for-other-experiments loop.
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(seed),
                     "--target-rounds", str(options.target_rounds)],
                    cwd=REPO, start_new_session=True, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                )
            print(json.dumps({"seed": seed, "dispatched_worker_pid": process.pid}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

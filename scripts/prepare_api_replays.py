#!/usr/bin/env python3
"""Prepare measured five-round replays without modifying preserved search runs.

Uses the existing checkpoint selector/evaluator. Exact previously measured pools
and single-factor measurements are reusable; no Test value is interpolated.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from alpha_research.api.records import compile_replay
from alpha_research.api.settings import REPO_ROOT, online_args
from alpha_research.canonical import canonical_form
from alpha_research.io import append_jsonl, read_json, write_json
from alpha_research.methods.ldm_continuous_discovery.method import ContinuousDiscoveryLDM, _result_from_dict
from alpha_research.runner import build_evaluator, resolve_repo_path
from alpha_research.types import Period


class MeasuredCache:
    def __init__(self, delegate, reports, audit_path):
        self.delegate = delegate
        self.audit_path = audit_path
        self.individual = {}
        self.pools = {}
        for report in reports["checkpoints"]:
            rows = report["selected_top5_validation_order"]
            expressions = [r["candidate"]["expression"] for r in rows]
            for field in ("validation_equal_weight_rank_pool", "test_equal_weight_rank_pool"):
                result = _result_from_dict(report[field])
                if result.success:
                    self.pools[self.key(expressions, result.period)] = (result, report["checkpoint_round"])
            for row in rows:
                for field in ("validation", "test"):
                    if row.get(field) and row[field].get("success"):
                        result = _result_from_dict(row[field])
                        self.individual[self.key([row["candidate"]["expression"]], result.period)] = result

    @staticmethod
    def key(expressions, period):
        return (tuple(sorted(canonical_form(e) for e in expressions)), str(period.start), str(period.end))

    def evaluate_many(self, factors, period):
        results = []
        for factor in factors:
            key = self.key([factor.expression], period)
            if key not in self.individual:
                self.individual[key] = self.delegate.evaluate(factor.expression, period)
            result = self.individual[key]
            if not result.success:
                raise RuntimeError(f"Real single-factor evaluation failed: {factor.name}")
            results.append(result)
        return tuple(results)

    def evaluate_pool(self, factors, period):
        key = self.key([f.expression for f in factors], period)
        if key in self.pools:
            result, source_round = self.pools[key]
            reused = True
        else:
            result = self.delegate.evaluate_pool(factors, period)
            source_round, reused = None, False
            if result.success:
                self.pools[key] = (result, None)
        if not result.success:
            raise RuntimeError("Real pool evaluation failed")
        append_jsonl(self.audit_path, {"event": "measured_pool", "period": period.to_dict(),
                     "expressions": [f.expression for f in factors], "reused": reused,
                     "source_checkpoint": source_round, "result": result.to_dict()})
        return result


def prepare(seed: int):
    source = REPO_ROOT / "runs/ldm_continuous_discovery" / f"ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}"
    work = REPO_ROOT / "runtime/api/replay_build" / f"seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    source_files = ["factor_sequence.json", "resume_state.json", "top5_combinations.json", "checkpoint_validation_ledger.jsonl"]
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in source_files}
    old = read_json(work / "source_hashes.json") if (work / "source_hashes.json").exists() else None
    if old is not None and old != hashes:
        raise RuntimeError("Source results changed; use a new replay build directory")
    for name in source_files:
        if not (work / name).exists():
            shutil.copy2(source / name, work / name)
    write_json(work / "source_hashes.json", hashes)
    args = online_args(seed, 100, work)
    evaluator = MeasuredCache(build_evaluator(args, resolve_repo_path(args.alphabench_root)),
                              read_json(work / "top5_combinations.json"), work / "measurement_provenance.jsonl")
    method = ContinuousDiscoveryLDM(evaluator=evaluator, profiler=None, generator=None,
        output_dir=work, objective="rank_ic", acquisition="random", random_seed=seed,
        checkpoint_rounds=list(range(0, 101, 5)), checkpoint_factor_budget=5,
        checkpoint_validation_period=Period.from_strings(args.val_start, args.val_end),
        checkpoint_test_period=Period.from_strings(args.test_start, args.test_end),
        checkpoint_max_correlation=.8)
    def emit(event):
        append_jsonl(work / "backfill_events.jsonl", event)
        if event.get("event") == "ldm_continuous_top5_checkpoint":
            print(json.dumps({"seed": seed, "round": event["checkpoint_round"],
                  "test_rank_ic": event["test_pool"]["metrics"]["rank_ic"]}), flush=True)
    method.backfill_checkpoint_reports_from_sequence(event_sink=emit)
    result = compile_replay(work, seed)
    result["source_files_sha256"] = hashes
    result["source_run"] = str(source.relative_to(REPO_ROOT))
    for name, digest in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    output = REPO_ROOT / "demo/replays" / f"seed{seed}.json"
    write_json(output, result)
    evidence = REPO_ROOT / "demo/replay_audits" / f"seed{seed}"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("top5_combinations.json", "measurement_provenance.jsonl"):
        if not (work / name).exists():
            continue
        with (work / name).open("rb") as source_handle, (evidence / (name + ".gz")).open("wb") as target_handle:
            with gzip.GzipFile(fileobj=target_handle, mode="wb", mtime=0) as compressed:
                shutil.copyfileobj(source_handle, compressed)
    write_json(evidence / "manifest.json", {"seed": seed, "source_files_sha256": hashes,
        "report_sha256": hashlib.sha256((work / "top5_combinations.json").read_bytes()).hexdigest(),
        "replay_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "checkpoint_rounds": list(range(0, 101, 5)), "llm_called": False,
        "original_runs_modified": False})
    print(json.dumps({"seed": seed, "frames": len(result["frames"]), "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, choices=[42, 123, 456], default=[42, 123, 456])
    for seed in parser.parse_args().seeds:
        prepare(seed)

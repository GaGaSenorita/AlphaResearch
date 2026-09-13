"""Fail-closed migration of completed scalar-UCB runs; never edits the source.

Features and real results are copied, not recalculated. The Python sampler is
replayed against EVERY logged choice (including failed evaluations) before any
ledger is written. This restores local sampling state, not remote LLM determinism.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from alpha_research.io import read_json, write_json
from alpha_research.llm import effective_model_name
from alpha_research.methods.ldm.acquire import softmax_sample


SOURCE_FILES = (
    "protocol.json", "search.json", "ldm_history.csv", "events.jsonl",
    "summary.json", "validation_selection.json",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"legacy compatibility mismatch: {label}")


def validate_protocol(source: dict[str, Any], expected: dict[str, Any]) -> None:
    if source.get("metadata", {}).get("method") not in {"ldm", "ldm_standard"}:
        raise ValueError("legacy import only accepts pure ldm_standard runs")
    for key in ("environment", "train", "validation", "test", "objective", "max_correlation"):
        _equal(source.get(key), expected.get(key), key)
    old, new = source["metadata"], expected["metadata"]
    for key in ("market", "evaluator", "qlib_provider"):
        _equal(old.get(key), new.get(key), f"metadata.{key}")
    for key in ("label", "forward_n", "label_horizon_days", "fast", "topk", "n_drop", "use_cache"):
        _equal(old["ffo"].get(key), new["ffo"].get(key), f"ffo.{key}")
    for key in (
        "search_objective", "profile_schema_version", "profile_window", "profiler",
        "reference_basket", "candidates_per_round", "acquisition", "evaluate_per_round",
        "max_refill_batches", "random_seed",
    ):
        _equal(old["ldm"].get(key), new["ldm"].get(key), f"ldm.{key}")
    # The fit-policy field is checked against the live implementation and the
    # source's actual training history below. Older runs can lack this field
    # only when no GP hyperparameter training has occurred.
    old_gp, new_gp = old["ldm"].get("gp") or {}, new["ldm"].get("gp") or {}
    _equal(
        {key: value for key, value in old_gp.items() if key != "fit_policy"},
        {key: value for key, value in new_gp.items() if key != "fit_policy"},
        "ldm.gp",
    )
    for key in ("base_url", "temperature", "max_tokens", "context_window", "reasoning_effort", "thinking_enabled"):
        _equal(old["llm"].get(key), new["llm"].get(key), f"llm.{key}")
    _equal(
        effective_model_name(old["llm"]["base_url"], old["llm"]["model"]),
        effective_model_name(new["llm"]["base_url"], new["llm"]["model"]),
        "effective_model",
    )
    # Only duration, method durability/reporting and the deliverable budget may change.


def _validate_gp_fit_policy(
    source: dict[str, Any], events: list[dict[str, Any]], *,
    observation_count: int, destination_policy: str,
) -> dict[str, Any]:
    """Do not turn a historical trained GP into a new-policy exact continuation."""
    source_gp = source.get("metadata", {}).get("ldm", {}).get("gp") or {}
    source_policy = source_gp.get("fit_policy")
    if source_policy == destination_policy:
        basis = "matching_explicit_fit_policy"
    else:
        fit_events = [row for row in events if row.get("event") in {"ldm_warmup", "ldm_round"}]
        explicitly_untrained = bool(fit_events) and all(row.get("gp_trained") is False for row in fit_events)
        train_iters = source_gp.get("train_iters")
        minimum = source_gp.get("min_fit_data")
        disabled = isinstance(train_iters, (int, float)) and train_iters == 0
        below_threshold = (
            isinstance(train_iters, (int, float)) and train_iters > 0
            and isinstance(minimum, (int, float)) and observation_count < minimum
        )
        if not explicitly_untrained or not (disabled or below_threshold):
            raise ValueError(
                "legacy GP fit policy is missing or incompatible for a trained or unverifiable GP run; "
                "historical reports remain usable, but this run cannot be imported as an exact continuation"
            )
        basis = "no_hyperparameter_training_verified"
    return {
        "source_fit_policy": source_policy,
        "destination_fit_policy": destination_policy,
        "compatibility_basis": basis,
    }


def replay_sampler(events: list[dict[str, Any]], *, rounds: int, seed: int,
                   temperature: float) -> tuple[dict[int, tuple], int]:
    rng = random.Random(seed)
    states = {0: rng.getstate()}
    round_rows = [e for e in events if e.get("event") == "ldm_round"]
    _equal([int(e["round_id"]) for e in round_rows], list(range(1, rounds + 1)), "round events")
    checked = 0
    for row in round_rows:
        rid = int(row["round_id"])
        _equal(row["status"], "evaluated", "round status")
        batches = [e for e in events if e.get("event") == "ldm_refill_batch" and e.get("round_id") == rid]
        _equal([e["refill_batch"] for e in batches], list(range(1, int(row["generation_batches"]) + 1)), "refill order")
        values = np.asarray(row["acquisition"], dtype=float)
        actual = [int(i) for i in row["selected_indices"]]
        replayed: list[int] = []
        offset = 0
        for batch in batches:
            count = int(batch["profiled"])
            remaining = list(range(offset, offset + count))
            batch_choices = [i for i in actual if offset <= i < offset + count]
            for expected_index in batch_choices:
                position = softmax_sample(values[remaining], rng, temperature)
                chosen = remaining.pop(position)
                _equal(chosen, expected_index, f"sampler replay round {rid}")
                replayed.append(chosen)
                checked += 1
            offset += count
        _equal(offset, len(values), "profiled acquisition size")
        _equal(replayed, actual, "all sampler choices")
        states[rid] = rng.getstate()
    return states, checked


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def import_legacy_run(method: Any, *, train_period: Any, target_rounds: int) -> dict[str, Any]:
    from .method import LEDGER_SCHEMA, REPORT_SCHEMA, _read_jsonl
    from alpha_research.methods.ldm.method import _safe_canonical

    source = method.legacy_source
    destination = method.output_dir
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("legacy source and destination must be separate run directories")
    if method.ledger_path.exists():
        raise ValueError("refusing to replace an existing continuous ledger")
    if method.legacy_expected_protocol is None:
        raise ValueError("legacy import requires a full expected protocol")
    hashes = {name: _digest(source / name) for name in SOURCE_FILES}
    protocol = read_json(source / "protocol.json")
    validate_protocol(protocol, method.legacy_expected_protocol)
    _equal(read_json(source / "summary.json")["status"], "completed", "source completion")
    _equal(method.acquisition, "ucb", "supported legacy acquisition")
    search = read_json(source / "search.json")
    completed = int(search["rounds_requested"])
    _equal(completed, int(protocol["search_rounds"]), "completed rounds")
    if target_rounds < completed:
        raise ValueError("target is below legacy completed round")
    events = _read_jsonl(source / "events.jsonl")
    schema_events = [e for e in events if e.get("event") == "ldm_profile_schema"]
    if len(schema_events) != 1:
        raise ValueError("source must contain exactly one profile schema event")
    schema = schema_events[0]
    _equal(schema["schema"]["version"], method.profiler.schema.version, "live profile schema")
    _equal(schema["schema"]["names"], list(method.profiler.schema.names), "feature coordinate order")
    _equal(schema["reference_expressions"], list(method.profiler.reference_expressions), "frozen reference basket")
    states, choices = replay_sampler(events, rounds=completed, seed=method.random_seed,
                                    temperature=method.softmax_temperature)
    with (source / "ldm_history.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    archive = search["archive"]
    _equal(len(history), len(archive), "archive/history size")
    successful_events = [e for e in events if e.get("event") == "train_evaluation"
                         and e["result"].get("success")
                         and e["result"]["metrics"].get(method.search_objective) is not None]
    _equal([e["candidate"] for e in successful_events], [a["candidate"] for a in archive], "successful evaluation order")
    signature = method._signature(train_period)
    gp_compatibility = _validate_gp_fit_policy(
        protocol, events, observation_count=len(history),
        destination_policy=signature["gp_fit_policy"],
    )
    grouped: dict[int, list[dict[str, Any]]] = {r: [] for r in range(completed + 1)}
    for index, (hist, item, event) in enumerate(zip(history, archive, successful_events), 1):
        candidate, result = item["candidate"], item["train"]
        rid = int(candidate["round_id"])
        if rid not in grouped:
            raise ValueError("candidate lies beyond completed legacy rounds")
        _equal(hist["schema_version"], method.profiler.schema.version, "history schema")
        _equal(hist["expression"], candidate["expression"], "expression order")
        _equal(int(hist["round_id"]), rid, "round order")
        _equal(result, event["result"], "archive/event train result")
        _equal(result["period"], train_period.to_dict(), "train result period")
        canonical = _safe_canonical(candidate["expression"])
        _equal(hist["canonical"], canonical, "canonical formula")
        score = float(hist["score"])
        if not math.isfinite(score) or not math.isclose(score, float(result["metrics"][method.search_objective]), rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("history/Train score mismatch or nonfinite result")
        feature = [float(hist[f"f{i}"]) for i in range(len(method.profiler.schema.names))]
        if not np.isfinite(feature).all():
            raise ValueError("nonfinite legacy profile")
        grouped[rid].append({
            "schema_version": LEDGER_SCHEMA, "record_type": "factor_evaluation",
            "attempt_id": f"legacy-round-{rid:06d}", "round_id": rid,
            "evaluation_index": index, "selection_order_within_round": len(grouped[rid]) + 1,
            "refill_batch": event.get("refill_batch"), "candidate": candidate,
            "canonical": canonical, "feature": feature, "search_objective": method.search_objective,
            "search_score": score, "train": result, "origin": "legacy_import",
        })
    _equal([a["candidate"]["round_id"] for a in archive], sorted(a["candidate"]["round_id"] for a in archive), "chronological history")
    if not grouped[0]:
        raise ValueError("missing legacy seed observations")
    ledger: list[dict[str, Any]] = []
    total = 0
    for rid, factors in grouped.items():
        if rid:
            _equal(len(factors), method.evaluate_per_round, f"verified count round {rid}")
        ledger.extend(factors)
        total += len(factors)
        commit = {
            "schema_version": LEDGER_SCHEMA, "record_type": "round_commit",
            "attempt_id": f"legacy-round-{rid:06d}", "round_id": rid,
            "target_round": int(target_rounds), "observation_count": len(factors),
            "history_observations": total, "rng_state": states[rid], "signature": signature,
            "origin": "legacy_import", "imported_at_unix": time.time(),
            "rng_state_origin": "replayed_and_verified_against_all_logged_choices",
            "source_gp_compatibility": gp_compatibility,
        }
        # Per-round LLM call totals were not recorded by the old method. Do not invent them.
        if rid == completed:
            commit["llm_calls_total"] = int(search["llm_calls"])
        elif rid == 0:
            commit["llm_calls_total"] = 0
        ledger.append(commit)

    validation_rows = []
    by_expression = {a["candidate"]["expression"]: a["candidate"] for a in archive}
    for item in read_json(source / "validation_selection.json")["evaluations"]:
        candidate, result = item["candidate"], item["validation"]
        _equal(candidate, by_expression.get(candidate["expression"]), "validation candidate")
        _equal(result["period"], method.checkpoint_validation_period.to_dict(), "validation period")
        validation_rows.append({
            "schema_version": REPORT_SCHEMA, "checkpoint_round_first_requested": completed,
            "period": method.checkpoint_validation_period.to_dict(),
            "canonical": _safe_canonical(candidate["expression"]),
            "candidate": candidate, "validation": result, "origin": "legacy_import",
        })
    manifest = {
        "schema_version": "alphaldm.legacy-import.v1", "source_dir": str(source),
        "source_sha256": hashes, "source_completed_round": completed,
        "source_factor_count": len(archive), "sampler_choices_verified": choices,
        "rng_policy": "replayed_and_verified", "llm_deterministic_reproduction_claimed": False,
        "source_factor_budget": protocol["factor_budget"],
        "new_factor_budget": method.checkpoint_factor_budget,
        "history_order_preserved": True, "train_observations_reevaluated": False,
        "validation_imported": len(validation_rows), "source_unchanged": True,
        "gp_compatibility": gp_compatibility,
        "imported_at_unix": time.time(),
    }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".legacy-import-", dir=destination) as temporary:
        staging = Path(temporary)
        preserved = staging / "legacy_source"
        preserved.mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(source / name, preserved / name)
            _equal(_digest(preserved / name), hashes[name], f"source copy hash {name}")
            _equal(_digest(source / name), hashes[name], f"source changed during import {name}")
        _write_jsonl(staging / "factor_ledger.jsonl", ledger)
        _write_jsonl(staging / "checkpoint_validation_ledger.jsonl", validation_rows)
        existing_copy = destination / "legacy_source"
        if existing_copy.exists():
            for name in SOURCE_FILES:
                _equal(_digest(existing_copy / name), hashes[name], "previous import source hash")
        else:
            os.replace(preserved, existing_copy)
        write_json(destination / "legacy_import.json", manifest)
        write_json(destination / "resume_state.json", ledger[-1])
        os.replace(staging / "checkpoint_validation_ledger.jsonl", destination / "checkpoint_validation_ledger.jsonl")
        # Publish a complete ledger last: interruption cannot expose a half-imported history.
        os.replace(staging / "factor_ledger.jsonl", method.ledger_path)
    return manifest

"""Small, measured presentation records shared by replay and live workers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "alphaldm.api.snapshot.v1"


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def factor_view(row: dict) -> dict:
    candidate = row["candidate"]
    return {"name": candidate["name"], "expression": candidate["expression"],
            "reason": candidate.get("reason", ""),
            "round": int(row.get("round_id", candidate.get("round_id", 0))),
            "evaluation_index": row.get("evaluation_index"),
            "train_rank_ic": number(row.get("train_metrics", {}).get("rank_ic"))}


def checkpoint_view(record: dict) -> dict:
    pool = record.get("test_equal_weight_rank_pool") or {}
    val_pool = record.get("validation_equal_weight_rank_pool") or {}
    test = number(pool.get("metrics", {}).get("rank_ic")) if pool.get("success") else None
    val = number(val_pool.get("metrics", {}).get("rank_ic")) if val_pool.get("success") else None
    selected = []
    for row in record.get("selected_top5_validation_order", []):
        selected.append({**factor_view(row), "validation_rank": row.get("validation_rank"),
                         "validation_rank_ic": number((row.get("validation") or {}).get("metrics", {}).get("rank_ic")),
                         "test_rank_ic": number((row.get("test") or {}).get("metrics", {}).get("rank_ic"))})
    return {"round": int(record["checkpoint_round"]), "rank_ic": test,
            "validation_rank_ic": val, "measured": test is not None,
            "qualified": test is not None and test >= .035,
            "top5": selected, "factor_count": record.get("available_factor_count")}


def snapshot(sequence: dict, reports: dict, *, committed_round: int, target: int,
             step: int = 5) -> dict:
    rows = sorted((r for r in sequence.get("factors", [])
                   if int(r["round_id"]) <= committed_round), key=lambda r: int(r["evaluation_index"]))
    factors = [factor_view(r) for r in rows]
    valid = [f for f in factors if f["train_rank_ic"] is not None]
    leaderboard = sorted(valid, key=lambda f: -f["train_rank_ic"])[:5]
    display_round = max(0, committed_round // step * step)
    if committed_round >= target:
        display_round = target
    rounds = list(range(0, display_round + 1, step))
    if display_round not in rounds:
        rounds.append(display_round)
    train = []
    for round_id in rounds:
        candidates = [f for f in valid if f["round"] <= round_id]
        if candidates:
            leader = max(candidates, key=lambda f: f["train_rank_ic"])
            train.append({"round": round_id, "rank_ic": leader["train_rank_ic"], "factor": leader})
    test = sorted((checkpoint_view(r) for r in reports.get("checkpoints", [])
                   if int(r["checkpoint_round"]) <= display_round and
                   (int(r["checkpoint_round"]) % step == 0 or int(r["checkpoint_round"]) == target)),
                  key=lambda r: r["round"])
    envelope = []
    best = None
    for point in test:
        if point["rank_ic"] is not None:
            best = point["rank_ic"] if best is None else max(best, point["rank_ic"])
            envelope.append({"round": point["round"], "rank_ic": best})
    measured = [p for p in test if p["measured"]]
    latest = measured[-1] if measured else None
    return {"schema": SCHEMA, "committed_round": max(0, committed_round),
            "display_round": display_round, "target_rounds": target,
            "factor_count": len(factors), "gp_observations": len(factors),
            "train": train, "test": test, "test_best_so_far": envelope,
            "best_factor": leaderboard[0] if leaderboard else None,
            "leaderboard": leaderboard, "recent_factors": factors[-6:][::-1],
            "latest_test": latest, "top5": latest["top5"] if latest else [],
            "threshold": .035,
            "test_policy": "Validation selects Top-5; Test is reporting only. The best-so-far envelope is retrospective."}


def compile_replay(run_dir: Path, seed: int) -> dict:
    sequence = read_json(run_dir / "factor_sequence.json")
    reports = read_json(run_dir / "top5_combinations.json")
    state = read_json(run_dir / "resume_state.json")
    if not sequence or not reports or not state:
        raise ValueError("Replay requires factor sequence, reports, and resume state")
    if int(state["round_id"]) != 100 or int(sequence["completed_round"]) != 100:
        raise ValueError("Replay must be based on a committed 100-round run")
    factors = sequence["factors"]
    if len(factors) != 342 or [int(r["evaluation_index"]) for r in factors] != list(range(1, 343)):
        raise ValueError("Replay factor order/count does not match 42 + 100 * 3")
    expected = set(range(0, 101, 5))
    checkpoints = {p["round"]: p for p in map(checkpoint_view, reports["checkpoints"])}
    missing = sorted(r for r in expected if r not in checkpoints or not checkpoints[r]["measured"])
    if missing:
        raise ValueError(f"Missing measured Test checkpoints: {missing}")
    for r in expected:
        p = checkpoints[r]
        if len(p["top5"]) != 5 or any(f["round"] > r for f in p["top5"]):
            raise ValueError(f"Invalid or future factor in R{r} selection")
    return {"schema": "alphaldm.api.replay.v1", "seed": seed,
            "source": "measured historical run; missing checkpoints retrospectively evaluated",
            "target_rounds": 100, "checkpoint_step": 5, "factors": [factor_view(row) for row in factors],
            "frames": [snapshot(sequence, reports, committed_round=r, target=100) for r in range(0, 101, 5)]}

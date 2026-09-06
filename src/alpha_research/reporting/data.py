"""Shared checkpoint extraction without evaluator or search calls."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


CHECKPOINT_STEP = 5


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def short_name(name: str, max_chars: int = 20) -> str:
    parts = str(name).split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + "_" + part
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > 2:
        lines = [lines[0], "_".join(lines[1:])]
    return "\n".join(lines)



def train_leaders_at_rounds(
    sequence: list[dict[str, Any]],
    rounds: list[int],
) -> list[dict[str, Any]]:
    """Return the best observed Train candidate available at each requested round."""
    leaders: list[dict[str, Any]] = []
    for round_id in rounds:
        eligible = [
            row for row in sequence
            if int(row["round_id"]) <= round_id
        ]
        if not eligible:
            continue
        leader = max(eligible, key=lambda row: float(row["search_score"]))
        leaders.append({
            "round": round_id,
            "factor_round": int(leader["round_id"]),
            "name": str(leader["candidate"]["name"]),
            "rank_ic": float(leader["search_score"]),
        })
    return leaders



def retrospective_test_best_so_far(
    rounds: list[int],
    values: list[float],
) -> list[dict[str, Any]]:
    """Build a labelled cumulative Test envelope without hiding raw audits.

    This is presentation-only post-processing. ``source_round`` makes the
    hindsight selection explicit and prevents the envelope from being mistaken
    for a metric observed at every later checkpoint or used by the search.
    """
    if len(rounds) != len(values):
        raise ValueError("Test checkpoint rounds and values must have equal length")
    records: list[dict[str, Any]] = []
    best_value = float("-inf")
    source_round: int | None = None
    for round_id, raw_value in zip(rounds, values):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite Test RankIC at checkpoint R{round_id}")
        improved = value > best_value
        if improved:
            best_value = value
            source_round = int(round_id)
        records.append({
            "round": int(round_id),
            "rank_ic": best_value,
            "source_round": source_round,
            "raw_rank_ic": value,
            "improved": improved,
        })
    return records



def regular_checkpoint_rows(
    checkpoints: list[dict[str, Any]],
    snapshot_round: int,
    step: int = CHECKPOINT_STEP,
    *,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    """Select the regular reporting grid while retaining legacy audits on disk."""
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    requested = set(range(0, snapshot_round + 1, step))
    requested.add(snapshot_round)
    selected = [
        row for row in sorted(checkpoints, key=lambda item: item["checkpoint_round"])
        if int(row["checkpoint_round"]) in requested
    ]
    rounds = [int(row["checkpoint_round"]) for row in selected]
    if len(set(rounds)) != len(rounds):
        raise ValueError("duplicate checkpoint rounds in reporting grid")
    if require_complete and set(rounds) != requested:
        raise ValueError(f"missing measured checkpoints: {sorted(requested - set(rounds))}")
    return selected



def measured_test_values(checkpoints: list[dict[str, Any]]) -> list[float]:
    """Reject failed or non-finite audits rather than silently plotting a placeholder."""
    values = []
    for row in checkpoints:
        result = row["test_equal_weight_rank_pool"]
        value = result.get("metrics", {}).get("rank_ic")
        if not result.get("success") or value is None or not math.isfinite(float(value)):
            raise ValueError(f"invalid measured Test audit at R{row['checkpoint_round']}")
        values.append(float(value))
    return values

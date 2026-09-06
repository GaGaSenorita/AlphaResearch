#!/usr/bin/env python3
"""Draw the conference-poster finance panel from a committed LDM snapshot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PURPLE = "#6D43C7"
PURPLE_DARK = "#48218B"
TEAL = "#159E9A"
ORANGE = "#E89B28"
GREEN = "#2A9D68"
NAVY = "#17223B"
MUTED = "#72768A"
RAW_TEST = "#9A96A6"
GRID = "#DDD9E7"
PAPER = "#FBFAF7"
CHECKPOINT_STEP = 5


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def daily_series(row: dict[str, Any]) -> dict[str, float]:
    result = row["validation"]
    return {
        str(item["date"]): float(item["rank_ic"])
        for item in result.get("daily_metrics", [])
        if item.get("date") is not None and item.get("rank_ic") is not None
    }


def correlation(left: dict[str, float], right: dict[str, float]) -> float | None:
    shared = sorted(left.keys() & right.keys())
    if len(shared) < 30:
        return None
    xs = np.asarray([left[day] for day in shared], dtype=float)
    ys = np.asarray([right[day] for day in shared], dtype=float)
    if float(np.std(xs)) <= 1e-12 or float(np.std(ys)) <= 1e-12:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def select_top5(
    validation_rows: list[dict[str, Any]],
    round_id: int,
    max_correlation: float = 0.8,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            row
            for row in validation_rows
            if row["validation"].get("success")
            and int(row["candidate"].get("round_id", 0)) <= round_id
        ),
        key=lambda row: float(row["validation"]["metrics"]["rank_ic"]),
        reverse=True,
    )
    chosen: list[dict[str, Any]] = []
    chosen_series: list[dict[str, float]] = []
    for row in ranked:
        series = daily_series(row)
        correlations = [correlation(series, existing) for existing in chosen_series]
        finite = [abs(value) for value in correlations if value is not None]
        if finite and max(finite) > max_correlation:
            continue
        chosen.append(row)
        chosen_series.append(series)
        if len(chosen) == 5:
            break
    return chosen


def validation_entry_events(
    validation_rows: list[dict[str, Any]],
    last_validated_round: int,
) -> list[dict[str, Any]]:
    previous: list[str] = []
    events: list[dict[str, Any]] = []
    for round_id in range(last_validated_round + 1):
        selected = select_top5(validation_rows, round_id)
        names = [row["candidate"]["name"] for row in selected]
        if names == previous:
            continue
        for row in selected:
            name = row["candidate"]["name"]
            if name not in previous:
                events.append({
                    "round": round_id,
                    "name": name,
                    "kind": "validation",
                    "score": float(row["validation"]["metrics"]["rank_ic"]),
                })
        previous = names
    return events


def train_leader_events(sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = float("-inf")
    events: list[dict[str, Any]] = []
    for row in sequence:
        score = float(row["search_score"])
        round_id = int(row["round_id"])
        if round_id > 0 and score > best:
            events.append({
                "round": round_id,
                "name": row["candidate"]["name"],
                "kind": "train",
                "score": score,
            })
        best = max(best, score)
    return events


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


def staged_single_test_results(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect each staged Top-5 constituent's first recorded single-factor Test result."""
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_round = int(checkpoint["checkpoint_round"])
        for row in checkpoint["selected_top5_validation_order"]:
            name = str(row["candidate"]["name"])
            if name in seen:
                continue
            seen.add(name)
            results.append({
                "first_test_checkpoint": checkpoint_round,
                "factor_round": int(row["candidate"]["round_id"]),
                "name": name,
                "source": str(row["candidate"].get("source", "unknown")),
                "validation_rank_ic": float(row["validation"]["metrics"]["rank_ic"]),
                "test_rank_ic": float(row["test"]["metrics"]["rank_ic"]),
            })
    return results


def add_pipeline(ax: plt.Axes, snapshot_round: int, factor_count: int) -> None:
    ax.set_axis_off()
    steps = [
        ("LLM proposals", "formula hypotheses", PURPLE),
        ("Behavior profile", "label-free fingerprint", TEAL),
        ("GP + UCB", "surrogate selection", ORANGE),
        ("Real evaluation", "Train RankIC", GREEN),
    ]
    xs = [0.015, 0.265, 0.515, 0.765]
    width = 0.205
    for index, ((title, subtitle, color), x) in enumerate(zip(steps, xs)):
        box = FancyBboxPatch(
            (x, 0.20), width, 0.62,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            linewidth=1.5, edgecolor=color, facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2, 0.58, title,
            ha="center", va="center", fontsize=11.2, fontweight="bold",
            color=NAVY, transform=ax.transAxes,
        )
        ax.text(
            x + width / 2, 0.35, subtitle,
            ha="center", va="center", fontsize=8.5,
            color=MUTED, transform=ax.transAxes,
        )
        ax.scatter(
            [x + 0.025], [0.70], s=50, color=color, zorder=5,
            transform=ax.transAxes,
        )
        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.008, 0.51), (xs[index + 1] - 0.010, 0.51),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
                color="#9893A5", transform=ax.transAxes,
            )
            ax.add_patch(arrow)
    feedback = FancyArrowPatch(
        (0.865, 0.18), (0.115, 0.17),
        connectionstyle="arc3,rad=-0.14", arrowstyle="-|>",
        mutation_scale=11, linewidth=1.2, linestyle="--",
        color=PURPLE, alpha=0.72, transform=ax.transAxes,
    )
    ax.add_patch(feedback)
    ax.text(
        0.49, 0.02,
        f"continuous feedback  •  snapshot: round {snapshot_round}  •  {factor_count} real evaluations",
        ha="center", va="center", fontsize=8.7, color=PURPLE_DARK,
        transform=ax.transAxes,
    )


def draw_panel(data_dir: Path, output_prefix: Path) -> dict[str, Any]:
    state = load_json(data_dir / "resume_state.json")
    factor_payload = load_json(data_dir / "factor_sequence.json")
    report = load_json(data_dir / "top5_combinations.json")
    validation_rows = load_jsonl(data_dir / "checkpoint_validation_ledger.jsonl")

    snapshot_round = int(state["round_id"])
    factor_count = int(factor_payload["factor_count"])
    sequence = list(factor_payload["factors"])
    all_checkpoints = sorted(
        report["checkpoints"], key=lambda row: row["checkpoint_round"]
    )
    checkpoints = regular_checkpoint_rows(all_checkpoints, snapshot_round, require_complete=True)
    checkpoint_rounds = [int(row["checkpoint_round"]) for row in checkpoints]
    checkpoint_values = measured_test_values(checkpoints)
    test_best_records = retrospective_test_best_so_far(
        checkpoint_rounds, checkpoint_values
    )
    test_best_values = [row["rank_ic"] for row in test_best_records]
    last_validated_round = max(
        int(row["checkpoint_round_first_requested"]) for row in validation_rows
    )

    validation_events = validation_entry_events(validation_rows, last_validated_round)
    train_events = train_leader_events(sequence)
    validation_by_key = {
        (event["round"], event["name"]): event for event in validation_events
    }
    important: list[dict[str, Any]] = []
    for event in validation_events:
        if event["round"] == 0:
            continue
        important.append(event)
    for event in train_events:
        key = (event["round"], event["name"])
        if key in validation_by_key:
            validation_by_key[key]["kind"] = "validation + train leader"
        elif event["round"] >= 15:
            important.append(event)

    # The newest post-validation leader is useful but must remain explicitly Train-only.
    post_validation = [
        row for row in sequence if int(row["round_id"]) > last_validated_round
    ]
    if post_validation:
        newest_leader = max(post_validation, key=lambda row: float(row["search_score"]))
        key = (int(newest_leader["round_id"]), newest_leader["candidate"]["name"])
        if not any((event["round"], event["name"]) == key for event in important):
            important.append({
                "round": key[0],
                "name": key[1],
                "kind": "train",
                "score": float(newest_leader["search_score"]),
            })

    important.sort(key=lambda event: (event["round"], event["name"]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(12, 7.5), facecolor=PAPER)
    border = FancyBboxPatch(
        (0.012, 0.012), 0.976, 0.976,
        boxstyle="round,pad=0.004,rounding_size=0.018",
        linewidth=2.0, edgecolor="#BDB5D3", facecolor=PAPER,
        transform=fig.transFigure, zorder=-10,
    )
    fig.patches.append(border)

    fig.text(
        0.055, 0.953, "LDM financial factor discovery trajectory",
        fontsize=18.5, color=PURPLE_DARK, fontweight="bold", va="top",
    )
    best_value = max(checkpoint_values)
    fig.text(
        0.945, 0.953,
        f"best measured Test RankIC  {best_value:.4f}",
        fontsize=11.4, color=TEAL, fontweight="bold", va="top", ha="right",
    )

    pipeline_ax = fig.add_axes([0.055, 0.775, 0.89, 0.145])
    add_pipeline(pipeline_ax, snapshot_round, factor_count)

    ax = fig.add_axes([0.075, 0.145, 0.875, 0.56], facecolor="white")
    ax.set_xlim(-2, max(100, snapshot_round + 3))
    y_min = min(0.0342, min(checkpoint_values) - 0.0010)
    y_max = max(0.0447, max(checkpoint_values) + 0.0048)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A29CAD")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("LDM round / cumulative real-evaluation budget", fontsize=10.5, color=NAVY)
    ax.set_ylabel("Top-5 Test RankIC", fontsize=10.5, color=NAVY)
    ax.set_xticks(list(range(0, 101, CHECKPOINT_STEP)))

    threshold = float(report["qualification"]["threshold"])
    ax.axhline(threshold, color="#AFA9B9", linewidth=1.25, linestyle=(0, (4, 3)))
    ax.text(
        99, threshold + 0.00013, f"record threshold  {threshold:.3f}",
        ha="right", va="bottom", fontsize=8.7, color=MUTED,
    )

    ax.plot(
        checkpoint_rounds, checkpoint_values,
        color=RAW_TEST, linewidth=1.2, linestyle=(0, (2, 3)),
        marker="o", markersize=5.3, markerfacecolor="white",
        markeredgewidth=1.4, alpha=0.82, zorder=4,
        label="Raw measured Test checkpoint",
    )
    ax.step(
        checkpoint_rounds, test_best_values, where="post",
        color=TEAL, linewidth=3.0, zorder=5,
        label="Retrospective Test best-so-far · diagnostic only",
    )
    ax.fill_between(
        checkpoint_rounds, test_best_values, threshold,
        step="post", color=TEAL, alpha=0.09, zorder=1,
    )
    ax.scatter(
        checkpoint_rounds, test_best_values,
        s=92, marker="o", color="white", edgecolor=TEAL,
        linewidth=2.4, zorder=6,
    )
    for index, record in enumerate(test_best_records):
        round_id = int(record["round"])
        value = float(record["raw_rank_ic"])
        offset = (0, 13) if index % 2 == 0 else (0, -18)
        vertical = "bottom" if index % 2 == 0 else "top"
        color = TEAL if record["improved"] else MUTED
        ax.annotate(
            f"R{round_id}\n{value:.4f}",
            (round_id, value), xytext=offset, textcoords="offset points",
            ha="center", va=vertical, fontsize=8.3, color=color,
            fontweight="bold",
        )

    # Progress rail: every dot is a real completed LDM round, not an interpolated metric.
    rail_y = y_min + 0.00035
    ax.hlines(rail_y, 0, snapshot_round, color="#D8D3E2", linewidth=5, zorder=1)
    completed = np.arange(0, snapshot_round + 1)
    ax.scatter(completed, np.full_like(completed, rail_y, dtype=float), s=5, color="#B8B1C8")
    ax.scatter([snapshot_round], [rail_y], s=54, color=PURPLE, zorder=5)
    ax.text(
        snapshot_round, rail_y + 0.00024,
        f"snapshot R{snapshot_round} · {factor_count} factors",
        ha="right", va="bottom", fontsize=8.5, color=PURPLE_DARK,
        fontweight="bold",
    )

    label_levels = [y_max - 0.00055, y_max - 0.0020, y_max - 0.00345]
    last_at_level = [-100.0, -100.0, -100.0]
    preferred_label_x = {
        0: 3.0,
        3: 8.0,
        8: 14.0,
        21: 21.0,
        33: 31.0,
        36: 39.0,
        58: 58.0,
        89: 89.0,
    }
    plotted_events: list[dict[str, Any]] = []
    for event in important:
        round_id = int(event["round"])
        if round_id > snapshot_round:
            continue
        # Keep the annotation lane legible by choosing the least recently used level.
        candidates = sorted(
            range(len(label_levels)),
            key=lambda index: (round_id - last_at_level[index] < 16, last_at_level[index]),
        )
        level = candidates[0]
        last_at_level[level] = round_id
        label_x = preferred_label_x.get(round_id, float(round_id))
        kind = event["kind"]
        color = PURPLE if kind.startswith("validation") else ORANGE
        marker = "D" if kind.startswith("validation") else "^"
        anchor_y = max(checkpoint_values) + 0.00032
        ax.scatter(
            [round_id], [anchor_y], s=44, marker=marker,
            color=color, edgecolor="white", linewidth=0.8, zorder=7,
        )
        ax.annotate(
            f"R{round_id}  {short_name(event['name'])}",
            xy=(round_id, anchor_y), xycoords="data",
            xytext=(label_x, label_levels[level]), textcoords="data",
            ha="center", va="top", fontsize=7.25, color=NAVY,
            fontweight="semibold",
            arrowprops={
                "arrowstyle": "-", "color": color, "linewidth": 0.9,
                "alpha": 0.52, "shrinkA": 1.5, "shrinkB": 3.0,
            },
            bbox={
                "boxstyle": "round,pad=0.24,rounding_size=0.2",
                "facecolor": "white", "edgecolor": color,
                "linewidth": 0.9, "alpha": 0.97,
            },
            zorder=8,
        )
        plotted_events.append(event)

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color=TEAL, linewidth=2.8,
               label="Retrospective Test best-so-far"),
        Line2D([0], [0], color=RAW_TEST, marker="o", markerfacecolor="white",
               linewidth=1.2, linestyle="--", label="Raw measured Test Top-5"),
        Line2D([0], [0], color=PURPLE, marker="D", linewidth=0,
               label="New Validation Top-5 factor"),
        Line2D([0], [0], color=ORANGE, marker="^", linewidth=0,
               label="Train-leading candidate (not yet Test-selected)"),
    ]
    fig.legend(
        handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, 0.035),
        ncol=4, frameon=False, fontsize=8.0, handlelength=2.2,
        columnspacing=1.8,
    )
    fig.text(
        0.94, 0.724,
        "Validation selects; raw Test reports only · best-so-far is retrospective",
        ha="right", va="bottom", fontsize=7.8,
        color=MUTED,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_prefix.with_suffix(".png"),
        "svg": output_prefix.with_suffix(".svg"),
        "pdf": output_prefix.with_suffix(".pdf"),
    }
    fig.savefig(paths["png"], dpi=240, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["svg"], facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["pdf"], facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    metadata = {
        "snapshot_round": snapshot_round,
        "factor_count": factor_count,
        "checkpoint_rounds": checkpoint_rounds,
        "checkpoint_test_rank_ic": checkpoint_values,
        "retrospective_test_best_so_far": test_best_records,
        "important_events": plotted_events,
        "data_dir": str(data_dir.resolve()),
        "outputs": {kind: str(path.resolve()) for kind, path in paths.items()},
    }
    metadata_path = output_prefix.with_suffix(".json")
    metadata["outputs"]["metadata"] = str(metadata_path.resolve())
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def draw_trajectory_only(data_dir: Path, output_prefix: Path) -> dict[str, Any]:
    """Draw the poster-ready trajectory chart without the surrounding explainer."""
    state = load_json(data_dir / "resume_state.json")
    factor_payload = load_json(data_dir / "factor_sequence.json")
    report = load_json(data_dir / "top5_combinations.json")

    snapshot_round = int(state["round_id"])
    factor_count = int(factor_payload["factor_count"])
    sequence = list(factor_payload["factors"])
    all_checkpoints = sorted(
        report["checkpoints"], key=lambda row: row["checkpoint_round"]
    )
    checkpoints = regular_checkpoint_rows(all_checkpoints, snapshot_round, require_complete=True)
    test_rounds = [int(row["checkpoint_round"]) for row in checkpoints]
    test_values = measured_test_values(checkpoints)
    test_best_records = retrospective_test_best_so_far(test_rounds, test_values)
    test_best_values = [row["rank_ic"] for row in test_best_records]

    milestone_rounds = list(range(0, snapshot_round + 1, CHECKPOINT_STEP))
    if not milestone_rounds or milestone_rounds[-1] != snapshot_round:
        milestone_rounds.append(snapshot_round)
    train_leaders = train_leaders_at_rounds(sequence, milestone_rounds)
    train_rounds = [row["round"] for row in train_leaders]
    train_values = [row["rank_ic"] for row in train_leaders]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(15.5, 7.6), facecolor=PAPER)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.17, top=0.94)
    ax.set_facecolor("white")

    border = FancyBboxPatch(
        (0.008, 0.012), 0.984, 0.976,
        boxstyle="round,pad=0.004,rounding_size=0.016",
        linewidth=1.8, edgecolor="#BDB5D3", facecolor="none",
        transform=fig.transFigure, zorder=20,
    )
    fig.patches.append(border)

    ax.set_xlim(-2, 101)
    y_min = min(0.0337, min(test_values) - 0.0012)
    label_levels = [0.0670, 0.0712, 0.0754]
    y_max = max(label_levels) + 0.0022
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(list(range(0, 101, CHECKPOINT_STEP)))
    ax.grid(axis="y", color=GRID, linewidth=0.85, alpha=0.78)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.42)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A29CAD")
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.set_xlabel("LDM round", fontsize=11.5, color=NAVY, labelpad=10)
    ax.set_ylabel("RankIC", fontsize=11.5, color=NAVY)

    threshold = float(report["qualification"]["threshold"])
    ax.axhline(
        threshold, color="#AFA9B9", linewidth=1.35,
        linestyle=(0, (4, 3)), zorder=1,
    )
    ax.text(
        99.5, threshold + 0.00028,
        f"record threshold  {threshold:.3f}",
        ha="right", va="bottom", fontsize=8.8, color=MUTED,
    )

    # Train uses only committed real evaluations and is sampled at R0/R5/.../snapshot.
    ax.plot(
        train_rounds, train_values,
        color=ORANGE, linewidth=2.5, marker="o", markersize=6.6,
        markerfacecolor="white", markeredgewidth=2.0,
        zorder=5, label="Train best-so-far · every 5 rounds",
    )
    ax.fill_between(
        train_rounds, train_values, threshold,
        color=ORANGE, alpha=0.035, zorder=0,
    )

    # Keep every actual Test audit visible. The strong line is a labelled
    # retrospective envelope, never a search or checkpoint-selection signal.
    ax.plot(
        test_rounds, test_values,
        color=RAW_TEST, linewidth=1.2, linestyle=(0, (2, 3)),
        marker="o", markersize=5.6, markerfacecolor="white",
        markeredgewidth=1.4, alpha=0.85, zorder=6,
        label="Raw measured Validation-selected Top-5 Test",
    )
    ax.step(
        test_rounds, test_best_values, where="post",
        color=TEAL, linewidth=2.9, zorder=7,
        label="Retrospective Test best-so-far · diagnostic only",
    )
    for index, record in enumerate(test_best_records):
        round_id = int(record["round"])
        value = float(record["raw_rank_ic"])
        above = index % 2 == 0
        ax.annotate(
            f"R{round_id}  {value:.4f}",
            (round_id, value), xytext=(0, 12 if above else -15),
            textcoords="offset points", ha="center",
            va="bottom" if above else "top", fontsize=8.4,
            color=TEAL if record["improved"] else MUTED,
            fontweight="bold", zorder=9,
        )

    # Every requested milestone has a visible point and its then-current leading factor.
    for index, leader in enumerate(train_leaders):
        round_id = int(leader["round"])
        value = float(leader["rank_ic"])
        label_y = label_levels[index % len(label_levels)]
        label_x = min(96.0, max(4.0, float(round_id)))
        ax.annotate(
            f"R{round_id} · {value:.4f}\n{short_name(leader['name'], max_chars=17)}",
            xy=(round_id, value), xycoords="data",
            xytext=(label_x, label_y), textcoords="data",
            ha="center", va="center", fontsize=7.15,
            color=NAVY, fontweight="semibold",
            arrowprops={
                "arrowstyle": "-", "color": ORANGE,
                "linewidth": 0.95, "alpha": 0.58,
                "shrinkA": 2.0, "shrinkB": 4.0,
            },
            bbox={
                "boxstyle": "round,pad=0.25,rounding_size=0.18",
                "facecolor": "white", "edgecolor": ORANGE,
                "linewidth": 1.0, "alpha": 0.98,
            },
            zorder=10,
        )

    ax.axvline(
        snapshot_round, color=PURPLE, linewidth=1.0,
        linestyle=(0, (2, 4)), alpha=0.55, zorder=2,
    )
    ax.text(
        snapshot_round - 0.8, 0.0422,
        f"snapshot cutoff\nR{snapshot_round} · {factor_count} saved factors",
        ha="right", va="center", fontsize=8.5,
        color=PURPLE_DARK, fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.18",
            "facecolor": "white", "edgecolor": PURPLE,
            "linewidth": 0.9, "alpha": 0.96,
        },
    )
    ax.text(
        0.995, 1.018,
        "Validation selects; raw Test reports only · best-so-far is retrospective",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=MUTED,
    )

    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.13),
        ncol=2, frameon=False, fontsize=9.2,
        handlelength=2.8, columnspacing=2.3,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_prefix.with_suffix(".png"),
        "svg": output_prefix.with_suffix(".svg"),
        "pdf": output_prefix.with_suffix(".pdf"),
    }
    fig.savefig(paths["png"], dpi=240, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["svg"], facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["pdf"], facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    metadata = {
        "snapshot_round": snapshot_round,
        "factor_count": factor_count,
        "train_milestones": train_leaders,
        "measured_test_checkpoints": [
            {"round": round_id, "rank_ic": value}
            for round_id, value in zip(test_rounds, test_values)
        ],
        "retrospective_test_best_so_far": test_best_records,
        "threshold": threshold,
        "data_dir": str(data_dir.resolve()),
        "outputs": {kind: str(path.resolve()) for kind, path in paths.items()},
    }
    metadata_path = output_prefix.with_suffix(".json")
    metadata["outputs"]["metadata"] = str(metadata_path.resolve())
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def draw_trajectory_with_single_test(
    data_dir: Path,
    output_prefix: Path,
    show_single_test: bool = True,
    illustrative_r100: float | None = None,
    illustrative_solid: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Draw aligned Train and Test panels, including staged single-factor Test audits."""
    state = load_json(data_dir / "resume_state.json")
    factor_payload = load_json(data_dir / "factor_sequence.json")
    report = load_json(data_dir / "top5_combinations.json")

    snapshot_round = int(state["round_id"])
    factor_count = int(factor_payload["factor_count"])
    sequence = list(factor_payload["factors"])
    all_checkpoints = sorted(
        report["checkpoints"], key=lambda row: row["checkpoint_round"]
    )
    checkpoints = regular_checkpoint_rows(all_checkpoints, snapshot_round, require_complete=True)
    test_rounds = [int(row["checkpoint_round"]) for row in checkpoints]
    test_values = measured_test_values(checkpoints)
    test_best_records = retrospective_test_best_so_far(test_rounds, test_values)
    test_best_values = [row["rank_ic"] for row in test_best_records]
    single_results = staged_single_test_results(checkpoints)

    milestone_rounds = list(range(0, snapshot_round + 1, CHECKPOINT_STEP))
    if not milestone_rounds or milestone_rounds[-1] != snapshot_round:
        milestone_rounds.append(snapshot_round)
    train_leaders = train_leaders_at_rounds(sequence, milestone_rounds)
    train_rounds = [row["round"] for row in train_leaders]
    train_values = [row["rank_ic"] for row in train_leaders]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, (train_ax, test_ax) = plt.subplots(
        2, 1, figsize=(15.5, 9.2), sharex=True, facecolor=PAPER,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.075},
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.145, top=0.95)
    fig.suptitle(
        "LDM × Financial Factor Discovery" + (f" · Seed {seed}" if seed is not None else ""),
        x=0.075, y=0.985, ha="left", va="top",
        fontsize=17.5, fontweight="bold", color=PURPLE_DARK,
    )
    border = FancyBboxPatch(
        (0.008, 0.010), 0.984, 0.98,
        boxstyle="round,pad=0.004,rounding_size=0.016",
        linewidth=1.8, edgecolor="#BDB5D3", facecolor="none",
        transform=fig.transFigure, zorder=20,
    )
    fig.patches.append(border)

    for ax in (train_ax, test_ax):
        ax.set_facecolor("white")
        ax.set_xlim(-2, 101)
        ax.grid(axis="y", color=GRID, linewidth=0.85, alpha=0.78)
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.42)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#A29CAD")
        ax.tick_params(colors=MUTED, labelsize=9.2)
        ax.axvline(
            snapshot_round, color=PURPLE, linewidth=1.0,
            linestyle=(0, (2, 4)), alpha=0.55, zorder=2,
        )

    # Upper panel: committed Train leaders, sampled at every five rounds.
    # Preserve the reference layout while allowing other seeds to exceed its scale.
    train_floor = min(0.0442, min(train_values) - 0.0010)
    label_floor = max(0.0664, max(train_values) + 0.0050)
    train_label_levels = [label_floor, label_floor + 0.0046, label_floor + 0.0092]
    train_ax.set_ylim(train_floor, train_label_levels[-1] + 0.0024)
    train_ax.set_ylabel("Train RankIC", fontsize=11.2, color=NAVY)
    train_ax.plot(
        train_rounds, train_values,
        color=ORANGE, linewidth=2.6, marker="o", markersize=5.4,
        markerfacecolor="white", markeredgewidth=2.0,
        zorder=5, label="Train best-so-far · every 5 rounds",
    )
    for index, leader in enumerate(train_leaders):
        round_id = int(leader["round"])
        value = float(leader["rank_ic"])
        train_ax.annotate(
            f"R{round_id} · {value:.4f}\n{short_name(leader['name'], max_chars=17)}",
            xy=(round_id, value), xycoords="data",
            xytext=(min(94.5, max(3.0, float(round_id))),
                    train_label_levels[-1 if index == 0 else index % len(train_label_levels)]),
            textcoords="data", ha="center", va="center",
            fontsize=6.5, color=NAVY, fontweight="bold",
            arrowprops={
                "arrowstyle": "-", "color": ORANGE,
                "linewidth": 0.9, "alpha": 0.58,
                "shrinkA": 2.0, "shrinkB": 4.0,
                "connectionstyle": "angle,angleA=0,angleB=90",
            },
            bbox={
                "boxstyle": "round,pad=0.23,rounding_size=0.16",
                "facecolor": "white", "edgecolor": ORANGE,
                "linewidth": 1.0, "alpha": 0.98,
            },
            zorder=10,
        )
    train_ax.text(
        0.995, 1.018,
        "Committed Train leader snapshots · every 5 rounds",
        transform=train_ax.transAxes, ha="right", va="bottom",
        fontsize=8.4, color=MUTED,
    )

    # Lower panel: every raw Validation-selected Top-5 audit stays visible. The
    # prominent line is only its retrospective cumulative envelope.
    threshold = float(report["qualification"]["threshold"])
    if show_single_test:
        single_values = [row["test_rank_ic"] for row in single_results]
        test_ax.set_ylim(min(single_values) - 0.0030, max(single_values) + 0.0033)
    else:
        label_padding = max(0.0035, (max(test_values) - min(test_values)) * 0.15)
        test_ax.set_ylim(
            min(0.0340, min(test_values) - 0.0012),
            max(max(test_values), illustrative_r100 or float("-inf")) + label_padding,
        )
    test_ax.set_ylabel("Test RankIC", fontsize=11.2, color=NAVY)
    test_ax.set_xlabel("LDM round", fontsize=11.5, color=NAVY, labelpad=9)
    test_ax.set_xticks(list(range(0, 101, CHECKPOINT_STEP)))
    test_ax.axhline(
        threshold, color="#AFA9B9", linewidth=1.35,
        linestyle=(0, (4, 3)), zorder=1,
    )
    test_ax.plot(
        test_rounds, test_values,
        color=RAW_TEST, linewidth=1.15, linestyle=(0, (2, 3)),
        marker="o", markersize=5.5, markerfacecolor="white",
        markeredgewidth=1.35, alpha=0.85, zorder=6,
        label="Validation-selected Top-5 · raw measured Test",
    )
    test_ax.step(
        test_rounds, test_best_values, where="post",
        color=TEAL, linewidth=3.0, zorder=7,
        label="Retrospective Test best-so-far · diagnostic only",
    )
    if not show_single_test and test_rounds[0] > 0:
        # Only draw a pre-measurement guide when the first real audit is after R0.
        test_ax.annotate(
            "", xy=(test_rounds[0], test_values[0]),
            xytext=(0, test_values[0]),
            arrowprops={
                "arrowstyle": "-|>", "color": "#B8B2C2",
                "linewidth": 1.35,
                "linestyle": "-" if illustrative_solid else (0, (2, 3)),
                "mutation_scale": 10,
            },
            zorder=3,
        )
        test_ax.text(
            1.5, test_values[0] + 0.00020,
            "pre-Test phase · no measurement",
            ha="left", va="bottom", fontsize=7.8, color=MUTED,
        )
    if illustrative_r100 is not None:
        test_ax.plot(
            [test_rounds[-1], 100], [test_values[-1], illustrative_r100],
            color=TEAL, linewidth=2.2,
            linestyle="-" if illustrative_solid else (0, (4, 3)),
            alpha=0.72, zorder=5,
            label="Illustrative projection · not measured",
        )
        test_ax.scatter(
            [100], [illustrative_r100], s=88, marker="o",
            facecolor="white", edgecolor=TEAL, linewidth=2.1,
            linestyle="--", alpha=0.82, zorder=8,
        )
        test_ax.annotate(
            f"R100  ≈{illustrative_r100:.4f}\nprojected · not measured",
            (100, illustrative_r100), xytext=(-8, 12),
            textcoords="offset points", ha="right", va="bottom",
            fontsize=8.1, color=TEAL, fontweight="bold", zorder=10,
        )
    for index, record in enumerate(test_best_records):
        round_id = int(record["round"])
        value = float(record["raw_rank_ic"])
        above = bool(record["improved"]) or index % 2 == 0
        envelope_gap = float(record["rank_ic"]) - value
        if 1e-7 < envelope_gap < 0.16 * (test_ax.get_ylim()[1] - test_ax.get_ylim()[0]):
            # Keep raw-audit labels out of the prominent cumulative envelope.
            above = False
        test_ax.annotate(
            f"R{round_id}\n{value:.4f}",
            (round_id, value), xytext=(0, 11 if above else -14),
            textcoords="offset points", ha="center",
            va="bottom" if above else "top", fontsize=7.0,
            color=TEAL if record["improved"] else MUTED,
            fontweight="bold", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.8},
        )

    if show_single_test:
        generated = [
            row for row in single_results
            if row["source"] == "ldm_continuous_discovery"
        ]
        seeds = [row for row in single_results if row not in generated]
        test_ax.scatter(
            [row["factor_round"] for row in seeds],
            [row["test_rank_ic"] for row in seeds],
            s=54, marker="D", color="#8D859D", edgecolor="white",
            linewidth=0.9, zorder=6, label="Seed single factor · measured Test",
        )
        test_ax.scatter(
            [row["factor_round"] for row in generated],
            [row["test_rank_ic"] for row in generated],
            s=62, marker="D", color=PURPLE, edgecolor="white",
            linewidth=0.9, zorder=8, label="LDM single factor · measured Test",
        )

        single_label_positions = {
            "LOW": (6.0, 0.0458),
            "LOW_REF": (7.0, 0.0372),
            "MA": (5.5, 0.0174),
            "PRIOR_LOW_TO_OPEN": (16.0, 0.0203),
            "LOW_INTRADAY_RETURN": (15.5, 0.0285),
            "RANGE_EXPANSION_CLOSE_LOCATION_REVERSAL": (34.0, 0.0290),
            "LOW_CLOSE_HIGH_VOLUME_REVERSAL": (60.5, 0.0240),
        }
        for row in single_results:
            name = row["name"]
            x = float(row["factor_round"])
            y = float(row["test_rank_ic"])
            label_x, label_y = single_label_positions.get(name, (x + 4.0, y + 0.002))
            color = PURPLE if row in generated else "#756D82"
            test_ax.annotate(
                f"R{row['factor_round']}  {short_name(name, max_chars=18)}\n{y:.4f}",
                xy=(x, y), xycoords="data",
                xytext=(label_x, label_y), textcoords="data",
                ha="center", va="center", fontsize=6.8,
                color=NAVY, fontweight="semibold",
                arrowprops={
                    "arrowstyle": "-", "color": color,
                    "linewidth": 0.85, "alpha": 0.55,
                    "shrinkA": 2.0, "shrinkB": 3.0,
                },
                bbox={
                    "boxstyle": "round,pad=0.22,rounding_size=0.15",
                    "facecolor": "white", "edgecolor": color,
                    "linewidth": 0.85, "alpha": 0.96,
                },
                zorder=9,
            )

    if show_single_test:
        snapshot_label_y = test_ax.get_ylim()[0] + 0.0009
        snapshot_label_va = "bottom"
    elif illustrative_r100 is not None:
        snapshot_label_y = test_ax.get_ylim()[0] + 0.00018
        snapshot_label_va = "bottom"
    else:
        snapshot_label_y = test_ax.get_ylim()[0] + 0.00018
        snapshot_label_va = "bottom"
    test_ax.text(
        snapshot_round - 0.8, snapshot_label_y,
        f"snapshot R{snapshot_round} · {factor_count} saved factors",
        ha="right", va=snapshot_label_va, fontsize=8.2,
        color=PURPLE_DARK, fontweight="bold",
    )
    test_ax.text(
        0.995, 1.018,
        "Raw Test audits · best-so-far is retrospective, never used for selection"
        f" · threshold {threshold:.3f}",
        transform=test_ax.transAxes, ha="right", va="bottom",
        fontsize=8.4, color=MUTED,
    )

    handles1, labels1 = train_ax.get_legend_handles_labels()
    handles2, labels2 = test_ax.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2, labels1 + labels2,
        loc="lower center", bbox_to_anchor=(0.5, 0.028),
        ncol=4, frameon=False, fontsize=8.6,
        handlelength=2.5, columnspacing=1.7,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_prefix.with_suffix(".png"),
        "svg": output_prefix.with_suffix(".svg"),
        "pdf": output_prefix.with_suffix(".pdf"),
    }
    fig.savefig(paths["png"], dpi=240, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["svg"], facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths["pdf"], facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    metadata = {
        "snapshot_round": snapshot_round,
        "seed": seed,
        "factor_count": factor_count,
        "train_milestones": train_leaders,
        "measured_top5_test_checkpoints": [
            {"round": round_id, "rank_ic": value}
            for round_id, value in zip(test_rounds, test_values)
        ],
        "available_checkpoint_rounds": [
            int(row["checkpoint_round"]) for row in all_checkpoints
        ],
        "displayed_checkpoint_schedule": "R0/R5/.../R100",
        "checkpoint_step": CHECKPOINT_STEP,
        "retrospective_test_best_so_far": test_best_records,
        "single_factor_points_displayed": show_single_test,
        "staged_single_factor_test_results": single_results if show_single_test else [],
        "illustrative_projection": (
            {
                "round": 100,
                "rank_ic": illustrative_r100,
                "measured": False,
                "line_style": "solid" if illustrative_solid else "dashed",
            }
            if illustrative_r100 is not None else None
        ),
        "threshold": threshold,
        "test_usage": "retrospective audit only; not used for search or selection",
        "data_dir": str(data_dir.resolve()),
        "outputs": {kind: str(path.resolve()) for kind, path in paths.items()},
    }
    metadata_path = output_prefix.with_suffix(".json")
    metadata["outputs"]["metadata"] = str(metadata_path.resolve())
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="replicate label for the split Train/Test chart")
    parser.add_argument(
        "--chart-only", action="store_true",
        help="draw only the trajectory chart, with 5-round Train milestones",
    )
    parser.add_argument(
        "--single-test", action="store_true",
        help="split Train/Test panels and add staged single-factor Test audits",
    )
    parser.add_argument(
        "--split-top5-only", action="store_true",
        help="split Train/Test panels and show only the staged Top-5 Test curve",
    )
    parser.add_argument(
        "--illustrative-r100", type=float,
        help="add a clearly labeled, non-measured R100 projection to the Test panel",
    )
    parser.add_argument(
        "--illustrative-solid", action="store_true",
        help="draw illustrative pre/post-Test guide segments as solid lines",
    )
    args = parser.parse_args()
    if args.split_top5_only:
        metadata = draw_trajectory_with_single_test(
            args.data_dir.resolve(), args.output_prefix.resolve(),
            show_single_test=False,
            illustrative_r100=args.illustrative_r100,
            illustrative_solid=args.illustrative_solid,
            seed=args.seed,
        )
    elif args.single_test:
        drawer = draw_trajectory_with_single_test
        metadata = drawer(args.data_dir.resolve(), args.output_prefix.resolve())
    elif args.chart_only:
        drawer = draw_trajectory_only
        metadata = drawer(args.data_dir.resolve(), args.output_prefix.resolve())
    else:
        drawer = draw_panel
        metadata = drawer(args.data_dir.resolve(), args.output_prefix.resolve())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

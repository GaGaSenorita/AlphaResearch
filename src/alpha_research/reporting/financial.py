"""Render aligned Train/Test trajectories using measured checkpoints only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .data import (
    CHECKPOINT_STEP, load_json, measured_test_values, regular_checkpoint_rows,
    retrospective_test_best_so_far, short_name, train_leaders_at_rounds,
)
from .style import GRID, MUTED, NAVY, ORANGE, PAPER, PURPLE, PURPLE_DARK, RAW_TEST, TEAL


def draw_financial_trajectory(
    data_dir: Path,
    output_prefix: Path,
    seed: int | None = None,
) -> dict[str, Any]:
    """Draw aligned Train and Test panels, with measured Validation-selected Top-5 Test audits."""
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
    label_padding = max(0.0035, (max(test_values) - min(test_values)) * 0.15)
    test_ax.set_ylim(
        min(0.0340, min(test_values) - 0.0012),
        max(test_values) + label_padding,
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
        "single_factor_points_displayed": False,
        "staged_single_factor_test_results": [],
        "illustrative_projection": None,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    metadata = draw_financial_trajectory(
        args.data_dir.resolve(), args.output_prefix.resolve(), seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0

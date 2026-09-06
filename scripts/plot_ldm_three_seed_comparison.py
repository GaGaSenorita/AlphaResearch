#!/usr/bin/env python3
"""Compare three completed continuous-LDM seeds in one Train/Test figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from plot_ldm_financial_mining import (
    GRID,
    MUTED,
    NAVY,
    PAPER,
    PURPLE_DARK,
    RAW_TEST,
    load_json,
    regular_checkpoint_rows,
    retrospective_test_best_so_far,
    train_leaders_at_rounds,
)


SEED_STYLES = {
    42: {"color": "#3558C8"},
    123: {"color": "#079A91"},
    456: {"color": "#DE7445"},
}


def _load_seed(seed: int, data_dir: Path) -> dict[str, Any]:
    state = load_json(data_dir / "resume_state.json")
    factors = load_json(data_dir / "factor_sequence.json")
    report = load_json(data_dir / "top5_combinations.json")

    snapshot_round = int(state["round_id"])
    if snapshot_round != 100:
        raise ValueError(f"seed {seed} is at R{snapshot_round}, expected R100")

    milestone_rounds = list(range(0, snapshot_round + 1, 10))
    train = train_leaders_at_rounds(list(factors["factors"]), milestone_rounds)

    checkpoints = regular_checkpoint_rows(
        sorted(report["checkpoints"], key=lambda row: row["checkpoint_round"]),
        snapshot_round,
        step=10,
    )
    test_rounds = [int(row["checkpoint_round"]) for row in checkpoints]
    if test_rounds != milestone_rounds:
        raise ValueError(
            f"seed {seed} Test checkpoints are {test_rounds}, expected {milestone_rounds}"
        )
    raw_test = [
        float(row["test_equal_weight_rank_pool"]["metrics"]["rank_ic"])
        for row in checkpoints
    ]
    test_best = retrospective_test_best_so_far(test_rounds, raw_test)
    return {
        "seed": seed,
        "data_dir": str(data_dir.resolve()),
        "snapshot_round": snapshot_round,
        "factor_count": int(factors["factor_count"]),
        "threshold": float(report["qualification"]["threshold"]),
        "train": train,
        "raw_test": [
            {"round": round_id, "rank_ic": value}
            for round_id, value in zip(test_rounds, raw_test)
        ],
        "test_best_so_far": test_best,
    }


def draw_comparison(
    seed_dirs: dict[int, Path],
    output_prefix: Path,
) -> dict[str, Any]:
    records = [_load_seed(seed, seed_dirs[seed]) for seed in sorted(seed_dirs)]
    if sorted(seed_dirs) != [42, 123, 456]:
        raise ValueError("this comparison requires seeds 42, 123, and 456")
    thresholds = {round(float(row["threshold"]), 12) for row in records}
    if len(thresholds) != 1:
        raise ValueError(f"seed thresholds differ: {sorted(thresholds)}")
    threshold = float(records[0]["threshold"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, (train_ax, test_ax) = plt.subplots(
        2,
        1,
        figsize=(14.2, 8.0),
        sharex=True,
        facecolor="#FFFFFF",
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.18},
    )
    fig.subplots_adjust(left=0.085, right=0.88, bottom=0.12, top=0.835)
    fig.text(
        0.085,
        0.95,
        "LDM financial factor discovery",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color="#17213A",
    )
    fig.text(
        0.085,
        0.902,
        "Three independent seeds · 100 LDM rounds · real evaluations only",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#778096",
    )

    for ax in (train_ax, test_ax):
        ax.set_facecolor("#FCFCFD")
        ax.set_xlim(0, 107)
        ax.set_xticks(list(range(0, 101, 10)))
        ax.grid(axis="y", color="#E6E9F0", linewidth=0.8, alpha=0.9)
        ax.grid(axis="x", color="#EEF0F5", linewidth=0.65, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#C8CEDA")
        ax.spines[["left", "bottom"]].set_linewidth(0.9)
        ax.tick_params(colors="#6F788C", labelsize=9.5, length=3.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    all_train = [float(item["rank_ic"]) for row in records for item in row["train"]]
    all_test = [
        float(item["rank_ic"])
        for row in records
        for item in row["test_best_so_far"]
    ]
    train_ax.set_ylim(min(all_train) - 0.0015, max(all_train) + 0.0018)
    test_ax.set_ylim(min(all_test) - 0.0018, max(all_test) + 0.0020)
    train_ax.set_ylabel("RankIC", fontsize=10.5, color="#4B556B", labelpad=9)
    test_ax.set_ylabel("RankIC", fontsize=10.5, color="#4B556B", labelpad=9)
    test_ax.set_xlabel("LDM round", fontsize=10.8, color="#4B556B", labelpad=10)

    train_ax.text(
        0.0, 1.055, "TRAIN  ·  best-so-far",
        transform=train_ax.transAxes, ha="left", va="bottom",
        fontsize=10.5, color="#29334A", fontweight="bold",
    )
    test_ax.text(
        0.0, 1.055, "TEST  ·  Validation-selected Top-5 best-so-far",
        transform=test_ax.transAxes, ha="left", va="bottom",
        fontsize=10.5, color="#29334A", fontweight="bold",
    )

    for row in records:
        seed = int(row["seed"])
        style = SEED_STYLES[seed]
        train_rounds = [int(item["round"]) for item in row["train"]]
        train_values = [float(item["rank_ic"]) for item in row["train"]]
        test_rounds = [int(item["round"]) for item in row["test_best_so_far"]]
        test_values = [float(item["rank_ic"]) for item in row["test_best_so_far"]]

        # No factor-name boxes or arrows: only the three seed trajectories.
        train_ax.plot(
            train_rounds,
            train_values,
            color=style["color"],
            linewidth=2.5,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.55,
            zorder=5,
        )
        test_ax.step(
            test_rounds,
            test_values,
            where="post",
            color=style["color"],
            linewidth=2.5,
            zorder=5,
        )
        test_ax.scatter(
            test_rounds,
            test_values,
            color="white",
            edgecolor=style["color"],
            marker="o",
            s=30,
            linewidth=1.55,
            zorder=6,
        )

        train_ax.text(
            102.0,
            train_values[-1],
            f"{train_values[-1]:.4f}",
            color=style["color"],
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="center",
        )
        test_ax.text(
            102.0,
            test_values[-1],
            f"{test_values[-1]:.4f}",
            color=style["color"],
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="center",
        )

    test_ax.axhline(
        threshold,
        color=RAW_TEST,
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    test_ax.axhspan(
        threshold,
        test_ax.get_ylim()[1],
        color="#ECF7F4",
        alpha=0.30,
        zorder=0,
    )
    test_ax.text(
        106.5,
        threshold + 0.00022,
        f"threshold {threshold:.3f}",
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#7D8495",
    )

    legend = [
        Line2D(
            [0],
            [0],
            color=SEED_STYLES[seed]["color"],
            linewidth=3.0,
            label=f"Seed {seed}",
        )
        for seed in (42, 123, 456)
    ]
    fig.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.88, 0.962),
        ncol=3,
        frameon=False,
        fontsize=9.3,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.text(
        0.88,
        0.045,
        "Test is a retrospective audit only; it is never used for search or selection.",
        ha="right",
        va="bottom",
        fontsize=8.4,
        color="#858C9C",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": output_prefix.with_suffix(".png"),
        "svg": output_prefix.with_suffix(".svg"),
        "pdf": output_prefix.with_suffix(".pdf"),
    }
    for kind, path in outputs.items():
        kwargs: dict[str, Any] = {
            "facecolor": "#FFFFFF",
            "bbox_inches": "tight",
            "pad_inches": 0.08,
        }
        if kind == "png":
            kwargs["dpi"] = 240
        fig.savefig(path, **kwargs)
    plt.close(fig)

    metadata = {
        "title": "LDM financial factor discovery",
        "seeds": records,
        "train_curve": "best-so-far from committed real evaluations at R0/R10/.../R100",
        "test_curve": (
            "retrospective best-so-far of real Validation-selected Top-5 Test audits; "
            "not used for search or selection"
        ),
        "train_factor_annotations_displayed": False,
        "raw_test_curves_displayed": False,
        "outputs": {kind: str(path.resolve()) for kind, path in outputs.items()},
    }
    metadata_path = output_prefix.with_suffix(".json")
    metadata["outputs"]["metadata"] = str(metadata_path.resolve())
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parse_seed_dir(value: str) -> tuple[int, Path]:
    try:
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected SEED=/absolute/run/directory") from exc
    return seed, Path(raw_path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-dir",
        action="append",
        required=True,
        metavar="SEED=DIR",
        help="completed continuous-discovery run directory; pass once per seed",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    seed_dirs = dict(_parse_seed_dir(value) for value in args.seed_dir)
    metadata = draw_comparison(seed_dirs, args.output_prefix.resolve())
    print(json.dumps(metadata["outputs"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

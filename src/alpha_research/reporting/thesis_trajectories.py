"""Publication figures for the three Stage I runs, from saved observations only.

Training uses every completed round and labels every incumbent improvement.
Complete factor names are annotated directly at every improvement in the plot.
Test shows all 21 measured checkpoints and their retrospective running maximum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from .data import (load_json, measured_test_values, regular_checkpoint_rows,
                   train_leaders_at_rounds)
from .output import FIGURE_FORMATS, save_figure

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 123, 456)
CHECKPOINTS = list(range(0, 101, 5))
TRAIN_LIMITS = (.035, .0775)
TEST_LIMITS = (.020, .053)
INK, MUTED = "#25282C", "#737A80"
TRAIN_COLOR, TEAL, BEST_COLOR = "#B95E19", "#566B86", "#9C3D89"


def incumbent_improvements(train: list[dict]) -> list[dict]:
    """Every strictly positive change, with no top-k or size threshold."""
    updates = []
    for previous, current in zip(train, train[1:]):
        gain = current["rank_ic"] - previous["rank_ic"]
        if gain < 0:
            raise ValueError("Training incumbent must not decrease")
        if gain > 0:
            updates.append({**current, "gain": gain})
    return updates


def wrap_factor_name(name: str, width: int = 25) -> str:
    # Preserve every character of the factor identifier, including underscores.
    lines, current = [], ""
    for token in name.split("_"):
        if current and len(current) + len(token) + 1 > width:
            lines.append(current + "_")
            current = token
        else:
            current = token if not current else current + "_" + token
    return "\n".join([*lines, current])


def load_seed(root: Path, seed: int) -> dict:
    run = root / "runs/ldm_continuous_discovery" / (
        f"ldm_continuous_discovery_openai-deepseek-v4-pro_2016-2025_seed{seed}")
    source = run / "factor_sequence.json"
    report_path = run / "top5_combinations.json"
    sequence = load_json(source)["factors"]
    report = load_json(report_path)
    checkpoints = regular_checkpoint_rows(report["checkpoints"], 100, require_complete=True)
    raw = measured_test_values(checkpoints)
    train = train_leaders_at_rounds(sequence, list(range(101)))
    if len(train) != 101 or not np.isfinite([p["rank_ic"] for p in train]).all():
        raise ValueError(f"Seed {seed}: incomplete or nonfinite training records")
    return {"seed": seed, "train": train, "raw": raw,
            "source_run_report": str(report_path.resolve()),
            "source_factor_sequence": str(source.resolve()),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (source, report_path)}}


def style_axes(ax):
    ax.set_xlim(-1, 104)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9DA4AA")
    ax.spines[["left", "bottom"]].set_linewidth(.55)
    ax.tick_params(labelsize=9, length=2.6, width=.55, color="#858B91")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="x", which="minor", length=1.5, width=.4, color="#B0B5B9")
    ax.grid(axis="y", color="#E5E8EA", linewidth=.55)
    ax.set_axisbelow(True)


def render_seed(data: dict, output: Path, *, output_format: str = "pdf") -> dict:
    seed, train = data["seed"], data["train"]
    raw = np.array(data["raw"])
    best = np.maximum.accumulate(raw)
    train_y = np.array([p["rank_ic"] for p in train])
    breakthroughs = incumbent_improvements(train)
    peak = int(raw.argmax())
    figure_height = 6.2
    positions = {
        42: [(18, .0615), (45, .0715), (79, .043)],
        123: [(12, .063), (35, .072), (24, .041), (57, .064),
              (56, .041), (82, .072), (88, .047)],
        456: [(12, .0625), (36, .072), (33, .041), (71, .0715), (84, .043)],
    }
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, (upper, lower) = plt.subplots(2, 1, figsize=(8.0, figure_height),
            sharex=True, gridspec_kw={"height_ratios": [1.65, 1], "hspace": .37})
        fig.subplots_adjust(left=.11, right=.979, top=.878, bottom=.09)
        fig.suptitle(f"AlphaLDM: training and test discovery trajectories (seed {seed})",
                     x=.11, y=.982, ha="left", va="top", fontsize=11, weight="bold")
        for ax in (upper, lower):
            style_axes(ax)

        upper.set_ylim(TRAIN_LIMITS)
        upper.set_yticks([.045, .050, .055, .060, .065])
        upper.tick_params(axis="x", labelbottom=False)
        upper.set_ylabel("Training RankIC", labelpad=7)
        upper.set_title("(a) Best-so-far single-factor training reward (2016-2020)",
                        fontsize=9.5, loc="left", pad=7)
        upper.step(range(101), train_y, where="post", color=TRAIN_COLOR, lw=1.65, zorder=3)
        upper.plot(CHECKPOINTS, train_y[::5], "o", ms=2.6, color=TRAIN_COLOR, zorder=4)
        for point, position in zip(breakthroughs, positions[seed], strict=True):
            upper.plot(point["round"], point["rank_ic"], "o", ms=3.4,
                       color=TRAIN_COLOR, zorder=5)
            name = wrap_factor_name(point["name"], width=22)
            upper.annotate(f"{name}\n(R{point['round']})",
                xy=(point["round"], point["rank_ic"]), xytext=position,
                textcoords="data", fontsize=8.5, linespacing=1.15,
                ha="center", va="center", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.96, pad=.6),
                arrowprops=dict(arrowstyle="-", color="#98918A", lw=.6,
                                shrinkA=3, shrinkB=3), zorder=6)
        upper.annotate(f"{train_y[-1]:.4f}", (100, train_y[-1]),
                       xytext=(0, -10), textcoords="offset points", ha="right",
                       va="top", fontsize=9, color=TRAIN_COLOR, weight="bold")

        lower.set_ylim(TEST_LIMITS)
        lower.set_yticks([.020, .030, .040, .050])
        lower.set_ylabel("Test RankIC", labelpad=7)
        lower.set_xlabel("Search round (Test checkpoints every 5 rounds)", labelpad=6)
        lower.set_title("(b) Validation-selected Top-5 combination on Test (2022-2025)",
                        fontsize=9.5, loc="left", pad=25)
        lower.axhline(.040, color="#999999", lw=.65, ls=(0, (1.2, 3.5)), zorder=1)
        raw_line, = lower.plot(CHECKPOINTS, raw, color=TEAL, lw=1.15,
                   marker="o", ms=3.5, markerfacecolor="white",
                   markeredgecolor=TEAL, markeredgewidth=.8, zorder=3,
                   label="Measured checkpoint")
        # Retain the original blue-grey raw trace; magenta dashes distinguish
        # best-so-far by both colour and line style, including coincident values.
        best_line, = lower.plot(CHECKPOINTS, best, color=BEST_COLOR,
                   lw=1.3, ls=(0, (5, 3.5)), zorder=4,
                   label="Best-so-far (retrospective)")
        lower.legend(handles=[raw_line, best_line], loc="lower left",
                     bbox_to_anchor=(0, 1.005), ncol=2, frameon=False,
                     fontsize=9, borderaxespad=0, handlelength=2.8,
                     columnspacing=1.8)
        lower.annotate(f"{raw[-1]:.4f}", (100, raw[-1]), xytext=(-2, -11),
                       textcoords="offset points", ha="right", va="top",
                       fontsize=9, color=TEAL)
        lower.annotate(f"{best[-1]:.4f}", (100, best[-1]), xytext=(-2, 8),
                       textcoords="offset points", ha="right", va="bottom",
                       fontsize=9, color=BEST_COLOR)
        prefix = output / f"stage1_seed{seed}_thesis"
        figure_path = save_figure(fig, prefix, output_format=output_format,
                                 facecolor="white", bbox_inches="tight", pad_inches=.08)
        plt.close(fig)
    return {"seed": seed, "figure": figure_path.name,
            "train_start": float(train_y[0]), "train_final": float(train_y[-1]),
            "test_initial": float(raw[0]), "test_final": float(raw[-1]),
            "test_checkpoint_maximum": float(raw.max()),
            "first_maximum_checkpoint": CHECKPOINTS[peak],
            "first_checkpoint_above_004": next((r for r, v in zip(CHECKPOINTS, raw) if v > .04), None),
            "training_rounds": list(range(101)), "train_rank_ic": train_y.tolist(),
            "checkpoint_rounds": CHECKPOINTS, "raw_test_rank_ic": raw.tolist(),
            "retrospective_test_maximum": best.tolist(),
            "annotated_breakthroughs": breakthroughs,
            "annotation_rule": "Every strictly positive round-to-round incumbent gain; full name and round directly annotated in the training plot.",
            "train_ylim": TRAIN_LIMITS, "test_ylim": TEST_LIMITS,
            "source_run_report": data["source_run_report"],
            "source_factor_sequence": data["source_factor_sequence"],
            "source_sha256": data["source_sha256"], "figure_size_inches": [8.0, figure_height]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/thesis_trajectories")
    parser.add_argument("--format", choices=FIGURE_FORMATS, default="pdf",
                        help="one output image format (default: pdf)")
    args = parser.parse_args(argv)
    # Verify every source before replacing any figure.
    datasets = [load_seed(args.research_root, seed) for seed in SEEDS]
    records = [render_seed(data, args.output_dir, output_format=args.format) for data in datasets]
    payload = {"figures": records, "note": "All measured values retained. Training at actual rounds; "
               "Test at five-round checkpoints. Peak is descriptive and never selects a checkpoint."}
    (args.output_dir / "stage1_thesis_figure_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
    for record in records:
        print(f"Seed {record['seed']}: Train {record['train_final']:.4f}; Test {record['test_final']:.4f}; "
              f"labelled rounds {[p['round'] for p in record['annotated_breakthroughs']]}")
    return 0

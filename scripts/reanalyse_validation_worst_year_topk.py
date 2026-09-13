#!/usr/bin/env python3
"""Select a frozen Top-k by Validation mean or worst year and evaluate it.

This is a post-search readout. It reuses saved Validation observations, keeps
the daily-RankIC correlation filter, and never changes the original run files. With a one-year Validation window,
worst-year RankIC equals mean Validation RankIC; the two choices select the
same factors. This readout does not convert historical searches to annual ones.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alpha_research.canonical import canonical_form
from alpha_research.evaluator import AlphaBenchFFOEvaluator, _series_correlation
from alpha_research.types import FactorCandidate, Period


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def validated_daily(result: dict, period: dict) -> list[dict]:
    if not result.get("success") or result.get("period") != period:
        raise ValueError("unsuccessful evaluation or mismatched period")
    rows = result.get("daily_metrics", [])
    if not rows or len(rows) != result["metrics"]["daily_count"]:
        raise ValueError("missing daily records or inconsistent count")
    seen, normalized = set(), []
    for row in rows:
        day = datetime.fromisoformat(str(row["date"])).date().isoformat()
        if day in seen or not period["start"] <= day <= period["end"]:
            raise ValueError("duplicate or out-of-period date")
        seen.add(day)
        if not all(math.isfinite(float(row[key])) for key in ("ic", "rank_ic")):
            raise ValueError("non-finite daily correlation")
        normalized.append({**row, "date": day})
    return normalized


def year_means(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        day = date.fromisoformat(row["date"])
        key = str(day.year)
        grouped.setdefault(key, []).append(float(row["rank_ic"]))
    return {key: statistics.mean(values) for key, values in sorted(grouped.items())}


def select_topk(evaluations: list[dict], period: dict, k: int, limit: float,
                objective: str = "worst-year") -> dict:
    start, end = date.fromisoformat(period["start"]), date.fromisoformat(period["end"])
    if start > end or start.month != 1 or end.month != 12:
        raise ValueError("this readout requires whole Validation calendar years")
    if k < 1 or not 0 <= limit <= 1:
        raise ValueError("invalid pool size or correlation limit")
    if objective not in ("worst-year", "mean"):
        raise ValueError("unknown Validation selection objective")
    expected = {str(year) for year in range(start.year, end.year + 1)}
    ranked, rejected, seen, series = [], [], set(), {}
    for item in evaluations:
        candidate, result = item["candidate"], item["validation"]
        canonical = canonical_form(candidate["expression"])
        if canonical in seen:
            continue
        seen.add(canonical)
        try:
            rows = validated_daily(result, period)
            years = year_means(rows)
            if set(years) != expected:
                raise ValueError("not all Validation years have valid observations")
            mean = statistics.mean(float(row["rank_ic"]) for row in rows)
            if not math.isclose(mean, result["metrics"]["rank_ic"], abs_tol=1e-10):
                raise ValueError("saved and recomputed Validation means disagree")
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"candidate": candidate, "reason": str(exc)})
            continue
        series[canonical] = {row["date"]: float(row["rank_ic"]) for row in rows}
        ranked.append({
            "candidate": candidate,
            "canonical": canonical,
            "validation_rankic": mean,
            "validation_yearly_rankic": years,
            "validation_worst_year_rankic": min(years.values()),
            "validation_daily_count": len(rows),
        })
    # Python's stable sort preserves archive order when scores tie.
    score_key = "validation_rankic" if objective == "mean" else "validation_worst_year_rankic"
    ranked.sort(key=lambda item: item[score_key], reverse=True)
    selected, skipped = [], []
    for rank, item in enumerate(ranked, 1):
        item["validation_rank"] = rank
        if len(selected) >= k:
            continue
        correlations = [
            _series_correlation(series[item["canonical"]], series[chosen["canonical"]])
            for chosen in selected
        ]
        finite = [abs(value) for value in correlations if value is not None]
        peak = max(finite) if finite else None
        if peak is not None and peak > limit:
            skipped.append({"candidate": item["candidate"], "max_abs_correlation": peak})
            continue
        selected.append(item)
    if len(selected) != k:
        raise ValueError(f"hard correlation filter supplied {len(selected)}/{k} factors")
    return {"selected": selected, "ranking": ranked, "rejected": rejected,
            "skipped_for_correlation": skipped}


def aggregate(rows: list[dict]) -> dict:
    output = {"daily_count": len(rows)}
    for key, ratio_key in (("ic", "icir"), ("rank_ic", "rank_icir")):
        values = [float(row[key]) for row in rows]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        output.update({key: mean, key + "_sample_sd": std,
                       ratio_key: mean / std if std > 0 else 0.0})
    return output


def describe(result: dict, period: dict) -> dict:
    rows = validated_daily(result, period)
    overall = aggregate(rows)
    for key in ("ic", "icir", "rank_ic", "rank_icir"):
        if not math.isclose(overall[key], result["metrics"][key], abs_tol=1e-8):
            raise ValueError(f"recomputed {key} disagrees with the FFO result")
    years = sorted({row["date"][:4] for row in rows})
    return {
        "overall": overall,
        "annual": {year: aggregate([r for r in rows if r["date"].startswith(year)])
                   for year in years},
        "yearly_rankic": year_means(rows),
        "worst_year_rankic": min(year_means(rows).values()),
    }


def calendar_span(period: dict) -> str:
    start, end = period["start"][:4], period["end"][:4]
    return start if start == end else f"{start}-{end}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--selection-objective", choices=("worst-year", "mean"),
                        default="worst-year")
    parser.add_argument("--archive-scope", choices=("all", "initial"), default="all")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output == run:
        raise ValueError("use a separate reanalysis directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run / "protocol.json").read_text())
    source = run / "validation_selection.json"
    validation = json.loads(source.read_text())
    evaluations = validation["evaluations"]
    if args.archive_scope == "initial":
        evaluations = [item for item in evaluations
                       if item["candidate"]["round_id"] == 0
                       and item["candidate"]["source"] == "alphabench_alpha158"]
        if len(evaluations) != 42:
            raise ValueError(f"expected 42 initial Alpha158 factors, found {len(evaluations)}")
    selection = select_topk(evaluations, protocol["validation"],
                            args.top_k, protocol["max_correlation"], args.selection_objective)
    objective = ("mean_validation_signed_rankic" if args.selection_objective == "mean"
                 else "minimum_of_validation_yearly_mean_signed_rankic")
    frozen = {
        "source_run": str(run),
        "source_search_split": protocol.get("metadata", {}).get("ldm", {}).get("split_reward"),
        "validation_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_objective": objective,
        "selection_period": protocol["validation"],
        "factor_budget": args.top_k,
        "max_abs_daily_validation_rankic_correlation": protocol["max_correlation"],
        "backfill_rejected_factors": False,
        "tie_break": "stable_saved_archive_order",
        "combination": "equal_weight_daily_cross_sectional_z_scores",
        "test_used_for_selection": False,
        **selection,
    }
    if args.archive_scope == "initial":
        frozen["archive_scope"] = "initial_alpha158_42"
    frozen_path = output / "selection.json"
    if frozen_path.exists():
        old = json.loads(frozen_path.read_text())
        # The shared-date helper sums over a set: process hash order can change
        # the last floating-point bit of a diagnostic correlation on reruns.
        same = all(old.get(key) == value for key, value in frozen.items()
                   if key != "skipped_for_correlation")
        old_skipped, new_skipped = old["skipped_for_correlation"], frozen["skipped_for_correlation"]
        same = same and len(old_skipped) == len(new_skipped) and all(
            left["candidate"] == right["candidate"] and math.isclose(
                left["max_abs_correlation"], right["max_abs_correlation"],
                rel_tol=0, abs_tol=1e-12)
            for left, right in zip(old_skipped, new_skipped)
        )
        if not same:
            raise ValueError("refusing to change an existing frozen selection")
        frozen = old
    else:
        frozen["frozen_at"] = datetime.now(timezone.utc).isoformat()
        write_json(frozen_path, frozen)
    print("Frozen factors:", [x["candidate"]["name"] for x in selection["selected"]], flush=True)
    if args.selection_only:
        return 0
    config = protocol["metadata"]["ffo"]
    evaluator = AlphaBenchFFOEvaluator(
        alphabench_root=protocol["metadata"]["alphabench"]["root"],
        base_url=config["url"], market=protocol["metadata"]["market"],
        label=config["label"], use_cache=config["use_cache"], fast=config["fast"],
        topk=config["topk"], n_drop=config["n_drop"], forward_n=config["forward_n"],
        max_attempts=config["max_attempts"], max_parallel=1,
    )
    if not evaluator.health_check():
        raise RuntimeError("FFO service is not healthy")
    factors = tuple(FactorCandidate(**item["candidate"]) for item in selection["selected"])
    summaries = {}
    for split in ("validation", "test"):
        path = output / f"{split}_pool.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            print(f"Evaluating frozen Top-{args.top_k} on {split}...", flush=True)
            result = evaluator.evaluate_pool(factors, Period.from_strings(
                protocol[split]["start"], protocol[split]["end"])).to_dict()
            if not result["success"]:
                raise RuntimeError(result["error"])
            write_json(path, result)
        summaries[split] = describe(result, protocol[split])
        print(split, summaries[split]["overall"], flush=True)
    previous = json.loads((run / "summary.json").read_text())
    report = {
        "status": "completed", "source_run": str(run),
        "selection_file": str(frozen_path), "factor_budget": args.top_k,
        "selection_objective": objective,
        "archive_scope": args.archive_scope,
        "selected_factors": selection["selected"], **summaries,
        "original_top30_test": previous["test"]["equal_weight_rank_pool"]["metrics"],
        "note": "Post-search reanalysis after the original Test report was observed; "
                "factor selection for this readout uses Validation only. Annual readout "
                "does not change the original search objective. With one Validation "
                "year, worst-year and mean selection are equivalent.",
    }
    write_json(output / "summary.json", report)
    criterion = ("the full-period mean daily signed Validation RankIC"
                 if args.selection_objective == "mean"
                 else "the minimum of calendar-year mean signed Validation RankICs")
    scope_label = "Initial-library " if args.archive_scope == "initial" else ""
    lines = [f"# {scope_label}Validation {args.selection_objective} Top-{args.top_k} reanalysis", "",
             f"Selection uses {criterion}. "
             f"The {protocol['max_correlation']} daily Validation RankIC correlation filter and equal-weight "
             "daily cross-sectional z-score combination are retained.", "",
             "| Factor | Validation mean RankIC | Validation worst-year RankIC |",
             "|---|---:|---:|"]
    for item in selection["selected"]:
        lines.append(f"| {item['candidate']['name']} | {item['validation_rankic']:.5f} | "
                     f"{item['validation_worst_year_rankic']:.5f} |")
    lines += ["", "| Period | IC | ICIR | RankIC | RankICIR |",
              "|---|---:|---:|---:|---:|"]
    periods = [(f"Validation {calendar_span(protocol['validation'])}", summaries["validation"]["overall"])]
    periods += [(f"Test {year}", values) for year, values in summaries["test"]["annual"].items()]
    periods += [(f"Test {calendar_span(protocol['test'])}", summaries["test"]["overall"])]
    for label, values in periods:
        lines.append(f"| {label} | " + " | ".join(f"{values[k]:.5f}" for k in
                     ("ic", "icir", "rank_ic", "rank_icir")) + " |")
    lines += ["", "| Period | Worst year | Worst-year RankIC | Positive years |",
              "|---|---|---:|---:|"]
    for split in ("validation", "test"):
        years = summaries[split]["yearly_rankic"]
        worst = min(years, key=years.get)
        positive = sum(value > 0 for value in years.values())
        lines.append(f"| {split.title()} | {worst} | {years[worst]:.5f} | "
                     f"{positive}/{len(years)} |")
    lines += ["", report["note"], "", "Ratios use daily sample standard deviations and are not annualised."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("Completed:", output / "report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the annual search contract before any evaluator calls or writes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from alpha_research.methods.split_robust.reward import YearlySplitSettings
from alpha_research.types import Period


def validate_train_years(period: Period | None, settings: YearlySplitSettings) -> None:
    if period is None or (
        period.start.year != settings.start_year
        or period.end.year != settings.end_year
        or period.start.month != 1
        or period.end.month != 12
    ):
        raise ValueError("Stage-2 Train must span January 2016 through December 2020")


def validate_annual_output(output_dir: str | Path) -> None:
    """Never overwrite or continue a saved run from a different period contract.

    Legacy results remain evidence of the objective actually used for search;
    their rewards must not be reinterpreted as annual observations.
    """
    output_dir = Path(output_dir)
    expected = YearlySplitSettings().to_dict()
    for name in ("protocol.json", "summary.json", "mobo_state.json", "pareto_archive.json"):
        path = output_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "summary.json":
            payload = payload.get("protocol", {})
        if name in {"protocol.json", "summary.json"}:
            settings = payload.get("metadata", {}).get("ldm", {}).get("split_reward", {})
        else:
            settings = payload.get("split_reward", {})
        if settings != expected:
            raise ValueError(
                f"{path}: saved split contract is not the current calendar-year "
                "contract; use a new output directory and retain the original archive"
            )
    for name in ("ldm_history.csv", "split_reward_history.csv"):
        path = output_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            columns = next(csv.reader(handle), [])
        required = {"worst_year", *(f"rankic_{year}" for year in expected["years"])}
        if not required.issubset(columns) or "worst_quarter" in columns:
            raise ValueError(
                f"{path}: saved history is not annual; use a new output directory "
                "and retain the original archive"
            )

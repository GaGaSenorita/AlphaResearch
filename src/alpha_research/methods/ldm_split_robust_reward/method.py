"""The LDM search under a cross-year stability reward."""

from __future__ import annotations

import csv
import math
from typing import Any

from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.methods.split_robust.reward import SplitScore, SplitSettings, split_score


SPLIT_OBJECTIVE = "split_robust"
FIXED_COLUMNS = (
    "expression", "score", "rank_ic", "rank_icir", "mean", "std", "worst",
    "sign_consistency", "noise_std", "dispersion_ratio", "usable_days", "rejected",
)


class SplitRobustLDM(AlphaLDM):
    name = "ldm_split_robust_reward"
    candidate_source = "ldm_standard"

    def __init__(self, *, split_settings: SplitSettings | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.search_objective != SPLIT_OBJECTIVE:
            raise ValueError(
                f"{self.name} requires search_objective={SPLIT_OBJECTIVE!r}, got "
                f"{self.search_objective!r}"
            )
        self.split_settings = split_settings or SplitSettings()
        self._split_records: list[dict[str, Any]] = []

    def _score(self, result: Any) -> float:
        outcome = split_score(result, self.split_settings)
        metrics = getattr(result, "metrics", None)
        row: dict[str, Any] = {
            "expression": getattr(result, "expression", ""),
            "score": outcome.score if math.isfinite(outcome.score) else None,
            "rank_ic": getattr(metrics, "rank_ic", None) if metrics else None,
            "rank_icir": getattr(metrics, "rank_icir", None) if metrics else None,
            "mean": outcome.mean,
            "std": outcome.std,
            "worst": outcome.worst,
            "sign_consistency": outcome.sign_consistency,
            "noise_std": outcome.noise_std,
            "dispersion_ratio": outcome.dispersion_ratio,
            "usable_days": outcome.usable_days,
            "rejected": outcome.rejected or "",
        }
        for year, value, days in zip(outcome.years, outcome.segment_means, outcome.segment_days):
            row[f"r_{year}"] = value
            row[f"n_{year}"] = days
        self._split_records.append(row)
        return outcome.score

    def search(self, **kwargs: Any) -> Any:
        self._split_records = []
        try:
            return super().search(**kwargs)
        finally:
            self._write_split_history()

    def _write_split_history(self) -> None:
        if not self._split_records:
            return
        years = sorted({
            int(key.split("_", 1)[1])
            for row in self._split_records
            for key in row
            if key.startswith("r_")
        })
        columns = list(FIXED_COLUMNS) + [f"r_{year}" for year in years] + [f"n_{year}" for year in years]
        path = self.output_dir / "split_reward_history.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in self._split_records:
                writer.writerow({column: row.get(column) for column in columns})

"""Observation store for the surrogate: fingerprint, realised score, identity.

Ported from ``ldm_agent/history.py``. This is D_0 in the flowchart and every
D_r that follows: append-only, persisted next to the run artifacts, and keyed by
canonical form so a re-proposed formula cannot be counted twice.

The schema version of the fingerprint is stored with the rows. Loading a history
written under a different ``profile.SCHEMA_VERSION`` is refused rather than
silently reinterpreted -- coordinate ``k`` meaning something else across a schema
change is exactly the failure the frozen contract exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class History:
    """Pandas-backed store of (fingerprint, score, canonical form)."""

    def __init__(self, dim: int, schema_version: str) -> None:
        self.dim = int(dim)
        self.schema_version = str(schema_version)
        self.feature_cols = [f"f{i}" for i in range(self.dim)]
        self._df = pd.DataFrame(
            columns=self.feature_cols + ["score", "canonical", "expression", "round_id"]
        )

    @property
    def n(self) -> int:
        return len(self._df)

    def add(
        self,
        feature,
        score: float,
        canonical: str,
        expression: str = "",
        round_id: int = 0,
    ) -> None:
        row = {c: float(v) for c, v in zip(self.feature_cols, np.asarray(feature, dtype=float))}
        row["score"] = float(score)
        row["canonical"] = canonical
        row["expression"] = expression
        row["round_id"] = int(round_id)
        self._df.loc[len(self._df)] = row

    def features(self) -> np.ndarray:
        return self._df[self.feature_cols].to_numpy(dtype=float)

    def scores(self) -> np.ndarray:
        return self._df["score"].to_numpy(dtype=float)

    def canonical_forms(self) -> set[str]:
        return set(self._df["canonical"].tolist())

    def best_score(self) -> float | None:
        if self._df.empty:
            return None
        return float(self._df["score"].max())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self._df.copy()
        frame["schema_version"] = self.schema_version
        frame.to_csv(path, index=False)

    @classmethod
    def load(cls, path: Path, dim: int, schema_version: str) -> "History":
        frame = pd.read_csv(path)
        stored = str(frame["schema_version"].iloc[0]) if "schema_version" in frame else ""
        if stored != str(schema_version):
            raise ValueError(
                f"history {path} was written under profile schema {stored!r}, "
                f"current schema is {schema_version!r}; the stored coordinates "
                "no longer mean what the surrogate expects"
            )
        history = cls(dim, schema_version)
        missing = [c for c in history.feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"history {path} is missing feature columns {missing[:3]}")
        history._df = frame.drop(columns=["schema_version"], errors="ignore")
        return history

"""Label-free behavioural fingerprints for formula alphas.

The LDM surrogate needs a fixed-length vector per candidate *before* the real
score is known. Anything derived from forward returns would make the embedding
as expensive as the evaluation it is supposed to save, so every coordinate here
is computed from the factor values alone -- cross-sectional shape, persistence,
turnover, and the relationship to a frozen reference basket.

Three properties are contractual, not incidental:

``FEATURE_NAMES`` order is frozen
    The GP indexes coordinates positionally. Reordering or inserting a name
    silently corrupts every stored observation, so ``SCHEMA_VERSION`` hashes the
    tuple and history files refuse to load under a different hash.

The reference basket is frozen
    ``*_to_ref`` coordinates are relative quantities. Correlating against a
    growing factor pool would change a candidate's fingerprint as the pool
    grows. The reference expressions are fixed for the whole run.

``profile`` is deterministic
    Same expression, same window, same vector -- no sampling, no wall-clock, no
    dependence on evaluation order. A noisy fingerprint makes the GP fit noise.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alpha_research.canonical import canonical_form
from alpha_research.formula import FormulaValidator
from alpha_research.types import Period


FEATURE_NAMES: tuple[str, ...] = (
    # Cross-sectional shape, averaged over days. Every coordinate here is
    # scale-free: a factor is consumed as a cross-sectional ranking, so its
    # units carry no information, and a raw-magnitude coordinate would place
    # "$close * $volume" a hundred thousand standard deviations from a ratio.
    "xs_skew_mean",
    "xs_kurt_mean",
    "xs_iqr_over_std",
    # Whether the factor is defined often enough to be tradeable.
    "coverage_ratio",
    "nan_ratio",
    "const_day_ratio",
    # Persistence: cross-sectional correlation with the same factor k days ago.
    "autocorr_1",
    "autocorr_5",
    "autocorr_20",
    # Relationship to the frozen reference basket.
    "maxcorr_to_ref",
    "meancorr_to_ref",
    "frac_ref_above_0p7",
)

FEATURE_DIM = len(FEATURE_NAMES)

SCHEMA_VERSION = hashlib.sha256("|".join(FEATURE_NAMES).encode("utf-8")).hexdigest()[:16]

# Coordinates are clipped before they reach the GP: a single degenerate factor
# with a 1e12 variance would otherwise dominate every ARD lengthscale.
_CLIP = 1e6
_EPS = 1e-12


@dataclass(frozen=True)
class ProfileSchema:
    """Serialisable description of the frozen coordinate contract."""

    names: tuple[str, ...] = FEATURE_NAMES
    version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "dimension": len(self.names), "names": list(self.names)}


def _validatable(expression: str) -> str:
    """Canonical call form, falling back to the raw text when unparsable."""
    try:
        return canonical_form(expression)
    except Exception:
        return expression


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return float(np.clip(number, -_CLIP, _CLIP))


class FactorProfiler:
    """Compute frozen-schema behavioural fingerprints via a direct Qlib read.

    The profiling window is deliberately separate from the training window. A
    fingerprint costs one Qlib feature read; the evaluation it feeds costs one
    FFO round trip. If the two windows were identical the surrogate would save
    nothing, so ``period`` is normally a tail slice of the training period.
    """

    def __init__(
        self,
        *,
        provider_uri: str | Path,
        market: str,
        period: Period,
        reference_expressions: Sequence[str],
        validator: FormulaValidator | None = None,
        min_observations: int = 30,
        top_quantile: float = 0.1,
    ) -> None:
        self.provider_uri = Path(provider_uri).expanduser().resolve()
        if not self.provider_uri.is_dir():
            raise FileNotFoundError(f"Qlib provider directory not found: {self.provider_uri}")
        self.market = market
        self.period = period
        self.reference_expressions = tuple(dict.fromkeys(str(e).strip() for e in reference_expressions if str(e).strip()))
        self.validator = validator
        self.min_observations = max(2, int(min_observations))
        self.top_quantile = float(top_quantile)
        self.schema = ProfileSchema()
        self._initialized = False
        self._reference_z: Any | None = None
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ setup

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        import qlib

        qlib.init(provider_uri=str(self.provider_uri), region="cn")
        self._initialized = True

    def _load(self, expressions: list[str]):
        """Read raw factor values; no label is requested at any point."""
        self._ensure_initialized()
        from qlib.data import D

        frame = D.features(
            D.instruments(self.market),
            expressions,
            start_time=self.period.start.isoformat(),
            end_time=self.period.end.isoformat(),
            freq="day",
        )
        if frame is None or frame.empty:
            raise RuntimeError("Qlib returned no observations for the profiling window")
        return frame.replace([float("inf"), float("-inf")], float("nan"))

    @staticmethod
    def _levels(index) -> tuple[Any, Any]:
        names = list(index.names)
        datetime_level = "datetime" if "datetime" in names else (1 if len(names) > 1 else 0)
        instrument_level = "instrument" if "instrument" in names else (0 if len(names) > 1 else None)
        return datetime_level, instrument_level

    def _daily_zscore(self, series):
        """Cross-sectional z-score per day, so correlations become dot products."""
        import pandas as pd  # noqa: F401  (imported for the pandas API on `series`)

        datetime_level, _ = self._levels(series.index)
        grouped = series.groupby(level=datetime_level)
        centered = series - grouped.transform("mean")
        scale = grouped.transform("std")
        return centered / scale.where(scale > _EPS)

    def _self_column(self, expression: str) -> str | None:
        """Reference column holding this very expression, if any."""
        target = _validatable(expression)
        for index, reference in enumerate(self.reference_expressions):
            if _validatable(reference) == target:
                return f"ref_{index}"
        return None

    def _reference_matrix(self):
        """Daily z-scored reference factors, computed once and reused."""
        if self._reference_z is not None:
            return self._reference_z
        if not self.reference_expressions:
            self._reference_z = None
            return None
        frame = self._load(list(self.reference_expressions))
        frame.columns = [f"ref_{i}" for i in range(frame.shape[1])]
        self._reference_z = frame.apply(self._daily_zscore)
        return self._reference_z

    # ---------------------------------------------------------------- profile

    def profile(self, expression: str) -> np.ndarray:
        expression = str(expression).strip()
        if expression in self._cache:
            return self._cache[expression].copy()
        if self.validator is not None:
            # canonical_form accepts infix arithmetic and rewrites it into the
            # call form the validator recognises, so a proposal written
            # "a / b" is judged on the same terms as "Div(a, b)".
            self.validator.validate(_validatable(expression))
        frame = self._load([expression])
        vector = self._features(frame.iloc[:, 0], expression)
        self._cache[expression] = vector
        return vector.copy()

    def profile_many(self, expressions: Sequence[str]) -> np.ndarray:
        return np.vstack([self.profile(expression) for expression in expressions])

    def _features(self, factor, expression: str) -> np.ndarray:
        import pandas as pd

        datetime_level, instrument_level = self._levels(factor.index)
        factor = factor.astype(float)
        total = int(len(factor))
        nan_ratio = float(factor.isna().mean()) if total else 1.0

        grouped = factor.groupby(level=datetime_level)
        counts = grouped.count()
        # Coverage is measured against the widest cross-section actually seen;
        # the universe size drifts over a decade, so a fixed denominator lies.
        widest = float(counts.max()) if len(counts) else 0.0
        coverage = float((counts / widest).mean()) if widest > 0 else 0.0

        valid_days = counts[counts >= self.min_observations].index
        const_day_ratio = 1.0
        if len(counts):
            nunique = grouped.nunique()
            const_day_ratio = float((nunique < 2).mean())

        usable = factor[factor.index.get_level_values(datetime_level).isin(valid_days)]
        if usable.empty:
            usable = factor

        stats = usable.groupby(level=datetime_level)
        xs_skew = float(stats.skew().mean())
        try:
            xs_kurt = float(stats.apply(lambda s: s.kurt()).mean())
        except Exception:
            xs_kurt = 0.0
        # IQR over standard deviation: 1.35 for a normal cross-section, lower
        # when the mass concentrates and outliers stretch the deviation. Unlike
        # a raw IQR this does not move when the factor is rescaled.
        spread = stats.std()
        iqr = stats.quantile(0.75) - stats.quantile(0.25)
        ratio = (iqr / spread.where(spread > _EPS)).replace([np.inf, -np.inf], np.nan)
        xs_iqr_over_std = float(ratio.mean(skipna=True))

        z = self._daily_zscore(usable)
        wide = z.unstack(level=instrument_level) if instrument_level is not None else None

        autocorrs: list[float] = []
        for lag in (1, 5, 20):
            autocorrs.append(self._lagged_cross_section_corr(wide, lag))

        maxcorr, meancorr, frac_high = self._reference_relationship(z, expression)

        values = [
            xs_skew,
            xs_kurt,
            xs_iqr_over_std,
            coverage,
            nan_ratio,
            const_day_ratio,
            autocorrs[0],
            autocorrs[1],
            autocorrs[2],
            maxcorr,
            meancorr,
            frac_high,
        ]
        vector = np.asarray([_finite(v) for v in values], dtype=float)
        if vector.shape != (FEATURE_DIM,):
            raise RuntimeError(f"profile produced {vector.shape}, expected ({FEATURE_DIM},)")
        return vector

    @staticmethod
    def _lagged_cross_section_corr(wide, lag: int) -> float:
        """Mean daily correlation between today's cross-section and t-lag.

        Measured on already z-scored values, so this is the average of
        ``mean(z_t * z_{t-lag})`` over the days where both sides exist.
        """
        if wide is None or wide.shape[0] <= lag:
            return 0.0
        current = wide.iloc[lag:]
        previous = wide.iloc[:-lag]
        previous.index = current.index
        product = (current * previous).mean(axis=1, skipna=True)
        value = float(product.mean(skipna=True))
        return value if math.isfinite(value) else 0.0

    def _reference_relationship(self, candidate_z, expression: str) -> tuple[float, float, float]:
        reference = self._reference_matrix()
        if reference is None or reference.empty:
            return 0.0, 0.0, 0.0
        # A seed profiled against a basket containing itself scores 1.0 by
        # construction, which is both uninformative and, because the whole
        # warm-up set is seeds, a coordinate with no variance to learn from.
        drop = self._self_column(expression)
        if drop is not None:
            reference = reference.drop(columns=[drop])
            if reference.empty:
                return 0.0, 0.0, 0.0
        aligned = reference.reindex(candidate_z.index)
        products = aligned.mul(candidate_z, axis=0)
        correlations = products.mean(axis=0, skipna=True).abs().dropna()
        if correlations.empty:
            return 0.0, 0.0, 0.0
        return (
            float(correlations.max()),
            float(correlations.mean()),
            float((correlations > 0.7).mean()),
        )


class MockFactorProfiler:
    """Deterministic pseudo-fingerprints for the no-network smoke config.

    Derived from a hash of the expression so repeated calls agree, which is the
    same contract the real profiler has to honour.
    """

    def __init__(self, **_kwargs: Any) -> None:
        self.schema = ProfileSchema()
        self.reference_expressions: tuple[str, ...] = ()

    def profile(self, expression: str) -> np.ndarray:
        digest = hashlib.sha256(str(expression).strip().encode("utf-8")).digest()
        raw = np.frombuffer(digest[:FEATURE_DIM], dtype=np.uint8).astype(float)
        return raw / 255.0

    def profile_many(self, expressions: Sequence[str]) -> np.ndarray:
        return np.vstack([self.profile(expression) for expression in expressions])

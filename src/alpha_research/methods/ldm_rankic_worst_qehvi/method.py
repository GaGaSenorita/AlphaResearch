"""AlphaLDM MOBO over average and worst-quarter Train RankIC.

The two objectives are computed from one full 2016--2020 Train evaluation:

    J_avg(f)   = mean of every valid daily signed RankIC
    J_worst(f) = min over 20 calendar-quarter mean signed RankIC values

They are never collapsed into a weighted scalar.  One independent GP is fitted
per objective and each eight-candidate proposal slate is searched exactly over
all three-candidate subsets using Monte Carlo qEHVI.  Objective normalisation
and the dominated reference point are frozen from the initial 42 observations;
Validation and Test are not read anywhere in this module.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from alpha_research.io import write_json
from alpha_research.methods.ldm.gp import LdmSurrogate
from alpha_research.methods.ldm.history import History
from alpha_research.methods.ldm.method import (
    AlphaLDM,
    _cross_validated_rank_correlation,
)
from alpha_research.methods.ldm.multi_objective import (
    hypervolume_2d,
    pareto_front,
    pareto_mask,
    q_expected_hypervolume_improvement_2d,
)
from alpha_research.methods.ldm_split_robust_reward.method import (
    FIXED_COLUMNS,
    QUARTER_DAYS_PREFIX,
    QUARTER_VALUE_PREFIX,
    QuarterlyWorstHistory,
)
from alpha_research.methods.split_robust.reward import SplitScore, SplitSettings, split_score
from alpha_research.types import EvaluationResult, FactorCandidate, SearchResult


OBJECTIVES = ("train_rankic", "worst_rankic")
MAXIMIZE = (True, True)
INITIAL_OBSERVATIONS = 42


@dataclass(frozen=True)
class FixedObjectiveNormalizer:
    """Per-objective z-score transform fitted once on the initial 42 factors."""

    center: np.ndarray
    scale: np.ndarray
    observation_count: int

    @classmethod
    def fit(cls, values: np.ndarray) -> "FixedObjectiveNormalizer":
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(OBJECTIVES):
            raise ValueError(f"objective matrix must have shape [n, 2], got {matrix.shape}")
        if len(matrix) < 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("normalisation requires at least two finite objective rows")
        center = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0, ddof=1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(center=center, scale=scale, observation_count=len(matrix))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.center) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.center

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "z_score",
            "fitted_on": "initial_42_train_observations_only",
            "observation_count": self.observation_count,
            "objectives": list(OBJECTIVES),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "frozen_after_warmup": True,
        }


class RankICWorstHistory(QuarterlyWorstHistory):
    """One row per factor, with both outputs and quarter-level provenance."""

    def __init__(self, dim: int, schema_version: str, quarters: tuple[str, ...]) -> None:
        super().__init__(dim=dim, schema_version=schema_version, quarters=quarters)
        self._df["pareto_nondominated"] = False

    def add_objectives(
        self,
        feature: np.ndarray,
        *,
        canonical: str,
        expression: str,
        round_id: int,
        outcome: SplitScore,
    ) -> None:
        train_rankic = outcome.train_rankic
        worst_rankic = outcome.worst_rankic
        if train_rankic is None or worst_rankic is None:
            raise ValueError("both train_rankic and worst_rankic must be finite")
        # ``score`` remains J_avg solely for the base SearchResult incumbent.
        # Candidate selection uses qEHVI over objective_scores(), never score.
        super().add_outcome(
            feature,
            float(train_rankic),
            canonical,
            expression,
            round_id,
            outcome,
        )
        mask = pareto_mask(self.objective_scores().tolist(), maximize=MAXIMIZE)
        # Direct column replacement keeps a Boolean dtype after ``loc`` appended
        # a row with a missing value for this derived column.
        self._df["pareto_nondominated"] = mask

    def objective_scores(self) -> np.ndarray:
        return self._df[list(OBJECTIVES)].to_numpy(dtype=float)


class RankICWorstSurrogate:
    """Two independent LDM GPs with frozen warm-up objective normalisation."""

    def __init__(
        self,
        *,
        feature_dim: int,
        history: RankICWorstHistory,
        gp_kwargs: dict[str, Any],
        reference_margin: float,
        initial_observations: int = INITIAL_OBSERVATIONS,
    ) -> None:
        self.history = history
        self.models = tuple(
            LdmSurrogate(feature_dim=feature_dim, **gp_kwargs)
            for _objective in OBJECTIVES
        )
        self.reference_margin = float(reference_margin)
        self.initial_observations = int(initial_observations)
        self.normalizer: FixedObjectiveNormalizer | None = None
        self.reference_point: tuple[float, float] | None = None

    def fit(self, features: np.ndarray, _primary_scores: np.ndarray) -> None:
        features = np.asarray(features, dtype=float)
        raw_scores = self.history.objective_scores()
        if len(raw_scores) != len(features):
            raise ValueError("two-objective scores and fingerprint history are misaligned")
        if self.normalizer is None:
            if len(raw_scores) != self.initial_observations:
                raise ValueError(
                    "RankIC-Worst normalisation must be fitted on exactly the initial "
                    f"{self.initial_observations} factors, got {len(raw_scores)}"
                )
            self.normalizer = FixedObjectiveNormalizer.fit(raw_scores)
            initial_normalized = self.normalizer.transform(raw_scores)
            reference = np.min(initial_normalized, axis=0) - self.reference_margin
            self.reference_point = (float(reference[0]), float(reference[1]))

        normalized = self.normalizer.transform(raw_scores)
        for objective_index, model in enumerate(self.models):
            model.fit(features, normalized[:, objective_index])

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base-loop compatibility: return J_avg posterior in raw RankIC units."""
        if self.normalizer is None:
            raise RuntimeError("surrogate normalisation has not been fitted")
        mean, std = self.models[0].predict(features)
        return (
            mean * self.normalizer.scale[0] + self.normalizer.center[0],
            std * self.normalizer.scale[0],
        )

    def predict_objectives_joint(
        self,
        features: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Normalized posterior means/full covariances for qEHVI."""
        if self.normalizer is None:
            raise RuntimeError("surrogate normalisation has not been fitted")
        predictions = [model.predict_joint(features) for model in self.models]
        return (
            [np.asarray(mean, dtype=float) for mean, _covariance in predictions],
            [
                np.asarray(covariance, dtype=float)
                for _mean, covariance in predictions
            ],
        )

    @property
    def trained(self) -> bool:
        return all(model.trained for model in self.models)

    @property
    def fitted_lengthscale(self) -> float:
        return float(self.models[0].fitted_lengthscale)


class RankICWorstQEHVIAlphaLDM(AlphaLDM):
    """Stage-2 Method-2 with exact finite-slate Monte Carlo qEHVI."""

    name = "ldm_rankic_worst_qehvi"
    candidate_source = "ldm_rankic_worst_qehvi"
    supported_acquisitions = frozenset({"qehvi"})

    def __init__(
        self,
        *,
        split_settings: SplitSettings | None = None,
        qehvi_reference_margin: float = 0.1,
        qehvi_n_samples: int = 128,
        **kwargs: Any,
    ) -> None:
        if str(kwargs.get("objective", "rank_ic")) != "rank_ic":
            raise ValueError(
                "ldm_rankic_worst_qehvi keeps Validation/Test selection at rank_ic"
            )
        if kwargs.get("search_objective") not in {None, "rank_ic"}:
            raise ValueError(
                "ldm_rankic_worst_qehvi fixes search to the two objectives "
                "train_rankic and worst_rankic; scalar overrides are not allowed"
            )
        if not math.isfinite(float(qehvi_reference_margin)) or qehvi_reference_margin <= 0:
            raise ValueError("qehvi_reference_margin must be finite and > 0")
        if int(qehvi_n_samples) < 1:
            raise ValueError("qehvi_n_samples must be >= 1")
        if str(kwargs.get("acquisition", "qehvi")).lower() != "qehvi":
            raise ValueError("ldm_rankic_worst_qehvi requires acquisition='qehvi'")

        kwargs["objective"] = "rank_ic"
        kwargs["search_objective"] = "rank_ic"
        kwargs["acquisition"] = "qehvi"
        super().__init__(**kwargs)
        if self.diversity_weight:
            raise ValueError(
                "diversity_weight must be 0: Method 2 selects only by joint qEHVI"
            )
        self.split_settings = split_settings or SplitSettings()
        self.qehvi_reference_margin = float(qehvi_reference_margin)
        self.qehvi_n_samples = int(qehvi_n_samples)
        self._qehvi_rng = np.random.default_rng(self.random_seed)
        self._history: RankICWorstHistory | None = None
        self._surrogate: RankICWorstSurrogate | None = None
        self._split_records: list[dict[str, Any]] = []
        self._outcome_cache: dict[int, tuple[EvaluationResult, SplitScore]] = {}

    def _split_outcome(self, result: EvaluationResult) -> SplitScore:
        key = id(result)
        cached = self._outcome_cache.get(key)
        if cached is not None and cached[0] is result:
            return cached[1]
        outcome = split_score(result, self.split_settings)
        if outcome.quarterly_rankic and outcome.usable_days:
            # Define J_avg from the same signed daily series as the quarter
            # objective.  Weighting quarter means by their valid-day counts is
            # exactly the mean over all valid Train days and avoids depending
            # on a separately reported aggregate field.
            train_rankic = math.fsum(
                value * days
                for value, days in zip(
                    outcome.quarterly_rankic, outcome.quarter_days
                )
            ) / outcome.usable_days
            outcome = replace(outcome, train_rankic=float(train_rankic))
        self._outcome_cache[key] = (result, outcome)
        return outcome

    def _score(self, result: Any) -> float:
        outcome = self._split_outcome(result)
        values = (outcome.train_rankic, outcome.worst_rankic)
        if not all(value is not None and math.isfinite(value) for value in values):
            return float("-inf")
        # Compatibility incumbent only; qEHVI never consumes this scalar.
        return float(outcome.train_rankic)

    def _new_history(self, *, dim: int, schema_version: str) -> RankICWorstHistory:
        history = RankICWorstHistory(dim, schema_version, self.split_settings.quarters)
        self._history = history
        return history

    def _new_surrogate(
        self,
        *,
        feature_dim: int,
        history: History,
    ) -> RankICWorstSurrogate:
        if not isinstance(history, RankICWorstHistory):
            raise TypeError("RankIC-Worst method requires RankICWorstHistory")
        surrogate = RankICWorstSurrogate(
            feature_dim=feature_dim,
            history=history,
            gp_kwargs=self.gp_kwargs,
            reference_margin=self.qehvi_reference_margin,
        )
        self._surrogate = surrogate
        return surrogate

    def _record_observation(
        self,
        history: History,
        *,
        feature: np.ndarray,
        score: float,
        result: EvaluationResult,
        canonical: str,
        expression: str,
        round_id: int,
    ) -> None:
        del score
        if not isinstance(history, RankICWorstHistory):
            raise TypeError("RankIC-Worst method requires RankICWorstHistory")
        outcome = self._split_outcome(result)
        if (
            outcome.train_rankic is None
            or outcome.worst_rankic is None
            or not math.isfinite(outcome.train_rankic)
            or not math.isfinite(outcome.worst_rankic)
        ):
            raise ValueError(outcome.rejected or "non-finite RankIC-Worst objectives")
        history.add_objectives(
            feature,
            canonical=canonical,
            expression=expression,
            round_id=round_id,
            outcome=outcome,
        )
        row: dict[str, Any] = {
            "expression": expression,
            "score": outcome.train_rankic,
            "train_rankic": outcome.train_rankic,
            "worst_rankic": outcome.worst_rankic,
            "worst_quarter": outcome.worst_quarter,
            "usable_days": outcome.usable_days,
            "rejected": outcome.rejected or "",
        }
        for quarter, value, days in zip(
            outcome.quarters, outcome.quarterly_rankic, outcome.quarter_days
        ):
            row[f"{QUARTER_VALUE_PREFIX}{quarter}"] = value
            row[f"{QUARTER_DAYS_PREFIX}{quarter}"] = days
        self._split_records.append(row)

    def _evaluated_record(
        self,
        *,
        expression: str,
        score: float,
        result: EvaluationResult,
    ) -> dict[str, Any]:
        del score
        outcome = self._split_outcome(result)
        return {
            "expression": expression,
            "score": outcome.train_rankic,
            "scores": {
                "train_rankic": outcome.train_rankic,
                "worst_rankic": outcome.worst_rankic,
            },
            "worst_quarter": outcome.worst_quarter,
        }

    def _generator_objective(self) -> str:
        return "joint Train RankIC and worst-quarter RankIC Pareto improvement"

    def _generator_best(
        self,
        candidate: FactorCandidate | None,
        result: EvaluationResult | None,
        score: float,
    ) -> dict[str, Any] | None:
        if candidate is None or result is None:
            return None
        return self._evaluated_record(
            expression=candidate.expression,
            score=score,
            result=result,
        )

    @staticmethod
    def _require_mobo_types(
        history: History,
        surrogate: Any,
    ) -> tuple[RankICWorstHistory, RankICWorstSurrogate]:
        if not isinstance(history, RankICWorstHistory):
            raise TypeError("RankIC-Worst method requires RankICWorstHistory")
        if not isinstance(surrogate, RankICWorstSurrogate):
            raise TypeError("RankIC-Worst method requires RankICWorstSurrogate")
        return history, surrogate

    def _candidate_batch_acquisition(
        self,
        *,
        surrogate: Any,
        history: History,
        features: np.ndarray,
        batch_size: int,
        rng: random.Random,
    ) -> dict[str, Any]:
        del rng
        typed_history, typed_surrogate = self._require_mobo_types(history, surrogate)
        if typed_history.n != len(typed_history.objective_scores()):
            raise RuntimeError("objective history is misaligned")

        typed_surrogate.fit(typed_history.features(), typed_history.scores())
        normalizer = typed_surrogate.normalizer
        reference = typed_surrogate.reference_point
        if normalizer is None or reference is None:
            raise RuntimeError("qEHVI warm-up normalisation/reference is missing")

        normalized_means, normalized_covariances = (
            typed_surrogate.predict_objectives_joint(features)
        )
        normalized_observed = normalizer.transform(typed_history.objective_scores())
        normalized_front = pareto_front(normalized_observed.tolist(), maximize=MAXIMIZE)
        batches, qehvi_values = q_expected_hypervolume_improvement_2d(
            means=normalized_means,
            covariance_matrices=normalized_covariances,
            pareto_points=normalized_front,
            reference=reference,
            maximize=MAXIMIZE,
            batch_size=batch_size,
            n_samples=self.qehvi_n_samples,
            rng=self._qehvi_rng,
        )
        if not batches:
            raise RuntimeError("qEHVI produced no candidate batches")
        selected_batch_index = int(np.argmax(qehvi_values))
        selected = batches[selected_batch_index]

        candidate_values = np.zeros(len(features), dtype=float)
        for candidate_index in range(len(features)):
            involving = [
                qehvi_values[index]
                for index, batch in enumerate(batches)
                if candidate_index in batch
            ]
            candidate_values[candidate_index] = max(involving) if involving else 0.0

        normalized_mean_matrix = np.column_stack(normalized_means)
        raw_mean_matrix = normalizer.inverse(normalized_mean_matrix)
        normalized_stds = [
            np.sqrt(np.clip(np.diag(covariance), 0.0, None))
            for covariance in normalized_covariances
        ]
        raw_stds = [
            normalized_stds[index] * normalizer.scale[index]
            for index in range(len(OBJECTIVES))
        ]
        current_hypervolume = hypervolume_2d(
            normalized_front,
            reference,
            maximize=MAXIMIZE,
        )
        return {
            "mean": raw_mean_matrix[:, 0],
            "std": raw_stds[0],
            "candidate_acquisition": candidate_values,
            "selected_indices": list(selected),
            "diagnostics": {
                "posterior_mean_worst_rankic": raw_mean_matrix[:, 1],
                "posterior_std_worst_rankic": raw_stds[1],
                "posterior_mean_normalized_train_rankic": normalized_means[0],
                "posterior_mean_normalized_worst_rankic": normalized_means[1],
                "candidate_max_batch_qehvi": candidate_values,
            },
            "event": {
                "acquisition": "monte_carlo_qehvi",
                "batch_selection": "exact_enumeration_of_finite_proposal_slate",
                "candidate_count": len(features),
                "batch_size": batch_size,
                "batch_count": len(batches),
                "selected_batch": list(selected),
                "selected_qehvi": float(qehvi_values[selected_batch_index]),
                "qehvi_n_samples": self.qehvi_n_samples,
                "normalized_reference_point": list(reference),
                "normalization": normalizer.to_dict(),
                "pareto_size_before_batch": len(normalized_front),
                "hypervolume_before_batch": current_hypervolume,
                "all_batch_qehvi": [
                    {"indices": list(batch), "value": float(value)}
                    for batch, value in zip(batches, qehvi_values)
                ],
            },
        }

    def _warmup_event_fields(
        self,
        history: History,
        surrogate: Any,
    ) -> dict[str, Any]:
        typed_history, typed_surrogate = self._require_mobo_types(history, surrogate)
        normalizer = typed_surrogate.normalizer
        reference = typed_surrogate.reference_point
        if normalizer is None or reference is None:
            raise RuntimeError("qEHVI warm-up did not freeze normalisation/reference")
        raw = typed_history.objective_scores()
        normalized = normalizer.transform(raw)
        front = pareto_front(normalized.tolist(), maximize=MAXIMIZE)
        worst_cv = _cross_validated_rank_correlation(
            typed_history.features(), raw[:, 1], self.gp_kwargs
        )
        return {
            "search_objectives": list(OBJECTIVES),
            "objective_directions": ["maximize", "maximize"],
            "normalization": normalizer.to_dict(),
            "normalized_reference_point": list(reference),
            "reference_rule": "componentwise_initial_min_minus_margin",
            "reference_margin": self.qehvi_reference_margin,
            "qehvi_n_samples": self.qehvi_n_samples,
            "warmup_pareto_size": len(front),
            "warmup_hypervolume": hypervolume_2d(
                front, reference, maximize=MAXIMIZE
            ),
            "gp_trained_by_objective": {
                objective: model.trained
                for objective, model in zip(OBJECTIVES, typed_surrogate.models)
            },
            "gp_lengthscale_by_objective": {
                objective: float(model.fitted_lengthscale)
                for objective, model in zip(OBJECTIVES, typed_surrogate.models)
            },
            "cv_rank_correlation_worst_rankic": worst_cv,
        }

    def _round_event_fields(self, history: History, surrogate: Any) -> dict[str, Any]:
        typed_history, typed_surrogate = self._require_mobo_types(history, surrogate)
        normalizer = typed_surrogate.normalizer
        reference = typed_surrogate.reference_point
        if normalizer is None or reference is None:
            return {}
        raw = typed_history.objective_scores()
        normalized = normalizer.transform(raw)
        front = pareto_front(normalized.tolist(), maximize=MAXIMIZE)
        return {
            "pareto_size": len(front),
            "normalized_hypervolume": hypervolume_2d(
                front, reference, maximize=MAXIMIZE
            ),
            "best_train_rankic": float(np.max(raw[:, 0])),
            "best_worst_rankic": float(np.max(raw[:, 1])),
        }

    def search(self, **kwargs: Any) -> SearchResult:
        train_period = kwargs.get("train_period")
        if train_period is None or (
            train_period.start.year != self.split_settings.start_year
            or train_period.end.year != self.split_settings.end_year
        ):
            raise ValueError(
                "ldm_rankic_worst_qehvi requires a Train period contained in "
                "calendar years 2016--2020"
            )
        self._split_records = []
        self._outcome_cache = {}
        result: SearchResult | None = None
        try:
            result = super().search(**kwargs)
            return result
        finally:
            self._write_method_artifacts(result)

    def _write_method_artifacts(self, result: SearchResult | None) -> None:
        self._write_split_history()
        if self._history is None or self._history.n == 0:
            return
        # Preserve partial observations too if an external service interrupts a run.
        self._history.save(self.output_dir / "ldm_history.csv")
        surrogate = self._surrogate
        if (
            surrogate is None
            or surrogate.normalizer is None
            or surrogate.reference_point is None
        ):
            return

        frame = self._history._df.copy()
        raw = self._history.objective_scores()
        mask = pareto_mask(raw.tolist(), maximize=MAXIMIZE)
        names = {
            candidate.expression: candidate.name
            for candidate, _evaluation in (result.archive if result else ())
        }
        records: list[dict[str, Any]] = []
        for row_index, (_, row) in enumerate(frame.iterrows()):
            quarterly = {
                quarter: float(row[f"{QUARTER_VALUE_PREFIX}{quarter}"])
                for quarter in self.split_settings.quarters
            }
            records.append(
                {
                    "name": names.get(str(row["expression"])),
                    "expression": str(row["expression"]),
                    "canonical": str(row["canonical"]),
                    "round_id": int(row["round_id"]),
                    "train_rankic": float(row["train_rankic"]),
                    "worst_rankic": float(row["worst_rankic"]),
                    "worst_quarter": str(row["worst_quarter"]),
                    "quarterly_rankic": quarterly,
                    "pareto_nondominated": bool(mask[row_index]),
                }
            )
        pareto_records = [
            record for record in records if record["pareto_nondominated"]
        ]
        normalized = surrogate.normalizer.transform(raw)
        normalized_front = pareto_front(normalized.tolist(), maximize=MAXIMIZE)
        write_json(
            self.output_dir / "pareto_archive.json",
            {
                "schema_version": "alphaldm.rankic_worst_qehvi.v1",
                "objectives": list(OBJECTIVES),
                "directions": ["maximize", "maximize"],
                "observation_count": self._history.n,
                "pareto_size": len(pareto_records),
                "factors": pareto_records,
            },
        )
        write_json(
            self.output_dir / "mobo_state.json",
            {
                "schema_version": "alphaldm.rankic_worst_qehvi.v1",
                "status": "complete" if result is not None else "partial",
                "objectives": list(OBJECTIVES),
                "directions": ["maximize", "maximize"],
                "surrogates": "two_independent_gaussian_processes",
                "acquisition": "monte_carlo_qehvi",
                "batch_selection": "exact_finite_slate_enumeration",
                "qehvi_n_samples": self.qehvi_n_samples,
                "normalization": surrogate.normalizer.to_dict(),
                "normalized_reference_point": list(surrogate.reference_point),
                "reference_rule": "componentwise_initial_min_minus_margin",
                "reference_margin": self.qehvi_reference_margin,
                "observation_count": self._history.n,
                "pareto_size": len(pareto_records),
                "normalized_hypervolume": hypervolume_2d(
                    normalized_front,
                    surrogate.reference_point,
                    maximize=MAXIMIZE,
                ),
                "all_observations": records,
            },
        )

    def _write_split_history(self) -> None:
        if not self._split_records:
            return
        quarters = self.split_settings.quarters
        columns = list(FIXED_COLUMNS)
        columns += [f"{QUARTER_VALUE_PREFIX}{quarter}" for quarter in quarters]
        columns += [f"{QUARTER_DAYS_PREFIX}{quarter}" for quarter in quarters]
        path = self.output_dir / "split_reward_history.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._split_records)

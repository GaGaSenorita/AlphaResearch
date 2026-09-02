"""AlphaLDM with Pareto BO over train RankIC and train turnover.

The method models the two objectives with independent GPs and selects from each
generated candidate reservoir using Monte Carlo expected hypervolume
improvement (EHVI):

    GP_rank_ic(X), GP_turnover(X) -> EHVI(Pareto(D), reference) -> softmax pick

RankIC is maximised and turnover is minimised. Both search objectives use Train
data only. Validation still selects the final pool on the configured RankIC
objective, and Test remains report only, preserving the leakage boundary.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from alpha_research.methods.ldm.acquire import novelty_bonus
from alpha_research.methods.ldm.gp import LdmSurrogate
from alpha_research.methods.ldm.history import History
from alpha_research.methods.ldm.method import (
    AlphaLDM,
    _cross_validated_rank_correlation,
)
from alpha_research.methods.ldm.multi_objective import (
    expected_hypervolume_improvement_2d,
    pareto_front,
)
from alpha_research.types import EvaluationResult, FactorCandidate


OBJECTIVES = ("rank_ic", "turnover")
MAXIMIZE = (True, False)


class RankICTurnoverHistory(History):
    """Base-compatible history with both realised objectives persisted."""

    def __init__(self, dim: int, schema_version: str) -> None:
        super().__init__(dim=dim, schema_version=schema_version)
        self._df["rank_ic"] = np.nan
        self._df["turnover"] = np.nan

    def add_objectives(
        self,
        feature: np.ndarray,
        *,
        rank_ic: float,
        turnover: float,
        canonical: str,
        expression: str,
        round_id: int,
    ) -> None:
        super().add(feature, rank_ic, canonical, expression, round_id)
        row = len(self._df) - 1
        self._df.loc[row, "rank_ic"] = float(rank_ic)
        self._df.loc[row, "turnover"] = float(turnover)

    def objective_scores(self) -> np.ndarray:
        return self._df[list(OBJECTIVES)].to_numpy(dtype=float)


class RankICTurnoverSurrogate:
    """Two independent existing LDM GPs over a shared fingerprint history."""

    def __init__(
        self,
        *,
        feature_dim: int,
        history: RankICTurnoverHistory,
        gp_kwargs: dict[str, Any],
    ) -> None:
        self.history = history
        self.models = tuple(
            LdmSurrogate(feature_dim=feature_dim, **gp_kwargs)
            for _objective in OBJECTIVES
        )

    def fit(self, features: np.ndarray, _primary_scores: np.ndarray) -> None:
        objective_scores = self.history.objective_scores()
        if len(objective_scores) != len(features):
            raise ValueError("two-objective scores and fingerprint history are misaligned")
        for objective_index, model in enumerate(self.models):
            model.fit(features, objective_scores[:, objective_index])

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base-loop compatibility: expose RankIC as the primary posterior."""
        return self.models[0].predict(features)

    def predict_objectives(
        self,
        features: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        predictions = [model.predict(features) for model in self.models]
        return (
            [np.asarray(mean, dtype=float) for mean, _std in predictions],
            [np.asarray(std, dtype=float) for _mean, std in predictions],
        )

    @property
    def trained(self) -> bool:
        return all(model.trained for model in self.models)

    @property
    def fitted_lengthscale(self) -> float:
        return float(self.models[0].fitted_lengthscale)


class RankICTurnoverEHVILDM(AlphaLDM):
    """Seeded AlphaLDM acquired by RankIC/turnover two-objective EHVI."""

    name = "ldm_rankic_turnover_ehvi"
    candidate_source = "ldm_rankic_turnover_ehvi"
    supported_acquisitions = frozenset({"ehvi"})

    def __init__(
        self,
        *,
        ehvi_reference_rank_ic: float = -0.10,
        ehvi_reference_turnover: float = 1.50,
        ehvi_n_samples: int = 128,
        **kwargs: Any,
    ) -> None:
        objective = str(kwargs.get("objective", "rank_ic"))
        if objective != "rank_ic":
            raise ValueError(
                "ldm_rankic_turnover_ehvi keeps the formal validation/test objective "
                f"at rank_ic, got {objective!r}"
            )
        search_objective = kwargs.get("search_objective")
        if search_objective not in {None, "rank_ic"}:
            raise ValueError(
                "ldm_rankic_turnover_ehvi fixes its search objectives to rank_ic and "
                f"turnover, got scalar override {search_objective!r}"
            )
        reference = (
            float(ehvi_reference_rank_ic),
            float(ehvi_reference_turnover),
        )
        if not all(np.isfinite(value) for value in reference):
            raise ValueError(f"EHVI reference point must be finite, got {reference!r}")
        if int(ehvi_n_samples) < 1:
            raise ValueError("ehvi_n_samples must be >= 1")
        requested_acquisition = str(kwargs.get("acquisition", "ehvi")).lower()
        if requested_acquisition != "ehvi":
            raise ValueError(
                "ldm_rankic_turnover_ehvi requires acquisition='ehvi', got "
                f"{requested_acquisition!r}"
            )

        kwargs["objective"] = "rank_ic"
        kwargs["search_objective"] = "rank_ic"
        kwargs["acquisition"] = "ehvi"
        super().__init__(**kwargs)
        self.ehvi_reference = reference
        self.ehvi_n_samples = int(ehvi_n_samples)
        self._ehvi_rng = np.random.default_rng(self.random_seed)

    @staticmethod
    def _objective_values(result: EvaluationResult) -> tuple[float, float]:
        return (
            result.metrics.value("rank_ic"),
            result.metrics.value("turnover"),
        )

    def _score(self, result: Any) -> float:
        values = self._objective_values(result)
        if not all(np.isfinite(value) for value in values):
            return float("-inf")
        # SearchResult and the generator still require one incumbent. RankIC is
        # the formal readout objective; EHVI, not this scalar, selects candidates.
        return float(values[0])

    def _new_history(self, *, dim: int, schema_version: str) -> RankICTurnoverHistory:
        return RankICTurnoverHistory(dim=dim, schema_version=schema_version)

    def _new_surrogate(
        self,
        *,
        feature_dim: int,
        history: History,
    ) -> RankICTurnoverSurrogate:
        if not isinstance(history, RankICTurnoverHistory):
            raise TypeError("two-objective method requires RankICTurnoverHistory")
        return RankICTurnoverSurrogate(
            feature_dim=feature_dim,
            history=history,
            gp_kwargs=self.gp_kwargs,
        )

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
        if not isinstance(history, RankICTurnoverHistory):
            raise TypeError("two-objective method requires RankICTurnoverHistory")
        rank_ic, turnover = self._objective_values(result)
        history.add_objectives(
            feature,
            rank_ic=rank_ic,
            turnover=turnover,
            canonical=canonical,
            expression=expression,
            round_id=round_id,
        )

    def _evaluated_record(
        self,
        *,
        expression: str,
        score: float,
        result: EvaluationResult,
    ) -> dict[str, Any]:
        rank_ic, turnover = self._objective_values(result)
        return {
            "expression": expression,
            "score": score,
            "scores": {"rank_ic": rank_ic, "turnover": turnover},
        }

    def _generator_objective(self) -> str:
        return "joint RankIC improvement and turnover reduction"

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

    def _candidate_acquisition(
        self,
        *,
        surrogate: Any,
        history: History,
        features: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        del rng
        if not isinstance(history, RankICTurnoverHistory):
            raise TypeError("two-objective method requires RankICTurnoverHistory")
        if not isinstance(surrogate, RankICTurnoverSurrogate):
            raise TypeError("two-objective method requires RankICTurnoverSurrogate")
        if not history.n:
            raise RuntimeError("EHVI requires the seeded RankIC/turnover warm-up history")

        surrogate.fit(history.features(), history.scores())
        means, standard_deviations = surrogate.predict_objectives(features)
        points = history.objective_scores()
        front = pareto_front(points.tolist(), maximize=MAXIMIZE)
        values = expected_hypervolume_improvement_2d(
            means=means,
            standard_deviations=standard_deviations,
            pareto_points=front,
            reference=self.ehvi_reference,
            maximize=MAXIMIZE,
            n_samples=self.ehvi_n_samples,
            rng=self._ehvi_rng,
        )
        if self.diversity_weight:
            spread = float(np.std(values))
            if spread > 1e-12:
                values = (values - float(np.mean(values))) / spread
            values = values + self.diversity_weight * novelty_bonus(
                features,
                history.features(),
            )
        return means[0], standard_deviations[0], values, {
            "posterior_mean_turnover": means[1],
            "posterior_std_turnover": standard_deviations[1],
        }

    def _warmup_event_fields(
        self,
        history: History,
        surrogate: Any,
    ) -> dict[str, Any]:
        if not isinstance(history, RankICTurnoverHistory):
            raise TypeError("two-objective method requires RankICTurnoverHistory")
        if not isinstance(surrogate, RankICTurnoverSurrogate):
            raise TypeError("two-objective method requires RankICTurnoverSurrogate")
        points = history.objective_scores()
        front = pareto_front(points.tolist(), maximize=MAXIMIZE)
        reference_is_strictly_worse = all(
            (
                point[index] > self.ehvi_reference[index]
                if MAXIMIZE[index]
                else point[index] < self.ehvi_reference[index]
            )
            for point in front
            for index in range(2)
        )
        if not reference_is_strictly_worse:
            raise RuntimeError(
                "EHVI reference point must be strictly worse than every warm-up "
                f"Pareto point: reference={self.ehvi_reference!r}, front={front!r}"
            )
        turnover_cv = _cross_validated_rank_correlation(
            history.features(),
            points[:, 1],
            self.gp_kwargs,
        )
        return {
            "search_objectives": list(OBJECTIVES),
            "objective_directions": ["maximize", "minimize"],
            "ehvi_reference_point": list(self.ehvi_reference),
            "ehvi_n_samples": self.ehvi_n_samples,
            "warmup_pareto_size": len(front),
            "gp_trained_by_objective": {
                objective: model.trained
                for objective, model in zip(OBJECTIVES, surrogate.models)
            },
            "gp_lengthscale_by_objective": {
                objective: float(model.fitted_lengthscale)
                for objective, model in zip(OBJECTIVES, surrogate.models)
            },
            "cv_rank_correlation_turnover": turnover_cv,
        }

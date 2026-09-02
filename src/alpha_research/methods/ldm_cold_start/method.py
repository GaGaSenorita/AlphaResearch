"""AlphaLDM with an empty initial observation set."""

from __future__ import annotations

import time
from typing import Any

from alpha_research.methods.ldm.method import AlphaLDM
from alpha_research.types import EvaluationResult, FactorCandidate


class AlphaLDMColdStart(AlphaLDM):
    """Skip the Alpha158 warm-up while keeping the remaining search unchanged."""

    name = "ldm_cold_start"
    candidate_source = "ldm_cold_start"

    def _warm_up(
        self,
        *,
        history: Any,
        surrogate: Any,
        archive: list[tuple[FactorCandidate, EvaluationResult]],
        evaluated_records: list[dict[str, Any]],
        train_period: Any,
        seed_factors: Any,
        emit: Any,
        cycle_id: str,
    ) -> tuple[FactorCandidate | None, EvaluationResult | None, float]:
        if seed_factors:
            raise ValueError(
                "AlphaLDMColdStart requires an empty D_0; use ldm_standard for "
                "an Alpha158 warm start"
            )
        started = time.monotonic()
        schema = self.profiler.schema
        emit({
            "event": "ldm_warmup",
            "cycle_id": cycle_id,
            "search_objective": self.search_objective,
            "seeds_offered": 0,
            "observations": 0,
            "feature_dim": len(schema.names),
            "schema_version": schema.version,
            "acquisition": self.acquisition,
            "gp_trained": surrogate.trained,
            "gp_lengthscale": float(surrogate.fitted_lengthscale),
            "cv_rank_correlation": None,
            "best_seed_expression": None,
            "best_seed_score": None,
            "seconds": round(time.monotonic() - started, 3),
            "warm_start": False,
        })
        return None, None, float("-inf")

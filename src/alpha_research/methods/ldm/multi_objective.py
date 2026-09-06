"""Two-objective Pareto and Monte Carlo EHVI utilities.

Adapted from ``small_molecule/strbo_v1/acquisition.py`` in the upstream LDM
case2 task.  That task fits one independent GP per objective and scores a finite
proposal pool by expected hypervolume improvement.  This module keeps the same
algorithm while accepting NumPy's generator directly and limiting the surface
to the two-objective path AlphaResearch uses.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def dominates(
    first: Sequence[float],
    second: Sequence[float],
    maximize: Sequence[bool],
) -> bool:
    """Return whether ``first`` Pareto-dominates ``second``."""
    if len(first) != len(second) or len(first) != len(maximize):
        raise ValueError(
            "point/maximize length mismatch: "
            f"{len(first)}/{len(second)}/{len(maximize)}"
        )
    at_least_one_better = False
    for first_value, second_value, is_maximized in zip(first, second, maximize):
        if is_maximized:
            if first_value < second_value:
                return False
            if first_value > second_value:
                at_least_one_better = True
        else:
            if first_value > second_value:
                return False
            if first_value < second_value:
                at_least_one_better = True
    return at_least_one_better


def pareto_front(
    points: Sequence[Sequence[float]],
    maximize: Sequence[bool],
) -> list[tuple[float, ...]]:
    """Return the non-dominated points, preserving first-seen order."""
    materialized = [tuple(float(value) for value in point) for point in points]
    for point in materialized:
        if len(point) != len(maximize):
            raise ValueError(
                f"point length {len(point)} does not match maximize length {len(maximize)}"
            )
    return [
        point
        for index, point in enumerate(materialized)
        if not any(
            other_index != index and dominates(other, point, maximize)
            for other_index, other in enumerate(materialized)
        )
    ]


def pareto_mask(
    points: Sequence[Sequence[float]],
    maximize: Sequence[bool],
) -> np.ndarray:
    """Return one Boolean per point indicating current non-dominance."""

    materialized = [tuple(float(value) for value in point) for point in points]
    for point in materialized:
        if len(point) != len(maximize):
            raise ValueError(
                f"point length {len(point)} does not match maximize length {len(maximize)}"
            )
    return np.asarray(
        [
            not any(
                other_index != index and dominates(other, point, maximize)
                for other_index, other in enumerate(materialized)
            )
            for index, point in enumerate(materialized)
        ],
        dtype=bool,
    )


def _hypervolume_2d_minimized(
    points: Sequence[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    """Exact 2-D hypervolume after conversion to smaller-is-better axes."""
    dominated = [
        point
        for point in points
        if point[0] < reference[0] and point[1] < reference[1]
    ]
    if not dominated:
        return 0.0
    front = pareto_front(dominated, maximize=(False, False))
    ordered = sorted(front, key=lambda point: point[0])
    volume = 0.0
    for index, point in enumerate(ordered):
        next_x = ordered[index + 1][0] if index + 1 < len(ordered) else reference[0]
        volume += (next_x - point[0]) * (reference[1] - point[1])
    return float(volume)


def _to_minimized(
    point: Sequence[float],
    maximize: Sequence[bool],
) -> tuple[float, float]:
    if len(point) != 2 or len(maximize) != 2:
        raise ValueError("two-objective hypervolume requires two coordinates")
    return tuple(
        -float(value) if is_maximized else float(value)
        for value, is_maximized in zip(point, maximize)
    )  # type: ignore[return-value]


def hypervolume_2d(
    points: Sequence[Sequence[float]],
    reference: Sequence[float],
    *,
    maximize: Sequence[bool],
) -> float:
    """Return exact dominated hypervolume for two objectives."""
    converted_reference = _to_minimized(reference, maximize)
    converted_points = [_to_minimized(point, maximize) for point in points]
    return _hypervolume_2d_minimized(converted_points, converted_reference)


def expected_hypervolume_improvement_2d(
    *,
    means: Sequence[np.ndarray],
    standard_deviations: Sequence[np.ndarray],
    pareto_points: Sequence[Sequence[float]],
    reference: Sequence[float],
    maximize: Sequence[bool],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate finite-pool two-objective EHVI with independent normal draws.

    The two GP posteriors are sampled independently, matching the LDM case2
    implementation.  Higher values always mean a more valuable candidate.
    """
    if len(reference) != 2 or len(maximize) != 2:
        raise ValueError("EHVI requires exactly two objectives")
    if len(means) != 2 or len(standard_deviations) != 2:
        raise ValueError("means and standard_deviations must each contain two arrays")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    mean_arrays = [np.asarray(values, dtype=float).ravel() for values in means]
    std_arrays = [np.asarray(values, dtype=float).ravel() for values in standard_deviations]
    if not (
        mean_arrays[0].shape
        == mean_arrays[1].shape
        == std_arrays[0].shape
        == std_arrays[1].shape
    ):
        raise ValueError("posterior mean/std shapes do not match")
    candidate_count = len(mean_arrays[0])
    if candidate_count == 0:
        return np.zeros((0,), dtype=float)

    converted_reference = _to_minimized(reference, maximize)
    converted_pareto = [_to_minimized(point, maximize) for point in pareto_points]
    converted_pareto = [
        point
        for point in converted_pareto
        if point[0] < converted_reference[0] and point[1] < converted_reference[1]
    ]
    converted_means = [
        -values if is_maximized else values
        for values, is_maximized in zip(mean_arrays, maximize)
    ]
    converted_stds = [np.abs(values) for values in std_arrays]

    current_volume = _hypervolume_2d_minimized(
        converted_pareto,
        converted_reference,
    )
    samples = [
        rng.normal(
            loc=mean[:, None],
            scale=std[:, None],
            size=(candidate_count, n_samples),
        )
        for mean, std in zip(converted_means, converted_stds)
    ]
    for objective_index, limit in enumerate(converted_reference):
        samples[objective_index] = np.minimum(
            samples[objective_index],
            limit - 1e-12,
        )

    values = np.zeros(candidate_count, dtype=float)
    for candidate_index in range(candidate_count):
        improvements = np.zeros(n_samples, dtype=float)
        for sample_index in range(n_samples):
            sampled_point = (
                float(samples[0][candidate_index, sample_index]),
                float(samples[1][candidate_index, sample_index]),
            )
            improvements[sample_index] = (
                _hypervolume_2d_minimized(
                    [*converted_pareto, sampled_point],
                    converted_reference,
                )
                - current_volume
            )
        values[candidate_index] = float(improvements.mean())
    return np.clip(values, 0.0, None)


def q_expected_hypervolume_improvement_2d(
    *,
    means: Sequence[np.ndarray],
    covariance_matrices: Sequence[np.ndarray],
    pareto_points: Sequence[Sequence[float]],
    reference: Sequence[float],
    maximize: Sequence[bool],
    batch_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Estimate discrete two-objective qEHVI for every candidate batch.

    Each objective is modelled by an independent GP, but samples from a single
    GP retain the full posterior covariance across candidates.  For an eight
    candidate slate and ``batch_size=3`` this returns all 56 batch values; the
    caller can therefore choose the exact best batch in the discrete slate.
    """

    if len(reference) != 2 or len(maximize) != 2:
        raise ValueError("qEHVI requires exactly two objectives")
    if len(means) != 2 or len(covariance_matrices) != 2:
        raise ValueError("means and covariance_matrices must each contain two arrays")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    mean_arrays = [np.asarray(values, dtype=float).ravel() for values in means]
    if mean_arrays[0].shape != mean_arrays[1].shape:
        raise ValueError("posterior mean shapes do not match")
    candidate_count = len(mean_arrays[0])
    if batch_size < 1 or batch_size > candidate_count:
        raise ValueError(
            f"batch_size must be in [1, {candidate_count}], got {batch_size}"
        )

    covariance_arrays: list[np.ndarray] = []
    for covariance in covariance_matrices:
        matrix = np.asarray(covariance, dtype=float)
        if matrix.shape != (candidate_count, candidate_count):
            raise ValueError(
                "posterior covariance shape does not match candidate count: "
                f"{matrix.shape} vs {(candidate_count, candidate_count)}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("posterior covariance contains non-finite values")
        matrix = 0.5 * (matrix + matrix.T)
        minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
        if minimum_eigenvalue < 0.0:
            matrix = matrix + np.eye(candidate_count) * (
                -minimum_eigenvalue + 1e-12
            )
        covariance_arrays.append(matrix)

    batches = list(combinations(range(candidate_count), int(batch_size)))
    if not batches:
        return [], np.zeros((0,), dtype=float)

    # Independent objectives, joint candidates within each objective.
    objective_samples = [
        rng.multivariate_normal(mean, covariance, size=n_samples, check_valid="raise")
        for mean, covariance in zip(mean_arrays, covariance_arrays)
    ]
    current_volume = hypervolume_2d(
        pareto_points,
        reference,
        maximize=maximize,
    )
    values = np.zeros(len(batches), dtype=float)
    for batch_index, batch in enumerate(batches):
        improvements = np.zeros(n_samples, dtype=float)
        for sample_index in range(n_samples):
            sampled_points = [
                tuple(
                    float(objective_samples[objective][sample_index, candidate])
                    for objective in range(2)
                )
                for candidate in batch
            ]
            improvements[sample_index] = max(
                0.0,
                hypervolume_2d(
                    [*pareto_points, *sampled_points],
                    reference,
                    maximize=maximize,
                )
                - current_volume,
            )
        values[batch_index] = float(improvements.mean())
    return batches, np.clip(values, 0.0, None)

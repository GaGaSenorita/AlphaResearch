"""Acquisition functions and softmax reweighted sampling.

Ported unchanged from the MLS-Bench LDM agent (``ldm_agent/acquire.py``). The
softmax step is what the flowchart calls Boltzmann sampling: the acquisition
values pick a distribution over the generated pool rather than an argmax, so a
confident-but-wrong surrogate cannot collapse the search onto one lineage.
"""

from __future__ import annotations

import math
import random

import numpy as np


def ucb(mean: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    return np.asarray(mean, dtype=float) + beta * np.asarray(std, dtype=float)


def ei(mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    improvement = mean - best - xi
    z = improvement / std
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    values = improvement * cdf + std * pdf
    return np.where(std <= 1e-12, np.maximum(improvement, 0.0), values)


def acquisition_values(
    mean: np.ndarray, std: np.ndarray, mode: str = "ucb", beta: float = 2.0, xi: float = 0.01
) -> np.ndarray:
    if mode == "ei":
        best = float(np.max(mean))
        return ei(mean, std, best, xi)
    return ucb(mean, std, beta)


def softmax_sample(values: np.ndarray, rng: random.Random, temperature: float = 1.0) -> int:
    """Scale-invariant softmax sampling over *values* (one per candidate)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        raise ValueError("softmax_sample over empty values")
    if len(v) == 1:
        return 0
    centered = v - v.mean()
    scale = float(np.std(centered)) + 1e-8
    logits = centered / scale
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - logits.max()
    p = np.exp(logits)
    total = float(p.sum())
    if total <= 0 or not np.isfinite(total):
        return int(rng.randrange(len(v)))
    p = p / total
    return int(rng.choices(range(len(v)), weights=p.tolist(), k=1)[0])


def novelty_bonus(features: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Distance from each candidate to the nearest fingerprint already observed.

    Scaled to unit spread so it can be added to acquisition values of any
    magnitude, and zero when there is nothing to be different from.

    The surrogate's own uncertainty term does not cover this. Sigma says "the
    model has not seen this region"; it does not say "the factor pool already
    contains something that behaves like this". A pool is scored as an
    equal-weight rank combination, so a candidate correlated with an incumbent
    costs a slot without adding signal -- and a search that hill-climbs one
    lineage produces exactly that. Measured on the first full run: 63 per cent
    of candidates were dropped by the validation correlation filter, against
    19-27 per cent for the chain-of-experience baseline.
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    history = np.atleast_2d(np.asarray(history, dtype=float))
    if history.size == 0 or history.shape[0] == 0:
        return np.zeros(features.shape[0], dtype=float)
    centre = history.mean(axis=0)
    spread = history.std(axis=0)
    scale = np.where(spread > 1e-8, spread, 1.0)
    a = (features - centre) / scale
    b = (history - centre) / scale
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
    nearest = distances.min(axis=1)
    deviation = float(np.std(nearest))
    if deviation < 1e-12:
        return np.zeros(len(nearest), dtype=float)
    return (nearest - float(np.mean(nearest))) / deviation

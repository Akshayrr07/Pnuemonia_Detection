"""Pure probability-aggregation helpers used by reproducible evaluation.

These helpers intentionally expose every non-metric transformation. There is
no class-specific branch, magic multiplier, or implicit weight selection here:
the neutral defaults are an unadjusted mean and equal weights. A non-neutral
viral boost is accepted only when a caller passes it explicitly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


VIRAL_CLASS_INDEX = 2
VIRAL_BOOST_NEUTRAL = 1.0


def _validate_probabilities(model_probs: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model_probs, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError("model_probs must have shape [models, samples, classes]")
    if probabilities.shape[0] == 0 or probabilities.shape[2] == 0:
        raise ValueError("model_probs must contain at least one model, sample, and class")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model_probs must contain only finite values")
    row_sums = probabilities.sum(axis=-1)
    if np.any(row_sums <= 0):
        raise ValueError("each model/sample probability row must have a positive sum")
    return probabilities


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _validate_viral_boost(viral_boost: float) -> float:
    try:
        boost = float(viral_boost)
    except (TypeError, ValueError) as exc:
        raise ValueError("viral_boost must be a positive finite number") from exc
    if not np.isfinite(boost) or boost <= 0:
        raise ValueError("viral_boost must be a positive finite number")
    return boost


def _validate_weights(weights: Optional[Sequence[float]], model_count: int) -> np.ndarray:
    if weights is None:
        return np.full(model_count, 1.0 / model_count, dtype=np.float64)

    if isinstance(weights, (str, bytes)):
        raise ValueError("weights must be a sequence of finite non-negative numbers")
    weight_values = np.asarray(weights, dtype=np.float64)
    if weight_values.ndim != 1 or weight_values.size != model_count:
        raise ValueError(f"weights must contain exactly {model_count} values")
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0):
        raise ValueError("weights must contain only finite non-negative numbers")
    if weight_values.sum() <= 0:
        raise ValueError("at least one weight must be positive")
    return weight_values / weight_values.sum()


def _apply_explicit_viral_boost(
    probabilities: np.ndarray,
    viral_boost: float,
) -> np.ndarray:
    """Apply only the caller-provided class prior, then renormalize."""

    normalised = _normalise(probabilities)
    boost = _validate_viral_boost(viral_boost)
    if boost == VIRAL_BOOST_NEUTRAL:
        return normalised
    normalised[:, VIRAL_CLASS_INDEX] *= boost
    return _normalise(normalised)


def soft_vote(
    model_probs: np.ndarray,
    *,
    viral_boost: float = VIRAL_BOOST_NEUTRAL,
) -> np.ndarray:
    """Average model probabilities and return normalized probabilities.

    ``viral_boost=1.0`` is the default and is a no-op. Any other value is an
    explicit, caller-visible prior adjustment, not a hidden post-hoc change.
    """

    probabilities = _validate_probabilities(model_probs)
    mean_probs = probabilities.mean(axis=0)
    return _apply_explicit_viral_boost(mean_probs, viral_boost)


def weighted_vote(
    model_probs: np.ndarray,
    *,
    weights: Optional[Sequence[float]] = None,
    viral_boost: float = VIRAL_BOOST_NEUTRAL,
) -> np.ndarray:
    """Combine probabilities with explicit weights and optional viral prior.

    Omitting ``weights`` selects an equal average. A supplied sequence is
    normalized once and then used directly; no accuracy lookup or fallback
    weights are applied.
    """

    probabilities = _validate_probabilities(model_probs)
    normalized_weights = _validate_weights(weights, probabilities.shape[0])
    weighted_probs = np.tensordot(
        normalized_weights,
        probabilities,
        axes=(0, 0),
    )
    return _apply_explicit_viral_boost(weighted_probs, viral_boost)


__all__ = [
    "VIRAL_BOOST_NEUTRAL",
    "VIRAL_CLASS_INDEX",
    "soft_vote",
    "weighted_vote",
]

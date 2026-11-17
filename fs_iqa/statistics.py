"""Finite-sample bounds and scalar calculations (no PyTorch dependency)."""

import math

import numpy as np
from scipy.stats import binom, norm


def quantile_indices(n, sigma, radius, confidence=0.999):
    """Return zero-based order-statistic indices, or None for missing tails.

    Each tail uses (1-confidence)/2 failure probability; a union bound
    gives the requested joint confidence. No finite endpoint is invented
    when the sample count is insufficient.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    alpha_tail = (1 - confidence) / 2
    p_low, p_high = norm.cdf(-radius / sigma), norm.cdf(radius / sigma)
    indices = np.arange(n)  # index = rank - 1
    lower = indices[binom.cdf(indices, n, p_low) <= alpha_tail]
    upper = indices[binom.sf(indices, n, p_high) <= alpha_tail]
    return (int(lower[-1]) if lower.size else None,
            int(upper[0]) if upper.size else None)


def interval_from_scores(scores, sigma, radius, confidence=0.999):
    """Median and simultaneous finite-sample feature-space bounds."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be nonempty and finite")
    values = np.sort(values)
    lower_index, upper_index = quantile_indices(len(values), sigma, radius, confidence)
    finite = lower_index is not None and upper_index is not None
    return {
        "median": float(np.median(values)),
        "lower_b": float(values[lower_index]) if lower_index is not None else None,
        "upper_b": float(values[upper_index]) if upper_index is not None else None,
        "interval_status": "finite" if finite else "abstained",
        "n_samples": len(values),
    }


def quadratic_radius(jacobian_norm, curvature, feature_radius):
    """Stable positive root of J*r + L*r**2/2 = feature_radius.

    The caller must decide whether the supplied sensitivity values are
    certified bounds or empirical estimates.
    """
    if not all(math.isfinite(v) for v in (jacobian_norm, curvature, feature_radius)):
        raise ValueError("radius inputs must be finite")
    if jacobian_norm <= 0 or curvature < 0 or feature_radius < 0:
        raise ValueError("require J > 0, L >= 0, feature_radius >= 0")
    return 2 * feature_radius / (
        jacobian_norm + math.hypot(jacobian_norm, math.sqrt(2 * curvature * feature_radius))
    )

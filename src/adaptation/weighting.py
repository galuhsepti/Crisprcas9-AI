"""
Domain-adaptation utilities for the Phase 9C experiment.

Implements INTERNAL-ONLY importance reweighting: sample weights derived from
the DeepSpCas9 training distribution that re-balance the training data toward
currently under-represented regions (high activity, high GC) WITHOUT ever
using the locked Moreno-Mateos test set.

The weights implement flattening (target = uniform) of one or two covariates:
    w_i = 1 / p_hat(z_i)
so rare regions are up-weighted and the common centre is down-weighted.
Weights are clipped to bound influence and normalised to mean 1 so that the
weighted fit is directly comparable to the canonical unweighted fit.
"""

from typing import Dict, List

import numpy as np

MIN_DENSITY = 1e-6
DEFAULT_N_BINS = 20
DEFAULT_CLIP_MIN = 0.1
DEFAULT_CLIP_MAX = 10.0


def _as_float_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")
    return arr


def inverse_density_weights(
    values: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX
) -> np.ndarray:
    """
    Inverse-density sample weights (target = uniform over the observed range).

    Args:
        values: 1D continuous values of one covariate (e.g. labels or GC).
        n_bins: Number of equal-width histogram bins (>= 1).
        clip_min: Lower bound on the relative weights.
        clip_max: Upper bound on the relative weights.

    Returns:
        Relative weights w_i = clip(c_i / mean(c)) where c_i = 1 / p_hat(z_i).
        Weights therefore measure *relative* up/down-weighting around mean 1
        (raw 1/p units are ~n_bins on average, so the clip must act on the
        normalised scale). Exact mean is recorded by the caller via
        ``weight_summary``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if clip_min <= 0:
        raise ValueError("clip_min must be > 0")
    if clip_max <= 0:
        raise ValueError("clip_max must be > 0")
    if clip_min > clip_max:
        raise ValueError("clip_min must be <= clip_max")

    values = _as_float_1d(values, 'values')
    n = values.size
    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        # Degenerate covariate: no spread => no reweighting.
        return np.ones(n)
    if (hi - lo) < 1e-12:
        return np.ones(n)

    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    density = counts / counts.sum()
    bin_idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    safe_density = np.maximum(density[bin_idx], MIN_DENSITY)

    w = 1.0 / safe_density
    w = w / float(np.mean(w))
    return np.clip(w, clip_min, clip_max)


def uniform_weights(values: np.ndarray) -> np.ndarray:
    """All-ones baseline weights (canonical unweighted fit comparison arm)."""
    values = _as_float_1d(values, 'values')
    return np.ones(values.size)


def combine_weights(
    *weight_arrays: np.ndarray,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX
) -> np.ndarray:
    """
    Element-wise product of weight arrays (e.g. label x GC), clipped and
    re-normalised to mean 1.
    """
    if not weight_arrays:
        raise ValueError("At least one weight array is required")
    n = weight_arrays[0].size
    combo = np.ones(n)
    for arr in weight_arrays:
        arr = _as_float_1d(arr, 'weight_array')
        if arr.size != n:
            raise ValueError("All weight arrays must have identical length")
        combo = combo * arr
    if clip_min <= 0 or clip_max <= 0 or clip_min > clip_max:
        raise ValueError("Invalid clip bounds")
    combo = combo / float(np.mean(combo))
    return np.clip(combo, clip_min, clip_max)


def label_reweighting_weights(
    y_true: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX
) -> np.ndarray:
    """Inverse-density weights over the activity label (flatten y)."""
    return inverse_density_weights(
        y_true, n_bins=n_bins, clip_min=clip_min, clip_max=clip_max)


def gc_reweighting_weights(
    gc: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX
) -> np.ndarray:
    """Inverse-density weights over GC content (flatten GC)."""
    return inverse_density_weights(
        gc, n_bins=n_bins, clip_min=clip_min, clip_max=clip_max)


def combined_reweighting_weights(
    y_true: np.ndarray,
    gc: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX
) -> np.ndarray:
    """Product of label x GC inverse-density weights, clipped + normalised."""
    wl = label_reweighting_weights(y_true, n_bins=n_bins,
                                   clip_min=clip_min, clip_max=clip_max)
    wg = gc_reweighting_weights(gc, n_bins=n_bins,
                                clip_min=clip_min, clip_max=clip_max)
    return combine_weights(wl, wg, clip_min=clip_min, clip_max=clip_max)


def weight_summary(weights: np.ndarray) -> Dict[str, float]:
    """
    Summary of a weight vector: mean/min/max and effective sample size.

    ESS = (sum(w))^2 / sum(w^2). ESS < n indicates that reweighting down-weights
    the effective amount of information in the training data.
    """
    w = _as_float_1d(weights, 'weights')
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    n = w.size
    ess = (np.sum(w) ** 2) / np.sum(w ** 2) if np.sum(w ** 2) > 0 else 0.0
    return {
        'n': int(n),
        'mean': float(np.mean(w)),
        'min': float(np.min(w)),
        'max': float(np.max(w)),
        'std': float(np.std(w, ddof=1)) if n > 1 else 0.0,
        'effective_sample_size': float(ess),
        'ess_ratio': float(ess) / n if n > 0 else 0.0,
    }
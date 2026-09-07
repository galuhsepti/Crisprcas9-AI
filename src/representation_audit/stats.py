"""
Phase 13 statistics.

Reproducible percentile bootstrap confidence intervals for paired comparison of
the Phase 13 representation model vs the canonical CNN on the SAME external
observations. This is a plain percentile bootstrap (NOT BCa). The RNG seed is
fixed for reproducibility.
"""

from typing import Dict, List, Sequence

import numpy as np


def reproducible_bootstrap_seed(seed: int = 20260906) -> np.random.Generator:
    """Return a fresh Generator seeded deterministically for bootstrap use."""
    return np.random.default_rng(seed)


def percentile_bootstrap_ci(
    values: Sequence[float],
    statistic,
    n_boot: int = 1000,
    beta: float = 0.95,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """
    Percentile bootstrap 100*beta% CI for a scalar statistic of `values`.

    Args:
        values: per-observation values (e.g. per-sample errors).
        statistic: callable(values) -> scalar point estimate.
        n_boot: number of bootstrap resamples.
        beta: coverage level (0.95 -> 95% CI).
        rng: deterministic NumPy Generator; a fresh seeded one is created if
            None so results are reproducible.

    Returns:
        dict with point, ci_lower, ci_upper, n_boot, n_samples, ci_method.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    if n < 2:
        raise ValueError("At least 2 observations are required")
    if rng is None:
        rng = reproducible_bootstrap_seed()
    point = float(statistic(values))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(values[idx])
    alpha = 1.0 - beta
    lo = 100.0 * alpha / 2.0
    hi = 100.0 * (1.0 - alpha / 2.0)
    ci_lo, ci_hi = np.percentile(boot, [lo, hi])
    return {
        "point": point,
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "n_boot": int(n_boot),
        "n_samples": int(n),
        "ci_method": "percentile (not BCa)",
    }


def bootstrap_delta_mae_ci(
    ae_new: Sequence[float],
    ae_canonical: Sequence[float],
    n_boot: int = 1000,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """
    Paired bootstrap CI for mean(delta_ae) = mean(ae_new - ae_canonical).

    Resamples (with replacement) the SAME paired observations together, so the
    pairing structure is preserved. mean of delta is the primary comparison of
    absolute error between the new representation and the canonical CNN.
    """
    ae_new = np.asarray(ae_new, dtype=float).ravel()
    ae_canonical = np.asarray(ae_canonical, dtype=float).ravel()
    if ae_new.shape != ae_canonical.shape:
        raise ValueError("ae_new and ae_canonical must have identical shapes")
    delta = ae_new - ae_canonical
    n = delta.size
    if n < 2:
        raise ValueError("At least 2 observations are required")
    if rng is None:
        rng = reproducible_bootstrap_seed()

    def _mean(x):
        return float(np.mean(x))

    point = _mean(delta)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = _mean(delta[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_delta_mae": float(point),
        "median_delta_ae": float(np.median(delta)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_boot": int(n_boot),
        "n_samples": int(n),
        "ci_method": "paired percentile bootstrap (not BCa)",
    }

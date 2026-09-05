"""
Statistical utilities for Phase 9A domain-shift diagnostics.

Provides descriptive statistics, standardized effect sizes and distribution
comparison tests. All functions are pure and deterministic.

Terminology guardrails
----------------------
- A statistically significant difference between two datasets is reported as
  a "distribution difference" or "distribution shift"; it is NOT interpreted
  as a causal explanation for model behavior.
- Effect sizes (Cohen's d, standardized mean difference) are reported
  together with p-values because significance alone is not informative for
  large n.
"""

from typing import Dict, List, Sequence
import numpy as np
from scipy import stats


def descriptive_stats(values: Sequence[float]) -> Dict[str, float]:
    """Basic descriptive statistics of a numeric array."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("values must not be empty")
    q = np.percentile(v, [1, 25, 50, 75, 99])
    return {
        'n': int(v.size),
        'mean': float(np.mean(v)),
        'median': float(np.median(v)),
        'std': float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        'min': float(np.min(v)),
        'max': float(np.max(v)),
        'q01': float(q[0]),
        'q25': float(q[1]),
        'q75': float(q[3]),
        'q99': float(q[4]),
    }


def proportion_near_zero(values: Sequence[float], threshold: float = 0.05) -> float:
    """Fraction of values within [0, threshold] (near-zero activity)."""
    v = np.asarray(values, dtype=float).ravel()
    return float(np.mean(v <= threshold))


def proportion_above(values: Sequence[float], threshold: float = 0.8) -> float:
    """Fraction of values strictly above threshold (high activity)."""
    v = np.asarray(values, dtype=float).ravel()
    return float(np.mean(v > threshold))


def ks_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """
    Two-sample Kolmogorov-Smirnov test for equality of continuous
    distributions. Returns the D statistic and p-value.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples must be non-empty")
    stat, p = stats.ks_2samp(a, b)
    return {'statistic': float(stat), 'p_value': float(p)}


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sample Welch t-test (unequal variances)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return {'t_statistic': float(stat), 'p_value': float(p)}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Standardized mean difference (Cohen's d) between two samples,
    pooled standard deviation. Reported as an effect size so large-n
    significance is not over-interpreted.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        raise ValueError("both samples need at least 2 values")
    na, nb = a.size, b.size
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0.0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def histogram_data(values: Sequence[float], bins: int = 30,
                   density: bool = True) -> Dict[str, List[float]]:
    """
    Histogram of values for reproducible plotting, returned as bin edges and
    densities/counts.
    """
    v = np.asarray(values, dtype=float).ravel()
    counts, edges = np.histogram(v, bins=bins, density=density)
    return {
        'counts_density': counts.tolist(),
        'bin_edges': edges.tolist()
    }


def ecdf_data(values: Sequence[float],
              n_points: int = 200) -> Dict[str, List[float]]:
    """Thinned empirical CDF for plotting."""
    v = np.sort(np.asarray(values, dtype=float).ravel())
    if v.size == 0:
        raise ValueError("values must not be empty")
    if v.size <= n_points:
        x = v
        y = np.arange(1, v.size + 1) / v.size
    else:
        idx = np.linspace(0, v.size - 1, n_points).astype(int)
        x = v[idx]
        y = np.arange(1, v.size + 1)[idx] / v.size
    return {'x': x.tolist(), 'y': y.tolist()}
"""
Phase 14 statistics.

Reproducible percentile bootstrap CIs and multiple-testing correction.

Between-domain (C1/C2): MAE subgroups are compared across INDEPENDENT domains.
Observations are never paired across datasets; each domain is resampled with
replacement independently and the CI covers mean(external) - mean(internal).

Within-domain associations (C3-C7): percentile bootstrap over the paired
observations within a single domain.

Percentile method is used throughout (NOT BCa). Seeds are fixed.
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def reproducible_bootstrap_seed(seed: int = 20260907) -> np.random.Generator:
    """Return a fresh Generator seeded deterministically for bootstrap use."""
    return np.random.default_rng(seed)


def percentile_bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    beta: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Percentile bootstrap 100*beta% CI for a scalar statistic of `values`.

    Args:
        values: per-observation values.
        statistic: callable(values) -> scalar point estimate.
        n_boot: number of bootstrap resamples.
        beta: coverage level.
        rng: deterministic NumPy Generator.

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


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def bootstrap_mean_delta_ci(
    external_values: Sequence[float],
    internal_values: Sequence[float],
    n_boot: int = 1000,
    beta: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Between-domain bootstrap CI for mean(external) - mean(internal).

    The two domains are INDEPENDENT samples: each is resampled with replacement
    separately. This is NOT a paired comparison; no observation-level pairing
    across datasets is implied. Used by C1/C2 for subgroup MAE.

    Returns:
        dict with mean_delta, ci_lower, ci_upper, n_boot, n_ext, n_int,
        comparison: "independent_domains (external - internal)".
    """
    ext = np.asarray(external_values, dtype=float).ravel()
    internal = np.asarray(internal_values, dtype=float).ravel()
    if ext.size < 2 or internal.size < 2:
        raise ValueError("Both domains require at least 2 observations")
    if rng is None:
        rng = reproducible_bootstrap_seed()

    n_ext, n_int = ext.size, internal.size
    point = _mean(ext) - _mean(internal)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        e = ext[rng.integers(0, n_ext, size=n_ext)]
        g = internal[rng.integers(0, n_int, size=n_int)]
        boot[i] = _mean(e) - _mean(g)
    alpha = 1.0 - beta
    lo, hi = np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "mean_delta": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "p_value": bootstrap_p_value(boot),
        "n_boot": int(n_boot),
        "n_external": int(n_ext),
        "n_internal": int(n_int),
        "delta_definition": "mean(external) - mean(internal)",
        "comparison": "independent_domains (external - internal), not paired",
        "ci_method": "independent percentiles (not paired bootstrap)",
    }


def bootstrap_p_value(boot_replicates: np.ndarray) -> float:
    """
    Two-sided bootstrap p-value from replicate extrema.

    p = 2 * min(P(boot <= 0), P(boot >= 0)) on the signed replicate
    distribution (centred on the observed statistic at zero), clipped to
    [1/n, 1]. Used only as the p-value companion to the percentile CI.
    """
    boot = np.asarray(boot_replicates, dtype=float).ravel()
    n = boot.size
    if n == 0:
        return 1.0
    frac_leq0 = float(np.mean(boot <= 0.0))
    frac_geq0 = float(np.mean(boot >= 0.0))
    p = 2.0 * min(frac_leq0, frac_geq0)
    p = float(np.clip(p, 1.0 / n, 1.0))
    return p


def spearman_with_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 1000,
    seed: int = 20260907,
) -> Dict[str, float]:
    """
    Spearman rank correlation with a percentile bootstrap 95% CI.

    Resampling is WITHIN a single domain, preserving observation pairs
    (legitimate for a one-sample correlation). Used by C3-C5 and C7.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    if x.size < 3:
        raise ValueError("At least 3 observations are required")
    from scipy import stats as sps

    rng = reproducible_bootstrap_seed(seed)
    point = float(sps.spearmanr(x, y)[0])
    n = x.size
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = sps.spearmanr(x[idx], y[idx])[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "spearman": point,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_boot": int(n_boot),
        "n_samples": int(n),
        "ci_method": "within-domain percentile bootstrap (not BCa)",
    }


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR-adjusted p-values.

    Guarantees: adjusted values are >= raw, monotone non-decreasing in rank
    order, and the BH procedure is exact for a fixed set of tests.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must be in [0, 1]")
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out.tolist()


def mean_pairwise_spearman(vectors: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """
    Mean pairwise Spearman correlation across a dict of parallel vectors.

    Used by C7: agreement of per-bin MAE patterns across models. Returns the
    mean of all pairwise rank correlations plus the number of pairs.
    """
    from scipy import stats as sps
    from itertools import combinations

    keys = list(vectors.keys())
    vals = {k: np.asarray(vectors[k], dtype=float).ravel() for k in keys}
    n0 = vals[keys[0]].size
    for k in keys[1:]:
        if vals[k].shape != (n0,):
            raise ValueError("All vectors must have identical length")
    corrs = []
    for a, b in combinations(keys, 2):
        corrs.append(float(sps.spearmanr(vals[a], vals[b])[0]))
    return {
        "mean_pairwise_spearman": float(np.mean(corrs)) if corrs else None,
        "n_pairs": int(len(corrs)),
        "pairwise": {f"{a}__{b}": c for (a, b), c in zip(combinations(keys, 2), corrs)},
    }


def convergence_bootstrap(
    per_model_errors: Dict[str, np.ndarray],
    y_bin: np.ndarray,
    n_bins: int,
    n_boot: int = 1000,
    seed: int = 20260907,
) -> Dict[str, float]:
    """
    C7: within-domain bootstrap of cross-model per-bin MAE consistency.

    Observations are resampled jointly (same rows across models), per-activity-
    bin MAE is recomputed per model, and the mean pairwise Spearman of the
    resulting bin-MAE vectors is recorded. Confidence interval via percentiles.
    """
    from itertools import combinations

    keys = list(per_model_errors.keys())
    base = np.asarray(per_model_errors[keys[0]]).ravel()
    n = base.size
    for k in keys[1:]:
        if np.asarray(per_model_errors[k]).ravel().shape != base.shape:
            raise ValueError("per-model error arrays must have identical length")
    if np.asarray(y_bin).ravel().shape != base.shape:
        raise ValueError("y_bin must be aligned with the error arrays")

    def _stat(idx):
        vecs = {}
        for k in keys:
            vecs[k] = [float(np.mean(np.abs(np.asarray(per_model_errors[k]).ravel()[idx])[y_bin[idx] == b]))
                       if np.any(y_bin[idx] == b) else np.nan for b in range(n_bins)]
        corrs = []
        for a, b in combinations(keys, 2):
            va, vb = np.asarray(vecs[a]), np.asarray(vecs[b])
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < 2:
                return np.nan
            corrs.append(stats_spearman(va[ok], vb[ok]))
        return float(np.mean(corrs)) if corrs else np.nan

    rng = reproducible_bootstrap_seed(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = _stat(idx)
    boot = boot[np.isfinite(boot)]
    point = _stat(np.arange(n))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot.size else (np.nan, np.nan)
    return {
        "mean_pairwise_spearman": float(point) if np.isfinite(point) else None,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "p_value": bootstrap_p_value(boot) if boot.size else None,
        "n_boot": int(n_boot),
        "n_bins": int(n_bins),
        "models": list(keys),
    }


def stats_spearman(a, b) -> float:
    from scipy import stats as sps
    return float(sps.spearmanr(a, b)[0])
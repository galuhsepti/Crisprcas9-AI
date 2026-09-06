"""
Paired statistics for the Phase 10 external evaluation.

All statistics are computed on the LOCKED Moreno-Mateos external set only,
per model family, pairing arm B (baseline) vs arm D1 (diversified) predictions
on the same samples. Variates:

    e_B, e_D1        absolute errors per sample
    d_i = |e_B| - |e_D1|   -> positive d_i means D1 improved this sample

A cluster (bootstrap) procedure is used for the CI of the mean of d_i because
the 810 external samples are not truly independent realisations.
"""

from typing import Dict, Optional

import numpy as np


def mean_pairwise_diff(err_b: np.ndarray, err_d1: np.ndarray) -> float:
    err_b = np.asarray(err_b, dtype=float)
    err_d1 = np.asarray(err_d1, dtype=float)
    return float(np.mean(np.abs(err_b) - np.abs(err_d1)))


def cohens_dz(err_b: np.ndarray, err_d1: np.ndarray) -> float:
    """
    Standardised mean difference for paired differences d_i:
        dz = mean(d) / sd(d)
    (Lakens 2013, eq. 4).
    """
    err_b = np.asarray(err_b, dtype=float)
    err_d1 = np.asarray(err_d1, dtype=float)
    d = np.abs(err_b) - np.abs(err_d1)
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def paired_t_test(err_b: np.ndarray, err_d1: np.ndarray) -> Dict[str, float]:
    from scipy import stats as sps
    err_b = np.asarray(err_b, dtype=float)
    err_d1 = np.asarray(err_d1, dtype=float)
    d = np.abs(err_b) - np.abs(err_d1)
    t, p = sps.ttest_1samp(d, 0.0)
    return {"t": float(t), "p": float(p),
            "df": float(len(d) - 1)}


def paired_wilcoxon(err_b: np.ndarray, err_d1: np.ndarray) -> Dict[str, float]:
    from scipy import stats as sps
    err_b = np.asarray(err_b, dtype=float)
    err_d1 = np.asarray(err_d1, dtype=float)
    d = np.abs(err_b) - np.abs(err_d1)
    if np.all(np.abs(d) < np.finfo(float).eps):
        return {"statistic": 0.0, "p": 1.0}
    stat, p = sps.wilcoxon(d)
    return {"statistic": float(stat), "p": float(p)}


def bootstrap_ci_meandiff(err_b: np.ndarray, err_d1: np.ndarray,
                          n_boot: int = 1000, seed: int = 42,
                          alpha: float = 0.05) -> Dict[str, float]:
    """
    Percentile bootstrap CI for mean(|e_B| - |e_D1|) over the paired external
    samples. Resamples the 810 samples with replacement (block of one) and
    recomputes the mean per-sample difference of absolute errors.

    Returns point (observed mean), ci_lower/ci_upper (alpha/2 and 1-alpha/2
    percentiles of the bootstrap distribution), ci_halfwidth and metadata.
    """
    err_b = np.asarray(err_b, dtype=float)
    err_d1 = np.asarray(err_d1, dtype=float)
    d = np.abs(err_b) - np.abs(err_d1)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = d[idx].mean()

    point = float(d.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))

    return {
        "point": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_halfwidth": float(0.5 * (hi - lo)),
        "ci_method": "percentile bootstrap",
        "n_boot": n_boot,
        "seed": seed,
        "mean_boot": float(boots.mean()),
    }


def paired_report(err_b: np.ndarray, err_d1: np.ndarray,
                  n_boot: int = 1000, seed: int = 42) -> Dict:
    """
    Complete paired difference report for absolute errors of one family:
        mean_pairwise_diff, cohens_dz, paired t-test, Wilcoxon, BCa CI.
    """
    report = {
        "n_samples": int(len(np.asarray(err_b, dtype=float))),
        "mean_pairwise_diff": mean_pairwise_diff(err_b, err_d1),
        "cohens_dz": cohens_dz(err_b, err_d1),
        "paired_t": paired_t_test(err_b, err_d1),
        "paired_wilcoxon": paired_wilcoxon(err_b, err_d1),
        "bootstrap_ci": bootstrap_ci_meandiff(err_b, err_d1, n_boot, seed),
    }
    report["verdict_inputs"] = {
        "primary_d": report["mean_pairwise_diff"],
        "ci_lower": report["bootstrap_ci"]["ci_lower"],
        "ci_upper": report["bootstrap_ci"]["ci_upper"],
    }
    return report
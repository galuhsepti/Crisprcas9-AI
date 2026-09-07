"""
Phase 14 error association analysis.

Per-sequence absolute/signed errors and their association with pre-registered
frozen covariates: GC (30-mer), guide GC, true activity, GC_train_z (frozen
internal-train z-score), activity_train_z (A_train_z), composition distance
(Euclidean distance of the 2-mer frequency vector from the internal-train mean
2-mer vector), and predicted activity.

Primary association statistics: Spearman correlation with a percentile
bootstrap 95% CI. Everything is associational; no causal reading.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from .statistics import spearman_with_bootstrap


def absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return np.abs(y_true - y_pred)


def signed_error(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return y_pred - y_true


def association_summary(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    covariates: Dict[str, np.ndarray],
    model: str = "model",
    n_boot: int = 1000,
    seed: int = 20260907,
    primary_error: str = "absolute",
) -> Dict[str, object]:
    """
    Spearman association between model error and each covariate.

    Returns per-covariate: spearman point estimate, percentile bootstrap 95%
    CI, p-value, n and a quality flag for ties/small n.
    """
    from scipy import stats as sps

    ae = absolute_error(y_true, y_pred)
    if primary_error == "absolute":
        err = ae
    elif primary_error == "signed":
        err = signed_error(y_true, y_pred)
    else:
        raise ValueError("primary_error must be 'absolute' or 'signed'")

    out: Dict[str, object] = {"model": model, "error_type": primary_error}
    for name, cov in covariates.items():
        c = np.asarray(cov, dtype=float).ravel()
        if c.shape != err.shape:
            out[name] = {
                "error": "covariate length mismatch",
            }
            continue
        mask = np.isfinite(c) & np.isfinite(err)
        if mask.sum() < 3:
            out[name] = {"n": int(mask.sum()), "error": "too few valid pairs"}
            continue
        rho, p = sps.spearmanr(err[mask], c[mask])
        boot = spearman_with_bootstrap(
            err[mask], c[mask], n_boot=n_boot, seed=seed
        )
        out[name] = {
            "spearman": float(rho),
            "p_value": float(p),
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
            "n": int(mask.sum()),
            "quality": "ok",
        }
    return out


def composition_distance(
    sequences: Sequence[str],
    train_mean_2mer: np.ndarray,
    k: int = 2,
) -> np.ndarray:
    """
    Euclidean distance of each sequence's 2-mer frequency vector from the
    frozen internal-train mean 2-mer vector (a defensible composition scalar).
    """
    from .composition import kmer_frequency_vector

    mu = np.asarray(train_mean_2mer, dtype=float)
    out = np.empty(len(sequences), dtype=float)
    for i, s in enumerate(sequences):
        v = kmer_frequency_vector([s], k=k)
        out[i] = float(np.linalg.norm(v - mu))
    return out
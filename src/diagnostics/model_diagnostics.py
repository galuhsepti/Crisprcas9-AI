"""
Prediction and error diagnostics for Phase 9A.

Analyzes predictions of the CANONICAL models (loaded, never retrained) on the
in-domain validation split and the external test set.

Conventions
-----------
- bias = mean(prediction - true); positive bias => overprediction.
- absolute error = |true - prediction|.
- Regression-style summaries use pixel-independent scalars only (no images).
- Observed patterns are "associations"/"patterns", not causal claims.
"""

from typing import Dict, List, Sequence
import numpy as np

from ..evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_pearson_correlation,
    calculate_spearman_correlation
)


def prediction_summary(y_true, y_pred) -> Dict[str, float]:
    """Descriptive stats of predictions plus error/correlation scalars."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")

    pearson_r, pearson_p = calculate_pearson_correlation(y_true, y_pred)
    spearman_r, spearman_p = calculate_spearman_correlation(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    residuals = y_pred - y_true

    return {
        'n': int(y_true.size),
        'prediction_mean': float(np.mean(y_pred)),
        'prediction_median': float(np.median(y_pred)),
        'prediction_std': float(np.std(y_pred, ddof=1)) if y_pred.size > 1 else 0.0,
        'prediction_min': float(np.min(y_pred)),
        'prediction_max': float(np.max(y_pred)),
        'prediction_q01': float(np.percentile(y_pred, 1)),
        'prediction_q99': float(np.percentile(y_pred, 99)),
        'mae': float(calculate_mae(y_true, y_pred)),
        'rmse': float(calculate_rmse(y_true, y_pred)),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'bias_mean_pred_minus_true': bias,
        'bias_median_residual': float(np.median(residuals)),
        'residual_std': float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0,
        'residual_q01': float(np.percentile(residuals, 1)),
        'residual_q99': float(np.percentile(residuals, 99))
    }


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin edges over the observed range of values."""
    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        return np.linspace(lo - 0.5, lo + 0.5, n_bins + 1)
    return np.linspace(lo, hi, n_bins + 1)


def error_by_bins(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    bin_var: Sequence[float],
    n_bins: int = 5
) -> List[Dict[str, float]]:
    """
    Error statistics within equal-width bins of ``bin_var``.

    Args:
        y_true: True activity
        y_pred: Predictions
        bin_var: Variable used for binning (activity, GC, prediction, ...)
        n_bins: Number of equal-width bins

    Returns:
        List of dicts, one per bin: range, n, MAE, RMSE, bias, mean true mean
        predicted.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    bin_var = np.asarray(bin_var, dtype=float).ravel()
    if not (y_true.shape == y_pred.shape == bin_var.shape):
        raise ValueError("y_true, y_pred, bin_var must have identical shapes")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = _bin_edges(bin_var, n_bins)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (bin_var >= lo) & (bin_var <= hi)
        else:
            mask = (bin_var >= lo) & (bin_var < hi)
        idx = np.where(mask)[0]
        if idx.size == 0:
            out.append({'bin': f'[{lo:.4f}, {hi:.4f})', 'n': 0,
                        'mae': None, 'rmse': None, 'bias': None,
                        'mean_true': None, 'mean_pred': None})
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        out.append({
            'bin': f'[{lo:.4f}, {hi:.4f})',
            'n': int(idx.size),
            'mae': float(calculate_mae(yt, yp)),
            'rmse': float(calculate_rmse(yt, yp)),
            'bias': float(np.mean(yp - yt)),
            'mean_true': float(np.mean(yt)),
            'mean_pred': float(np.mean(yp))
        })
    return out


def model_agreement(y_pred_a, y_pred_b) -> Dict[str, float]:
    """
    Agreement between two models' prediction vectors: Pearson and Spearman
    correlations plus mean absolute prediction difference.
    Agreement of predictions does not imply biological correctness.
    """
    a = np.asarray(y_pred_a, dtype=float).ravel()
    b = np.asarray(y_pred_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("predictions must have identical shapes")
    p_r, p_p = calculate_pearson_correlation(a, b)
    s_r, s_p = calculate_spearman_correlation(a, b)
    return {
        'pearson': float(p_r),
        'pearson_p': float(p_p),
        'spearman': float(s_r),
        'spearman_p': float(s_p),
        'mae_between_predictions': float(np.mean(np.abs(a - b))),
        'max_abs_difference': float(np.max(np.abs(a - b)))
    }


def prediction_vs_true_regression(y_true, y_pred) -> Dict[str, float]:
    """
    Ordinary least-squares fit of prediction ~ true activity (slope,
    intercept). A slope < 1 / intercept > 0 indicates regression-to-the-mean
    (overprediction of low / underprediction of high activity).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    return {
        'slope': float(slope),
        'intercept': float(intercept)
    }
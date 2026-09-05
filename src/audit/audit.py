"""
Audit utilities for Phase 9D (final generalization audit).

Read-only analysis helpers that operate on *already generated* predictions.
No fitting, tuning, or optimization is performed. All functions reuse the
canonical evaluation/statistics conventions from Phases 5-9C:
- bias = mean(prediction - true); positive bias => overprediction.
- Standard deviations use ddof=1 (sample).
- Correlations report association; they are NOT evidence of calibration.
"""

from typing import Dict, List, Sequence

import numpy as np

from ..evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_all_metrics,
)

DEFAULT_ACTIVITY_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _check_shapes(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return y_true, y_pred


def activity_bin_stats(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    edges: Sequence[float] = DEFAULT_ACTIVITY_EDGES
) -> List[Dict]:
    """
    Per-activity-bin error statistics on a COMMON true-activity grid.

    The same grid is used across models and datasets so that internal
    (validation) and external (Moreno-Mateos) behaviour can be compared on
    identical bins. R^2 is intentionally NOT reported within bins: when
    within-bin true variance is small, bin-level R^2 is dominated by noise.
    MAE/RMSE/bias and per-bin SDs are reported instead.

    Args:
        y_true: True activity (bounded [0, 1]).
        y_pred: Model predictions.
        edges: Monotonically increasing bin edges on the true-activity axis.

    Returns:
        List of per-bin dicts: bin range, n, mean true/pred, bias, MAE, RMSE,
        prediction_std, true_std.
    """
    edges = sorted(float(e) for e in edges)
    if len(edges) < 2:
        raise ValueError("At least two edges are required")
    y_true, y_pred = _check_shapes(y_true, y_pred)

    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (y_true >= lo) & (y_true <= hi)
        else:
            mask = (y_true >= lo) & (y_true < hi)
        idx = np.where(mask)[0]
        n = int(idx.size)
        if n == 0:
            out.append({
                'bin': f'[{lo:.2f}, {hi:.2f}]',
                'n': 0,
                'mean_true': None,
                'mean_pred': None,
                'bias': None,
                'mae': None,
                'rmse': None,
                'true_std': None,
                'prediction_std': None,
            })
            continue
        yt, yp = y_true[idx], y_pred[idx]
        out.append({
            'bin': f'[{lo:.2f}, {hi:.2f}]',
            'n': n,
            'mean_true': float(np.mean(yt)),
            'mean_pred': float(np.mean(yp)),
            'bias': float(np.mean(yp - yt)),
            'mae': float(calculate_mae(yt, yp)),
            'rmse': float(calculate_rmse(yt, yp)),
            'true_std': float(np.std(yt, ddof=1)) if n > 1 else 0.0,
            'prediction_std': float(np.std(yp, ddof=1)) if n > 1 else 0.0,
        })
    return out


def dispersion_summary(
    y_true: Sequence[float],
    y_pred: Sequence[float]
) -> Dict[str, float]:
    """
    Dispersion/association summary for Audit B.

    Reports target vs prediction mean/SD, the SD ratio, and Pearson/Spearman/
    Kendall. A SD ratio well below 1 with moderate-to-weak correlation is the
    signature of prediction under-dispersion (order-preserving compression).
    """
    y_true, y_pred = _check_shapes(y_true, y_pred)
    m = calculate_all_metrics(y_true, y_pred)
    return {
        'n': int(y_true.size),
        'true_mean': float(np.mean(y_true)),
        'true_std': float(np.std(y_true, ddof=1)) if y_true.size > 1 else 0.0,
        'prediction_mean': float(np.mean(y_pred)),
        'prediction_std': float(np.std(y_pred, ddof=1)) if y_pred.size > 1 else 0.0,
        'sd_ratio_pred_over_true': (float(np.std(y_pred, ddof=1) / np.std(y_true, ddof=1))
                                    if y_true.size > 1 and np.std(y_true, ddof=1) > 0 else None),
        'pearson': m['pearson_corr'],
        'pearson_p': m['pearson_p'],
        'spearman': m['spearman_corr'],
        'spearman_p': m['spearman_p'],
        'kendall': m['kendall_corr'],
        'kendall_p': m['kendall_p'],
    }
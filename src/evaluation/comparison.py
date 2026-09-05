"""
Model comparison utilities for CRISPR-Cas9 sgRNA prediction.

Provides paired significance tests for comparing two regression models on the
same samples, and percentile-bootstrap confidence intervals for scalar
performance metrics.

Interpretation notes
--------------------
- Paired tests (paired t-test, Wilcoxon signed-rank) are applied to per-sample
  errors of two models on the SAME split. They test whether the mean/median of
  the per-sample error difference is non-zero.
- On the DeepSpCas9 validation split, the CNN was used for early
  stopping / model selection (D-006), so significance tests there are
  favourably biased for the CNN and must not be used for a fair model
  comparison. Fair significance comparisons use the Moreno-Mateos test set,
  which no model touched.
- P-values from these tests are approximate; they rely on the per-sample
  errors being comparable across models and do not account for all sources of
  dependence. They should be reported together with effect sizes and
  confidence intervals, and multiple comparisons should be acknowledged.
- MAPE is never used (see metrics.calculate_mape warning).
"""

from typing import Callable, Dict, Optional, Tuple
import numpy as np
from scipy import stats

from .metrics import (
    calculate_r2,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    calculate_mae,
    calculate_rmse
)


_PER_SAMPLE_METRIC_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    'absolute_error': lambda y_true, y_pred: np.abs(y_true - y_pred),
    'squared_error': lambda y_true, y_pred: (y_true - y_pred) ** 2,
}


def paired_error_tests(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    error_type: str = 'squared_error'
) -> Dict[str, float]:
    """
    Paired significance tests comparing two models' per-sample errors.

    Args:
        y_true: True target values
        y_pred_a: Predictions of model A
        y_pred_b: Predictions of model B
        error_type: 'squared_error' (paired t-test basis) or 'absolute_error'
            (Wilcoxon signed-rank basis). Both statistics are reported for
            the chosen error.

    Returns:
        Dictionary with:
            - 'mean_error_a', 'mean_error_b'
            - 'mean_diff' (error_a - error_b)
            - 't_stat', 't_p_value' (paired t-test; positive t -> model A
              has larger error)
            - 'wilcoxon_stat', 'wilcoxon_p_value' (signed-rank test)
            - 'n'
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred_a = np.asarray(y_pred_a, dtype=float).ravel()
    y_pred_b = np.asarray(y_pred_b, dtype=float).ravel()

    if not (y_true.shape == y_pred_a.shape == y_pred_b.shape):
        raise ValueError("y_true, y_pred_a, y_pred_b must have identical shapes")
    if error_type not in _PER_SAMPLE_METRIC_FUNCS:
        raise ValueError(
            f"Unknown error_type '{error_type}'. "
            f"Use one of {sorted(_PER_SAMPLE_METRIC_FUNCS)}"
        )

    err_func = _PER_SAMPLE_METRIC_FUNCS[error_type]
    err_a = err_func(y_true, y_pred_a)
    err_b = err_func(y_true, y_pred_b)
    diff = err_a - err_b

    n = len(y_true)
    if n < 2:
        raise ValueError("At least 2 samples are required")

    # Paired t-test on the difference of per-sample errors.
    t_stat, t_p = stats.ttest_rel(err_a, err_b)
    # Signed-rank test (zero differences dropped by the implementation).
    if np.all(np.abs(diff) < np.finfo(float).eps):
        w_stat, w_p = 0.0, 1.0
    else:
        w_stat, w_p = stats.wilcoxon(diff)

    return {
        'n': int(n),
        'error_type': error_type,
        'mean_error_a': float(np.mean(err_a)),
        'mean_error_b': float(np.mean(err_b)),
        'mean_diff': float(np.mean(diff)),   # A - B; if > 0, A has larger error
        't_stat': float(t_stat),
        't_p_value': float(t_p),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_p_value': float(w_p)
    }


def _metric_value(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """Compute the scalar value of a supported metric."""
    if metric == 'r2':
        return float(calculate_r2(y_true, y_pred))
    if metric == 'pearson':
        return float(calculate_pearson_correlation(y_true, y_pred)[0])
    if metric == 'spearman':
        return float(calculate_spearman_correlation(y_true, y_pred)[0])
    if metric == 'mae':
        return float(calculate_mae(y_true, y_pred))
    if metric == 'rmse':
        return float(calculate_rmse(y_true, y_pred))
    raise ValueError(
        f"Unsupported metric '{metric}'. "
        "Use one of: r2, pearson, spearman, mae, rmse"
    )


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Percentile bootstrap 95% confidence interval for a metric.

    The interval is the (alpha/2, 1-alpha/2) percentile interval of the
    bootstrap resampling distribution. It is computed without bias-correction
    or acceleration (it is NOT a BCa / bias-corrected percentile bootstrap).

    Args:
        y_true: True target values
        y_pred: Predicted values
        metric: 'r2', 'pearson', 'spearman', 'mae' or 'rmse'
        n_boot: Number of bootstrap resamples
        random_state: Random seed for reproducibility
        alpha: Two-sided significance level (e.g. 0.05 -> 95% CI)

    Returns:
        Dictionary with point estimate and the (alpha/2, 1-alpha/2)
        percentile confidence interval.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = len(y_true)
    if n < 2:
        raise ValueError("At least 2 samples are required")

    rng = np.random.default_rng(random_state)
    point = _metric_value(metric, y_true, y_pred)

    boot_values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = _metric_value(
            metric, y_true[idx], y_pred[idx]
        )

    lo = 100.0 * (alpha / 2)
    hi = 100.0 * (1 - alpha / 2)
    ci_lo, ci_hi = np.percentile(boot_values, [lo, hi])

    return {
        'metric': metric,
        'point': float(point),
        'ci_lower': float(ci_lo),
        'ci_upper': float(ci_hi),
        'n_boot': int(n_boot),
        'n_samples': int(n)
    }
"""
Evaluation metrics module for CRISPR-Cas9 sgRNA prediction.

This module provides functions to evaluate regression models
for predicting sgRNA activity.
"""

from typing import Dict, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    return mean_squared_error(y_true, y_pred)


def calculate_pearson_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation, p-value)
    """
    corr, p_value = stats.pearsonr(y_true, y_pred)
    return corr, p_value


def calculate_spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Important for ranking sgRNAs by activity.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation, p-value)
    """
    corr, p_value = stats.spearmanr(y_true, y_pred)
    return corr, p_value


def calculate_kendall_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Kendall tau correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation, p-value)
    """
    corr, p_value = stats.kendalltau(y_true, y_pred)
    return corr, p_value


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R-squared value
    """
    return r2_score(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value (as fraction)
        
    Warning:
        MAPE is NOT a primary metric for this project. sgRNA activity targets
        can be close to zero, which makes the per-sample percentage error
        explode and yields unstable, near-arbitrarily large values. Never use
        MAPE for model selection or to conclude one model is better. Primary
        metrics are MAE, RMSE, R², Pearson and Spearman correlations.
    """
    return mean_absolute_percentage_error(y_true, y_pred)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    pearson_corr, pearson_p = calculate_pearson_correlation(y_true, y_pred)
    spearman_corr, spearman_p = calculate_spearman_correlation(y_true, y_pred)
    kendall_corr, kendall_p = calculate_kendall_correlation(y_true, y_pred)
    
    return {
        'mae': calculate_mae(y_true, y_pred),
        'mse': calculate_mse(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'kendall_corr': kendall_corr,
        'kendall_p': kendall_p,
        'mape': calculate_mape(y_true, y_pred)
    }


def calculate_metrics_by_activity_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics stratified by activity level.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_bins: Number of bins to create
        
    Returns:
        Dictionary with metrics for each bin
    """
    # Create bins based on true values
    bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_indices = np.digitize(y_true, bins) - 1
    
    metrics_by_bin = {}
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_metrics = calculate_all_metrics(y_true[mask], y_pred[mask])
            bin_metrics['count'] = int(mask.sum())
            bin_metrics['activity_range'] = f"{bins[i]:.3f}-{bins[i+1]:.3f}"
            metrics_by_bin[f'bin_{i+1}'] = bin_metrics
    
    return metrics_by_bin


def calculate_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Calculate ranking quality metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        k: Top-k to evaluate
        
    Returns:
        Dictionary with ranking metrics
    """
    # Sort by predicted values (descending)
    pred_rank = np.argsort(-y_pred)
    true_rank = np.argsort(-y_true)
    
    # Top-k precision
    top_k_pred = set(pred_rank[:k])
    top_k_true = set(true_rank[:k])
    precision_at_k = len(top_k_pred & top_k_true) / k
    
    # Normalized Discounted Cumulative Gain (NDCG)
    # Relevance is the true activity value
    dcg = 0.0
    for i, idx in enumerate(pred_rank[:k]):
        relevance = y_true[idx]
        dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Ideal DCG
    ideal_relevance = np.sort(y_true)[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        f'precision_at_{k}': precision_at_k,
        f'ndcg_at_{k}': ndcg
    }


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Format metrics into a readable report.
    
    Args:
        metrics: Dictionary with metrics
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Model Evaluation Metrics")
    lines.append("=" * 50)
    lines.append(f"MAE:            {metrics.get('mae', 0):.4f}")
    lines.append(f"RMSE:           {metrics.get('rmse', 0):.4f}")
    lines.append(f"R²:             {metrics.get('r2', 0):.4f}")
    lines.append(f"Pearson r:      {metrics.get('pearson_corr', 0):.4f} (p={metrics.get('pearson_p', 0):.2e})")
    lines.append(f"Spearman ρ:     {metrics.get('spearman_corr', 0):.4f} (p={metrics.get('spearman_p', 0):.2e})")
    lines.append(f"Kendall τ:      {metrics.get('kendall_corr', 0):.4f} (p={metrics.get('kendall_p', 0):.2e})")
    lines.append("=" * 50)
    
    return "\n".join(lines)

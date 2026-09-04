"""
Unit tests for evaluation metrics module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mse,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    calculate_r2,
    calculate_mape,
    calculate_all_metrics,
    calculate_metrics_by_activity_bin,
    calculate_ranking_metrics,
    format_metrics_report
)


class TestMAE:
    """Tests for Mean Absolute Error."""
    
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert calculate_mae(y_true, y_pred) == 0.0
    
    def test_known_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert calculate_mae(y_true, y_pred) == 1.0


class TestRMSE:
    """Tests for Root Mean Squared Error."""
    
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert calculate_rmse(y_true, y_pred) == 0.0
    
    def test_known_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert calculate_rmse(y_true, y_pred) == 1.0


class TestPearsonCorrelation:
    """Tests for Pearson correlation."""
    
    def test_perfect_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        corr, p_value = calculate_pearson_correlation(y_true, y_pred)
        assert corr == pytest.approx(1.0)
    
    def test_perfect_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        corr, p_value = calculate_pearson_correlation(y_true, y_pred)
        assert corr == pytest.approx(-1.0)


class TestSpearmanCorrelation:
    """Tests for Spearman correlation."""
    
    def test_perfect_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        corr, p_value = calculate_spearman_correlation(y_true, y_pred)
        assert corr == pytest.approx(1.0)
    
    def test_monotonic_relationship(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 3.5, 5.5])  # Monotonically increasing
        corr, p_value = calculate_spearman_correlation(y_true, y_pred)
        assert corr == pytest.approx(1.0)


class TestR2:
    """Tests for R-squared."""
    
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert calculate_r2(y_true, y_pred) == pytest.approx(1.0)
    
    def test_worse_than_mean(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        assert calculate_r2(y_true, y_pred) < 0


class TestAllMetrics:
    """Tests for calculate_all_metrics."""
    
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = calculate_all_metrics(y_true, y_pred)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['r2'] == pytest.approx(1.0)
        assert metrics['pearson_corr'] == pytest.approx(1.0)
        assert metrics['spearman_corr'] == pytest.approx(1.0)


class TestRankingMetrics:
    """Tests for ranking metrics."""
    
    def test_perfect_ranking(self):
        y_true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        metrics = calculate_ranking_metrics(y_true, y_pred, k=3)
        
        assert metrics['precision_at_3'] == 1.0
        assert metrics['ndcg_at_3'] == pytest.approx(1.0)


class TestFormatReport:
    """Tests for report formatting."""
    
    def test_format_metrics(self):
        metrics = {
            'mae': 0.1,
            'rmse': 0.2,
            'r2': 0.8,
            'pearson_corr': 0.85,
            'pearson_p': 0.001,
            'spearman_corr': 0.82,
            'spearman_p': 0.002,
            'kendall_corr': 0.78,
            'kendall_p': 0.003
        }
        report = format_metrics_report(metrics)
        
        assert 'MAE' in report
        assert 'RMSE' in report
        assert 'R²' in report
        assert 'Pearson' in report
        assert 'Spearman' in report


class TestMAPENotPrimary:
    """Document that MAPE is not a primary metric (unstable near zero target)."""

    def test_mape_explodes_when_target_near_zero(self):
        """A target close to zero makes MAPE explode to a huge value."""
        y_true = np.array([0.001, 0.5, 0.9])
        y_pred = np.array([0.05, 0.45, 0.85])
        mape = calculate_mape(y_true, y_pred)
        # The per-sample error on the near-zero target is ~4900%
        assert mape > 5.0

    def test_primary_metrics_remain_stable(self):
        """The primary metrics stay well-behaved for the same data."""
        y_true = np.array([0.001, 0.5, 0.9])
        y_pred = np.array([0.05, 0.45, 0.85])
        assert calculate_mae(y_true, y_pred) < 0.1
        assert calculate_rmse(y_true, y_pred) < 0.1
        assert -1.0 < calculate_r2(y_true, y_pred) < 1.0
        assert -1.0 < calculate_pearson_correlation(y_true, y_pred)[0] < 1.0

    def test_format_report_does_not_emit_mape(self):
        """The formatted primary report must not show MAPE."""
        y_true = np.array([0.001, 0.5, 0.9])
        y_pred = np.array([0.05, 0.45, 0.85])
        metrics = calculate_all_metrics(y_true, y_pred)
        report = format_metrics_report(metrics)
        assert 'MAPE' not in report

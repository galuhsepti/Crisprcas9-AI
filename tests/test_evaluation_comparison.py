"""
Unit tests for the model comparison utility (Phase 6).
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.comparison import paired_error_tests, bootstrap_metric_ci


@pytest.fixture
def regression_data():
    """Realistic activities in [0, 1] with several near-zero values."""
    np.random.seed(3)
    n = 60
    y_true = np.abs(np.random.rand(n))
    y_true[y_true < 0.02] = 0.02
    y_pred_reasonable = y_true + 0.08 * np.random.randn(n)
    y_pred_bad = np.full_like(y_true, 0.5)
    return y_true, y_pred_reasonable, y_pred_bad


class TestPairedErrorTests:
    """Tests for paired_error_tests."""

    def test_bad_model_has_larger_squared_error(self, regression_data):
        y_true, y_pred_reasonable, y_pred_bad = regression_data
        res = paired_error_tests(y_true, y_pred_bad, y_pred_reasonable,
                                 error_type='squared_error')
        # Bad model (A) must have larger mean squared error than reasonable (B).
        assert res['mean_error_a'] > res['mean_error_b']
        assert res['mean_diff'] > 0
        assert res['n'] == len(y_true)

    def test_identical_predictions_no_difference(self, regression_data):
        y_true, y_pred_reasonable, _ = regression_data
        res = paired_error_tests(y_true, y_pred_reasonable, y_pred_reasonable,
                                 error_type='squared_error')
        assert res['mean_diff'] == pytest.approx(0.0)
        assert res['wilcoxon_p_value'] == pytest.approx(1.0)

    def test_reasonable_model_beats_naive(self, regression_data):
        y_true, y_pred_reasonable, y_pred_bad = regression_data
        # A = reasonable, B = bad -> A should have smaller error, p small.
        res = paired_error_tests(y_true, y_pred_reasonable, y_pred_bad,
                                 error_type='squared_error')
        assert res['mean_diff'] < 0

    def test_shape_mismatch_raises(self, regression_data):
        y_true, y_pred_reasonable, _ = regression_data
        with pytest.raises(ValueError):
            paired_error_tests(y_true, y_pred_reasonable[:10], y_pred_reasonable)

    def test_unknown_error_type_raises(self, regression_data):
        y_true, y_pred_reasonable, y_pred_bad = regression_data
        with pytest.raises(ValueError):
            paired_error_tests(y_true, y_pred_reasonable, y_pred_bad,
                               error_type='relative_error')


class TestBootstrapMetricCI:
    """Tests for bootstrap_metric_ci."""

    def test_ci_contains_point_estimate(self, regression_data):
        y_true, y_pred_reasonable, _ = regression_data
        res = bootstrap_metric_ci(y_true, y_pred_reasonable, metric='r2',
                                  n_boot=200, random_state=42)
        assert res['ci_lower'] <= res['point'] <= res['ci_upper']

    def test_reproducible_with_same_seed(self, regression_data):
        y_true, y_pred_reasonable, _ = regression_data
        r1 = bootstrap_metric_ci(y_true, y_pred_reasonable, metric='pearson',
                                 n_boot=200, random_state=42)
        r2 = bootstrap_metric_ci(y_true, y_pred_reasonable, metric='pearson',
                                 n_boot=200, random_state=42)
        assert r1['ci_lower'] == r2['ci_lower']
        assert r1['ci_upper'] == r2['ci_upper']

    def test_perfect_predictions_r2_ci_near_one(self):
        y_true = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0])
        res = bootstrap_metric_ci(y_true, y_true, metric='r2', n_boot=200,
                                  random_state=1)
        assert res['point'] == pytest.approx(1.0)
        assert res['ci_lower'] >= 0.99

    def test_unknown_metric_raises(self, regression_data):
        y_true, y_pred_reasonable, _ = regression_data
        with pytest.raises(ValueError):
            bootstrap_metric_ci(y_true, y_pred_reasonable, metric='mape',
                                n_boot=10)
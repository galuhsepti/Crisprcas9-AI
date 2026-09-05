"""Unit tests for the Phase 9D audit-only helpers (no model fitting)."""

import numpy as np
import pytest

from src.audit import activity_bin_stats, dispersion_summary, DEFAULT_ACTIVITY_EDGES


class TestActivityBinStats:
    def test_default_grid_five_bins(self):
        y_true = np.linspace(0, 1, 1000)
        y_pred = y_true * 0.5 + 0.25
        bins = activity_bin_stats(y_true, y_pred)
        assert len(bins) == 5
        assert [b['n'] for b in bins] == [200, 200, 200, 200, 200]

    def test_edge_inclusivity_last_bin(self):
        y_true = np.array([0.0, 0.199, 0.2, 0.399, 0.4, 1.0])
        y_pred = np.ones(6) * 0.5
        bins = activity_bin_stats(y_true, y_pred, [0.0, 0.2, 0.4, 1.0])
        assert [b['n'] for b in bins] == [2, 2, 2]
        assert bins[2]['mean_true'] == pytest.approx(0.7)

    def test_per_bin_statistics(self):
        y_true = np.array([0.05, 0.15, 0.1, 0.18])
        y_pred = np.array([0.2, 0.19, 0.21, 0.22])
        bins = activity_bin_stats(y_true, y_pred,
                                  [0.0, 0.2, 0.4, 1.0])
        b = bins[0]
        assert b['n'] == 4
        assert b['mean_true'] == pytest.approx(np.mean(y_true))
        assert b['mean_pred'] == pytest.approx(np.mean(y_pred))
        assert b['bias'] == pytest.approx(np.mean(y_pred - y_true))
        assert b['mae'] == pytest.approx(np.mean(np.abs(y_pred - y_true)))
        assert b['rmse'] == pytest.approx(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        assert b['true_std'] == pytest.approx(np.std(y_true, ddof=1))
        assert b['prediction_std'] == pytest.approx(np.std(y_pred, ddof=1))

    def test_empty_bin(self):
        y_true = np.array([0.1, 0.3, 0.5])
        y_pred = np.ones(3)
        bins = activity_bin_stats(y_true, y_pred, [0.0, 0.2, 0.4, 0.6, 1.0])
        assert [b['n'] for b in bins] == [1, 1, 1, 0]
        assert bins[3]['mae'] is None

    def test_single_sample_std_is_zero(self):
        y_true = np.array([0.45])
        y_pred = np.array([0.44])
        bins = activity_bin_stats(y_true, y_pred, [0.4, 0.6])
        assert bins[0]['true_std'] == 0.0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            activity_bin_stats(np.zeros(5), np.zeros(3))

    def test_edges_validation(self):
        with pytest.raises(ValueError):
            activity_bin_stats(np.zeros(5), np.zeros(5), [0.2])


class TestDispersionSummary:
    def test_identity_predictions(self):
        y = np.linspace(0.1, 0.9, 200)
        d = dispersion_summary(y, y)
        assert d['sd_ratio_pred_over_true'] == pytest.approx(1.0)
        assert d['pearson'] == pytest.approx(1.0)
        assert d['spearman'] == pytest.approx(1.0)
        assert d['kendall'] == pytest.approx(1.0)

    def test_compressed_predictions(self):
        y = np.linspace(0.1, 0.9, 500)
        p = y * 0.2 + 0.4  # strong order preservation, tight dynamic range
        d = dispersion_summary(y, p)
        assert d['sd_ratio_pred_over_true'] < 0.5
        assert d['pearson'] > 0.9
        # High correlation + low SD ratio: order preserved, range compressed.
        assert d['spearman'] > 0.9

    def test_true_std_zero_edge(self):
        y = np.full(10, 0.5)
        p = np.full(10, 0.5)
        d = dispersion_summary(y, p)
        assert d['sd_ratio_pred_over_true'] is None
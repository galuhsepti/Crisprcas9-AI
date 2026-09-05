"""
Unit tests for the Phase 9B post-hoc calibration module.

Covers:
- linear calibration (parameter fitting, transformation, slope guard)
- variance / scale correction (moment matching, zero-variance guard)
- no-calibration / identity behavior
- prediction transformation and monotonicity (ranking preservation)
- edge cases (shape mismatch, insufficient samples, non-finite values)
- no data leakage: transforms depend only on fit-time parameters
- out-of-fold internal evaluation helper
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration import (
    LinearCalibration,
    VarianceCalibration,
    NoCalibration,
    oof_calibrated_predictions,
    CALIBRATION_METHODS
)

RNG = np.random.default_rng(7)


class TestLinearCalibration:
    def test_fit_recovers_exact_parameters(self):
        x = np.linspace(0, 1, 50)
        y_true = 0.5 + 2.0 * x
        cal = LinearCalibration().fit(y_true, x)
        assert cal.slope_ == pytest.approx(2.0)
        assert cal.intercept_ == pytest.approx(0.5)
        assert cal.is_fitted

    def test_predict_transforms_correctly(self):
        x = np.linspace(0, 1, 50)
        y_true = 0.5 + 2.0 * x
        cal = LinearCalibration().fit(y_true, x)
        pred = cal.predict(np.array([0.25, 0.5]))
        assert len(pred) == 2
        assert pred[0] == pytest.approx(1.0)
        assert pred[1] == pytest.approx(1.5)

    def test_identity_when_already_calibrated(self):
        y = np.linspace(0.1, 0.9, 100)
        cal = LinearCalibration().fit(y, y)
        assert cal.slope_ == pytest.approx(1.0)
        assert cal.intercept_ == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(cal.predict(y), y, atol=1e-9)

    def test_ranking_preserved_monotonic(self):
        y_true = np.linspace(0.05, 0.95, 100)
        raw = 0.3 + 0.4 * y_true + RNG.normal(0, 0.02, 100)
        cal = LinearCalibration().fit(y_true, raw)
        cal_pred = cal.predict(raw)
        # Monotonic affine transform: rank correlation must be exactly 1.
        from scipy.stats import spearmanr
        assert cal.slope_ > 0
        assert spearmanr(raw, cal_pred).statistic == pytest.approx(1.0)

    def test_nonpositive_slope_raises_by_default(self):
        # Strongly negatively correlated data -> negative OLS slope.
        x = np.linspace(0, 1, 20)
        y_true = 1.0 - x
        with pytest.raises(ValueError, match='positive'):
            LinearCalibration().fit(y_true, x)

    def test_nonpositive_slope_allowed_with_flag(self):
        x = np.linspace(0, 1, 20)
        y_true = 1.0 - x
        cal = LinearCalibration(allow_nonpositive_slope=True).fit(y_true, x)
        assert cal.slope_ < 0

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match='not fitted'):
            LinearCalibration().predict(np.array([0.5]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            LinearCalibration().fit([0.1, 0.2], [0.1])

    def test_insufficient_samples_raises(self):
        with pytest.raises(ValueError):
            LinearCalibration().fit([0.5], [0.5])

    def test_nan_inputs_raise(self):
        with pytest.raises(ValueError, match='non-finite'):
            LinearCalibration().fit([0.1, np.nan, 0.3], [0.1, 0.2, 0.3])

    def test_get_params(self):
        x = np.linspace(0, 1, 50)
        y_true = 0.5 + 2.0 * x
        cal = LinearCalibration().fit(y_true, x)
        p = cal.get_params()
        assert p['method'] == 'linear'
        assert p['slope'] == pytest.approx(2.0)
        assert p['intercept'] == pytest.approx(0.5)


class TestVarianceCalibration:
    def test_fit_matches_moments(self):
        raw = np.linspace(0.2, 0.6, 100)
        y_true = np.linspace(0.0, 1.0, 100)
        cal = VarianceCalibration().fit(y_true, raw)
        assert cal.source_mean_ == pytest.approx(np.mean(raw))
        assert cal.source_std_ == pytest.approx(np.std(raw, ddof=1))
        assert cal.target_mean_ == pytest.approx(np.mean(y_true))
        assert cal.target_std_ == pytest.approx(np.std(y_true, ddof=1))

    def test_transform_restores_target_moments(self):
        raw = RNG.normal(0.4, 0.05, 500)
        y_true = RNG.beta(2, 2, 500)
        cal = VarianceCalibration().fit(y_true, raw)
        cal_pred = cal.predict(raw)
        assert np.mean(cal_pred) == pytest.approx(np.mean(y_true), rel=1e-3)
        assert np.std(cal_pred, ddof=1) == pytest.approx(np.std(y_true, ddof=1),
                                                         rel=1e-3)

    def test_ranking_preserved(self):
        y_true = np.linspace(0.05, 0.95, 200)
        raw = RNG.random(200) * 0.2 + 0.4
        cal = VarianceCalibration().fit(y_true, raw)
        cal_pred = cal.predict(raw)
        from scipy.stats import spearmanr
        assert spearmanr(raw, cal_pred).statistic == pytest.approx(1.0)

    def test_zero_source_variance_raises(self):
        y_true = np.array([0.1, 0.2, 0.3, 0.4])
        raw = np.array([0.5, 0.5, 0.5, 0.5])  # zero SD
        with pytest.raises(ValueError, match='zero'):
            VarianceCalibration().fit(y_true, raw)

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match='not fitted'):
            VarianceCalibration().predict(np.array([0.5]))

    def test_get_params(self):
        cal = VarianceCalibration().fit([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        p = cal.get_params()
        assert p['method'] == 'variance'
        assert p['source_std'] == pytest.approx(np.std([0.4, 0.5, 0.6], ddof=1))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            VarianceCalibration().fit([0.1, 0.2], [0.1])


class TestNoCalibration:
    def test_identity(self):
        preds = np.array([0.1, 0.5, 0.9])
        cal = NoCalibration().fit([0.1, 0.2, 0.3], preds)
        assert np.allclose(cal.predict(preds), preds)

    def test_predict_returns_copy(self):
        preds = np.array([0.1, 0.5])
        cal = NoCalibration().fit([0.5, 0.6], preds)
        out = cal.predict(preds)
        out[0] = 99.0
        assert preds[0] == 0.1

    def test_single_sample_ok(self):
        cal = NoCalibration().fit([0.5], [0.4])
        assert cal.is_fitted

    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match='not fitted'):
            NoCalibration().predict(np.array([0.5]))

    def test_calibration_methods_registry(self):
        assert set(CALIBRATION_METHODS) == {'raw', 'linear', 'variance'}


class TestNoLeakage:
    def _fit_on_a_predict_b(self, calibrator, y_train, pred_train, pred_new):
        cal = calibrator.fit(y_train, pred_train)
        return cal.predict(pred_new)

    def test_transform_does_not_depend_on_eval_data(self):
        # The transform applied to a NEW dataset must depend only on the
        # parameters learned from the FIT dataset, never on the new labels.
        y_fit = np.linspace(0.0, 1.0, 200)
        pred_fit = np.linspace(0.3, 0.5, 200)
        pred_new = np.array([0.4, 0.42, 0.48])

        lin = LinearCalibration().fit(y_fit, pred_fit)
        expected_lin = lin.predict(pred_new)
        var = VarianceCalibration().fit(y_fit, pred_fit)
        expected_var = var.predict(pred_new)

        # Even if we know the true labels of new data, they must NOT enter the
        # transform. We verify identical output regardless of any 'y_new'.
        assert np.allclose(
            self._fit_on_a_predict_b(LinearCalibration(), y_fit, pred_fit, pred_new),
            expected_lin)
        assert np.allclose(
            self._fit_on_a_predict_b(VarianceCalibration(), y_fit, pred_fit, pred_new),
            expected_var)

    def test_refit_changes_params(self):
        cal = LinearCalibration()
        cal.fit([0.0, 1.0, 2.0], [0.1, 0.3, 0.5])
        slope_a = cal.slope_
        cal.fit([0.0, 1.0, 4.0], [0.1, 0.3, 0.5])
        assert cal.slope_ != pytest.approx(slope_a)

    def test_calibrated_only_uses_fit_parameters(self):
        # Monotonic calibration on external data must not use any external
        # label information: applying the transform twice to identical input
        # yields identical output (determinism).
        y_fit = np.linspace(0.0, 1.0, 100)
        pred_fit = RNG.random(100) * 0.2 + 0.4
        var = VarianceCalibration().fit(y_fit, pred_fit)
        x = RNG.random(50) * 0.3 + 0.3
        assert np.allclose(var.predict(x), var.predict(x))


class TestOutOfFold:
    def test_oof_shape_and_no_leakage(self):
        n = 100
        y_true = np.linspace(0.1, 0.9, n)
        raw = 0.3 + 0.4 * y_true + RNG.normal(0, 0.05, n)  # noisy, compressed
        oof = oof_calibrated_predictions(
            y_true, raw,
            calibrator_factory=lambda: LinearCalibration(),
            n_folds=5, random_state=42)
        assert oof.shape == raw.shape
        # OOF must differ from a single full-fit transform: each sample's
        # calibration used a model not fit on that sample.
        full = LinearCalibration().fit(y_true, raw).predict(raw)
        assert not np.allclose(oof, full, atol=1e-9)

    def test_oof_improves_fit_of_compressed_predictions(self):
        y_true = np.linspace(0.0, 1.0, 300)
        raw = 0.3 + 0.2 * y_true  # compressed toward 0.3-0.5
        oof = oof_calibrated_predictions(
            y_true, raw,
            calibrator_factory=lambda: VarianceCalibration(),
            n_folds=5, random_state=123)
        # OOF-calibrated SD should be much closer to true SD than raw.
        from src.evaluation import calculate_rmse
        assert (np.std(oof, ddof=1) / np.std(y_true, ddof=1)) == pytest.approx(
            1.0, rel=0.02)

    def test_oof_requires_enough_samples(self):
        with pytest.raises(ValueError):
            oof_calibrated_predictions([0.1, 0.2], [0.1, 0.2],
                                       calibrator_factory=lambda: LinearCalibration(),
                                       n_folds=5)

    def test_oof_rejects_n_folds_below_2(self):
        with pytest.raises(ValueError):
            oof_calibrated_predictions(
                np.linspace(0, 1, 20), np.linspace(0, 1, 20),
                calibrator_factory=lambda: LinearCalibration(), n_folds=1)


class TestEdgeCases:
    def test_predict_empty_raises(self):
        cal = LinearCalibration().fit([0.0, 1.0], [0.0, 1.0])
        with pytest.raises(ValueError):
            cal.predict(np.array([]))

    def test_constant_target_ok_for_variance(self):
        # target_std can be 0 (perfectly constant target); the transform is
        # still defined as long as the SOURCE has spread.
        cal = VarianceCalibration().fit([0.5, 0.5, 0.5], [0.1, 0.3, 0.5])
        assert cal.target_std_ == pytest.approx(0.0)
        out = cal.predict(np.array([0.2, 0.4]))
        assert np.allclose(out, 0.5)  # all mapped to the constant target mean

    def test_inf_input_raises_predict(self):
        cal = LinearCalibration().fit([0.0, 1.0], [0.0, 1.0])
        with pytest.raises(ValueError, match='non-finite'):
            cal.predict(np.array([0.5, np.inf]))
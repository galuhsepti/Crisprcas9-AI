"""
Variance / scale-correction calibration.

Corrects only the location and spread of predictions to match the target
distribution learned from internal calibration data:

    y_cal = target_mean + (y_pred - source_mean) * (target_std / source_std)

The multiplier is positive, so the transform is monotonic and preserves the
ranking of predictions exactly. Unlike :class:`LinearCalibration`, this
method forces the prediction standard deviation to equal the target standard
deviation, isolating the pure scale-compression hypothesis.
"""

from typing import Any, Dict

import numpy as np

from .base import Calibrator
from .utils import validate_fit_inputs, validate_predict_input


class VarianceCalibration(Calibrator):
    """
    Variance / scale correction.

    Args:
        eps: Source prediction standard deviations at or below this value are
            treated as zero and rejected (scale correction is undefined).
    """

    NAME = 'variance'

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.source_mean_ = None
        self.source_std_ = None
        self.target_mean_ = None
        self.target_std_ = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> 'VarianceCalibration':
        y_true, y_pred = validate_fit_inputs(y_true, y_pred)
        source_mean = float(np.mean(y_pred))
        source_std = float(np.std(y_pred, ddof=1))
        target_mean = float(np.mean(y_true))
        target_std = float(np.std(y_true, ddof=1))
        if not (np.isfinite(source_mean) and np.isfinite(source_std)
                and np.isfinite(target_mean) and np.isfinite(target_std)):
            raise ValueError("Calibration statistics are non-finite")
        if source_std <= self.eps:
            raise ValueError(
                f"Prediction std is effectively zero ({source_std:.3e}); "
                "variance/scale calibration is undefined."
            )
        self.source_mean_ = source_mean
        self.source_std_ = source_std
        self.target_mean_ = target_mean
        self.target_std_ = target_std
        self._is_fitted = True
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Calibrator is not fitted yet. Call fit() first.")
        y_pred = validate_predict_input(y_pred)
        scale = self.target_std_ / self.source_std_
        return self.target_mean_ + (y_pred - self.source_mean_) * scale

    def get_params(self) -> Dict[str, Any]:
        if not self._is_fitted:
            return {}
        return {
            'method': self.NAME,
            'source_mean': self.source_mean_,
            'source_std': self.source_std_,
            'target_mean': self.target_mean_,
            'target_std': self.target_std_,
        }
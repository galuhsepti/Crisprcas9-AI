"""
Linear (affine) post-hoc calibration.

    y_cal = intercept + slope * y_pred

Parameters are fit with ordinary least squares (y_pred ~ y_true) on internal
calibration data. The transform is monotonic iff slope > 0, in which case the
ranking of predictions is preserved exactly (Pearson and Spearman correlations
with the true labels are unchanged by construction).
"""

from typing import Any, Dict

import numpy as np

from .base import Calibrator
from .utils import validate_fit_inputs, validate_predict_input


class LinearCalibration(Calibrator):
    """
    Linear calibration: OLS fit of ``y_true ~ y_pred``.

    Args:
        allow_nonpositive_slope: If False (default), fitting raises when the
            OLS slope is <= 0 because such a transform would invert ranking.
            Set to True only if rank inversion is explicitly acceptable.
    """

    NAME = 'linear'

    def __init__(self, allow_nonpositive_slope: bool = False) -> None:
        super().__init__()
        self.allow_nonpositive_slope = allow_nonpositive_slope
        self.intercept_ = None
        self.slope_ = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> 'LinearCalibration':
        y_true, y_pred = validate_fit_inputs(y_true, y_pred)
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        if slope <= 0 and not self.allow_nonpositive_slope:
            raise ValueError(
                f"Calibration slope must be positive to preserve ranking, "
                f"got {slope:.6f}. Set allow_nonpositive_slope=True only if "
                "rank inversion is acceptable."
            )
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            raise ValueError("Calibration parameters are non-finite")
        self.slope_ = float(slope)
        self.intercept_ = float(intercept)
        self._is_fitted = True
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Calibrator is not fitted yet. Call fit() first.")
        y_pred = validate_predict_input(y_pred)
        return self.intercept_ + self.slope_ * y_pred

    def get_params(self) -> Dict[str, Any]:
        if not self._is_fitted:
            return {}
        return {
            'method': self.NAME,
            'slope': self.slope_,
            'intercept': self.intercept_,
        }
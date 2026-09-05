"""
Trivial 'no calibration' mapping.
"""

from typing import Any, Dict

import numpy as np

from .base import Calibrator
from .utils import validate_fit_inputs, validate_predict_input


class NoCalibration(Calibrator):
    """
    Identity mapping: calibrated predictions equal raw predictions.

    Used as the 'raw' baseline / control arm of the Phase 9B experiment.
    """

    NAME = 'raw'

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> 'NoCalibration':
        # No parameters are learned, so a single sample is acceptable.
        validate_fit_inputs(y_true, y_pred, min_samples=1)
        self._is_fitted = True
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Calibrator is not fitted yet. Call fit() first.")
        y_pred = validate_predict_input(y_pred)
        return y_pred.copy()

    def get_params(self) -> Dict[str, Any]:
        return {'method': self.NAME}
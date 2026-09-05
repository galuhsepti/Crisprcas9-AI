"""
Post-hoc prediction calibration for the Phase 9B hypothesis test.

Provides three rank-preserving (monotonic or identity) calibration methods:

- :class:`NoCalibration`       - 'raw' baseline (identity)
- :class:`LinearCalibration`   - linear / affine OLS calibration
- :class:`VarianceCalibration` - variance / scale correction

and the out-of-fold evaluation helper (:func:`oof_calibrated_predictions`).
"""

from .base import Calibrator
from .noop import NoCalibration
from .linear import LinearCalibration
from .variance import VarianceCalibration
from .oof import oof_calibrated_predictions

CALIBRATION_METHODS = {
    'raw': NoCalibration,
    'linear': LinearCalibration,
    'variance': VarianceCalibration,
}

__all__ = [
    'Calibrator',
    'NoCalibration',
    'LinearCalibration',
    'VarianceCalibration',
    'oof_calibrated_predictions',
    'CALIBRATION_METHODS',
]
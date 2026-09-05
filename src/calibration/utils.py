"""
Shared input validation helpers for calibrators.
"""

from typing import Tuple

import numpy as np


MIN_FIT_SAMPLES = 2


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")


def validate_fit_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_samples: int = MIN_FIT_SAMPLES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize (true, raw prediction) calibration inputs.

    Returns raveled float arrays of identical shape.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if y_true.size < min_samples:
        raise ValueError(
            f"At least {min_samples} samples are required to fit calibration "
            f"(got {y_true.size})"
        )
    _check_finite(y_true, 'y_true')
    _check_finite(y_pred, 'y_pred')
    return y_true, y_pred


def validate_predict_input(y_pred: np.ndarray) -> np.ndarray:
    """
    Validate and standardize predictions to be transformed.

    Returns a raveled float array.
    """
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_pred.size == 0:
        raise ValueError("y_pred must contain at least one value")
    _check_finite(y_pred, 'y_pred')
    return y_pred
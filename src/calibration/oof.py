"""
Out-of-fold internal evaluation of calibration mappings.

When the full calibration dataset has no further held-out subset, applying a
calibration method out-of-fold (fit on K-1 folds, predict on the held-out
fold) yields a leakage-safe estimate of how the procedure behaves on internal
data it was not fit on.
"""

from typing import Callable

import numpy as np
from sklearn.model_selection import KFold

from .base import Calibrator
from .utils import validate_fit_inputs


def oof_calibrated_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibrator_factory: Callable[[], Calibrator],
    n_folds: int = 5,
    random_state: int = 4242
) -> np.ndarray:
    """
    Out-of-fold calibration predictions over the same dataset.

    For each fold, a fresh calibrator is fit on the complement folds and
    applied to the held-out fold. The result contains only predictions for
    samples that were not used to fit the calibrator that produced them.

    Args:
        y_true: True target values of the internal calibration dataset.
        y_pred: Raw model predictions on the SAME samples.
        calibrator_factory: Zero-argument callable returning a fresh
            Calibrator instance (a new instance per fold).
        n_folds: Number of folds (>= 2).
        random_state: Seed for the KFold shuffle.

    Returns:
        Array of calibrated predictions, same shape as y_pred.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    y_true, y_pred = validate_fit_inputs(y_true, y_pred)
    n = y_pred.size
    if n < n_folds:
        raise ValueError(
            f"At least {n_folds} samples are required for {n_folds}-fold "
            f"out-of-fold evaluation (got {n})"
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.empty_like(y_pred, dtype=float)
    for tr_idx, te_idx in kf.split(y_pred):
        calibrator = calibrator_factory()
        calibrator.fit(y_true[tr_idx], y_pred[tr_idx])
        oof[te_idx] = calibrator.predict(y_pred[te_idx])
    return oof
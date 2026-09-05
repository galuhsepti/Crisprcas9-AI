"""
Base interface for post-hoc prediction calibrators.

A calibrator learns a mapping from raw model predictions to calibrated
predictions using ONLY internal calibration data (never an external test
set). Every calibrator in this module is monotonic (rank-preserving) or
trivially the identity, so calibration never adds ranking information.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Calibrator(ABC):
    """
    Abstract post-hoc calibration transform.

    Concrete subclasses implement :meth:`fit` (learn parameters from internal
    calibration data) and :meth:`predict` (apply the transform). The learned
    transform must be monotonic so that the ranking of predictions is
    preserved exactly.
    """

    NAME = 'base'

    def __init__(self) -> None:
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> 'Calibrator':
        """Fit parameters from (true, raw prediction) pairs (internal data only)."""

    @abstractmethod
    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply the learned transform to raw predictions."""

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return fitted parameters (empty dict if not fitted)."""
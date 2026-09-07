"""
Phase 13 evaluation.

Controlled evaluation for the representation audit.

Internal evaluation reports descriptive metrics on the canonical validation set
(for model-selection context only; it does NOT prove external generalization).

Final external evaluation runs ONCE on the locked Moreno-Mateos test set AFTER
all model/design decisions are frozen. It is guarded by an explicit state
machine so external evaluation cannot occur before the experiment is frozen.

Prediction-compression analysis uses the SAME predefined activity bins as
earlier phases and does not create bins after inspecting Phase 13 results.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..evaluation.metrics import (
    calculate_all_metrics,
    calculate_mae,
    calculate_rmse,
)
from ..evaluation.comparison import paired_error_tests
from ..audit.audit import activity_bin_stats, dispersion_summary
from .stats import bootstrap_delta_mae_ci, reproducible_bootstrap_seed

DEFAULT_ACTIVITY_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Experiment state machine (guards against premature external evaluation)
# ---------------------------------------------------------------------------
class ExperimentStateMachine:
    """
    Deterministic transition guard for the Phase 13 experiment.

    Allowed path:
        DESIGN -> FEASIBILITY -> TRAINING -> INTERNAL_EVALUATION
        -> FREEZE -> FINAL_EXTERNAL_EVALUATION -> COMPLETE

    The only way to reach FINAL_EXTERNAL_EVALUATION is through FREEZE, which
    requires an explicit call to freeze(). Any attempt to run external
    evaluation before freeze raises RuntimeError.
    """

    _ORDER = [
        "DESIGN",
        "FEASIBILITY",
        "TRAINING",
        "INTERNAL_EVALUATION",
        "FREEZE",
        "FINAL_EXTERNAL_EVALUATION",
        "COMPLETE",
    ]

    def __init__(self, start: str = "DESIGN"):
        if start not in self._ORDER:
            raise ValueError(f"Unknown state '{start}'")
        self.state = start
        self.frozen = False

    def transition(self, target: str) -> None:
        if target not in self._ORDER:
            raise ValueError(f"Unknown target state '{target}'")
        current_idx = self._ORDER.index(self.state)
        target_idx = self._ORDER.index(target)
        # Allow only forward (or equal) transitions; never skip FREEZE to reach
        # FINAL_EXTERNAL_EVALUATION.
        if target_idx < current_idx:
            raise RuntimeError(
                f"Illegal backward transition {self.state} -> {target}"
            )
        if target == "FINAL_EXTERNAL_EVALUATION":
            if current_idx < self._ORDER.index("FREEZE"):
                raise RuntimeError(
                    "External evaluation is forbidden before the experiment is "
                    "frozen. Transition through FREEZE first."
                )
        self.state = target

    def freeze(self) -> None:
        self.transition("FREEZE")
        self.frozen = True

    def require(self, *states: str) -> None:
        if self.state not in states:
            raise RuntimeError(
                f"Current state '{self.state}' is not one of {states}"
            )


# ---------------------------------------------------------------------------
# Internal evaluation (validation set, descriptive)
# ---------------------------------------------------------------------------
def run_internal_evaluation(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    label: str = "model",
) -> Dict[str, object]:
    """
    Descriptive metric summary on the validation set.

    Validation is model-selection context only and must NOT be used to conclude
    external generalization.
    """
    metrics = calculate_all_metrics(y_true, y_pred)
    dispersion = dispersion_summary(y_true, y_pred)
    bins = activity_bin_stats(y_true, y_pred, edges=DEFAULT_ACTIVITY_EDGES)
    res = y_true - y_pred
    return {
        "label": label,
        "metrics": metrics,
        "dispersion": dispersion,
        "activity_bins": bins,
        "residual_mean": float(np.mean(res)),
        "residual_std": float(np.std(res, ddof=1)) if res.size > 1 else 0.0,
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred, ddof=1)) if np.size(y_pred) > 1 else 0.0,
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true, ddof=1)) if np.size(y_true) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Final external evaluation (locked Moreno-Mateos, only after freeze)
# ---------------------------------------------------------------------------
def run_final_external_evaluation(
    y_true: Sequence[float],
    y_pred_new: Sequence[float],
    y_pred_canonical: Optional[Sequence[float]],
    state_machine: ExperimentStateMachine,
    n_boot: int = 1000,
    bootstrap_seed: int = 20260906,
    paired_test_error_type: str = "absolute_error",
) -> Dict[str, object]:
    """
    Run the ONE permitted external evaluation.

    Guards:
      - Requires state == FREEZE (or later), enforced by the state machine.
      - Primary external metric is MAE; secondary RMSE, R2, Pearson, Spearman.
      - The canonical external predictions, when provided, are used for a
        paired per-example comparison and a paired bootstrap CI for
        mean(delta AE_new - AE_canonical).
    """
    try:
        state_machine.transition("FINAL_EXTERNAL_EVALUATION")
    except RuntimeError:
        raise

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred_new = np.asarray(y_pred_new, dtype=float).ravel()

    result: Dict[str, object] = {}
    result["metrics_new"] = calculate_all_metrics(y_true, y_pred_new)
    result["dispersion_new"] = dispersion_summary(y_true, y_pred_new)
    result["activity_bins_new"] = activity_bin_stats(
        y_true, y_pred_new, edges=DEFAULT_ACTIVITY_EDGES
    )
    res_new = y_true - y_pred_new
    result["residual_mean_new"] = float(np.mean(res_new))
    result["residual_std_new"] = float(np.std(res_new, ddof=1)) if res_new.size > 1 else 0.0
    result["prediction_mean_new"] = float(np.mean(y_pred_new))
    result["prediction_std_new"] = float(np.std(y_pred_new, ddof=1)) if np.size(y_pred_new) > 1 else 0.0
    result["target_mean_new"] = float(np.mean(y_true))
    result["target_std_new"] = float(np.std(y_true, ddof=1)) if np.size(y_true) > 1 else 0.0

    # Paired comparison vs canonical, if canonical predictions are provided.
    if y_pred_canonical is not None:
        y_pred_canonical = np.asarray(y_pred_canonical, dtype=float).ravel()
        if y_pred_canonical.shape != y_pred_new.shape:
            raise ValueError(
                "Canonical and new predictions must be aligned sequence-by-sequence"
            )
        result["metrics_canonical"] = calculate_all_metrics(y_true, y_pred_canonical)
        result["dispersion_canonical"] = dispersion_summary(y_true, y_pred_canonical)

        ae_new = np.abs(y_true - y_pred_new)
        ae_canonical = np.abs(y_true - y_pred_canonical)
        rng = reproducible_bootstrap_seed(bootstrap_seed)
        result["delta_mae_bootstrap"] = bootstrap_delta_mae_ci(
            ae_new, ae_canonical, n_boot=n_boot, rng=rng
        )
        # Paired statistical test on per-sample absolute error (new vs canonical).
        result["paired_test"] = paired_error_tests(
            y_true, y_pred_new, y_pred_canonical, error_type=paired_test_error_type
        )

    state_machine.transition("COMPLETE")
    return result


# ---------------------------------------------------------------------------
# Prediction compression analysis
# ---------------------------------------------------------------------------
def prediction_compression_analysis(
    y_true_external: Sequence[float],
    y_pred_new: Sequence[float],
    y_pred_canonical: Sequence[float],
) -> Dict[str, object]:
    """
    Compare prediction dispersion of the new and canonical models vs the true
    external target dispersion, and report per-activity-bin errors on the
    SAME predefined bins.
    """
    y_true = np.asarray(y_true_external, dtype=float).ravel()
    y_pred_new = np.asarray(y_pred_new, dtype=float).ravel()
    y_pred_canonical = np.asarray(y_pred_canonical, dtype=float).ravel()

    true_std = float(np.std(y_true, ddof=1)) if y_true.size > 1 else 0.0
    new_std = float(np.std(y_pred_new, ddof=1)) if y_pred_new.size > 1 else 0.0
    canon_std = float(np.std(y_pred_canonical, ddof=1)) if y_pred_canonical.size > 1 else 0.0

    return {
        "target_std": true_std,
        "new_pred_std": new_std,
        "canonical_pred_std": canon_std,
        "new_sd_over_target_sd": new_std / true_std if true_std > 0 else None,
        "canonical_sd_over_target_sd": canon_std / true_std if true_std > 0 else None,
        "new_bins": activity_bin_stats(y_true, y_pred_new, edges=DEFAULT_ACTIVITY_EDGES),
        "canonical_bins": activity_bin_stats(
            y_true, y_pred_canonical, edges=DEFAULT_ACTIVITY_EDGES
        ),
    }

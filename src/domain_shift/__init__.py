"""
Phase 14 - Domain-Shift Attribution / Robustness Audit.

Read-only analysis of frozen canonical artifacts to identify which observable
domain-shift characteristics are most consistently associated with external
model error. No training, no tuning, no calibration, no data expansion.

Associational only: results are interpreted as "consistent with domain shift"
being an important contributor, never as causal mechanisms.
"""

from .config import Phase14DomainShiftConfig
from .state_machine import Phase14StateMachine
from .stratification import construct_gc_edges, bin_indices
from .statistics import (
    reproducible_bootstrap_seed,
    percentile_bootstrap_ci,
    bootstrap_mean_delta_ci,
    bh_adjust,
)

__version__ = "0.1.0"
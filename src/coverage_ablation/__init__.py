"""Phase 12 - Controlled Data-Coverage Ablation (analysis-only feasibility gate)."""

from . import analysis, config, matching, stats  # noqa: F401

from .config import (  # noqa: F401
    ARMS,
    ARM_DEFINITIONS,
    DECISION_GATES,
    HIGH_ACTIVITY_THRESHOLD,
    MODELS,
    PRIMARY_CONTRASTS,
    PROPOSED_TOLERANCES,
    SEED,
    STOP_CONDITIONS,
    VALIDATION_POLICY,
)
from .matching import (  # noqa: F401
    activity_matching_report,
    decide_gate,
    feasibility_audit,
    pool_equivalence_audit,
    sample_size_report,
    sequence_matching_report,
)
from .analysis import arm_pool_metadata, high_activity_summary  # noqa: F401
from .stats import contrast_report  # noqa: F401

__all__ = [
    "ARMS", "ARM_DEFINITIONS", "DECISION_GATES", "HIGH_ACTIVITY_THRESHOLD",
    "MODELS", "PRIMARY_CONTRASTS", "PROPOSED_TOLERANCES", "SEED",
    "STOP_CONDITIONS", "VALIDATION_POLICY",
    "activity_matching_report", "decide_gate", "feasibility_audit",
    "pool_equivalence_audit", "sample_size_report", "sequence_matching_report",
    "arm_pool_metadata", "high_activity_summary", "contrast_report",
    "analysis", "config", "matching", "stats",
]
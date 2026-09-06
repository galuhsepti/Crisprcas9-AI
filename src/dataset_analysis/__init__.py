"""Phase 11 - targeted dataset analysis (analysis-only)."""

from .analysis import (  # noqa: F401
    ACTIVITY_EDGES,
    HYPOTHESES,
    VERDICTS,
    activity_descriptive,
    arm_descriptor,
    classify_shift_addressed,
    coverage_by_bin,
    coverage_expansion,
    decision_gate,
    duplicate_and_conflict_report,
    high_activity_coverage,
    hypothesis_assessment,
    js_divergence,
    label_quality_audit,
    pool_composition_table,
    sequence_diversity_report,
    summarize_generated_external,
    wasserstein_1d,
)
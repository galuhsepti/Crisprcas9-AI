"""
Phase 12 - Controlled Data-Coverage Ablation: pre-registered protocol constants.

Everything here is frozen BEFORE any arm construction or any model training.
No value in this module may depend on Moreno-Mateos or on any external result.
The locked Phase 10 / Phase 11 results may be *quoted* by the runner, never
referenced to choose a design constant.

Matching tolerances below are PROPOSED values, documented so that any future
run interprets them identically. If a tolerance cannot be met, the protocol
requires reporting the quantitative mismatch instead of claiming a match.
"""

import os
from pathlib import Path

from ..multidataset.config import (
    CANONICAL_MODEL_FILES,
    EFFECT_THRESHOLD,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    GEOMETRY_TRANSFORMATION,
    LABEL_HARMONIZATION,
    PRIMARY_MODEL,
    PRIMARY_ARM,
    BASELINE_ARM,
)

PHASE = "12"
EXPERIMENT_NAME_PREFIX = "phase12_controlled_data_coverage_ablation"

RESULTS_DIR = Path("results/experiments")
POOLS_DIR = Path("results/pools")
FIGURES_DIR = Path("results/figures")

# ---------------------------------------------------------------------------
# Activity conventions (identical to Phase 10/11; never re-derived)
# ---------------------------------------------------------------------------
ACTIVITY_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
HIGH_ACTIVITY_THRESHOLD = 0.8
ACTIVITY_QUANTILES = [5, 10, 25, 50, 75, 90, 95]

# Fixed depth of composition summaries.
KMER_VALUES = [2, 3]

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
CANONICAL_SOURCE = "deepspcas9"
CANONICAL_CONTEXT = ("DeepSpCas9 pooled SpCas9 screen (human), NGS "
                     "modified-read fraction (modFreq)")
ADDITIONAL_SOURCE = "deephf"
ADDITIONAL_CONTEXT = ("DeepHF pooled SpCas9 screens (human, HEK293T/HCT116/T "
                      "cells); edited-read fraction")

# ---------------------------------------------------------------------------
# Proposed matching tolerances (documented; only relevant if feasibility passes)
# ---------------------------------------------------------------------------
PROPOSED_TOLERANCES = {
    "sample_size_rel_pct": 10.0,          # |n_b - n_a|/n_a*100 <= 10%
    "activity_wasserstein": 0.05,         # 1-D W between label samples
    "activity_bin_max_delta": 0.05,       # max |fraction_b - fraction_a| per canonical bin
    "activity_cohens_d": 0.20,            # |d| for labels
    "gc_cohens_d": 0.20,                  # |d| for sequence GC
    "kmer3_js_divergence": 0.02,          # JSD(3-mer) between pools
    "note": ("Tolerances are pre-defined descriptive thresholds. They are "
             "NOT a single composite score; each dimension is reported "
             "independently."),
}

# ---------------------------------------------------------------------------
# Pre-registered arms (Section 5 of the protocol)
# ---------------------------------------------------------------------------
ARMS = ["A", "B", "C", "D", "E"]

ARM_DEFINITIONS = {
    "A": {
        "name": "canonical baseline",
        "composition": "original DeepSpCas9 training split (canonical, 85/15 seed 42)",
        "role": "reference",
    },
    "B": {
        "name": "sample-size control",
        "goal": ("isolate sample count: larger n with approximately the SAME "
                 "activity distribution AND the SAME sequence/domain "
                 "distribution as the canonical baseline"),
        "status": "pre-registered; construction hinges on feasibility audit",
    },
    "C": {
        "name": "activity-coverage control",
        "goal": ("isolate activity coverage: approximately same total sample "
                 "count as the chosen diversified control, deliberately "
                 "broader activity coverage, diversity controlled as far as "
                 "feasible"),
        "status": "pre-registered; construction hinges on feasibility audit",
    },
    "D": {
        "name": "diversity control",
        "goal": ("isolate sequence/domain diversity: approximately same total "
                 "sample count, comparable activity distribution, increased "
                 "sequence/domain diversity"),
        "status": "pre-registered; construction hinges on feasibility audit",
    },
    "E": {
        "name": "full diversified reference",
        "composition": "Phase 10 diversified pool (DeepSpCas9 train + DeepHF, 56,894)",
        "status": "reference only; construction frozen from Phase 10; not redesigned",
    },
}

PRIMARY_CONTRASTS = [
    ("B", "A", "sample-size effect"),
    ("C", "A", "activity-coverage effect"),
    ("D", "A", "sequence/domain-diversity effect"),
    ("E", "A", "full diversification effect"),
]

# ---------------------------------------------------------------------------
# Feasibility audit
# ---------------------------------------------------------------------------
STOP_CONDITIONS = {
    1: "No valid matched control can be constructed.",
    2: "Matching requires Moreno-Mateos.",
    3: "Matching requires selecting subsets based on external performance.",
    4: "Label harmonization is unresolved.",
    5: "Dataset provenance is incomplete.",
    6: "Sequence normalization is ambiguous.",
    7: "Activity coverage cannot be controlled adequately.",
    8: "Sequence/domain diversity cannot be separated from activity coverage.",
    9: "Sample size cannot be controlled.",
    10: "Canonical model protocol cannot be preserved.",
    11: "A new architecture would be required to run the comparison fairly.",
}

DECISION_GATES = {
    "A": "ACTIVITY COVERAGE IS THE STRONGEST SUPPORTED LEVER",
    "B": "SEQUENCE/DOMAIN DIVERSITY IS THE STRONGEST SUPPORTED LEVER",
    "C": "SAMPLE SIZE IS THE STRONGEST SUPPORTED LEVER",
    "D": "MULTIPLE LEVERS CONTRIBUTE; NO SINGLE DRIVER IDENTIFIED",
    "E": "FACTORS REMAIN NON-IDENTIFIABLE",
    "F": "CONTROLLED ABLATION INFEASIBLE WITH AVAILABLE DATA",
}

# Model scope (Section 13/14): canonical families only, canonical configs.
MODELS = ["random_forest", "xgboost", "cnn"]
MODEL_SCOPE_NOTE = (
    "Only Random Forest / XGBoost / CNN with the Phases 3-5 canonical "
    "configurations are allowed. No new architecture, no per-arm tuning. "
    "The only manipulated variable is training-data composition."
)
VALIDATION_POLICY = (
    "All arms are evaluated on the shared canonical DeepSpCas9 validation "
    "split (1,518 rows) with identically the Phases 3-5 protocol used for "
    "every Phase 10 arm."
)

# ---------------------------------------------------------------------------
# Deterministic seeds (defined pre-training; never chosen from results)
# ---------------------------------------------------------------------------
SEED = 42
ALT_SEEDS = [7, 2023, 12345]

# Exposure guard for the Moreno-Mateos freeze.
MORENO_MATEOS_FREEZE_POLICY = (
    "Moreno-Mateos is COMPLETELY LOCKED. It must not be read, matched against, "
    "inspect, or used to select any subset, seed, threshold, hyperparameter, "
    "arm, or analysis. Phase 10/11 result JSONs may be quoted as historical "
    "records only. No new external evaluation may be performed unless the "
    "entire protocol above is frozen first, and then only once, as a "
    "confirmatory reference."
)
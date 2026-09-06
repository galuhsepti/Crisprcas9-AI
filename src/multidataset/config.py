"""
Phase 10 - Multi-Dataset Training-Domain Diversification: pre-registered
protocol, constants, and the deterministic verdict rule.

Pre-registration discipline (written BEFORE any Moreno-Mateos evaluation):
-------------------------------------------------------------------------
- The only additional training dataset is DeepHF (59,852 raw sgRNAs; official
  DeepHF 21-mer schema, Wt_Efficiency label). It was chosen and fixed locked
  on documented provenance and label semantics, NOT on external performance.
- The only diversified arm is D1 (DeepSpCas9 + DeepHF). No arm was selected
  by looking at Moreno-Mateos results.
- Moreno-Mateos is evaluated exactly ONCE per pre-registered arm/model after
  the entire protocol below is frozen.
"""

# ----------------------------------------------------------------------------
# Sequence geometry (must match config.yaml: guide [4:24], PAM [24:27])
# ----------------------------------------------------------------------------
CONTEXT_LENGTH = 30
GUIDE_START = 4
GUIDE_LENGTH = 20
PAM_START = 24
PAM_LENGTH = 3

# DeepHF provides only the 20 bp guide + 3 bp PAM (official DeepHF training
# schema; no genomic flanking sequence is available in the source). Under the
# canonical 30-mer geometry the flanking positions [0:4] and [27:30] cannot be
# recovered. Documented transformation: constant nucleotide fill.
FLANK5_PADDING = "AAAA"   # canonical 5' context [0:4]
FLANK3_PADDING = "AAA"    # canonical 3' context [27:30]

GEOMETRY_TRANSFORMATION = {
    "dataset": "deephf",
    "description": ("DeepHF provides guide(20 bp)+PAM(3 bp) only. Canonical "
                    "30-mer is constructed as FLANK5_PADDING + guide + PAM + "
                    "FLANK3_PADDING, i.e. positions [4:24]=guide, [24:27]=PAM, "
                    "[0:4]='AAAA', [27:30]='AAA'."),
    "justification": (
        "The source format itself contains no genomic flank; a constant, "
        "information-free fill avoids fabricating covariate structure. Phase 7 "
        "showed the guide+PAM region carries ~70% of the predictive signal for "
        "the canonical model families. At evaluation time the locked Moreno-"
        "Mateos targets carry genuine flanks, so the diversified model must "
        "still process true 30-mers."),
    "flank5_padding": FLANK5_PADDING,
    "flank3_padding": FLANK3_PADDING,
    "limitation": ("DeepHF rows carry a systematic constant-flank signature; "
                   "the CNN/tabular models may or may not key on it. This is "
                   "documented and bounded by the guide-centred evidence of "
                   "Phase 7."),
}

# ----------------------------------------------------------------------------
# Data sources (paths relative to project root)
# ----------------------------------------------------------------------------
DEEP_SPCAS9_PATH = "data/raw/DeepSpCas9.csv"
MORENO_MATEOS_PATH = "data/raw/Moreno-Mateos.csv"
DEEP_HF_PATH = ("data/external/Benchmarking-CRISPR-on-tools/"
                "Training/DeepHF_training.xlsx")

# ----------------------------------------------------------------------------
# Canonical model artifacts (used ONLY to read frozen hyperparameters)
# ----------------------------------------------------------------------------
CANONICAL_MODEL_FILES = {
    "random_forest": "models/rf_baseline_fixed_20260905_001107.pkl",
    "xgboost": "models/xgboost_baseline_20260905_002839.pkl",
    "cnn": "models/cnn_baseline_20260905_011720.pt",
}

MODEL_FAMILIES = ["random_forest", "xgboost", "cnn"]
MODEL_LABELS = {
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "cnn": "CNN",
}

# ----------------------------------------------------------------------------
# Experimental arms (frozen)
# ----------------------------------------------------------------------------
ARMS = ["B", "D1"]
ARM_DEFINITIONS = {
    "B": "DeepSpCas9 canonical training split only (reproduces Phases 3-5).",
    "D1": "DeepSpCas9 canonical training split + DeepHF (PRIMARY DIVERSIFIED "
           "ARM - pre-registered).",
}
PRIMARY_ARM = "D1"
BASELINE_ARM = "B"

# ----------------------------------------------------------------------------
# Primary endpoint (pre-registered, defined before external evaluation)
# ----------------------------------------------------------------------------
# Primary endpoint per model family:
#   d_external = mean(|e_B| - |e_D1|) over the locked Moreno-Mateos samples,
#   computed on per-sample absolute error.
#   d > 0  <=>  MAE(D1) < MAE(B)  <=> diversified training improves MAE.
# The primary model family of the thesis is the CNN (Phase 5); RF and XGBoost
# are pre-specified supporting families.
PRIMARY_MODEL = "cnn"
PRIMARY_ENDPOINT = {
    "name": "Change in external MAE (d = mean(|e_B| - |e_D1|))",
    "model_family": PRIMARY_MODEL,
    "direction": "positive d = diversified improves absolute error",
    "delta_mae_relation": "d = - (MAE_D1 - MAE_B)",
}
SECONDARY_ENDPOINTS = [
    "delta RMSE", "delta R2", "delta Pearson", "delta Spearman",
    "prediction dispersion (pred SD ratio)", "activity-range bias (per-bin)",
]

EFFECT_THRESHOLD = 0.01        # |d| must reach this to count as an effect
CONTRADICTION_THRESHOLD = 0.02 # a supporting family worsening by >= this
                               # blocks the 'SUPPORTED' designation
CI_HALFWIDTH_GUARD = 0.03      # bootstrap CI half-width >= this -> insufficient
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

# ----------------------------------------------------------------------------
# Label harmonization protocol (frozen)
# ----------------------------------------------------------------------------
# Both DeepSpCas9 activity (modFreq) and DeepHF Wt_Efficiency are continuous
# on-target SpCas9 editing-efficiency fractions on a [0, 1] scale. The simplest
# scientifically defensible harmonization is RAW-SCALE (identity): neither
# dataset is transformed. Within-dataset mean differences (0.42 vs 0.73) are
# documented as protocol-level label-distribution shift and are themselves part
# of the diversification being tested.
LABEL_HARMONIZATION = {
    "method": "raw_scale_identity",
    "method_rationale": (
        "both labels are measured 0-1 fractions of on-target SpCas9 editing "
        "efficiency (DeepSpCas9: modification frequency from pooled NGS; "
        "DeepHF: Wt_Efficiency from pooled deep-sequencing of edited reads; "
        "see Supplementary Data1 'Xiang' row of the benchmark repository). "
        "No monotonic rescaling is imposed because the quantities share the "
        "same operational scale; any within-dataset recentring would erase "
        "part of the domain difference this phase must test."),
    "per_dataset": {
        "deepspcas9": {"raw_meaning": "modification frequency (0-1)",
                       "higher_is_better": True, "was_normalized": False,
                       "continuous": True, "same_biological_quantity": True},
        "deephf": {"raw_meaning": "Wt_Efficiency (0-1), wild-type tracrRNA "
                                  "editing efficiency",
                   "higher_is_better": True, "was_normalized": False,
                   "continuous": True, "same_biological_quantity": True},
    },
    "known_difference": ("DeepSpCas9 labels concentrate below 0.95 (mean "
                         "0.42) while DeepHF labels are high-activity "
                         "enriched (mean 0.73) - a documented protocol-level "
                         "distribution shift, not a correctness issue."),
}

# ----------------------------------------------------------------------------
# Contamination policy (frozen)
# ----------------------------------------------------------------------------
# A candidate dataset is eligible only if it has ZERO exact-sequence overlap
# (30-mer and 20-mer guide, forward and reverse-complement) with the LOCKED
# Moreno-Mateos external test. This was verified for DeepHF before inclusion;
# candidates that failed semantic/geometry checks are excluded regardless.
DUPLICATE_POLICY = (
    "Within-corpus exact-duplicate 30-mers are excluded with reporting. "
    "Cross-corpus same-guide co-occurrences (406 DeepSpCas9 guides also "
    "present in DeepHF) are RETAINED as distinct independent experimental "
    "measurements with source metadata, NOT averaged: the two laboratories "
    "measured these guides in different assays, and averaging would erase "
    "provenance. No within-corpus label conflicts exist."
)

# ----------------------------------------------------------------------------
# Verdict rule (pre-registered; deterministic function, see decide_verdict)
# ----------------------------------------------------------------------------
VERDICT_RULE = {
    "SUPPORTED": ("primary (CNN) d > 0 with bootstrap CI excluding 0 and "
                  "|d| >= EFFECT_THRESHOLD, and no supporting family shows a "
                  "contradicting degradation >= CONTRADICTION_THRESHOLD."),
    "PARTIAL_SUPPORT": ("primary improves but effect size below "
                        "EFFECT_THRESHOLD, or mixed/contradictory across "
                        "families, or statistics marginal."),
    "NOT_SUPPORTED": ("primary d <= 0 with CI at or below zero (diversification "
                      "does not improve external MAE)."),
    "INSUFFICIENT_EVIDENCE": ("bootstrap CI too wide relative to the effect "
                              "(CI half-width >= CI_HALFWIDTH_GUARD) to "
                              "distinguish any meaningful improvement."),
}


def decide_verdict(primary_d, primary_ci_lower, primary_ci_upper,
                   family_deltas_mae):
    """
    Frozen, pre-registered decision rule. Takes only the post-hoc external
    statistics and maps them to one of the four verdicts.

    Args:
        primary_d: mean(|e_B| - |e_D1|) for the primary (CNN) family. Positive
            means the diversified arm improved absolute error.
        primary_ci_lower / primary_ci_upper: bootstrap 95% CI for primary_d.
        family_deltas_mae: dict mapping each family to delta MAE (D1 - B).
            Negative delta MAE = improvement.

    Returns:
        str: "SUPPORTED", "PARTIAL_SUPPORT", "NOT_SUPPORTED" or
             "INSUFFICIENT_EVIDENCE".
    """
    half_width = 0.5 * (primary_ci_upper - primary_ci_lower)
    if half_width >= CI_HALFWIDTH_GUARD * 2:
        return "INSUFFICIENT_EVIDENCE"

    contradicted = any(
        fam != PRIMARY_MODEL and delta_mae > CONTRADICTION_THRESHOLD
        for fam, delta_mae in family_deltas_mae.items())

    if primary_d >= EFFECT_THRESHOLD and primary_ci_lower > 0 and not contradicted:
        return "SUPPORTED"
    if primary_d >= EFFECT_THRESHOLD and primary_ci_lower > 0 and contradicted:
        return "PARTIAL_SUPPORT"
    if primary_d > 0 and primary_d < EFFECT_THRESHOLD:
        return "PARTIAL_SUPPORT"
    if primary_d <= 0:
        return "NOT_SUPPORTED"
    return "PARTIAL_SUPPORT"
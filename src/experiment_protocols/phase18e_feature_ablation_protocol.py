"""Locked Phase 18E XGBoost feature-ablation preregistration.

This module describes future Phase 18F work. It cannot fit models, generate
predictions, or access the locked external dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.bioinformatics.sequence_features import SequenceFeatureExtractor
from src.experiment_protocols.multidomain_protocol import policies
from src.experiment_protocols.phase18c_access import phase18c_access_guard

ROOT = Path(__file__).resolve().parents[2]
PHASE = "18E"
PROTOCOL_VERSION = "phase18e-xgboost-feature-ablation-v1"
FUTURE_PHASE = "Phase 18F - Locked XGBoost Feature-Ablation Experiment"
DOMAINS = ("deepspcas9", "crispron_xiang_luo")
DOMAIN_CODES = {DOMAINS[0]: "A", DOMAINS[1]: "B"}
FEATURE_FAMILIES = (
    "GC_AND_SKEW",
    "NUCLEOTIDE_COMPOSITION",
    "DINUCLEOTIDE_FREQUENCY",
    "GLOBAL_KMER",
    "ENTROPY_COMPLEXITY",
    "POSITION_SPECIFIC_NUCLEOTIDE",
)
OPTIONAL_FAMILY_ONLY = (
    "POSITION_SPECIFIC_NUCLEOTIDE",
    "GLOBAL_KMER",
)
CANONICAL_FEATURE_COUNT = 197
CANONICAL_FEATURE_SHA256 = (
    "9adf5c5e285338ac3cf4bfd3987b1245ae61f7a18d2cbf78f6b96d6079c84dab"
)
SPLIT_SHA256 = (
    "b7e043b30b3d070e1a41c461ad83774aec21a6abbfe9948cf1387a4ae49ab4a8"
)
PHASE18B_PROTOCOL_SHA256 = (
    "79ec4f03fe7314bc54a337c0d235c4f40e93b441aa40633828e1a50a3ee4d49b"
)
PHASE18B_PROTOCOL_PATH = ROOT / (
    "results/phase18b_multidomain_protocol_20260913_180731.json"
)
PHASE18C_EXECUTION_LOCK_PATH = ROOT / (
    "results/phase18c/"
    "79ec4f03fe7314bc54a337c0d235c4f40e93b441aa40633828e1a50a3ee4d49b/"
    "campaign_20260914_141742/execution_lock.json"
)
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000
NOMINAL_CONFIDENCE = 0.95
CONFIRMATORY_INTERVAL_N = len(FEATURE_FAMILIES) * len(DOMAINS) * 2
BONFERRONI_ALPHA = (1.0 - NOMINAL_CONFIDENCE) / CONFIRMATORY_INTERVAL_N
SIMULTANEOUS_CONFIDENCE = 1.0 - BONFERRONI_ALPHA
SIMULTANEOUS_QUANTILES = (
    BONFERRONI_ALPHA / 2.0,
    1.0 - BONFERRONI_ALPHA / 2.0,
)

EXPECTED_XGBOOST_CONFIGURATION = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": 4,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "verbosity": 0,
    "early_stopping": False,
    "hyperparameter_tuning": False,
    "eval_set": None,
    "sample_weight": None,
}

FAMILY_HYPOTHESES = {
    "GC_AND_SKEW": (
        "GC and skew summaries provide predictive information not recoverable "
        "from the remaining sequence representations."
    ),
    "NUCLEOTIDE_COMPOSITION": (
        "Global nucleotide composition and heterogeneity provide unique "
        "predictive information beyond GC, k-mers, and positional identity."
    ),
    "DINUCLEOTIDE_FREQUENCY": (
        "Global dinucleotide frequencies provide unique predictive "
        "information "
        "despite overlap with the global 2-mer family."
    ),
    "GLOBAL_KMER": (
        "Global 2-mer and 3-mer frequencies provide unique order-local motif "
        "information beyond composition and spacer-position indicators."
    ),
    "ENTROPY_COMPLEXITY": (
        "K-mer entropy and complexity summaries provide unique predictive "
        "information beyond their underlying k-mer frequencies."
    ),
    "POSITION_SPECIFIC_NUCLEOTIDE": (
        "Spacer-position nucleotide indicators provide unique predictive "
        "information beyond order-agnostic sequence summaries."
    ),
}


def serialize(payload: object) -> str:
    """Return canonical JSON for deterministic hashing and artifacts."""
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def fingerprint(payload: object) -> str:
    return hashlib.sha256(serialize(payload).encode("utf-8")).hexdigest()


@contextmanager
def phase18e_protocol_guard():
    """Reject locked-data access and training calls for this process."""
    previous_profile = sys.getprofile()

    def no_training(frame, event, arg):
        if event == "call" and frame.f_code.co_name in {
            "fit",
            "partial_fit",
            "cross_validate",
            "hyperparameter_tuning",
            "predict",
            "train",
            "backward",
            "step",
        }:
            module = frame.f_globals.get("__name__", "")
            if module.startswith(
                ("torch", "sklearn", "xgboost", "src.models")
            ):
                raise RuntimeError(
                    "Phase 18E model training/prediction prohibited"
                )
        if previous_profile:
            previous_profile(frame, event, arg)

    with phase18c_access_guard():
        sys.setprofile(no_training)
        try:
            yield
        finally:
            sys.setprofile(previous_profile)


def canonical_feature_names() -> tuple[str, ...]:
    """Read ordered names from the exact registered unfitted extractor."""
    names = tuple(SequenceFeatureExtractor().get_feature_names())
    if len(names) != CANONICAL_FEATURE_COUNT or len(set(names)) != len(names):
        raise RuntimeError("Canonical XGBoost feature identity/count changed")
    if fingerprint(list(names)) != CANONICAL_FEATURE_SHA256:
        raise RuntimeError("Canonical XGBoost feature ordering changed")
    return names


def feature_family_registry() -> dict[str, list[str]]:
    """Partition every canonical feature using implementation-defined names."""
    names = canonical_feature_names()
    composition = {
        "freq_A",
        "freq_C",
        "freq_G",
        "freq_T",
        "purine_content",
        "pyrimidine_content",
        "heterogeneity",
    }
    entropy = {
        "k2_entropy",
        "k2_complexity",
        "k3_entropy",
        "k3_complexity",
    }
    registry = {
        "GC_AND_SKEW": list(names[:10]),
        "NUCLEOTIDE_COMPOSITION": [n for n in names if n in composition],
        "DINUCLEOTIDE_FREQUENCY": [n for n in names if n.startswith("dinuc_")],
        "GLOBAL_KMER": [
            n
            for n in names
            if n.startswith(("k2_", "k3_")) and n not in entropy
        ],
        "ENTROPY_COMPLEXITY": [n for n in names if n in entropy],
        "POSITION_SPECIFIC_NUCLEOTIDE": [
            n for n in names if n.startswith("guide_pos_")
        ],
    }
    validate_feature_family_registry(registry, names)
    return registry


def validate_feature_family_registry(
    registry: Mapping[str, Sequence[str]],
    canonical_names: Sequence[str] | None = None,
) -> None:
    """Require complete, disjoint, order-preserving family membership."""
    names = tuple(canonical_names or canonical_feature_names())
    if set(registry) != set(FEATURE_FAMILIES):
        raise ValueError("Feature-family identities changed")
    flattened = [
        name for family in FEATURE_FAMILIES for name in registry[family]
    ]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Feature-family membership overlaps")
    if set(flattened) != set(names):
        raise ValueError("Feature-family registry is incomplete")
    positions = {name: index for index, name in enumerate(names)}
    if any(
        list(registry[family])
        != sorted(registry[family], key=positions.__getitem__)
        for family in FEATURE_FAMILIES
    ):
        raise ValueError("Within-family canonical feature order changed")
    counts = {family: len(registry[family]) for family in FEATURE_FAMILIES}
    if counts != {
        "GC_AND_SKEW": 10,
        "NUCLEOTIDE_COMPOSITION": 7,
        "DINUCLEOTIDE_FREQUENCY": 16,
        "GLOBAL_KMER": 80,
        "ENTROPY_COMPLEXITY": 4,
        "POSITION_SPECIFIC_NUCLEOTIDE": 80,
    }:
        raise ValueError("Canonical feature-family counts changed")


def canonical_xgboost_configuration() -> dict:
    """Lock the actual Phase 18C XGBoost policy with seed convention 42."""
    registered = dict(policies()["tabular"]["xgboost"])
    expected_registered = {
        key: EXPECTED_XGBOOST_CONFIGURATION[key] for key in registered
    }
    if registered != expected_registered:
        raise RuntimeError("Phase 18C XGBoost configuration changed")
    return dict(EXPECTED_XGBOOST_CONFIGURATION)


def verify_split_lock(payload: Mapping[str, object] | None = None) -> dict:
    """Verify reuse of the immutable Phase 18C split without rebuilding it."""
    if payload is None:
        payload = json.loads(
            PHASE18B_PROTOCOL_PATH.read_text(encoding="utf-8")
        )
    if payload.get("protocol_sha256") != PHASE18B_PROTOCOL_SHA256:
        raise RuntimeError("Approved Phase 18B protocol hash mismatch")
    if payload.get("split_sha256") != SPLIT_SHA256:
        raise RuntimeError("Approved Phase 18C split hash mismatch")
    policy = payload.get("policy", {}).get("split", {})
    if (
        policy.get("unit") != "spacer20"
        or policy.get("ratios")
        != {"train": 0.70, "validation": 0.15, "internal_test": 0.15}
        or "all group relatives held out" not in policy.get("bridge", "")
    ):
        raise RuntimeError("Phase 18C split policy changed")
    return {
        "source_artifact": PHASE18B_PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "source_protocol_sha256": PHASE18B_PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "grouping_unit": "exact forward spacer20 at 30-mer positions [4:24]",
        "partitions": {
            "train": 0.70,
            "validation": 0.15,
            "internal_test": 0.15,
        },
        "reuse_without_modification": True,
        "bridge_excluded": True,
        "quarantine_excluded": True,
        "new_split_generation": False,
    }


def verify_feature_source_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Compare allowlisted feature sources to Phase 18B/18C locks."""
    if payload is None:
        payload = json.loads(
            PHASE18B_PROTOCOL_PATH.read_text(encoding="utf-8")
        )
    phase18b_expected = payload.get("protected_source_hashes", {})
    execution = json.loads(
        PHASE18C_EXECUTION_LOCK_PATH.read_text(encoding="utf-8")
    )
    phase18c_expected = execution.get("implementation_hashes", {})
    phase18b_paths = (
        "src/bioinformatics/sequence_features.py",
        "src/models/xgboost_model.py",
        "src/experiment_protocols/multidomain_protocol.py",
    )
    phase18c_paths = (
        "src/bioinformatics/gc_content.py",
        "src/bioinformatics/nucleotide_composition.py",
        "src/bioinformatics/kmer.py",
        "src/bioinformatics/positional_features.py",
        "src/multidomain/data.py",
    )
    observed = {}
    for relative, expected in (
        *((path, phase18b_expected) for path in phase18b_paths),
        *((path, phase18c_expected) for path in phase18c_paths),
    ):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if expected.get(relative) != digest:
            raise RuntimeError(f"Locked Phase 18C source changed: {relative}")
        observed[relative] = digest
    return observed


def metric_policy() -> dict:
    return {
        "primary": "spearman",
        "regression_guardrail": "rmse",
        "secondary": ["pearson", "r2", "mae"],
        "domain_aggregation": "none",
        "effects": {
            "delta_spearman": "spearman_ablation - spearman_full",
            "delta_rmse": "rmse_ablation - rmse_full",
            "relative_rmse_degradation": (
                "(rmse_ablation - rmse_full) / rmse_full"
            ),
        },
        "degradation_directions": {
            "delta_spearman": "negative",
            "delta_rmse": "positive",
            "relative_rmse_degradation": "positive",
        },
        "native_label_scales_retained": True,
        "raw_domain_scale_comparison": False,
    }


def bootstrap_policy() -> dict:
    return {
        "method": (
            "paired percentile bootstrap over internal-test spacer groups"
        ),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "rng": "numpy.random.Generator(PCG64(42))",
        "nominal_confidence": NOMINAL_CONFIDENCE,
        "primary_pairing": (
            "Each drop-one-family arm versus FULL in the same domain only; "
            "the same sampled groups index both prediction vectors."
        ),
        "resampling_unit": (
            "sorted unique exact forward spacer20 groups; sample groups with "
            "replacement and retain every observation of each sampled group"
        ),
        "interval": "percentile with linear quantiles",
        "invalid_replicate": (
            "Any undefined/nonfinite primary metric is reported; one or more "
            "invalid replicates makes that comparison "
            "UNSTABLE_OR_INCONCLUSIVE."
        ),
    }


def multiple_comparison_policy() -> dict:
    return {
        "familywise_alpha": 0.05,
        "method": "Bonferroni simultaneous percentile intervals",
        "confirmatory_interval_n": CONFIRMATORY_INTERVAL_N,
        "scope": (
            "6 drop-one families x 2 domains x 2 endpoints "
            "(Spearman and RMSE)"
        ),
        "per_interval_alpha": BONFERRONI_ALPHA,
        "per_interval_confidence": SIMULTANEOUS_CONFIDENCE,
        "two_sided_quantiles": list(SIMULTANEOUS_QUANTILES),
        "secondary_metrics": "nominal 95% descriptive intervals only",
        "family_only_arms": (
            "nominal 95% descriptive intervals only; excluded from primary "
            "contribution categories"
        ),
    }


def contribution_decision_rules() -> dict:
    return {
        "threshold_policy": (
            "No minimum effect-size threshold is claimed because no "
            "biological "
            "or practical equivalence margin is justified prospectively. "
            "Classification uses the locked simultaneous confidence intervals."
        ),
        "ESSENTIAL_OR_STRONG_CONTRIBUTOR": (
            "Adjusted upper CI(delta Spearman) < 0 and adjusted lower "
            "CI(delta RMSE) > 0. 'Essential' refers only to predictive "
            "information."
        ),
        "MODERATE_CONTRIBUTOR": (
            "Exactly one adjusted interval supports degradation, the other "
            "point "
            "estimate also points toward degradation, and neither endpoint "
            "supports improvement."
        ),
        "LITTLE_UNIQUE_CONTRIBUTION": (
            "Neither adjusted interval supports degradation or improvement. "
            "This "
            "means no clear unique contribution was detected, not equivalence."
        ),
        "POTENTIALLY_REDUNDANT": (
            "At least one adjusted interval supports improvement after "
            "removal, and neither endpoint supports degradation. Redundancy, "
            "not absence "
            "of biological information, is the permitted interpretation."
        ),
        "UNSTABLE_OR_INCONCLUSIVE": (
            "Undefined/nonfinite metrics, invalid bootstrap replicates, or "
            "statistically supported conflicting endpoint directions."
        ),
        "precedence": [
            "UNSTABLE_OR_INCONCLUSIVE",
            "ESSENTIAL_OR_STRONG_CONTRIBUTOR",
            "MODERATE_CONTRIBUTOR",
            "POTENTIALLY_REDUNDANT",
            "LITTLE_UNIQUE_CONTRIBUTION",
        ],
    }


def domain_consistency_decision_rules() -> dict:
    return {
        "contributor_classes": [
            "ESSENTIAL_OR_STRONG_CONTRIBUTOR",
            "MODERATE_CONTRIBUTOR",
        ],
        "weak_classes": [
            "LITTLE_UNIQUE_CONTRIBUTION",
            "POTENTIALLY_REDUNDANT",
        ],
        "CROSS_DOMAIN_CONSISTENT": (
            "Both domains receive a contributor class with concordant "
            "degradation "
            "directions."
        ),
        "DOMAIN_A_ENRICHED": (
            "Only DeepSpCas9 receives a contributor class; this is an "
            "evidence "
            "pattern, not a formal cross-domain effect-size difference."
        ),
        "DOMAIN_B_ENRICHED": (
            "Only Xiang/Luo receives a contributor class; this is an evidence "
            "pattern, not a formal cross-domain effect-size difference."
        ),
        "WEAK_IN_BOTH": (
            "Both domains receive weak classes; this does not establish "
            "equivalence."
        ),
        "UNSTABLE_OR_INCONCLUSIVE": (
            "Either domain is unstable/inconclusive or endpoint directions "
            "conflict."
        ),
        "raw_label_scale_comparison": False,
    }


def bootstrap_group_draws(
    group_ids: Sequence[str], replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[tuple[str, ...], ...]:
    """Materialize a synthetic/audit bootstrap plan without predictions."""
    groups = tuple(sorted(set(group_ids)))
    if not groups or len(groups) != len(group_ids):
        raise ValueError(
            "Bootstrap group IDs must be sorted-agnostic and unique"
        )
    if type(replicates) is not int or replicates <= 0:
        raise ValueError(
            "Bootstrap replicate count must be a positive integer"
        )
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(
        0, len(groups), size=(replicates, len(groups))
    )
    return tuple(tuple(groups[index] for index in row) for row in indices)


def experiment_matrix(
    registry: Mapping[str, Sequence[str]] | None = None,
) -> list[dict]:
    """Return the exact future Phase 18F matrix without executing it."""
    registry = feature_family_registry() if registry is None else registry
    validate_feature_family_registry(registry)
    configuration = canonical_xgboost_configuration()
    metrics = metric_policy()
    bootstrap = bootstrap_policy()
    rows = []
    for domain in DOMAINS:
        code = DOMAIN_CODES[domain]
        common = {
            "domain": domain,
            "canonical_xgboost_configuration": configuration,
            "split_hash": SPLIT_SHA256,
            "metric_policy": metrics,
            "bootstrap_policy": bootstrap,
            "fit_seed": 42,
        }
        rows.append(
            dict(
                common,
                experiment_id=f"18F_XGB_{code}_FULL",
                experiment_type="FULL_CONTROL",
                feature_family_removed=None,
                feature_families_retained=list(FEATURE_FAMILIES),
                number_of_features_remaining=CANONICAL_FEATURE_COUNT,
                scientific_hypothesis=(
                    "Locked full-feature within-domain reference for paired "
                    "drop-one comparisons."
                ),
                inferential_role="corresponding within-domain control",
            )
        )
        for family in FEATURE_FAMILIES:
            rows.append(
                dict(
                    common,
                    experiment_id=f"18F_XGB_{code}_DROP_{family}",
                    experiment_type="DROP_ONE_FAMILY",
                    feature_family_removed=family,
                    feature_families_retained=[
                        item for item in FEATURE_FAMILIES if item != family
                    ],
                    number_of_features_remaining=(
                        CANONICAL_FEATURE_COUNT - len(registry[family])
                    ),
                    scientific_hypothesis=FAMILY_HYPOTHESES[family],
                    inferential_role="primary paired comparison versus FULL",
                )
            )
        for family in OPTIONAL_FAMILY_ONLY:
            question = (
                "whether spacer-position identity alone is sufficient "
                "for much "
                "of within-domain predictive performance"
                if family == "POSITION_SPECIFIC_NUCLEOTIDE"
                else "whether order-agnostic 2/3-mer spectra alone are "
                "sufficient "
                "for much of within-domain predictive performance"
            )
            rows.append(
                dict(
                    common,
                    experiment_id=f"18F_XGB_{code}_ONLY_{family}",
                    experiment_type="FAMILY_ONLY",
                    feature_family_removed=None,
                    feature_families_retained=[family],
                    number_of_features_remaining=len(registry[family]),
                    scientific_hypothesis=question,
                    inferential_role=(
                        "secondary descriptive sufficiency analysis; not a "
                        "primary contribution category"
                    ),
                )
            )
    validate_experiment_matrix(rows, registry)
    return rows


def validate_experiment_matrix(
    matrix: Sequence[Mapping[str, object]],
    registry: Mapping[str, Sequence[str]] | None = None,
) -> None:
    registry = feature_family_registry() if registry is None else registry
    expected_n = len(DOMAINS) * (
        1 + len(FEATURE_FAMILIES) + len(OPTIONAL_FAMILY_ONLY)
    )
    if len(matrix) != expected_n:
        raise ValueError("Future experiment matrix size changed")
    ids = [row.get("experiment_id") for row in matrix]
    if len(ids) != len(set(ids)):
        raise ValueError("Future experiment IDs are not unique")
    required = {
        "experiment_id",
        "domain",
        "feature_family_removed",
        "number_of_features_remaining",
        "canonical_xgboost_configuration",
        "split_hash",
        "metric_policy",
        "bootstrap_policy",
        "scientific_hypothesis",
    }
    if any(not required <= set(row) for row in matrix):
        raise ValueError("Future experiment matrix field missing")
    for row in matrix:
        if row["domain"] not in DOMAINS or row["split_hash"] != SPLIT_SHA256:
            raise ValueError("Future experiment domain/split changed")
        if row["canonical_xgboost_configuration"] != (
            EXPECTED_XGBOOST_CONFIGURATION
        ):
            raise ValueError("Future XGBoost configuration changed")
        if row["experiment_type"] == "DROP_ONE_FAMILY":
            family = row["feature_family_removed"]
            expected = CANONICAL_FEATURE_COUNT - len(registry[family])
            if row["number_of_features_remaining"] != expected:
                raise ValueError("Drop-one feature count changed")


def build_protocol() -> dict:
    """Build and validate the Phase 18E design-only protocol."""
    with phase18e_protocol_guard():
        names = canonical_feature_names()
        registry = feature_family_registry()
        split = verify_split_lock()
        source_hashes = verify_feature_source_integrity()
        matrix = experiment_matrix(registry)
        configuration = canonical_xgboost_configuration()
        payload = {
            "phase": PHASE,
            "protocol_version": PROTOCOL_VERSION,
            "final_decision": "READY_FOR_PHASE18F",
            "future_phase": FUTURE_PHASE,
            "scope": {
                "design_and_preregistration_only": True,
                "model_training_occurred": False,
                "predictions_generated": False,
                "feature_ablations_executed": False,
                "phase18f_started": False,
                "locked_external_accessed": False,
            },
            "scientific_question": (
                "Which engineered feature families contribute unique "
                "predictive information to XGBoost in each development "
                "domain? This isolates representation contribution, not all "
                "algorithmic "
                "reasons for the "
                "observed XGBoost-CNN performance difference."
            ),
            "feature_space": {
                "canonical_count": len(names),
                "canonical_feature_sha256": CANONICAL_FEATURE_SHA256,
                "canonical_ordered_names": list(names),
                "family_order": list(FEATURE_FAMILIES),
                "family_registry": registry,
                "family_counts": {
                    family: len(registry[family])
                    for family in FEATURE_FAMILIES
                },
                "partition_complete": True,
                "partition_disjoint": True,
            },
            "xgboost_configuration": configuration,
            "seed_policy": {
                "fit_seeds": [42],
                "reason": (
                    "The Phase 18C configuration is deterministic because row "
                    "and feature sampling are both 1.0; Phase 18C observed "
                    "identical outcomes for seeds 42-46. Repeating "
                    "deterministic "
                    "refits would "
                    "not estimate stochastic variability. Seed 42 preserves "
                    "the "
                    "established project convention."
                ),
                "redundant_five_seed_refits": False,
            },
            "split_policy": split,
            "metric_policy": metric_policy(),
            "bootstrap_policy": bootstrap_policy(),
            "multiple_comparison_policy": multiple_comparison_policy(),
            "contribution_decision_rules": contribution_decision_rules(),
            "domain_consistency_decision_rules": (
                domain_consistency_decision_rules()
            ),
            "redundancy_interpretation": (
                "GC, composition, k-mer, and positional features overlap. A "
                "small drop-one effect can indicate compensation by retained "
                "families; it does not show that the removed family lacks "
                "predictive or biological information. Ablation measures "
                "predictive information "
                "contribution, not causality, molecular necessity, or "
                "mechanism."
            ),
            "optional_family_only_policy": {
                "included": list(OPTIONAL_FAMILY_ONLY),
                "rationale": (
                    "These two 80-feature representations answer the distinct "
                    "sufficiency question of positional identity versus "
                    "global "
                    "motif "
                    "spectrum. Smaller family-only arms and all combinations "
                    "are "
                    "excluded to avoid weakly interpretable subset search."
                ),
                "confirmatory": False,
            },
            "experiment_matrix": matrix,
            "budget": {
                "domains": 2,
                "full_controls": 2,
                "drop_one_ablations": 12,
                "optional_family_only_models": 4,
                "total_registered_fits": 18,
                "expected_burden": (
                    "Low: Phase 18C recorded about 6.75 seconds for 10 "
                    "XGBoost fits on the same domains/configuration, so 18 "
                    "fits "
                    "should be on the order of tens of seconds for fitting on "
                    "comparable "
                    "hardware; 10,000-replicate bootstrap analysis may "
                    "dominate."
                ),
            },
            "excluded_searches": [
                "individual-feature leave-one-out",
                "recursive feature elimination",
                "SHAP-guided feature selection",
                "importance-guided cherry-picked removals",
                "all feature-family combinations",
                "post-ablation rescue tuning",
                "individual guide_pos_19_G removal",
            ],
            "interpretation_language": {
                "required": (
                    "Removing this feature family reduced predictive "
                    "performance."
                ),
                "prohibited": "This feature family causes CRISPR activity.",
                "feature_importance_is_causality": False,
            },
            "canonical_integrity": {
                "feature_sources_match_phase18b_and_phase18c_locks": True,
                "allowlisted_source_hashes": source_hashes,
                "split_reused_without_modification": True,
                "canonical_feature_identity_locked": True,
            },
            "stop_conditions": [
                "locked external dataset access attempt",
                "canonical feature identity/order/count mismatch",
                "feature-family overlap or incomplete partition",
                "Phase 18C split hash or policy mismatch",
                "canonical XGBoost configuration mismatch",
                "bridge or quarantine inclusion",
                "training or prediction attempt during Phase 18E",
            ],
        }
        validate_protocol(payload)
        return payload


def validate_protocol(payload: Mapping[str, object]) -> None:
    if payload.get("final_decision") != "READY_FOR_PHASE18F":
        raise ValueError("Phase 18E final decision changed")
    scope = payload.get("scope", {})
    if not scope.get("design_and_preregistration_only") or any(
        scope.get(key)
        for key in (
            "model_training_occurred",
            "predictions_generated",
            "feature_ablations_executed",
            "phase18f_started",
            "locked_external_accessed",
        )
    ):
        raise ValueError("Phase 18E no-execution boundary changed")
    feature_space = payload["feature_space"]
    validate_feature_family_registry(
        feature_space["family_registry"],
        feature_space["canonical_ordered_names"],
    )
    if feature_space["canonical_feature_sha256"] != CANONICAL_FEATURE_SHA256:
        raise ValueError("Canonical feature hash changed")
    if payload["xgboost_configuration"] != EXPECTED_XGBOOST_CONFIGURATION:
        raise ValueError("XGBoost lock changed")
    if payload["split_policy"]["split_sha256"] != SPLIT_SHA256:
        raise ValueError("Split lock changed")
    validate_experiment_matrix(
        payload["experiment_matrix"], feature_space["family_registry"]
    )
    if payload["budget"]["total_registered_fits"] != 18:
        raise ValueError("Compute budget changed")
    if not math.isclose(
        payload["multiple_comparison_policy"]["familywise_alpha"], 0.05
    ):
        raise ValueError("Multiplicity policy changed")


def lock_protocol(payload: Mapping[str, object]) -> dict:
    if "protocol_sha256" in payload:
        raise ValueError("Protocol is already locked")
    validate_protocol(payload)
    result = dict(payload, protocol_sha256=fingerprint(payload))
    validate_locked_protocol(result)
    return result


def validate_locked_protocol(payload: Mapping[str, object]) -> None:
    body = {
        key: value
        for key, value in payload.items()
        if key != "protocol_sha256"
    }
    if payload.get("protocol_sha256") != fingerprint(body):
        raise ValueError("Protocol serialization hash mismatch")
    validate_protocol(body)

"""Phase 18G positional-region preregistration for future Phase 18H.

This module verifies existing metadata and defines a protocol. It cannot fit
models, generate predictions, or execute positional ablations.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

from src.bioinformatics.sequence_features import SequenceFeatureExtractor
from src.experiment_protocols.phase18c_access import phase18c_access_guard
from src.experiment_protocols.phase18e_feature_ablation_protocol import (
    CANONICAL_FEATURE_COUNT,
    CANONICAL_FEATURE_SHA256,
    EXPECTED_XGBOOST_CONFIGURATION,
    SPLIT_SHA256,
    canonical_feature_names,
    canonical_xgboost_configuration,
    feature_family_registry,
    fingerprint,
    verify_feature_source_integrity,
    verify_split_lock,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE = "18G"
PROTOCOL_VERSION = "phase18g-positional-region-ablation-v1"
FUTURE_PHASE = "Phase 18H - Locked Positional-Region Ablation Experiment"
PHASE18F_COMMIT = "0fc7273eba47e4a7ee242a945db66de4fe183100"
PHASE18E_PROTOCOL_SHA256 = (
    "79d56d051ffbc91b0a096483aa1be64a95f577cd57b94148d8734bbacf92f8df"
)
PHASE18F_SUMMARY_PATH = ROOT / (
    "results/phase18f/"
    "79d56d051ffbc91b0a096483aa1be64a95f577cd57b94148d8734bbacf92f8df/"
    "campaign_20260914_231905/summary.json"
)
PHASE18F_SUMMARY_SHA256 = (
    "97594a947b18ba30b10c60d8af674a3cc9ca6fadf042f9a427f0fe71f1dcd9c0"
)
POSITIONAL_FEATURE_COUNT = 80
POSITIONAL_FEATURE_SHA256 = (
    "73e23152a66ec80f6e52a9d37324b4cf0bc705123d82720bcd5e3ee8ce88f939"
)
DOMAINS = ("deepspcas9", "crispron_xiang_luo")
DOMAIN_CODES = {DOMAINS[0]: "A", DOMAINS[1]: "B"}
NUCLEOTIDES = ("A", "C", "G", "T")
REGION_DEFINITIONS = (
    ("REGION_1_PAM_DISTAL", tuple(range(0, 5))),
    ("REGION_2_MID_DISTAL", tuple(range(5, 10))),
    ("REGION_3_MID_PROXIMAL", tuple(range(10, 15))),
    ("REGION_4_PAM_PROXIMAL", tuple(range(15, 20))),
)
EXPECTED_REGION_FEATURE_HASHES = {
    "REGION_1_PAM_DISTAL": (
        "5a255dbe0eda288cbcde6f80f3cce232a555194b5d446b79af96641ad8d0504e"
    ),
    "REGION_2_MID_DISTAL": (
        "1f8db099357909e2874a3a79f40d3d228bf81e1bbff183adb858d2906437e5d4"
    ),
    "REGION_3_MID_PROXIMAL": (
        "1cf04d6394dfd6bcfeab3c1f796e8005ec72a3fd649956c4706d04aa71e1eb0e"
    ),
    "REGION_4_PAM_PROXIMAL": (
        "a00bfdff1ec3297b6ecb8b23368e94d84ff77a2e19ff55b0dbffec73490397eb"
    ),
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
FAMILYWISE_ALPHA = 0.05
CONFIRMATORY_INTERVAL_N = len(REGION_DEFINITIONS) * len(DOMAINS) * 2
PER_INTERVAL_ALPHA = FAMILYWISE_ALPHA / CONFIRMATORY_INTERVAL_N
SIMULTANEOUS_CONFIDENCE = 1.0 - PER_INTERVAL_ALPHA
SIMULTANEOUS_QUANTILES = (
    PER_INTERVAL_ALPHA / 2.0,
    1.0 - PER_INTERVAL_ALPHA / 2.0,
)


def serialize(payload: object) -> str:
    """Return canonical JSON for deterministic protocol artifacts."""
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def phase18g_protocol_guard():
    """Reject locked-data access and model execution for this process."""
    previous_profile = sys.getprofile()

    def no_model_execution(frame, event, arg):
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
                    "Phase 18G model training/prediction prohibited"
                )
        if previous_profile:
            previous_profile(frame, event, arg)

    with phase18c_access_guard():
        sys.setprofile(no_model_execution)
        try:
            yield
        finally:
            sys.setprofile(previous_profile)


def positional_feature_names() -> tuple[str, ...]:
    """Return and verify all canonical spacer-position indicator names."""
    observed = tuple(feature_family_registry()["POSITION_SPECIFIC_NUCLEOTIDE"])
    expected = tuple(
        f"guide_pos_{position}_{nucleotide}"
        for position in range(20)
        for nucleotide in NUCLEOTIDES
    )
    if observed != expected or len(observed) != POSITIONAL_FEATURE_COUNT:
        raise RuntimeError("Canonical positional feature identity changed")
    if fingerprint(list(observed)) != POSITIONAL_FEATURE_SHA256:
        raise RuntimeError("Canonical positional feature ordering changed")
    return observed


def position_to_features() -> dict[int, tuple[str, ...]]:
    """Map each spacer position to exactly four ordered nucleotide features."""
    names = positional_feature_names()
    mapping = {
        position: tuple(names[position * 4:(position + 1) * 4])
        for position in range(20)
    }
    if (
        tuple(name for position in mapping for name in mapping[position])
        != names
    ):
        raise RuntimeError("Position-to-feature mapping is incomplete")
    return mapping


def orientation_audit() -> dict:
    """Verify spacer orientation from the canonical extractor geometry."""
    extractor = SequenceFeatureExtractor()
    observed = {
        "context_length": extractor.context_length,
        "five_prime_context_indices": [0, 1, 2, 3],
        "guide_start_index": extractor.guide_start,
        "guide_end_exclusive": extractor.guide_end,
        "pam_start_index": extractor.pam_start,
        "pam_end_exclusive": extractor.pam_end,
        "three_prime_context_indices": [27, 28, 29],
    }
    expected = {
        "context_length": 30,
        "five_prime_context_indices": [0, 1, 2, 3],
        "guide_start_index": 4,
        "guide_end_exclusive": 24,
        "pam_start_index": 24,
        "pam_end_exclusive": 27,
        "three_prime_context_indices": [27, 28, 29],
    }
    if observed != expected or extractor.guide_length != 20:
        raise RuntimeError("Canonical spacer/PAM geometry changed")
    return {
        **observed,
        "guide_length": 20,
        "sequence_direction": "canonical forward 30-mer, 5-prime to 3-prime",
        "guide_position_0": ("30-mer index 4; spacer 5-prime end; PAM-distal"),
        "guide_position_19": (
            "30-mer index 23; spacer 3-prime end immediately before PAM; "
            "PAM-proximal"
        ),
        "pam_distal_positions": list(range(0, 5)),
        "pam_proximal_positions": list(range(15, 20)),
        "orientation_verified_from_implementation": True,
    }


def region_registry() -> dict[str, dict]:
    """Build and verify the four equal, preregistered positional regions."""
    mapping = position_to_features()
    registry = {}
    for region, positions in REGION_DEFINITIONS:
        features = tuple(
            feature for position in positions for feature in mapping[position]
        )
        registry[region] = {
            "ordinal": len(registry) + 1,
            "positions": list(positions),
            "position_count": len(positions),
            "feature_names": list(features),
            "feature_count": len(features),
            "feature_name_sha256": fingerprint(list(features)),
        }
    validate_region_registry(registry)
    return registry


def validate_region_registry(
    registry: Mapping[str, Mapping[str, object]],
) -> None:
    """Require equal, complete, disjoint, order-preserving regions."""
    expected_names = [item[0] for item in REGION_DEFINITIONS]
    if list(registry) != expected_names:
        raise ValueError("Positional region identities or order changed")
    all_positions = []
    all_features = []
    for region, expected_positions in REGION_DEFINITIONS:
        record = registry[region]
        positions = list(record.get("positions", []))
        features = list(record.get("feature_names", []))
        expected_features = [
            f"guide_pos_{position}_{nucleotide}"
            for position in expected_positions
            for nucleotide in NUCLEOTIDES
        ]
        if positions != list(expected_positions):
            raise ValueError(f"Position boundary changed for {region}")
        if features != expected_features:
            raise ValueError(f"Feature membership changed for {region}")
        if len(positions) != 5 or len(features) != 20:
            raise ValueError(f"Region size changed for {region}")
        if record.get("feature_name_sha256") != (
            EXPECTED_REGION_FEATURE_HASHES[region]
        ):
            raise ValueError(f"Feature hash changed for {region}")
        all_positions.extend(positions)
        all_features.extend(features)
    if all_positions != list(range(20)) or len(set(all_positions)) != 20:
        raise ValueError("Regions do not partition all spacer positions")
    if tuple(all_features) != positional_feature_names():
        raise ValueError("Regions do not partition all positional features")
    if len(set(all_features)) != POSITIONAL_FEATURE_COUNT:
        raise ValueError("Positional regions overlap")


def metric_policy() -> dict:
    return {
        "primary": "spearman",
        "regression_guardrail": "rmse",
        "secondary": ["pearson", "r2", "mae"],
        "domain_aggregation": "none",
        "effects": {
            "delta_spearman": "spearman_drop_region - spearman_full",
            "delta_rmse": "rmse_drop_region - rmse_full",
            "relative_rmse_degradation": (
                "(rmse_drop_region - rmse_full) / rmse_full"
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
        "nominal_confidence": 0.95,
        "primary_pairing": (
            "Each drop-region arm versus the reused Phase 18F FULL prediction "
            "in the same domain; one sampled-group draw indexes both vectors."
        ),
        "resampling_unit": (
            "sorted unique exact forward spacer20 groups; sample groups with "
            "replacement and retain every observation of each sampled group"
        ),
        "interval": "percentile with linear quantiles",
        "invalid_replicate": (
            "Any undefined or nonfinite primary metric is reported; one or "
            "more invalid replicates makes that comparison "
            "UNSTABLE_OR_INCONCLUSIVE."
        ),
    }


def multiple_comparison_policy() -> dict:
    return {
        "familywise_alpha": FAMILYWISE_ALPHA,
        "method": "Bonferroni simultaneous percentile intervals",
        "confirmatory_interval_n": CONFIRMATORY_INTERVAL_N,
        "scope": ("4 regions x 2 domains x 2 endpoints (Spearman and RMSE)"),
        "per_interval_alpha": PER_INTERVAL_ALPHA,
        "per_interval_confidence": SIMULTANEOUS_CONFIDENCE,
        "per_interval_confidence_percent": (SIMULTANEOUS_CONFIDENCE * 100.0),
        "two_sided_quantiles": list(SIMULTANEOUS_QUANTILES),
        "secondary_metrics": "nominal 95% descriptive intervals only",
        "cross_region_rankings": (
            "descriptive only; no unregistered pairwise superiority tests"
        ),
    }


def regional_contribution_decision_rules() -> dict:
    return {
        "threshold_policy": (
            "No post-result effect threshold. Classification uses locked "
            "Bonferroni simultaneous confidence-interval directions."
        ),
        "STRONG_REGIONAL_CONTRIBUTOR": (
            "Adjusted upper CI(delta Spearman) < 0 and adjusted lower "
            "CI(delta RMSE) > 0."
        ),
        "MODERATE_REGIONAL_CONTRIBUTOR": (
            "Exactly one adjusted interval supports degradation, the other "
            "point estimate is directionally concordant, and neither endpoint "
            "supports improvement."
        ),
        "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION": (
            "Neither adjusted endpoint supports degradation or improvement; "
            "this is not equivalence or absence of information."
        ),
        "POTENTIALLY_REDUNDANT_REGION": (
            "At least one adjusted endpoint supports improvement after "
            "removal "
            "and neither endpoint supports degradation."
        ),
        "UNSTABLE_OR_INCONCLUSIVE": (
            "Undefined or nonfinite metrics, invalid bootstrap replicates, or "
            "statistically supported conflict between endpoint directions."
        ),
        "precedence": [
            "UNSTABLE_OR_INCONCLUSIVE",
            "STRONG_REGIONAL_CONTRIBUTOR",
            "MODERATE_REGIONAL_CONTRIBUTOR",
            "POTENTIALLY_REDUNDANT_REGION",
            "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
        ],
    }


def _supported(interval: Sequence[float], direction: str) -> bool:
    lower, upper = map(float, interval)
    if not all(math.isfinite(value) for value in (lower, upper)):
        raise ValueError("Classification interval must be finite")
    return upper < 0 if direction == "negative" else lower > 0


def classify_regional_contribution(record: Mapping[str, object]) -> str:
    """Apply the locked regional confidence-interval direction rules."""
    required = {
        "delta_spearman",
        "delta_rmse",
        "adjusted_delta_spearman_ci",
        "adjusted_delta_rmse_ci",
        "invalid_bootstrap_n",
    }
    if not required <= set(record):
        raise ValueError("Regional classification record is incomplete")
    delta_spearman = float(record["delta_spearman"])
    delta_rmse = float(record["delta_rmse"])
    if (
        int(record["invalid_bootstrap_n"]) != 0
        or not math.isfinite(delta_spearman)
        or not math.isfinite(delta_rmse)
    ):
        return "UNSTABLE_OR_INCONCLUSIVE"
    spearman_ci = record["adjusted_delta_spearman_ci"]
    rmse_ci = record["adjusted_delta_rmse_ci"]
    degradation = (
        _supported(spearman_ci, "negative"),
        _supported(rmse_ci, "positive"),
    )
    improvement = (
        _supported(spearman_ci, "positive"),
        _supported(rmse_ci, "negative"),
    )
    if any(degradation) and any(improvement):
        return "UNSTABLE_OR_INCONCLUSIVE"
    if all(degradation):
        return "STRONG_REGIONAL_CONTRIBUTOR"
    if sum(degradation) == 1 and delta_spearman < 0 and delta_rmse > 0:
        return "MODERATE_REGIONAL_CONTRIBUTOR"
    if any(improvement) and not any(degradation):
        return "POTENTIALLY_REDUNDANT_REGION"
    return "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION"


def cross_domain_region_decision_rules() -> dict:
    return {
        "contributor_classes": [
            "STRONG_REGIONAL_CONTRIBUTOR",
            "MODERATE_REGIONAL_CONTRIBUTOR",
        ],
        "weak_classes": [
            "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
            "POTENTIALLY_REDUNDANT_REGION",
        ],
        "CROSS_DOMAIN_CONSISTENT": (
            "Both domains receive a regional contributor class with "
            "concordant degradation directions."
        ),
        "DOMAIN_A_ENRICHED": (
            "Only DeepSpCas9 receives a contributor class; this is evidence "
            "asymmetry, not a native-scale effect-size test."
        ),
        "DOMAIN_B_ENRICHED": (
            "Only Xiang/Luo receives a contributor class; this is evidence "
            "asymmetry, not a native-scale effect-size test."
        ),
        "WEAK_IN_BOTH": (
            "Both domains receive weak classes; this does not establish "
            "equivalence or absence of regional information."
        ),
        "UNSTABLE_OR_INCONCLUSIVE": (
            "Either domain is inconclusive or endpoint directions conflict."
        ),
        "raw_rmse_scale_comparison": False,
    }


def classify_cross_domain_region(domain_a: str, domain_b: str) -> str:
    """Classify one region's evidence pattern across the two domains."""
    contributors = {
        "STRONG_REGIONAL_CONTRIBUTOR",
        "MODERATE_REGIONAL_CONTRIBUTOR",
    }
    weak = {
        "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
        "POTENTIALLY_REDUNDANT_REGION",
    }
    known = contributors | weak | {"UNSTABLE_OR_INCONCLUSIVE"}
    if domain_a not in known or domain_b not in known:
        raise ValueError("Unknown regional contribution classification")
    if "UNSTABLE_OR_INCONCLUSIVE" in {domain_a, domain_b}:
        return "UNSTABLE_OR_INCONCLUSIVE"
    if domain_a in contributors and domain_b in contributors:
        return "CROSS_DOMAIN_CONSISTENT"
    if domain_a in contributors and domain_b in weak:
        return "DOMAIN_A_ENRICHED"
    if domain_b in contributors and domain_a in weak:
        return "DOMAIN_B_ENRICHED"
    return "WEAK_IN_BOTH"


def _reference_record(record: Mapping[str, object]) -> dict:
    model_path = ROOT / str(record["model_artifact"])
    prediction_path = ROOT / str(record["prediction_artifact"])
    if not model_path.is_file() or not prediction_path.is_file():
        raise RuntimeError("Phase 18F reference artifact is unavailable")
    if _sha256(model_path) != record["model_sha256"]:
        raise RuntimeError("Phase 18F reference model hash mismatch")
    if _sha256(prediction_path) != record["prediction_sha256"]:
        raise RuntimeError("Phase 18F reference prediction hash mismatch")
    return {
        "source_experiment_id": record["experiment_id"],
        "model_artifact": record["model_artifact"],
        "model_sha256": record["model_sha256"],
        "prediction_artifact": record["prediction_artifact"],
        "prediction_sha256": record["prediction_sha256"],
        "feature_count": record["feature_count"],
        "feature_name_sha256": record["feature_name_sha256"],
        "seed": record["seed"],
        "split_sha256": record["split_sha256"],
        "xgboost_configuration": record["xgboost_configuration"],
        "metrics": record["metrics"],
        "compatibility_verified": True,
    }


def verify_phase18f_references() -> dict[str, dict[str, dict]]:
    """Verify reusable Phase 18F references without loading predictions."""
    if _sha256(PHASE18F_SUMMARY_PATH) != PHASE18F_SUMMARY_SHA256:
        raise RuntimeError("Phase 18F summary hash mismatch")
    payload = json.loads(PHASE18F_SUMMARY_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("completion_status") != "COMPLETED"
        or payload.get("completed_fits") != 18
        or payload.get("failed_fits") != 0
        or payload.get("skipped_fits") != 0
        or payload.get("feature_space_sha256") != CANONICAL_FEATURE_SHA256
        or payload.get("split_sha256") != SPLIT_SHA256
        or payload.get("phase18e_protocol_sha256") != PHASE18E_PROTOCOL_SHA256
        or payload.get("locked_external_accessed") is not False
        or payload.get("bridge_excluded") is not True
        or payload.get("quarantine_excluded") is not True
        or payload.get("analysis_recovery", {}).get("additional_fits") != 0
        or payload.get("analysis_recovery", {}).get("additional_predictions")
        != 0
    ):
        raise RuntimeError("Phase 18F completion or isolation record changed")
    if (
        payload.get("domain_consistency", {}).get(
            "POSITION_SPECIFIC_NUCLEOTIDE"
        )
        != "CROSS_DOMAIN_CONSISTENT"
    ):
        raise RuntimeError("Phase 18F positional conclusion changed")

    references = {}
    for domain in DOMAINS:
        code = DOMAIN_CODES[domain]
        full = payload["full_controls"][domain]
        drop_all = payload["runs"][
            f"18F_XGB_{code}_DROP_POSITION_SPECIFIC_NUCLEOTIDE"
        ]
        if (
            full["feature_count"] != CANONICAL_FEATURE_COUNT
            or full["feature_name_sha256"] != CANONICAL_FEATURE_SHA256
            or drop_all["feature_count"] != 117
            or drop_all["feature_family_removed"]
            != "POSITION_SPECIFIC_NUCLEOTIDE"
            or drop_all["seed"] != 42
            or full["seed"] != 42
            or full["split_sha256"] != SPLIT_SHA256
            or drop_all["split_sha256"] != SPLIT_SHA256
            or full["xgboost_configuration"] != EXPECTED_XGBOOST_CONFIGURATION
            or drop_all["xgboost_configuration"]
            != EXPECTED_XGBOOST_CONFIGURATION
        ):
            raise RuntimeError("Phase 18F reference configuration mismatch")
        effect = payload["drop_one_effects"][domain][
            "POSITION_SPECIFIC_NUCLEOTIDE"
        ]
        if effect["contribution_classification"] != (
            "ESSENTIAL_OR_STRONG_CONTRIBUTOR"
        ):
            raise RuntimeError("Phase 18F whole-family result changed")
        references[domain] = {
            "full": _reference_record(full),
            "drop_all_position_specific": _reference_record(drop_all),
            "whole_family_classification": effect[
                "contribution_classification"
            ],
            "whole_family_effect": {
                key: effect[key]
                for key in (
                    "delta_spearman",
                    "delta_rmse",
                    "relative_rmse_degradation",
                )
            },
        }
    return references


def experiment_matrix(
    regions: Mapping[str, Mapping[str, object]] | None = None,
    references: Mapping[str, Mapping[str, Mapping]] | None = None,
) -> list[dict]:
    """Return the exact Phase 18H registry without executing any arm."""
    regions = region_registry() if regions is None else regions
    validate_region_registry(regions)
    references = (
        verify_phase18f_references() if references is None else references
    )
    configuration = canonical_xgboost_configuration()
    metrics = metric_policy()
    bootstrap = bootstrap_policy()
    rows = []
    for domain in DOMAINS:
        code = DOMAIN_CODES[domain]
        common = {
            "domain": domain,
            "xgboost_configuration": configuration,
            "split_hash": SPLIT_SHA256,
            "seed": 42,
            "metric_policy": metrics,
            "bootstrap_policy": bootstrap,
        }
        full_reference = references[domain]["full"]
        rows.append(
            dict(
                common,
                experiment_id=f"18H_REF_{code}_FULL_PHASE18F",
                experiment_type="REUSED_FULL_REFERENCE",
                region_removed=None,
                positions_removed=[],
                feature_names_removed=[],
                remaining_feature_count=CANONICAL_FEATURE_COUNT,
                hypothesis=(
                    "Reused locked full-feature baseline for all "
                    "within-domain "
                    "paired regional comparisons."
                ),
                requires_new_fit=False,
                confirmatory=False,
                source_reference=full_reference,
            )
        )
        for region, region_record in regions.items():
            rows.append(
                dict(
                    common,
                    experiment_id=f"18H_XGB_{code}_DROP_{region}",
                    experiment_type="DROP_POSITIONAL_REGION",
                    region_removed=region,
                    positions_removed=region_record["positions"],
                    feature_names_removed=region_record["feature_names"],
                    remaining_feature_count=(
                        CANONICAL_FEATURE_COUNT
                        - int(region_record["feature_count"])
                    ),
                    hypothesis=(
                        f"{region} contains positional information with "
                        "unique "
                        "predictive contribution conditional on all retained "
                        "canonical features."
                    ),
                    requires_new_fit=True,
                    confirmatory=True,
                    source_reference=full_reference,
                )
            )
        whole_reference = references[domain]["drop_all_position_specific"]
        rows.append(
            dict(
                common,
                experiment_id=(
                    f"18H_REF_{code}_DROP_ALL_POSITION_SPECIFIC_PHASE18F"
                ),
                experiment_type="REUSED_WHOLE_POSITIONAL_REFERENCE",
                region_removed="ALL_POSITION_SPECIFIC_NUCLEOTIDE",
                positions_removed=list(range(20)),
                feature_names_removed=list(positional_feature_names()),
                remaining_feature_count=117,
                hypothesis=(
                    "Reused locked whole-family result supplies context for "
                    "regional effects without assuming regional additivity."
                ),
                requires_new_fit=False,
                confirmatory=False,
                source_reference=whole_reference,
            )
        )
    validate_experiment_matrix(rows, regions)
    return rows


def validate_experiment_matrix(
    matrix: Sequence[Mapping[str, object]],
    regions: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    regions = region_registry() if regions is None else regions
    validate_region_registry(regions)
    if len(matrix) != 12:
        raise ValueError("Phase 18H registry must contain exactly 12 rows")
    ids = [row.get("experiment_id") for row in matrix]
    if len(set(ids)) != 12:
        raise ValueError("Phase 18H experiment IDs are not unique")
    required = {
        "experiment_id",
        "domain",
        "region_removed",
        "positions_removed",
        "feature_names_removed",
        "remaining_feature_count",
        "xgboost_configuration",
        "split_hash",
        "seed",
        "metric_policy",
        "bootstrap_policy",
        "hypothesis",
        "requires_new_fit",
        "confirmatory",
    }
    if any(not required <= set(row) for row in matrix):
        raise ValueError("Phase 18H matrix field is missing")
    for row in matrix:
        if row["domain"] not in DOMAINS or row["split_hash"] != SPLIT_SHA256:
            raise ValueError("Phase 18H domain or split changed")
        if row["xgboost_configuration"] != EXPECTED_XGBOOST_CONFIGURATION:
            raise ValueError("Phase 18H XGBoost configuration changed")
        if row["seed"] != 42:
            raise ValueError("Phase 18H seed changed")
        if row["experiment_type"] == "DROP_POSITIONAL_REGION":
            region = row["region_removed"]
            if region not in regions:
                raise ValueError("Unknown Phase 18H positional region")
            if (
                row["positions_removed"] != regions[region]["positions"]
                or row["feature_names_removed"]
                != regions[region]["feature_names"]
                or row["remaining_feature_count"] != 177
                or row["requires_new_fit"] is not True
                or row["confirmatory"] is not True
            ):
                raise ValueError("Drop-region matrix specification changed")
    if sum(bool(row["requires_new_fit"]) for row in matrix) != 8:
        raise ValueError("Phase 18H new-fit budget changed")
    if (
        sum(
            row["experiment_type"] == "REUSED_FULL_REFERENCE" for row in matrix
        )
        != 2
    ):
        raise ValueError("Reused FULL reference inventory changed")
    if (
        sum(
            row["experiment_type"] == "REUSED_WHOLE_POSITIONAL_REFERENCE"
            for row in matrix
        )
        != 2
    ):
        raise ValueError("Reused whole-family reference inventory changed")


def build_protocol() -> dict:
    """Build and validate the design-only Phase 18G protocol."""
    with phase18g_protocol_guard():
        names = canonical_feature_names()
        positional = positional_feature_names()
        orientation = orientation_audit()
        regions = region_registry()
        split = verify_split_lock()
        source_hashes = verify_feature_source_integrity()
        references = verify_phase18f_references()
        matrix = experiment_matrix(regions, references)
        payload = {
            "phase": PHASE,
            "protocol_version": PROTOCOL_VERSION,
            "final_decision": "READY_FOR_PHASE18H",
            "future_phase": FUTURE_PHASE,
            "scope": {
                "design_and_preregistration_only": True,
                "model_training_occurred": False,
                "predictions_generated": False,
                "feature_ablations_executed": False,
                "phase18h_started": False,
                "locked_external_accessed": False,
            },
            "scientific_question": (
                "Is unique predictive information in the 20-mer spacer "
                "broadly "
                "distributed across fixed positional regions or concentrated "
                "toward particular regions such as the PAM-proximal segment?"
            ),
            "phase18f_context": {
                "commit": PHASE18F_COMMIT,
                "summary_path": PHASE18F_SUMMARY_PATH.relative_to(
                    ROOT
                ).as_posix(),
                "summary_sha256": PHASE18F_SUMMARY_SHA256,
                "full_performance": {
                    domain: references[domain]["full"]["metrics"]
                    for domain in DOMAINS
                },
                "whole_positional_classification": (
                    "ESSENTIAL_OR_STRONG_CONTRIBUTOR"
                ),
                "domain_consistency": "CROSS_DOMAIN_CONSISTENT",
                "interpretation": (
                    "Strong unique predictive contribution conditional on "
                    "retained features; not biological causality."
                ),
                "analysis_recovery_additional_fits": 0,
                "analysis_recovery_additional_predictions": 0,
            },
            "feature_space": {
                "canonical_count": len(names),
                "canonical_feature_sha256": CANONICAL_FEATURE_SHA256,
                "positional_count": len(positional),
                "positional_feature_sha256": POSITIONAL_FEATURE_SHA256,
                "positional_ordered_names": list(positional),
                "position_to_features": {
                    str(position): list(features)
                    for position, features in position_to_features().items()
                },
            },
            "orientation": orientation,
            "regions": regions,
            "region_design_rationale": (
                "Four contiguous equal-width regions were fixed before Phase "
                "18H. Each covers five positions and 20 indicators. "
                "Boundaries "
                "were not adjusted around Phase 18D importance observations."
            ),
            "phase18f_references": references,
            "whole_family_reference_policy": {
                "reuse": True,
                "new_drop_all_fit_required": False,
                "role": (
                    "Secondary locked context only; regional effects are not "
                    "assumed to sum to the whole-family effect."
                ),
            },
            "optional_region_only_policy": {
                "registered": False,
                "registered_fit_count": 0,
                "decision": "EXCLUDED_FROM_PHASE18H",
                "rationale": (
                    "Region-only models answer a distinct sufficiency "
                    "question and are unnecessary for the primary drop-region "
                    "question. They require a separate preregistration if "
                    "later justified."
                ),
            },
            "domain_policy": {
                "domains": list(DOMAINS),
                "pooled_labels": False,
                "identical_region_definitions": True,
                "native_label_scales_retained": True,
                "raw_rmse_cross_domain_comparison": False,
            },
            "split_policy": split,
            "xgboost_configuration": canonical_xgboost_configuration(),
            "seed_policy": {
                "fit_seeds": [42],
                "redundant_repeats": False,
                "reason": (
                    "Phase 18C observed identical XGBoost outcomes at seeds "
                    "42-46 with row and feature sampling fixed at 1.0; Phase "
                    "18F retained deterministic seed 42."
                ),
            },
            "metric_policy": metric_policy(),
            "bootstrap_policy": bootstrap_policy(),
            "multiple_comparison_policy": multiple_comparison_policy(),
            "regional_contribution_decision_rules": (
                regional_contribution_decision_rules()
            ),
            "cross_domain_region_decision_rules": (
                cross_domain_region_decision_rules()
            ),
            "distribution_interpretation": {
                "BROADLY_DISTRIBUTED": (
                    "At least two regions receive strong or moderate regional "
                    "contributor classifications within a domain."
                ),
                "CONCENTRATED_SINGLE_REGION": (
                    "Exactly one region receives a contributor classification "
                    "and all other regions receive weak classifications "
                    "within "
                    "a domain."
                ),
                "NO_CLEAR_REGIONAL_LOCALIZATION": (
                    "No region receives a contributor classification, or any "
                    "required regional result is unstable/inconclusive."
                ),
            },
            "preregistered_questions": [
                (
                    "Does removal of REGION_4_PAM_PROXIMAL cause the largest "
                    "descriptive degradation among the four fixed regions?"
                ),
                (
                    "Does removal of REGION_1_PAM_DISTAL show weaker "
                    "descriptive "
                    "degradation than REGION_4_PAM_PROXIMAL?"
                ),
                (
                    "Do at least two regions show confirmatory unique "
                    "predictive "
                    "contribution, supporting broadly distributed information?"
                ),
                (
                    "Are region-level contribution classes consistent between "
                    "DeepSpCas9 and Xiang/Luo?"
                ),
                (
                    "Does the fixed REGION_4_PAM_PROXIMAL, which contains "
                    "Phase "
                    "18D-highlighted positions 17-19, retain confirmatory "
                    "group-level predictive contribution?"
                ),
            ],
            "question_inference_boundary": (
                "Largest/weaker cross-region statements are descriptive point-"
                "effect orderings because no pairwise region-superiority "
                "tests "
                "are registered. Region-versus-FULL contribution "
                "classifications "
                "are confirmatory under the 16-interval Bonferroni family."
            ),
            "experiment_matrix": matrix,
            "budget": {
                "matrix_rows": len(matrix),
                "required_new_fits": 8,
                "reused_full_references": 2,
                "reused_whole_family_references": 2,
                "total_reused_references": 4,
                "optional_region_only_fits": 0,
                "bootstrap_comparisons": 8,
                "bootstrap_comparison_replicates": 80_000,
                "expected_burden": (
                    "Low fitting burden: Phase 18F recorded about 18 seconds "
                    "across 18 comparable fits, so eight new fits are "
                    "expected "
                    "to take roughly 8-15 seconds on similar hardware. The "
                    "10,000-replicate paired bootstrap may dominate runtime."
                ),
            },
            "excluded_searches": [
                "20 leave-one-position-out experiments",
                "80 individual nucleotide-feature removals",
                "guide_pos_19_G-specific experiments",
                "importance-guided position selection",
                "recursive positional feature elimination",
                "post-result region-boundary changes",
                "unregistered region-only models",
            ],
            "interpretation_language": {
                "required": [
                    "predictive contribution",
                    "associated positional information",
                    "regional predictive importance",
                ],
                "prohibited_claims": [
                    "molecular causality",
                    "Cas9 binding mechanism",
                    "seed-region causality",
                    "PAM-proximal mechanistic necessity",
                ],
                "biological_context_is_motivation_only": True,
            },
            "canonical_integrity": {
                "feature_sources_match_phase18b_and_phase18c_locks": True,
                "allowlisted_source_hashes": source_hashes,
                "canonical_feature_identity_locked": True,
                "positional_feature_identity_locked": True,
                "orientation_verified_from_implementation": True,
                "phase18f_reference_artifact_hashes_verified": True,
                "split_reused_without_modification": True,
            },
            "stop_conditions": [
                "locked external dataset access attempt",
                "canonical or positional feature identity/order/hash mismatch",
                "orientation or region membership mismatch",
                "Phase 18C/18F split hash or policy mismatch",
                "canonical XGBoost configuration mismatch",
                "Phase 18F reference artifact or metadata mismatch",
                "bridge or quarantine inclusion",
                "training, prediction, or ablation attempt during Phase 18G",
            ],
        }
        validate_protocol(payload)
        return payload


def validate_protocol(payload: Mapping[str, object]) -> None:
    if payload.get("final_decision") != "READY_FOR_PHASE18H":
        raise ValueError("Phase 18G final decision changed")
    scope = payload.get("scope", {})
    if not scope.get("design_and_preregistration_only") or any(
        scope.get(key)
        for key in (
            "model_training_occurred",
            "predictions_generated",
            "feature_ablations_executed",
            "phase18h_started",
            "locked_external_accessed",
        )
    ):
        raise ValueError("Phase 18G no-execution boundary changed")
    feature_space = payload["feature_space"]
    if (
        feature_space["canonical_count"] != CANONICAL_FEATURE_COUNT
        or feature_space["canonical_feature_sha256"]
        != CANONICAL_FEATURE_SHA256
        or feature_space["positional_count"] != POSITIONAL_FEATURE_COUNT
        or feature_space["positional_feature_sha256"]
        != POSITIONAL_FEATURE_SHA256
        or tuple(feature_space["positional_ordered_names"])
        != positional_feature_names()
    ):
        raise ValueError("Phase 18G feature lock changed")
    validate_region_registry(payload["regions"])
    if payload["split_policy"]["split_sha256"] != SPLIT_SHA256:
        raise ValueError("Phase 18G split lock changed")
    if payload["xgboost_configuration"] != EXPECTED_XGBOOST_CONFIGURATION:
        raise ValueError("Phase 18G XGBoost lock changed")
    validate_experiment_matrix(
        payload["experiment_matrix"], payload["regions"]
    )
    if payload["budget"]["required_new_fits"] != 8:
        raise ValueError("Phase 18H fit budget changed")
    multiplicity = payload["multiple_comparison_policy"]
    if (
        multiplicity["confirmatory_interval_n"] != 16
        or not math.isclose(multiplicity["familywise_alpha"], 0.05)
        or not math.isclose(multiplicity["per_interval_confidence"], 0.996875)
    ):
        raise ValueError("Phase 18G multiplicity lock changed")
    if payload["optional_region_only_policy"]["registered"] is not False:
        raise ValueError("Unapproved region-only experiments were registered")


def lock_protocol(payload: Mapping[str, object]) -> dict:
    if "protocol_sha256" in payload:
        raise ValueError("Protocol is already locked")
    locked = dict(payload)
    locked["protocol_sha256"] = fingerprint(payload)
    validate_locked_protocol(locked)
    return locked


def validate_locked_protocol(payload: Mapping[str, object]) -> None:
    digest = payload.get("protocol_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Phase 18G protocol hash is missing")
    unlocked = dict(payload)
    unlocked.pop("protocol_sha256")
    if fingerprint(unlocked) != digest:
        raise ValueError("Phase 18G protocol hash mismatch")
    validate_protocol(unlocked)

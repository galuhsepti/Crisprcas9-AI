"""Deterministic Phase 18G protocol tests; no fitting or predictions."""

import ast
import copy
import json

import pytest

from src.bioinformatics.sequence_features import SequenceFeatureExtractor
from src.experiment_protocols.phase18g_positional_region_protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CANONICAL_FEATURE_COUNT,
    CANONICAL_FEATURE_SHA256,
    CONFIRMATORY_INTERVAL_N,
    DOMAINS,
    EXPECTED_REGION_FEATURE_HASHES,
    EXPECTED_XGBOOST_CONFIGURATION,
    PER_INTERVAL_ALPHA,
    POSITIONAL_FEATURE_COUNT,
    POSITIONAL_FEATURE_SHA256,
    REGION_DEFINITIONS,
    ROOT,
    SIMULTANEOUS_CONFIDENCE,
    SIMULTANEOUS_QUANTILES,
    SPLIT_SHA256,
    build_protocol,
    classify_cross_domain_region,
    classify_regional_contribution,
    experiment_matrix,
    fingerprint,
    lock_protocol,
    orientation_audit,
    phase18g_protocol_guard,
    position_to_features,
    positional_feature_names,
    region_registry,
    serialize,
    validate_experiment_matrix,
    validate_locked_protocol,
    validate_region_registry,
    verify_phase18f_references,
)


@pytest.fixture(scope="module")
def regions():
    return region_registry()


@pytest.fixture(scope="module")
def locked():
    return lock_protocol(build_protocol())


def test_exact_80_positional_feature_identity_and_hash():
    observed = positional_feature_names()
    expected = tuple(
        f"guide_pos_{position}_{nucleotide}"
        for position in range(20)
        for nucleotide in "ACGT"
    )
    assert observed == expected
    assert len(observed) == POSITIONAL_FEATURE_COUNT == 80
    assert len(set(observed)) == 80
    assert fingerprint(list(observed)) == POSITIONAL_FEATURE_SHA256
    assert observed[0] == "guide_pos_0_A"
    assert observed[-1] == "guide_pos_19_T"


def test_position_to_feature_mapping_is_exact_and_complete():
    mapping = position_to_features()
    assert list(mapping) == list(range(20))
    for position, features in mapping.items():
        assert features == tuple(
            f"guide_pos_{position}_{nucleotide}" for nucleotide in "ACGT"
        )
    assert len({name for names in mapping.values() for name in names}) == 80


def test_orientation_is_verified_from_canonical_extractor():
    audit = orientation_audit()
    extractor = SequenceFeatureExtractor()
    assert extractor.guide_start == audit["guide_start_index"] == 4
    assert extractor.guide_end == audit["guide_end_exclusive"] == 24
    assert extractor.pam_start == audit["pam_start_index"] == 24
    assert extractor.pam_end == audit["pam_end_exclusive"] == 27
    assert "PAM-distal" in audit["guide_position_0"]
    assert "PAM-proximal" in audit["guide_position_19"]
    assert audit["pam_distal_positions"] == list(range(0, 5))
    assert audit["pam_proximal_positions"] == list(range(15, 20))


def test_regions_are_equal_complete_disjoint_and_locked(regions):
    validate_region_registry(regions)
    assert list(regions) == [name for name, _ in REGION_DEFINITIONS]
    positions = []
    features = []
    for region, expected_positions in REGION_DEFINITIONS:
        record = regions[region]
        assert record["positions"] == list(expected_positions)
        assert record["position_count"] == 5
        assert record["feature_count"] == 20
        assert record["feature_name_sha256"] == (
            EXPECTED_REGION_FEATURE_HASHES[region]
        )
        positions.extend(record["positions"])
        features.extend(record["feature_names"])
    assert positions == list(range(20))
    assert len(set(positions)) == 20
    assert tuple(features) == positional_feature_names()
    assert len(set(features)) == 80


def test_region_boundary_and_membership_tampering_fails(regions):
    changed = copy.deepcopy(regions)
    changed["REGION_1_PAM_DISTAL"]["positions"][0] = 1
    with pytest.raises(ValueError, match="boundary"):
        validate_region_registry(changed)
    changed = copy.deepcopy(regions)
    changed["REGION_4_PAM_PROXIMAL"]["feature_names"].pop()
    with pytest.raises(ValueError, match="membership"):
        validate_region_registry(changed)


def test_canonical_and_positional_feature_locks(locked):
    feature_space = locked["feature_space"]
    assert feature_space["canonical_count"] == CANONICAL_FEATURE_COUNT == 197
    assert feature_space["canonical_feature_sha256"] == (
        CANONICAL_FEATURE_SHA256
    )
    assert feature_space["positional_count"] == 80
    assert feature_space["positional_feature_sha256"] == (
        POSITIONAL_FEATURE_SHA256
    )


def test_phase18f_references_are_compatible_and_reused():
    references = verify_phase18f_references()
    assert set(references) == set(DOMAINS)
    for domain in DOMAINS:
        assert references[domain]["full"]["compatibility_verified"] is True
        assert references[domain]["full"]["feature_count"] == 197
        assert (
            references[domain]["drop_all_position_specific"][
                "compatibility_verified"
            ]
            is True
        )
        assert (
            references[domain]["drop_all_position_specific"]["feature_count"]
            == 117
        )
        assert references[domain]["whole_family_classification"] == (
            "ESSENTIAL_OR_STRONG_CONTRIBUTOR"
        )


def test_matrix_is_deterministic_and_has_exact_fit_budget(regions):
    references = verify_phase18f_references()
    first = experiment_matrix(regions, references)
    second = experiment_matrix(regions, references)
    assert serialize(first) == serialize(second)
    validate_experiment_matrix(first, regions)
    assert len(first) == 12
    assert len({row["experiment_id"] for row in first}) == 12
    assert sum(row["requires_new_fit"] for row in first) == 8
    assert (
        sum(
            row["experiment_type"] == "DROP_POSITIONAL_REGION" for row in first
        )
        == 8
    )
    assert (
        sum(row["experiment_type"] == "REUSED_FULL_REFERENCE" for row in first)
        == 2
    )
    assert (
        sum(
            row["experiment_type"] == "REUSED_WHOLE_POSITIONAL_REFERENCE"
            for row in first
        )
        == 2
    )


def test_drop_region_rows_remove_20_and_retain_177(regions):
    matrix = experiment_matrix(regions, verify_phase18f_references())
    drops = [
        row
        for row in matrix
        if row["experiment_type"] == "DROP_POSITIONAL_REGION"
    ]
    for row in drops:
        region = regions[row["region_removed"]]
        assert row["positions_removed"] == region["positions"]
        assert row["feature_names_removed"] == region["feature_names"]
        assert len(row["positions_removed"]) == 5
        assert len(row["feature_names_removed"]) == 20
        assert row["remaining_feature_count"] == 177
        assert row["confirmatory"] is True


def test_split_and_xgboost_configuration_are_locked(locked):
    assert locked["split_policy"]["split_sha256"] == SPLIT_SHA256
    assert locked["split_policy"]["reuse_without_modification"] is True
    assert locked["split_policy"]["bridge_excluded"] is True
    assert locked["split_policy"]["quarantine_excluded"] is True
    assert locked["xgboost_configuration"] == EXPECTED_XGBOOST_CONFIGURATION
    assert locked["xgboost_configuration"]["early_stopping"] is False
    assert locked["xgboost_configuration"]["hyperparameter_tuning"] is False


def test_seed_and_bootstrap_policies_are_locked(locked):
    assert locked["seed_policy"]["fit_seeds"] == [42]
    assert locked["seed_policy"]["redundant_repeats"] is False
    bootstrap = locked["bootstrap_policy"]
    assert bootstrap["replicates"] == BOOTSTRAP_REPLICATES == 10_000
    assert bootstrap["seed"] == BOOTSTRAP_SEED == 42
    assert bootstrap["rng"] == "numpy.random.Generator(PCG64(42))"
    assert "spacer20" in bootstrap["resampling_unit"]


def test_multiple_comparison_calculation_is_exact(locked):
    policy = locked["multiple_comparison_policy"]
    assert CONFIRMATORY_INTERVAL_N == 16
    assert PER_INTERVAL_ALPHA == pytest.approx(0.003125)
    assert SIMULTANEOUS_CONFIDENCE == pytest.approx(0.996875)
    assert SIMULTANEOUS_QUANTILES == pytest.approx((0.0015625, 0.9984375))
    assert policy["confirmatory_interval_n"] == 16
    assert policy["per_interval_confidence_percent"] == pytest.approx(99.6875)


def _classification_record(
    delta_spearman=-0.1,
    delta_rmse=0.1,
    spearman_ci=(-0.2, -0.01),
    rmse_ci=(0.01, 0.2),
    invalid=0,
):
    return {
        "delta_spearman": delta_spearman,
        "delta_rmse": delta_rmse,
        "adjusted_delta_spearman_ci": spearman_ci,
        "adjusted_delta_rmse_ci": rmse_ci,
        "invalid_bootstrap_n": invalid,
    }


@pytest.mark.parametrize(
    "record,expected",
    [
        (_classification_record(), "STRONG_REGIONAL_CONTRIBUTOR"),
        (
            _classification_record(spearman_ci=(-0.2, 0.1)),
            "MODERATE_REGIONAL_CONTRIBUTOR",
        ),
        (
            _classification_record(
                delta_spearman=0.0,
                delta_rmse=0.0,
                spearman_ci=(-0.1, 0.1),
                rmse_ci=(-0.1, 0.1),
            ),
            "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
        ),
        (
            _classification_record(
                delta_spearman=0.1,
                delta_rmse=-0.1,
                spearman_ci=(0.01, 0.2),
                rmse_ci=(-0.2, -0.01),
            ),
            "POTENTIALLY_REDUNDANT_REGION",
        ),
        (
            _classification_record(
                spearman_ci=(-0.2, -0.01),
                rmse_ci=(-0.2, -0.01),
            ),
            "UNSTABLE_OR_INCONCLUSIVE",
        ),
        (
            _classification_record(invalid=1),
            "UNSTABLE_OR_INCONCLUSIVE",
        ),
    ],
)
def test_regional_contribution_classification_rules(record, expected):
    assert classify_regional_contribution(record) == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (
            "STRONG_REGIONAL_CONTRIBUTOR",
            "MODERATE_REGIONAL_CONTRIBUTOR",
            "CROSS_DOMAIN_CONSISTENT",
        ),
        (
            "MODERATE_REGIONAL_CONTRIBUTOR",
            "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
            "DOMAIN_A_ENRICHED",
        ),
        (
            "POTENTIALLY_REDUNDANT_REGION",
            "STRONG_REGIONAL_CONTRIBUTOR",
            "DOMAIN_B_ENRICHED",
        ),
        (
            "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
            "POTENTIALLY_REDUNDANT_REGION",
            "WEAK_IN_BOTH",
        ),
        (
            "UNSTABLE_OR_INCONCLUSIVE",
            "STRONG_REGIONAL_CONTRIBUTOR",
            "UNSTABLE_OR_INCONCLUSIVE",
        ),
    ],
)
def test_cross_domain_region_classification_rules(left, right, expected):
    assert classify_cross_domain_region(left, right) == expected


def test_region_only_and_individual_searches_are_excluded(locked):
    optional = locked["optional_region_only_policy"]
    assert optional["registered"] is False
    assert optional["registered_fit_count"] == 0
    excluded = " ".join(locked["excluded_searches"])
    assert "20 leave-one-position-out" in excluded
    assert "80 individual" in excluded
    assert "guide_pos_19_G" in excluded
    assert "importance-guided" in excluded


@pytest.mark.parametrize(
    "name", ["Moreno-Mateos.csv", "MORENO.csv", "mateos.xlsx"]
)
def test_locked_dataset_prohibition_fails_before_io(tmp_path, name):
    with phase18g_protocol_guard():
        with pytest.raises(PermissionError):
            (tmp_path / name).open("rb")


def test_no_training_or_prediction_enforcement():
    namespace = {"__name__": "src.models.synthetic_phase18g_test"}
    exec("def fit():\n    raise AssertionError('must not run')", namespace)
    exec("def predict():\n    raise AssertionError('must not run')", namespace)
    with phase18g_protocol_guard():
        with pytest.raises(RuntimeError, match="prohibited"):
            namespace["fit"]()
    with phase18g_protocol_guard():
        with pytest.raises(RuntimeError, match="prohibited"):
            namespace["predict"]()


def test_protocol_serialization_and_hash_are_deterministic(locked):
    second = lock_protocol(build_protocol())
    assert serialize(locked) == serialize(second)
    decoded = json.loads(serialize(locked))
    validate_locked_protocol(decoded)
    changed = copy.deepcopy(decoded)
    changed["regions"]["REGION_1_PAM_DISTAL"]["positions"][0] = 1
    with pytest.raises(ValueError, match="hash"):
        validate_locked_protocol(changed)


def test_ready_conditions_and_no_execution_scope(locked):
    assert locked["final_decision"] == "READY_FOR_PHASE18H"
    assert locked["budget"]["required_new_fits"] == 8
    assert locked["budget"]["total_reused_references"] == 4
    scope = locked["scope"]
    assert scope["design_and_preregistration_only"] is True
    assert scope["model_training_occurred"] is False
    assert scope["predictions_generated"] is False
    assert scope["feature_ablations_executed"] is False
    assert scope["phase18h_started"] is False
    assert scope["locked_external_accessed"] is False


def test_source_contains_no_model_execution_implementation():
    for relative in (
        "src/experiment_protocols/phase18g_positional_region_protocol.py",
        "scripts/run_phase18g_positional_region_protocol.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                calls.append(node.func.attr)
        assert not any(
            name.startswith(("torch", "sklearn", "xgboost", "src.models"))
            for name in imports
        )
        assert set(calls).isdisjoint(
            {
                "fit",
                "partial_fit",
                "predict",
                "train",
                "backward",
                "step",
            }
        )

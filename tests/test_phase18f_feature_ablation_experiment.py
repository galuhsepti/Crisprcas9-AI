"""Focused Phase 18F preflight tests; no model fitting or predictions."""

import copy
import json

import numpy as np
import pytest

from scripts.complete_phase18f_feature_ablation import (
    vectorized_paired_bootstrap,
)
from src.experiment_protocols.phase18c_access import resolve_phase18c_path
from src.experiment_protocols.phase18e_feature_ablation_protocol import (
    CANONICAL_FEATURE_SHA256,
    EXPECTED_XGBOOST_CONFIGURATION,
    FEATURE_FAMILIES,
    SPLIT_SHA256,
    canonical_feature_names,
    feature_family_registry,
)
from src.feature_ablation.phase18f import (
    PHASE18E_PROTOCOL_SHA256,
    bonferroni_interval_quantiles,
    classify_contribution,
    classify_domain_consistency,
    load_phase18e_protocol,
    paired_bootstrap,
    phase18f_output_paths,
    selected_feature_names,
    validate_matrix,
    validate_prediction_repeatability,
)


@pytest.fixture(scope="module")
def protocol():
    return load_phase18e_protocol()


def test_phase18e_protocol_sha_lock(protocol):
    assert protocol["protocol_sha256"] == PHASE18E_PROTOCOL_SHA256


def test_feature_space_sha_and_registry_identity(protocol):
    assert protocol["feature_space"]["canonical_feature_sha256"] == (
        CANONICAL_FEATURE_SHA256
    )
    assert protocol["feature_space"]["canonical_ordered_names"] == list(
        canonical_feature_names()
    )
    assert protocol["feature_space"]["family_registry"] == (
        feature_family_registry()
    )
    memberships = [
        set(protocol["feature_space"]["family_registry"][family])
        for family in FEATURE_FAMILIES
    ]
    assert sum(map(len, memberships)) == 197
    assert len(set().union(*memberships)) == 197


def test_exact_drop_counts_and_matrix_lock(protocol):
    matrix = validate_matrix(protocol)
    assert len(matrix) == 18
    counts = {
        row["feature_family_removed"]: row["number_of_features_remaining"]
        for row in matrix
        if row["domain"] == "deepspcas9"
        and row["experiment_type"] == "DROP_ONE_FAMILY"
    }
    assert counts == {
        "GC_AND_SKEW": 187,
        "NUCLEOTIDE_COMPOSITION": 190,
        "DINUCLEOTIDE_FREQUENCY": 181,
        "GLOBAL_KMER": 117,
        "ENTROPY_COMPLEXITY": 193,
        "POSITION_SPECIFIC_NUCLEOTIDE": 117,
    }
    for row in matrix:
        selected = selected_feature_names(row)
        assert len(selected) == row["number_of_features_remaining"]
        assert row["split_hash"] == SPLIT_SHA256
        assert row["canonical_xgboost_configuration"] == (
            EXPECTED_XGBOOST_CONFIGURATION
        )


def test_matrix_tampering_is_rejected(protocol):
    changed = copy.deepcopy(protocol)
    changed["experiment_matrix"][0]["fit_seed"] = 43
    with pytest.raises(RuntimeError, match="seed"):
        validate_matrix(changed)


@pytest.mark.parametrize(
    "name", ["Moreno-Mateos.csv", "MORENO.csv", "mateos.xlsx"]
)
def test_locked_dataset_prohibition_uses_synthetic_paths(tmp_path, name):
    with pytest.raises(PermissionError):
        resolve_phase18c_path(tmp_path / name)


def test_bridge_quarantine_and_internal_test_are_closed_before_freeze():
    from src.feature_ablation.phase18f import Phase18FData

    data = Phase18FData({}, frozenset({"synthetic"}), object())
    for partition in ("bridge", "quarantine", "validation"):
        with pytest.raises(PermissionError):
            data.partition("deepspcas9", partition)
    with pytest.raises(PermissionError, match="closed"):
        data.partition("deepspcas9", "internal_test")


def test_deterministic_prediction_validator():
    predictions = np.asarray([0.1, 0.2, 0.3])
    validate_prediction_repeatability(predictions, predictions.copy())
    with pytest.raises(RuntimeError, match="repeatability"):
        validate_prediction_repeatability(predictions, predictions[::-1])
    with pytest.raises(RuntimeError, match="Nonfinite"):
        validate_prediction_repeatability([0.1, np.nan], [0.1, np.nan])


def test_paired_bootstrap_is_deterministic():
    labels = np.arange(12, dtype=float)
    groups = [f"g-{index // 2}" for index in range(12)]
    full = labels + np.asarray([0, 1, -1, 1, -1, 1] * 2, dtype=float)
    comparison = labels + np.asarray([1, -1, 2, -2, 1, -1] * 2, dtype=float)
    predictions = {"drop": comparison}
    types = {"drop": "DROP_ONE_FAMILY"}
    first, arrays_first = paired_bootstrap(
        labels, groups, full, predictions, types, iterations=20
    )
    second, arrays_second = paired_bootstrap(
        labels, groups, full, predictions, types, iterations=20
    )
    assert first == second
    assert arrays_first.keys() == arrays_second.keys()
    assert all(
        np.array_equal(arrays_first[key], arrays_second[key])
        for key in arrays_first
    )


def test_recovery_bootstrap_matches_locked_method_for_unique_groups():
    labels = np.arange(20, dtype=float)
    groups = [f"g-{index:02d}" for index in range(20)]
    full = labels + np.tile([0.0, 1.0, -1.0, 0.5], 5)
    predictions = {
        "drop": labels + np.tile([1.0, -2.0, 0.5, -0.5], 5),
        "only": labels + np.tile([3.0, -1.0, 2.0, -2.0], 5),
    }
    types = {"drop": "DROP_ONE_FAMILY", "only": "FAMILY_ONLY"}
    _, locked = paired_bootstrap(
        labels, groups, full, predictions, types, iterations=40
    )
    _, recovered = vectorized_paired_bootstrap(
        labels,
        groups,
        full,
        predictions,
        types,
        iterations=40,
        chunk_size=7,
    )
    assert locked.keys() == recovered.keys()
    for key in locked:
        np.testing.assert_allclose(locked[key], recovered[key], atol=1e-15)


def test_bonferroni_interval_calculation():
    lower, upper = bonferroni_interval_quantiles()
    assert lower == pytest.approx(0.0010416666666666675)
    assert upper == pytest.approx(0.9989583333333333)
    assert 1 - 0.05 / 24 == pytest.approx(0.9979166666666667)


def contribution_record(
    delta_spearman=-0.1,
    delta_rmse=0.1,
    spearman_ci=(-0.2, -0.01),
    rmse_ci=(0.01, 0.2),
):
    return {
        "delta_spearman": delta_spearman,
        "delta_rmse": delta_rmse,
        "adjusted_delta_spearman_ci": spearman_ci,
        "adjusted_delta_rmse_ci": rmse_ci,
        "invalid_bootstrap_n": 0,
    }


@pytest.mark.parametrize(
    "record,expected",
    [
        (contribution_record(), "ESSENTIAL_OR_STRONG_CONTRIBUTOR"),
        (
            contribution_record(spearman_ci=(-0.2, 0.1)),
            "MODERATE_CONTRIBUTOR",
        ),
        (
            contribution_record(
                delta_spearman=0.0,
                delta_rmse=0.0,
                spearman_ci=(-0.1, 0.1),
                rmse_ci=(-0.1, 0.1),
            ),
            "LITTLE_UNIQUE_CONTRIBUTION",
        ),
        (
            contribution_record(
                delta_spearman=0.1,
                delta_rmse=-0.1,
                spearman_ci=(0.01, 0.2),
                rmse_ci=(-0.2, -0.01),
            ),
            "POTENTIALLY_REDUNDANT",
        ),
        (
            contribution_record(
                spearman_ci=(-0.2, -0.01), rmse_ci=(-0.2, -0.01)
            ),
            "UNSTABLE_OR_INCONCLUSIVE",
        ),
    ],
)
def test_contribution_classification(record, expected):
    assert classify_contribution(record) == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (
            "ESSENTIAL_OR_STRONG_CONTRIBUTOR",
            "MODERATE_CONTRIBUTOR",
            "CROSS_DOMAIN_CONSISTENT",
        ),
        (
            "MODERATE_CONTRIBUTOR",
            "LITTLE_UNIQUE_CONTRIBUTION",
            "DOMAIN_A_ENRICHED",
        ),
        (
            "POTENTIALLY_REDUNDANT",
            "MODERATE_CONTRIBUTOR",
            "DOMAIN_B_ENRICHED",
        ),
        (
            "LITTLE_UNIQUE_CONTRIBUTION",
            "POTENTIALLY_REDUNDANT",
            "WEAK_IN_BOTH",
        ),
        (
            "UNSTABLE_OR_INCONCLUSIVE",
            "MODERATE_CONTRIBUTOR",
            "UNSTABLE_OR_INCONCLUSIVE",
        ),
    ],
)
def test_domain_consistency_classification(left, right, expected):
    assert classify_domain_consistency(left, right) == expected


def test_artifact_paths_are_phase18f_isolated():
    results, models = phase18f_output_paths("20260915_000000")
    assert "phase18f" in results.parts
    assert "phase18f" in models.parts
    assert "phase18c" not in results.parts
    assert results.name == models.name == "campaign_20260915_000000"


def test_protocol_json_is_roundtrip_stable(protocol):
    encoded = json.dumps(protocol, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == protocol

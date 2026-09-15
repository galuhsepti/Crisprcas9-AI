"""Focused Phase 18H tests; real model fitting is prohibited here."""

import copy

import numpy as np
import pytest

from src.experiment_protocols.phase18e_feature_ablation_protocol import (
    CANONICAL_FEATURE_SHA256,
    EXPECTED_XGBOOST_CONFIGURATION,
    SPLIT_SHA256,
    canonical_feature_names,
)
from src.experiment_protocols.phase18g_positional_region_protocol import (
    REGION_DEFINITIONS,
    SIMULTANEOUS_QUANTILES,
)
from src.feature_ablation.phase18h import (
    PHASE18G_PROTOCOL_SHA256,
    Phase18HData,
    bonferroni_interval_quantiles,
    classify_distribution,
    load_phase18g_protocol,
    paired_bootstrap,
    phase18h_output_paths,
    require_execution_confirmation,
    run_preflight,
    selected_feature_names,
    validate_matrix,
    validate_prediction_repeatability,
)
from src.models.xgboost_model import XGBoostModel


@pytest.fixture(scope="module")
def protocol():
    return load_phase18g_protocol()


def test_authoritative_protocol_and_matrix_are_locked(protocol):
    assert protocol["protocol_sha256"] == PHASE18G_PROTOCOL_SHA256
    assert protocol["feature_space"]["canonical_feature_sha256"] == (
        CANONICAL_FEATURE_SHA256
    )
    matrix = validate_matrix(protocol)
    assert len(matrix) == 12
    assert sum(row["requires_new_fit"] for row in matrix) == 8
    assert sum(not row["requires_new_fit"] for row in matrix) == 4
    assert all(row["split_hash"] == SPLIT_SHA256 for row in matrix)
    assert all(
        row["xgboost_configuration"] == EXPECTED_XGBOOST_CONFIGURATION
        for row in matrix
    )


def test_only_locked_drop_region_rows_are_trainable(protocol):
    matrix = validate_matrix(protocol)
    trainable = [row for row in matrix if row["requires_new_fit"]]
    assert [row["experiment_id"] for row in trainable] == [
        f"18H_XGB_{code}_DROP_{region}"
        for code in ("A", "B")
        for region, _ in REGION_DEFINITIONS
    ]
    assert all(
        row["experiment_type"] == "DROP_POSITIONAL_REGION"
        for row in trainable
    )
    assert all(row["remaining_feature_count"] == 177 for row in trainable)


def test_region_selection_removes_exactly_20_in_canonical_order(protocol):
    canonical = canonical_feature_names()
    for row in validate_matrix(protocol):
        if not row["requires_new_fit"]:
            continue
        selected = selected_feature_names(row, canonical)
        assert len(row["feature_names_removed"]) == 20
        assert len(selected) == 177
        assert selected == tuple(
            name
            for name in canonical
            if name not in set(row["feature_names_removed"])
        )
        assert not set(selected) & set(row["feature_names_removed"])


def test_matrix_tampering_is_rejected(protocol):
    changed = copy.deepcopy(protocol)
    changed["experiment_matrix"][1]["requires_new_fit"] = False
    with pytest.raises(RuntimeError, match="matrix|fit budget"):
        validate_matrix(changed)


def test_bridge_quarantine_and_internal_test_are_closed_before_freeze():
    data = Phase18HData({}, frozenset({"synthetic"}), object())
    for partition in ("bridge", "quarantine", "validation"):
        with pytest.raises(PermissionError):
            data.partition("deepspcas9", partition)
    with pytest.raises(PermissionError, match="closed"):
        data.partition("deepspcas9", "internal_test")


def test_execution_interlock_rejects_missing_or_wrong_digests(monkeypatch):
    monkeypatch.setattr(
        "src.feature_ablation.phase18h.implementation_sha256",
        lambda: "implementation-lock",
    )
    for execute, protocol, implementation in (
        (False, PHASE18G_PROTOCOL_SHA256, "implementation-lock"),
        (True, "wrong", "implementation-lock"),
        (True, PHASE18G_PROTOCOL_SHA256, "wrong"),
    ):
        with pytest.raises(PermissionError):
            require_execution_confirmation(execute, protocol, implementation)
    require_execution_confirmation(
        True, PHASE18G_PROTOCOL_SHA256, "implementation-lock"
    )


def test_prediction_repeatability_validator():
    predictions = np.asarray([0.1, 0.2, 0.3])
    validate_prediction_repeatability(predictions, predictions.copy())
    with pytest.raises(RuntimeError, match="repeatability"):
        validate_prediction_repeatability(predictions, predictions[::-1])
    with pytest.raises(RuntimeError, match="Nonfinite"):
        validate_prediction_repeatability([0.1, np.nan], [0.1, np.nan])


def test_phase18h_bootstrap_is_deterministic_with_repeated_groups():
    labels = np.arange(12, dtype=float)
    groups = [f"g-{index // 2}" for index in range(12)]
    full = labels + np.asarray([0, 1, -1, 1, -1, 1] * 2, dtype=float)
    comparison = labels + np.asarray([1, -1, 2, -2, 1, -1] * 2, dtype=float)
    first, arrays_first = paired_bootstrap(
        labels, groups, full, {"drop": comparison}, iterations=20
    )
    second, arrays_second = paired_bootstrap(
        labels, groups, full, {"drop": comparison}, iterations=20
    )
    assert first == second
    assert arrays_first.keys() == arrays_second.keys()
    assert all(
        np.array_equal(arrays_first[key], arrays_second[key])
        for key in arrays_first
    )
    assert set(first["drop"]["adjusted_99_6875_ci"]) == {
        "delta_spearman",
        "delta_rmse",
    }


def test_bonferroni_interval_calculation():
    assert bonferroni_interval_quantiles() == pytest.approx(
        SIMULTANEOUS_QUANTILES
    )
    assert SIMULTANEOUS_QUANTILES == pytest.approx(
        (0.0015625, 0.9984375)
    )


@pytest.mark.parametrize(
    "classes,expected",
    [
        (
            [
                "STRONG_REGIONAL_CONTRIBUTOR",
                "MODERATE_REGIONAL_CONTRIBUTOR",
                "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
                "POTENTIALLY_REDUNDANT_REGION",
            ],
            "BROADLY_DISTRIBUTED",
        ),
        (
            [
                "STRONG_REGIONAL_CONTRIBUTOR",
                "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
                "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
                "POTENTIALLY_REDUNDANT_REGION",
            ],
            "CONCENTRATED_SINGLE_REGION",
        ),
        (
            [
                "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
                "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
                "POTENTIALLY_REDUNDANT_REGION",
                "POTENTIALLY_REDUNDANT_REGION",
            ],
            "NO_CLEAR_REGIONAL_LOCALIZATION",
        ),
        (
            [
                "STRONG_REGIONAL_CONTRIBUTOR",
                "UNSTABLE_OR_INCONCLUSIVE",
                "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
                "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
            ],
            "NO_CLEAR_REGIONAL_LOCALIZATION",
        ),
    ],
)
def test_distribution_classification(classes, expected):
    assert classify_distribution(classes) == expected


def test_artifact_paths_are_phase18h_isolated():
    results, models = phase18h_output_paths("20260915_120000")
    assert "phase18h" in results.parts
    assert "phase18h" in models.parts
    assert "phase18f" not in results.parts
    assert results.name == models.name == "campaign_20260915_120000"


def test_preflight_cannot_fit_or_predict(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("preflight attempted model execution")

    monkeypatch.setattr(XGBoostModel, "fit", prohibited)
    monkeypatch.setattr(XGBoostModel, "predict", prohibited)
    result = run_preflight()
    assert result["status"] == "PREFLIGHT_PASSED_NO_TRAINING"
    assert result["matrix_rows"] == 12
    assert result["planned_new_fits"] == 8
    assert result["reused_references"] == 4
    assert result["model_training_occurred"] is False
    assert result["predictions_generated"] is False
    assert result["output_created"] is False


def test_runner_without_execute_never_reaches_campaign(monkeypatch):
    from scripts import run_phase18h_positional_region_ablation as runner

    expected = {"status": "PREFLIGHT_PASSED_NO_TRAINING"}
    monkeypatch.setattr(runner, "run_preflight", lambda: expected)

    def prohibited(*args, **kwargs):
        raise AssertionError("default runner reached campaign execution")

    monkeypatch.setattr(runner, "run_campaign", prohibited)
    assert runner.main([]) == expected

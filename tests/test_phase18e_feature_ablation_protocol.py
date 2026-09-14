"""Deterministic Phase 18E protocol tests; no fitting or predictions."""

import ast
import copy
import json

import pytest

from src.bioinformatics.sequence_features import SequenceFeatureExtractor
from src.experiment_protocols.phase18e_feature_ablation_protocol import (
    BOOTSTRAP_REPLICATES,
    CANONICAL_FEATURE_COUNT,
    CANONICAL_FEATURE_SHA256,
    DOMAINS,
    EXPECTED_XGBOOST_CONFIGURATION,
    FEATURE_FAMILIES,
    PHASE18B_PROTOCOL_SHA256,
    ROOT,
    SPLIT_SHA256,
    bootstrap_group_draws,
    build_protocol,
    canonical_feature_names,
    canonical_xgboost_configuration,
    experiment_matrix,
    feature_family_registry,
    fingerprint,
    lock_protocol,
    phase18e_protocol_guard,
    serialize,
    validate_feature_family_registry,
    validate_locked_protocol,
    verify_split_lock,
)


@pytest.fixture(scope="module")
def registry():
    return feature_family_registry()


@pytest.fixture(scope="module")
def locked():
    return lock_protocol(build_protocol())


def test_canonical_197_feature_identity_and_order():
    implementation_names = tuple(
        SequenceFeatureExtractor().get_feature_names()
    )
    assert canonical_feature_names() == implementation_names
    assert len(implementation_names) == CANONICAL_FEATURE_COUNT == 197
    assert len(set(implementation_names)) == 197
    assert fingerprint(list(implementation_names)) == CANONICAL_FEATURE_SHA256


def test_feature_family_partition_is_complete_and_disjoint(registry):
    validate_feature_family_registry(registry)
    names = set(canonical_feature_names())
    memberships = [set(registry[family]) for family in FEATURE_FAMILIES]
    assert set().union(*memberships) == names
    assert sum(map(len, memberships)) == len(names)
    for index, left in enumerate(memberships):
        for right in memberships[index + 1:]:
            assert left.isdisjoint(right)


def test_exact_feature_family_counts(registry):
    assert {family: len(names) for family, names in registry.items()} == {
        "GC_AND_SKEW": 10,
        "NUCLEOTIDE_COMPOSITION": 7,
        "DINUCLEOTIDE_FREQUENCY": 16,
        "GLOBAL_KMER": 80,
        "ENTROPY_COMPLEXITY": 4,
        "POSITION_SPECIFIC_NUCLEOTIDE": 80,
    }


def test_overlap_and_incomplete_registry_fail(registry):
    overlapping = copy.deepcopy(registry)
    overlapping["GLOBAL_KMER"].append(registry["GC_AND_SKEW"][0])
    with pytest.raises(ValueError, match="overlaps"):
        validate_feature_family_registry(overlapping)
    incomplete = copy.deepcopy(registry)
    incomplete["GLOBAL_KMER"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        validate_feature_family_registry(incomplete)


def test_drop_one_family_counts_and_matrix_determinism(registry):
    first = experiment_matrix(registry)
    second = experiment_matrix(registry)
    assert serialize(first) == serialize(second)
    assert len(first) == 18
    assert len({row["experiment_id"] for row in first}) == 18
    for row in first:
        if row["experiment_type"] == "DROP_ONE_FAMILY":
            family = row["feature_family_removed"]
            assert row["number_of_features_remaining"] == (
                197 - len(registry[family])
            )
    assert sum(row["experiment_type"] == "FULL_CONTROL" for row in first) == 2
    assert (
        sum(row["experiment_type"] == "DROP_ONE_FAMILY" for row in first) == 12
    )
    assert sum(row["experiment_type"] == "FAMILY_ONLY" for row in first) == 4


def test_split_hash_lock_with_synthetic_fixture():
    fixture = {
        "protocol_sha256": PHASE18B_PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "policy": {
            "split": {
                "unit": "spacer20",
                "ratios": {
                    "train": 0.70,
                    "validation": 0.15,
                    "internal_test": 0.15,
                },
                "bridge": "all group relatives held out",
            }
        },
    }
    assert verify_split_lock(fixture)["reuse_without_modification"] is True
    fixture["split_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="split hash"):
        verify_split_lock(fixture)


def test_canonical_xgboost_configuration_lock():
    configuration = canonical_xgboost_configuration()
    assert configuration == EXPECTED_XGBOOST_CONFIGURATION
    assert configuration["subsample"] == 1.0
    assert configuration["colsample_bytree"] == 1.0
    assert configuration["early_stopping"] is False
    assert configuration["hyperparameter_tuning"] is False
    assert configuration["random_state"] == 42


@pytest.mark.parametrize(
    "name", ["Moreno-Mateos.csv", "MORENO.csv", "mateos.xlsx"]
)
def test_locked_dataset_prohibition_fails_before_io(tmp_path, name):
    with phase18e_protocol_guard():
        with pytest.raises(PermissionError):
            (tmp_path / name).open("rb")


def test_no_training_or_prediction_enforcement():
    namespace = {"__name__": "src.models.synthetic_phase18e_test"}
    exec("def fit():\n    raise AssertionError('must not run')", namespace)
    exec("def predict():\n    raise AssertionError('must not run')", namespace)
    with phase18e_protocol_guard():
        with pytest.raises(RuntimeError, match="prohibited"):
            namespace["fit"]()
    with phase18e_protocol_guard():
        with pytest.raises(RuntimeError, match="prohibited"):
            namespace["predict"]()


def test_bootstrap_plan_is_deterministic_and_order_independent():
    groups = ["group-c", "group-a", "group-b"]
    first = bootstrap_group_draws(groups, replicates=20)
    second = bootstrap_group_draws(list(reversed(groups)), replicates=20)
    assert first == second
    assert len(first) == 20
    assert all(len(draw) == 3 for draw in first)
    with pytest.raises(ValueError):
        bootstrap_group_draws(["duplicate", "duplicate"], replicates=2)
    assert BOOTSTRAP_REPLICATES == 10_000


def test_protocol_serialization_roundtrip_and_lock(locked):
    assert serialize(locked) == serialize(lock_protocol(build_protocol()))
    decoded = json.loads(serialize(locked))
    validate_locked_protocol(decoded)
    assert set(decoded["feature_space"]["family_registry"]) == set(
        FEATURE_FAMILIES
    )
    assert set(row["domain"] for row in decoded["experiment_matrix"]) == set(
        DOMAINS
    )
    with pytest.raises(ValueError):
        serialize({"bad": float("nan")})


def test_locked_protocol_rejects_tampering(locked):
    changed = copy.deepcopy(locked)
    changed["experiment_matrix"][0]["number_of_features_remaining"] = 1
    with pytest.raises(ValueError, match="hash"):
        validate_locked_protocol(changed)


def test_source_contains_no_training_or_prediction_implementation():
    for relative in (
        "src/experiment_protocols/phase18e_feature_ablation_protocol.py",
        "scripts/run_phase18e_feature_ablation_protocol.py",
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

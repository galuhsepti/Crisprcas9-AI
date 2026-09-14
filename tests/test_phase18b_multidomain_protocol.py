"""Deterministic protocol tests; no model fitting or external-data access."""

import ast
import copy
import hashlib
import json
import pytest

from src.experiment_protocols.multidomain_protocol import (
    DOMAINS,
    EXPECTED_COUNTS,
    ROOT,
    SEEDS,
    audit_only_guard,
    build_protocol,
    classify_outcome,
    experiment_matrix,
    fingerprint,
    group_key,
    lock_protocol,
    make_split_manifest,
    policies,
    safe_path,
    serialize,
    split_for_group,
    validate_leakage,
    validate_locked_artifact,
    validate_policy,
    verify_integrity,
    verify_locked_inputs,
)

SEQ = "ACGT" + "ACGT" * 5 + "AGG" + "TTA"
CONTEXT_VARIANT = "TGCA" + SEQ[4:24] + "CGG" + "AAC"
OTHER = "ACGT" + "TGCA" * 5 + "TGG" + "TTA"


@pytest.fixture(scope="module")
def locked():
    return lock_protocol(build_protocol())


def test_canonical_integrity_real_and_corrected_rf_hash():
    snapshot = verify_integrity()
    assert len(snapshot) == 5
    assert all(r["matches_expected"] for r in snapshot.values())
    rf = snapshot["models/rf_baseline_fixed_20260905_001107.pkl"]
    assert rf["sha256"] == (
        "1f521bed090c091dd7a35e060f1166eaaa8e864ad4be09708a285750740280b6"
    )


def test_integrity_fails_for_missing_mismatched_or_empty_manifest(tmp_path):
    fixture = tmp_path / "synthetic.txt"
    fixture.write_text("synthetic", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity"):
        verify_integrity({str(fixture): "0" * 64})
    with pytest.raises(RuntimeError, match="integrity"):
        verify_integrity({str(tmp_path / "absent"): "0" * 64})
    with pytest.raises(ValueError):
        verify_integrity({})


@pytest.mark.parametrize(
    "name", ["Moreno-Mateos.csv", "MORENO.csv", "mateos.xlsx"]
)
def test_locked_path_rejected_before_actual_io(tmp_path, name):
    # This nonexistent synthetic path is rejected before any underlying open.
    with pytest.raises(PermissionError):
        safe_path(tmp_path / name)
    with audit_only_guard():
        with pytest.raises(PermissionError):
            (tmp_path / name).open("rb")


def test_unknown_dataset_and_canonical_write_fail_before_io():
    with audit_only_guard():
        with pytest.raises(PermissionError):
            (ROOT / "data" / "synthetic_unapproved.csv").open("rb")
        with pytest.raises(PermissionError):
            (ROOT / "models" / "synthetic_new_model.pt").open("wb")
        with pytest.raises(PermissionError):
            (ROOT / "results" / "predictions" / "synthetic.csv").open("wb")


def test_profile_guard_rejects_synthetic_fit_before_body():
    namespace = {"__name__": "src.models.synthetic_guard_test"}
    exec(
        "def fit():\n    raise AssertionError('body must never run')",
        namespace,
    )
    with audit_only_guard():
        with pytest.raises(
            RuntimeError, match="training/calibration prohibited"
        ):
            namespace["fit"]()


def test_exact_30mer_grouping_distinguishes_context():
    assert group_key(SEQ, "30mer") == SEQ
    assert group_key(SEQ, "30mer") != group_key(CONTEXT_VARIANT, "30mer")


def test_spacer20_grouping_is_stronger_and_geometry_exact():
    assert group_key(SEQ) == "ACGT" * 5
    assert group_key(SEQ) == group_key(CONTEXT_VARIANT)
    assert group_key(SEQ) != group_key(OTHER)


@pytest.mark.parametrize("sequence", ["A" * 20, "N" * 30, SEQ.lower(), None])
def test_geometry_rejects_mutation(sequence):
    with pytest.raises(ValueError):
        group_key(sequence)


def test_deterministic_split_independent_of_order_and_domain():
    inputs = {DOMAINS[0]: [SEQ, OTHER], DOMAINS[1]: [CONTEXT_VARIANT]}
    first = make_split_manifest(inputs, [], [])
    second = make_split_manifest(
        {d: list(reversed(v)) for d, v in inputs.items()}, [], []
    )
    assert serialize(first) == serialize(second)
    shared = hashlib.sha256(group_key(SEQ).encode()).hexdigest()
    assert len({r["split"] for r in first if r["group_sha256"] == shared}) == 1
    assert split_for_group(group_key(SEQ)) in {
        "train",
        "validation",
        "internal_test",
    }
    with pytest.raises(ValueError, match="seed"):
        split_for_group(group_key(SEQ), seed=43)


@pytest.mark.parametrize("same_exact", [True, False])
def test_cross_domain_leakage_detected_for_exact_and_spacer(same_exact):
    sequence = SEQ if same_exact else CONTEXT_VARIANT
    rows = make_split_manifest(
        {DOMAINS[0]: [SEQ], DOMAINS[1]: [sequence]}, [], []
    )
    rows[0]["split"] = "train"
    rows[1]["split"] = "internal_test"
    with pytest.raises(ValueError, match="leakage"):
        validate_leakage(rows, set())


def test_bridge_holds_both_domains_and_context_relatives():
    rows = make_split_manifest(
        {DOMAINS[0]: [SEQ, CONTEXT_VARIANT], DOMAINS[1]: [SEQ, OTHER]},
        [SEQ, OTHER],
        [SEQ],
    )
    assert sum(r["split"] == "bridge" for r in rows) == 2
    assert sum(r["split"] == "quarantine" for r in rows) == 2
    held = {r["group_sha256"] for r in rows}
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="Bridge"):
        validate_leakage(rows, held)


def test_duplicate_identity_and_invalid_bridge_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        make_split_manifest({DOMAINS[0]: [SEQ, SEQ], DOMAINS[1]: []}, [], [])
    with pytest.raises(ValueError, match="subset"):
        make_split_manifest({DOMAINS[0]: [SEQ], DOMAINS[1]: []}, [], [SEQ])


def test_experiment_matrix_and_exact_seed_lock():
    matrix = experiment_matrix()
    assert len(matrix) == 35
    assert len({r["experiment_id"] for r in matrix}) == 35
    assert SEEDS == (42, 43, 44, 45, 46)
    assert sum(r["arm"] == "D" for r in matrix) == 5
    assert {r["arm"] for r in matrix} == {
        "A",
        "B",
        "D",
        "random_forest_A",
        "random_forest_B",
        "xgboost_A",
        "xgboost_B",
    }
    validate_policy(policies(), matrix)
    matrix[0]["seed"] = 999
    with pytest.raises(ValueError, match="seed"):
        validate_policy(policies(), matrix)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("metrics", "primary", ["mae"]),
        ("metrics", "domain_aggregation", "raw_mean"),
        ("loss", "domain_weights", {DOMAINS[0]: 0.9, DOMAINS[1]: 0.1}),
        ("loss", "domain_weights", {DOMAINS[0]: -0.5, DOMAINS[1]: 1.5}),
        (
            "loss",
            "domain_weights",
            {DOMAINS[0]: float("nan"), DOMAINS[1]: 0.5},
        ),
        ("loss", "variance_source", "all_data"),
        ("training", "seeds", [42]),
        ("outcomes", "rho_preservation_margin", 0.9),
        ("outcomes", "relative_rmse_harm_margin", 0.0),
    ],
)
def test_metric_loss_seed_and_success_policies_reject_drift(
    section, key, value
):
    policy = policies()
    policy[section][key] = value
    with pytest.raises(ValueError, match=section):
        validate_policy(policy, experiment_matrix())


def evidence(
    rho=0.02, rho_ci=(0.005, 0.035), gain=0.01, gain_ci=(-0.01, 0.03)
):
    return {
        d: {
            "seeds": list(SEEDS),
            "rho": [rho] * 5,
            "rho_ci": list(rho_ci),
            "rmse_gain": [gain] * 5,
            "rmse_gain_ci": list(gain_ci),
        }
        for d in DOMAINS
    }


def test_success_requires_both_domain_preservation_and_one_improvement():
    assert classify_outcome(evidence()) == "MULTI_DOMAIN_BENEFIT"
    preserved = evidence(
        rho=0, rho_ci=(-0.01, 0.01), gain=0, gain_ci=(-0.02, 0.02)
    )
    assert classify_outcome(preserved) == "NO_CLEAR_BENEFIT"
    result = evidence()
    result[DOMAINS[1]]["rho_ci"] = [-0.011, 0.035]
    assert classify_outcome(result) == "NO_CLEAR_BENEFIT"


@pytest.mark.parametrize(
    "metric,values,ci",
    [
        ("rho", -0.04, [-0.06, -0.011]),
        ("rmse_gain", -0.05, [-0.08, -0.021]),
    ],
)
def test_negative_transfer_not_masked_by_other_domain(metric, values, ci):
    result = evidence(rho=0.2, rho_ci=(0.1, 0.3))
    result[DOMAINS[1]][metric] = [values] * 5
    result[DOMAINS[1]][metric + "_ci"] = ci
    assert classify_outcome(result) == "NEGATIVE_TRANSFER"


def test_negative_transfer_strict_boundary_and_precedence():
    result = evidence(rho=-0.02, rho_ci=(-0.03, -0.01))
    assert classify_outcome(result) == "NO_CLEAR_BENEFIT"
    assert (
        classify_outcome(result, violations=["leakage"])
        == "PROTOCOL_VIOLATION"
    )
    result[DOMAINS[0]]["rho"] = [0.0, 0.1, 0.0, 0.0, 0.0]
    assert classify_outcome(result) == "UNSTABLE_RESULT"


def test_instability_missing_nonfinite_invalid_ci_and_bootstrap():
    for key, value in [
        ("seeds", [42]),
        ("rho", [float("nan")] * 5),
        ("rho_ci", [0.04, 0.03]),
        ("invalid_bootstrap_n", 1),
    ]:
        result = evidence()
        result[DOMAINS[0]][key] = value
        assert classify_outcome(result) == "UNSTABLE_RESULT"


def test_four_of_five_seed_preservation_gate():
    result = evidence()
    result[DOMAINS[0]]["rho"] = [-0.011, -0.011, 0.03, 0.03, 0.03]
    assert classify_outcome(result) == "NO_CLEAR_BENEFIT"


def test_real_manifest_counts_and_holding(locked):
    assert locked["split_counts"] == EXPECTED_COUNTS
    assert len(locked["split_manifest"]) == 10117 + 10592
    assert locked["group_counts"] == {DOMAINS[0]: 10116, DOMAINS[1]: 10592}
    assert locked["budget"]["cnn_steps_per_epoch"] == 467
    assert locked["overlap"]["B_new_nonoverlapping"] == 10544
    assert all("label" not in r for r in locked["split_manifest"])


def test_protocol_serialization_determinism_and_roundtrip(locked):
    second = lock_protocol(build_protocol())
    assert serialize(second) == serialize(locked)
    decoded = json.loads(serialize(locked))
    validate_locked_artifact(decoded)
    assert serialize({"b": 2, "a": 1}) == serialize({"a": 1, "b": 2})
    with pytest.raises(ValueError):
        serialize({"bad": float("nan")})


def test_manifest_tampering_and_approved_digest_rejected(locked):
    tampered = copy.deepcopy(locked)
    original = tampered["split_manifest"][0]["split"]
    tampered["split_manifest"][0]["split"] = (
        "train" if original != "train" else "internal_test"
    )
    with pytest.raises(ValueError, match="hash"):
        validate_locked_artifact(tampered)
    with pytest.raises(ValueError, match="Approved"):
        validate_locked_artifact(locked, approved_digest="0" * 64)


def test_locked_preflight_and_label_mutation_detection(locked):
    verify_locked_inputs(locked, locked["protocol_sha256"])
    changed = copy.deepcopy(locked)
    del changed["protocol_sha256"]
    changed["label_fingerprints"][DOMAINS[0]] = "0" * 64
    changed = lock_protocol(changed)
    with pytest.raises(RuntimeError, match="label_fingerprints"):
        verify_locked_inputs(changed, changed["protocol_sha256"])


def test_source_policy_contains_no_model_or_training_implementation():
    for relative in (
        "src/experiment_protocols/multidomain_protocol.py",
        "scripts/run_phase18b_multidomain_protocol.py",
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
            n.startswith(("torch", "sklearn", "xgboost", "src.models"))
            for n in imports
        )
        assert set(calls).isdisjoint(
            {"fit", "predict", "backward", "train", "step"}
        )
    assert fingerprint(policies()) == fingerprint(
        json.loads(serialize(policies()))
    )

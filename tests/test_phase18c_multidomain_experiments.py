"""Synthetic Phase 18C implementation tests; no fitting or real predictions."""

import copy

import numpy as np
import pytest
import torch

from scripts import run_phase18c_multidomain_experiments as runner
from src.evaluation.phase18c_statistics import (
    DomainComparison,
    bridge_metrics,
    hierarchical_paired_bootstrap,
    paired_effects,
    regression_metrics,
)
from src.experiment_protocols.multidomain_protocol import (
    DOMAINS,
    SEEDS,
    fingerprint,
)
from src.multidomain.campaign import (
    APPROVED_PROTOCOL_SHA256,
    _tree_model,
    implementation_digest,
    output_root,
    require_execution_confirmation,
)
from src.multidomain.cnn import DomainAwareCNN, parameter_counts
from src.multidomain.data import (
    DomainRows,
    LockedDevelopmentData,
    TrainingStatistics,
    label_fingerprint,
    materialize_cnn,
    materialize_tabular,
    training_statistics,
    validate_geometry,
)
from src.multidomain.training import (
    CheckpointController,
    deterministic_epoch_batches,
    normalized_mse,
)

SEQ_A = "ACGTACGTACGTACGTACGTACGTAGGTTA"
SEQ_B = "TGCATGCATGCATGCATGCATGCATGGAAC"


def moments(mean=0.5, sd=0.2):
    return TrainingStatistics(mean, sd, sd**2, 4)


def synthetic_data():
    domains = {}
    for domain in DOMAINS:
        sequences = np.asarray([SEQ_A, SEQ_B])
        labels = np.asarray([0.1, 0.9], dtype=np.float64)
        identifiers = np.asarray(["a", "b"])
        domains[domain] = DomainRows(
            sequences=sequences,
            labels=labels,
            sequence_ids=identifiers,
            partition_indices={
                "train": np.asarray([0]),
                "validation": np.asarray([1]),
                "internal_test": np.asarray([0, 1]),
                "bridge": np.asarray([0, 1]),
                "quarantine": np.asarray([], dtype=np.int64),
            },
        )
    return LockedDevelopmentData(
        domains, APPROVED_PROTOCOL_SHA256, ["synthetic-run"]
    )


def test_training_statistics_use_float64_population_variance():
    result = training_statistics([1, 2, 3, 4])
    assert result.mean == 2.5
    assert result.variance == 1.25
    assert result.sd == np.sqrt(1.25)


@pytest.mark.parametrize("labels", [[], [1, 1], [1, np.nan], [[1, 2]]])
def test_training_statistics_reject_invalid_or_degenerate_labels(labels):
    with pytest.raises(ValueError):
        training_statistics(labels)


def test_geometry_accepts_exact_uppercase_30mers():
    validate_geometry([SEQ_A, SEQ_B])


@pytest.mark.parametrize("sequence", ["A" * 29, "N" * 30, SEQ_A.lower(), None])
def test_geometry_rejects_drift(sequence):
    with pytest.raises(ValueError):
        validate_geometry([sequence])


def test_label_fingerprint_matches_locked_float_hex_format():
    labels = np.asarray([0.25, 0.5], dtype=np.float64)
    expected = fingerprint(
        sorted([(SEQ_A, (0.25).hex()), (SEQ_B, (0.5).hex())])
    )
    assert label_fingerprint([SEQ_A, SEQ_B], labels) == expected


def test_cnn_materialization_has_locked_shape_and_dtype():
    features = materialize_cnn([SEQ_A, SEQ_B])
    assert features.shape == (2, 30, 4)
    assert features.dtype == np.float32
    assert np.all(features.sum(axis=2) == 1)


def test_tabular_materialization_has_locked_feature_count():
    features, names = materialize_tabular([SEQ_A, SEQ_B])
    assert features.shape == (2, 197)
    assert len(names) == len(set(names)) == 197
    assert np.all(np.isfinite(features))


def test_locked_data_allows_development_partitions():
    sequences, labels, identifiers = synthetic_data().partition(
        DOMAINS[0], "train"
    )
    assert sequences.tolist() == [SEQ_A]
    assert labels.tolist() == [0.1]
    assert identifiers.tolist() == ["a"]


def test_locked_data_fingerprints_immutable_raw_labels():
    observed = synthetic_data().label_fingerprints()
    expected = label_fingerprint([SEQ_A, SEQ_B], [0.1, 0.9])
    assert observed == {domain: expected for domain in DOMAINS}


@pytest.mark.parametrize("partition", ["internal_test", "bridge"])
def test_locked_data_keeps_evaluation_closed_before_freeze(partition):
    with pytest.raises(PermissionError, match="checkpoint freeze"):
        synthetic_data().partition(DOMAINS[0], partition)


def test_locked_data_allows_evaluation_after_freeze(tmp_path):
    checkpoint = tmp_path / "synthetic-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic")
    data = synthetic_data()
    freeze = data.freeze_evaluation({"synthetic-run": checkpoint})
    rows = data.partition(
        DOMAINS[1], "internal_test", evaluation_freeze=freeze
    )
    assert len(rows[0]) == 2


def test_locked_data_never_materializes_quarantine():
    with pytest.raises(PermissionError, match="Quarantine"):
        synthetic_data().partition(DOMAINS[1], "quarantine")


def test_evaluation_freeze_requires_exact_checkpoint_inventory():
    with pytest.raises(RuntimeError, match="inventory"):
        synthetic_data().freeze_evaluation({})


def test_evaluation_freeze_is_bound_to_issuing_data(tmp_path):
    checkpoint = tmp_path / "synthetic-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic")
    first = synthetic_data()
    second = synthetic_data()
    freeze = first.freeze_evaluation({"synthetic-run": checkpoint})
    with pytest.raises(PermissionError, match="checkpoint freeze"):
        second.partition(DOMAINS[0], "internal_test", evaluation_freeze=freeze)


def test_encoder_and_head_parameter_counts_are_exact():
    model = DomainAwareCNN({DOMAINS[0]: moments(), DOMAINS[1]: moments()}, 42)
    counts = parameter_counts(model)
    assert counts["encoder"] == 17920
    assert counts[f"head_{DOMAINS[0]}"] == 65
    assert counts[f"head_{DOMAINS[1]}"] == 65
    assert counts["total"] == 18050


def test_single_domain_parameter_count_is_exact():
    model = DomainAwareCNN({DOMAINS[0]: moments()}, 42)
    assert parameter_counts(model)["total"] == 17985


def test_domain_heads_have_paired_initialization_across_arms():
    stats = {DOMAINS[0]: moments(), DOMAINS[1]: moments(50, 20)}
    model_a = DomainAwareCNN({DOMAINS[0]: stats[DOMAINS[0]]}, 43)
    model_b = DomainAwareCNN({DOMAINS[1]: stats[DOMAINS[1]]}, 43)
    model_d = DomainAwareCNN(stats, 43)
    assert torch.equal(
        model_a.heads[DOMAINS[0]].weight, model_d.heads[DOMAINS[0]].weight
    )
    assert torch.equal(
        model_b.heads[DOMAINS[1]].weight, model_d.heads[DOMAINS[1]].weight
    )


def test_encoder_initialization_is_paired_across_arms():
    model_a = DomainAwareCNN({DOMAINS[0]: moments()}, 44)
    model_d = DomainAwareCNN(
        {DOMAINS[0]: moments(), DOMAINS[1]: moments(50, 20)}, 44
    )
    for left, right in zip(
        model_a.encoder.parameters(), model_d.encoder.parameters()
    ):
        assert torch.equal(left, right)


def test_domain_routing_preserves_order_and_native_units():
    model = DomainAwareCNN(
        {DOMAINS[0]: moments(0.5, 0.2), DOMAINS[1]: moments(50, 20)}, 42
    )
    for head in model.heads.values():
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    model.eval()
    inputs = torch.zeros((3, 30, 4), dtype=torch.float32)
    output = model(inputs, [DOMAINS[1], DOMAINS[0], DOMAINS[1]])
    assert output.reshape(-1).tolist() == [50.0, 0.5, 50.0]


@pytest.mark.parametrize(
    "domains",
    [[DOMAINS[0]], [DOMAINS[0], "unknown"], []],
)
def test_domain_routing_rejects_missing_or_unknown_ids(domains):
    model = DomainAwareCNN({DOMAINS[0]: moments()}, 42)
    with pytest.raises(ValueError):
        model(torch.zeros((2, 30, 4)), domains)


def test_deterministic_batches_repeat_exactly():
    identifiers = [f"id-{index:02d}" for index in range(9)]
    first = deterministic_epoch_batches(identifiers, 4, 7, 42, 0, 1)
    second = deterministic_epoch_batches(identifiers, 4, 7, 42, 0, 1)
    assert np.array_equal(first, second)
    assert first.shape == (7, 4)


def test_batch_rng_changes_by_seed_domain_epoch_and_cycle():
    identifiers = [f"id-{index:02d}" for index in range(20)]
    reference = deterministic_epoch_batches(identifiers, 4, 2, 42, 0, 1)
    assert not np.array_equal(
        reference, deterministic_epoch_batches(identifiers, 4, 2, 43, 0, 1)
    )
    assert not np.array_equal(
        reference, deterministic_epoch_batches(identifiers, 4, 2, 42, 1, 1)
    )
    assert not np.array_equal(
        reference, deterministic_epoch_batches(identifiers, 4, 2, 42, 0, 2)
    )


def test_batch_cycle_is_without_replacement_until_exhaustion():
    identifiers = [f"id-{index:02d}" for index in range(5)]
    batches = deterministic_epoch_batches(identifiers, 3, 2, 42, 0, 1).reshape(
        -1
    )
    assert len(set(batches[:5].tolist())) == 5


@pytest.mark.parametrize("identifiers", [["b", "a"], ["a", "a"], []])
def test_batches_require_sorted_unique_nonempty_ids(identifiers):
    with pytest.raises(ValueError):
        deterministic_epoch_batches(identifiers, 2, 1, 42, 0, 1)


def test_normalized_mse_uses_training_variance():
    prediction = torch.tensor([1.0, 3.0])
    labels = torch.tensor([0.0, 1.0])
    value = normalized_mse(prediction, labels, moments(0, 2))
    assert value.item() == pytest.approx(0.625)


def test_checkpoint_retains_earliest_exact_tie():
    controller = CheckpointController(patience=3)
    controller.observe(1, 1.0, {"value": 1})
    controller.observe(2, 0.9, {"value": 2})
    controller.observe(3, 0.9, {"value": 3})
    assert controller.best_epoch == 2
    assert controller.best_state == {"value": 2}


def test_patience_resets_only_for_locked_significant_improvement():
    controller = CheckpointController(patience=2, min_improvement=0.0001)
    assert controller.observe(1, 1.0, {}) is False
    assert controller.observe(2, 0.99995, {}) is False
    assert controller.observe(3, 0.99989, {}) is False
    assert controller.non_reset_epochs == 0
    assert controller.observe(4, 0.99985, {}) is False
    assert controller.observe(5, 0.99984, {}) is True


def test_regression_metrics_compute_all_locked_endpoints():
    values = regression_metrics([0, 1, 2, 3], [0.1, 0.9, 2.1, 2.9])
    assert set(values) == {"spearman", "pearson", "r2", "mae", "rmse"}
    assert values["spearman"] == 1.0


@pytest.mark.parametrize(
    "truth,prediction",
    [([1, 1], [1, 2]), ([1, 2], [2, 2]), ([1, np.nan], [1, 2])],
)
def test_regression_metrics_reject_undefined_correlations(truth, prediction):
    with pytest.raises(ValueError):
        regression_metrics(truth, prediction)


def test_paired_effects_use_d_minus_baseline_and_relative_gain():
    truth = np.arange(6, dtype=float)
    baseline = np.asarray([0, 1, 4, 2, 5, 3], dtype=float)
    shared = np.asarray([0, 1, 2, 4, 3, 5], dtype=float)
    rho, gain = paired_effects(truth, baseline, shared)
    assert rho > 0
    assert np.isfinite(gain)


def comparisons():
    truth = np.arange(20, dtype=float)
    groups = np.asarray([f"g-{index // 2:02d}" for index in range(20)])
    result = {}
    for domain_index, domain in enumerate(DOMAINS):
        baseline = {}
        shared = {}
        for seed in SEEDS:
            generator = np.random.default_rng(seed + domain_index * 100)
            baseline[seed] = truth + generator.normal(0, 3, len(truth))
            shared[seed] = truth + generator.normal(0, 2, len(truth))
        result[domain] = DomainComparison(groups, truth, baseline, shared)
    return result


def test_hierarchical_bootstrap_is_deterministic_and_paired():
    first = hierarchical_paired_bootstrap(comparisons(), iterations=20)
    second = hierarchical_paired_bootstrap(comparisons(), iterations=20)
    assert first == second
    assert set(first["evidence"]) == set(DOMAINS)
    assert all(
        len(record["rho"]) == 5 for record in first["evidence"].values()
    )


def test_hierarchical_bootstrap_rejects_seed_drift():
    with pytest.raises(ValueError, match="seed"):
        hierarchical_paired_bootstrap(comparisons(), iterations=2, seed=43)


def test_undefined_observed_effect_is_reported_as_unstable():
    invalid = comparisons()
    changed = dict(invalid)
    record = invalid[DOMAINS[0]]
    constant = {
        seed: np.ones_like(record.labels, dtype=float) for seed in SEEDS
    }
    changed[DOMAINS[0]] = DomainComparison(
        record.group_ids,
        record.labels,
        constant,
        record.multidomain_predictions,
    )
    result = hierarchical_paired_bootstrap(changed, iterations=2)
    assert result["outcome"] == "UNSTABLE_RESULT"


def test_bridge_metrics_enforce_exact_population():
    with pytest.raises(ValueError, match="41"):
        bridge_metrics([1, 2], [1, 2], [1, 2], [1, 2])


def test_bridge_metrics_include_average_tie_rank_disagreement():
    labels = np.arange(41, dtype=float)
    result = bridge_metrics(labels, labels + 1, labels, labels[::-1])
    assert result["head_to_head_spearman"] == -1.0
    assert result["mean_absolute_percentile_rank_disagreement"] > 0


@pytest.mark.parametrize("family", ["random_forest", "xgboost"])
def test_tree_factories_resolve_locked_parameters_without_fitting(family):
    model, resolved = _tree_model(family, 45)
    assert model.is_fitted is False
    assert set(resolved) == set(model.model.get_params())
    assert resolved["random_state"] == 45
    assert resolved["n_jobs"] == 4


def test_output_root_is_confined_and_not_created():
    path = output_root(APPROVED_PROTOCOL_SHA256)
    assert path.name == APPROVED_PROTOCOL_SHA256
    with pytest.raises(ValueError):
        output_root("0" * 64)


@pytest.mark.parametrize(
    "execute,protocol_digest,implementation_sha256",
    [
        (False, APPROVED_PROTOCOL_SHA256, implementation_digest()),
        (True, None, implementation_digest()),
        (True, "0" * 64, implementation_digest()),
        (True, APPROVED_PROTOCOL_SHA256, "0" * 64),
    ],
)
def test_execution_interlock_rejects_incomplete_confirmation(
    execute, protocol_digest, implementation_sha256
):
    with pytest.raises(PermissionError):
        require_execution_confirmation(
            execute, protocol_digest, implementation_sha256
        )


def test_execution_interlock_accepts_full_digest_only():
    require_execution_confirmation(
        True, APPROVED_PROTOCOL_SHA256, implementation_digest()
    )


def test_runner_defaults_to_preflight_and_never_calls_campaign(monkeypatch):
    report = {"status": "PREFLIGHT_PASSED_NO_TRAINING"}
    monkeypatch.setattr(runner, "run_preflight", lambda: copy.deepcopy(report))

    def unexpected_campaign(_, __):
        raise AssertionError("default runner reached campaign execution")

    monkeypatch.setattr(runner, "run_campaign", unexpected_campaign)
    assert runner.main([]) == report

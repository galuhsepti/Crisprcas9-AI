"""Preflight and execution orchestration for the locked Phase 18C campaign."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
import xgboost

from src.evaluation.phase18c_statistics import (
    DomainComparison,
    bridge_metrics,
    hierarchical_paired_bootstrap,
    regression_metrics,
)
from src.experiment_protocols.multidomain_protocol import (
    DOMAINS,
    ROOT,
    SEEDS,
    fingerprint,
    serialize,
    verify_integrity,
    verify_locked_inputs,
)
from src.experiment_protocols.phase18c_access import phase18c_access_guard
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.multidomain.cnn import DomainAwareCNN, parameter_counts
from src.multidomain.data import (
    LockedDevelopmentData,
    materialize_cnn,
    materialize_tabular,
    load_locked_development_data,
)
from src.multidomain.training import (
    TrainingInstability,
    process_peak_memory_bytes,
    train_cnn,
)

APPROVED_PROTOCOL_SHA256 = (
    "79ec4f03fe7314bc54a337c0d235c4f40e93b441aa40633828e1a50a3ee4d49b"
)
APPROVED_SPLIT_SHA256 = (
    "b7e043b30b3d070e1a41c461ad83774aec21a6abbfe9948cf1387a4ae49ab4a8"
)
PROTOCOL_PATH = (
    ROOT / "results/phase18b_multidomain_protocol_20260913_180731.json"
)
IMPLEMENTATION_PATHS = (
    "src/multidomain/__init__.py",
    "src/multidomain/data.py",
    "src/multidomain/cnn.py",
    "src/multidomain/training.py",
    "src/multidomain/campaign.py",
    "src/evaluation/phase18c_statistics.py",
    "src/evaluation/metrics.py",
    "src/experiment_protocols/phase18c_access.py",
    "src/bioinformatics/gc_content.py",
    "src/bioinformatics/nucleotide_composition.py",
    "src/bioinformatics/kmer.py",
    "src/bioinformatics/positional_features.py",
    "scripts/run_phase18c_multidomain_experiments.py",
    "requirements.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hashes() -> dict[str, str]:
    """Hash the additive implementation without traversing the repository."""
    return {
        relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_PATHS
    }


def implementation_digest() -> str:
    """Fingerprint the complete Phase 18C execution source manifest."""
    return fingerprint(implementation_hashes())


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "platform": platform.platform(),
    }


def load_locked_protocol() -> dict:
    """Read only the approved protocol artifact under the access guard."""
    with phase18c_access_guard():
        return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _serializable_parameter(value):
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _serializable_parameter(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serializable_parameter(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _tree_model(family: str, seed: int):
    if seed not in SEEDS:
        raise ValueError("Unregistered run seed")
    if family == "random_forest":
        model = RandomForestModel(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=4,
        )
        expected = {
            "bootstrap": True,
            "criterion": "squared_error",
            "max_depth": 20,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "min_samples_split": 5,
            "n_estimators": 200,
            "n_jobs": 4,
            "random_state": seed,
        }
    elif family == "xgboost":
        model = XGBoostModel(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=4,
            random_state=seed,
        )
        model.model.set_params(tree_method="hist", device="cpu")
        expected = {
            "colsample_bytree": 1.0,
            "device": "cpu",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "n_estimators": 100,
            "n_jobs": 4,
            "objective": "reg:squarederror",
            "random_state": seed,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "subsample": 1.0,
            "tree_method": "hist",
        }
    else:
        raise ValueError(f"Unregistered tabular family: {family}")
    raw_resolved = model.model.get_params()
    if any(raw_resolved.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Resolved {family} parameters violate the lock")
    resolved = _serializable_parameter(raw_resolved)
    return model, resolved


def resolved_matrix_parameters(payload: dict) -> dict:
    """Resolve every registered run parameter set before any fitting."""
    resolved = {}
    cnn_policy = payload["policy"]["training"]
    for specification in payload["experiment_matrix"]:
        experiment_id = specification["experiment_id"]
        family = specification["model_family"]
        if family == "cnn":
            resolved[experiment_id] = {
                "domains": specification["training_domains"],
                "seed": specification["seed"],
                "batch_size": 32,
                "steps_per_epoch": 467,
                "max_epochs": cnn_policy["max_epochs"],
                "patience": cnn_policy["patience"],
                "min_improvement": cnn_policy["min_improvement"],
                "optimizer": cnn_policy["optimizer"],
                "learning_rate": cnn_policy["learning_rate"],
                "betas": cnn_policy["betas"],
                "epsilon": cnn_policy["epsilon"],
                "weight_decay": cnn_policy["weight_decay"],
            }
        else:
            _, parameters = _tree_model(family, specification["seed"])
            resolved[experiment_id] = parameters
    expected_ids = {
        specification["experiment_id"]
        for specification in payload["experiment_matrix"]
    }
    if set(resolved) != expected_ids:
        raise RuntimeError("Resolved run matrix is incomplete")
    return resolved


def _synthetic_statistics():
    from src.multidomain.data import TrainingStatistics

    return {
        DOMAINS[0]: TrainingStatistics(0.5, 0.2, 0.04, 2),
        DOMAINS[1]: TrainingStatistics(50.0, 20.0, 400.0, 2),
    }


def output_root(protocol_hash: str) -> Path:
    """Resolve the sole allowed Phase 18C output root without creating it."""
    if protocol_hash != APPROVED_PROTOCOL_SHA256:
        raise ValueError("Unapproved protocol output namespace")
    approved = (ROOT / "results" / "phase18c" / protocol_hash).resolve()
    candidate = approved.resolve()
    if candidate != approved or not candidate.is_relative_to(ROOT / "results"):
        raise PermissionError("Phase 18C output escaped its approved root")
    return candidate


def run_preflight() -> dict:
    """Validate all locks and constructors without fitting or prediction."""
    with phase18c_access_guard():
        payload = load_locked_protocol()
        verify_locked_inputs(payload, APPROVED_PROTOCOL_SHA256)
        if payload["split_sha256"] != APPROVED_SPLIT_SHA256:
            raise RuntimeError("Approved split digest mismatch")
        statistics = _synthetic_statistics()
        model_a = DomainAwareCNN({DOMAINS[0]: statistics[DOMAINS[0]]}, 42)
        model_b = DomainAwareCNN({DOMAINS[1]: statistics[DOMAINS[1]]}, 42)
        model_d = DomainAwareCNN(statistics, 42)
        counts = {
            "A": parameter_counts(model_a),
            "B": parameter_counts(model_b),
            "D": parameter_counts(model_d),
        }
        if (
            counts["A"]["encoder"] != 17920
            or counts["A"]["total"] != 17985
            or counts["B"]["total"] != 17985
            or counts["D"]["total"] != 18050
        ):
            raise RuntimeError("CNN parameter count mismatch")
        if not torch.equal(
            model_a.heads[DOMAINS[0]].weight,
            model_d.heads[DOMAINS[0]].weight,
        ) or not torch.equal(
            model_b.heads[DOMAINS[1]].weight,
            model_d.heads[DOMAINS[1]].weight,
        ):
            raise RuntimeError("Paired CNN head initialization mismatch")
        tabular_parameters = {}
        for family in ("random_forest", "xgboost"):
            _, tabular_parameters[family] = _tree_model(family, 42)
        if payload["budget"]["cnn_steps_per_epoch"] != 467:
            raise RuntimeError("CNN step budget mismatch")
        return {
            "status": "PREFLIGHT_PASSED_NO_TRAINING",
            "protocol_sha256": payload["protocol_sha256"],
            "split_sha256": payload["split_sha256"],
            "split_counts": payload["split_counts"],
            "planned_runs": len(payload["experiment_matrix"]),
            "cnn_steps_per_epoch": payload["budget"]["cnn_steps_per_epoch"],
            "cnn_parameter_counts": counts,
            "tabular_parameters_seed42": tabular_parameters,
            "resolved_matrix_parameters": resolved_matrix_parameters(payload),
            "environment": environment_versions(),
            "implementation_hashes": implementation_hashes(),
            "implementation_sha256": implementation_digest(),
            "output_root": output_root(payload["protocol_sha256"]).as_posix(),
            "output_created": False,
            "model_training_occurred": False,
            "predictions_generated": False,
        }


def require_execution_confirmation(
    execute: bool,
    protocol_confirmation: str | None,
    implementation_confirmation: str | None,
) -> None:
    """Require explicit mode plus exact scientific and implementation locks."""
    if not execute:
        raise PermissionError(
            "Training requires the explicit --execute interlock"
        )
    if protocol_confirmation != APPROVED_PROTOCOL_SHA256:
        raise PermissionError(
            "Training requires the full approved protocol digest"
        )
    if implementation_confirmation != implementation_digest():
        raise PermissionError(
            "Training requires the exact preflight implementation digest"
        )


def _partition_cache(
    data: LockedDevelopmentData,
    partition: str,
    *,
    evaluation_freeze: object | None = None,
) -> dict:
    cache = {}
    for domain in DOMAINS:
        sequences, labels, identifiers = data.partition(
            domain,
            partition,
            evaluation_freeze=evaluation_freeze,
        )
        cache[domain] = {
            "sequences": sequences,
            "labels": labels,
            "ids": identifiers,
        }
    return cache


def _cnn_features(cache: dict) -> dict[str, np.ndarray]:
    return {
        domain: materialize_cnn(cache[domain]["sequences"].tolist())
        for domain in DOMAINS
    }


def _tabular_features(cache: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    features = {}
    names = None
    for domain in DOMAINS:
        features[domain], domain_names = materialize_tabular(
            cache[domain]["sequences"].tolist()
        )
        if names is None:
            names = domain_names
        elif names != domain_names:
            raise RuntimeError("Tabular feature order changed between domains")
    return features, names or []


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialize(payload))


def _progress_logger(path: Path):
    path.touch(exist_ok=False)

    def append(event):
        line = json.dumps(event, sort_keys=True, allow_nan=False)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    return append


def _predict_cnn(model: DomainAwareCNN, features, domain: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(features, dtype=torch.float32)
        return model(inputs, [domain] * len(inputs)).numpy().reshape(-1)


def _aggregate_seed_metrics(payload: dict, results: dict) -> dict:
    aggregates = {}
    for specification in payload["experiment_matrix"]:
        arm = specification["arm"]
        experiment_id = specification["experiment_id"]
        for domain in specification["prediction_heads"]:
            key = f"{arm}:{domain}"
            aggregates.setdefault(key, []).append(
                results[experiment_id]["domains"][domain]["internal_test"]
            )
    summary = {}
    for key, records in aggregates.items():
        if len(records) != len(SEEDS) or any(
            record["status"] != "COMPUTED" for record in records
        ):
            summary[key] = {"status": "UNSTABLE_RESULT"}
            continue
        summary[key] = {"status": "COMPUTED", "metrics": {}}
        for metric in ("spearman", "pearson", "r2", "mae", "rmse"):
            values = [record[metric] for record in records]
            summary[key]["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "values_by_seed": dict(zip(map(str, SEEDS), values)),
            }
    return summary


def _bridge_ranges(records: dict) -> dict:
    summary = {}
    for model_type in ("single_domain", "multidomain"):
        model_records = [records[str(seed)][model_type] for seed in SEEDS]
        if any(record["status"] != "COMPUTED" for record in model_records):
            summary[model_type] = {"status": "UNSTABLE_RESULT"}
            continue
        fields = {
            "head_to_head_spearman": [
                record["head_to_head_spearman"] for record in model_records
            ],
            "head_to_head_kendall": [
                record["head_to_head_kendall"] for record in model_records
            ],
            "mean_absolute_percentile_rank_disagreement": [
                record["mean_absolute_percentile_rank_disagreement"]
                for record in model_records
            ],
        }
        for domain in DOMAINS:
            for metric in (
                "head_to_own_label_spearman",
                "head_to_own_label_pearson",
            ):
                fields[f"{metric}:{domain}"] = [
                    record[metric][domain] for record in model_records
                ]
        summary[model_type] = {
            "status": "COMPUTED",
            "ranges": {
                field: [float(min(values)), float(max(values))]
                for field, values in fields.items()
            },
        }
    return summary


def _bridge_contrasts(records: dict) -> dict:
    """Report registered D-head diagnostics relative to paired A/B CNNs."""
    contrasts = {}
    for seed in SEEDS:
        baseline = records[str(seed)]["single_domain"]
        shared = records[str(seed)]["multidomain"]
        if baseline["status"] != "COMPUTED" or shared["status"] != "COMPUTED":
            contrasts[str(seed)] = {"status": "UNSTABLE_RESULT"}
            continue
        result = {
            "status": "COMPUTED",
            "delta_head_to_head_spearman": shared["head_to_head_spearman"]
            - baseline["head_to_head_spearman"],
            "delta_head_to_head_kendall": shared["head_to_head_kendall"]
            - baseline["head_to_head_kendall"],
            "delta_rank_disagreement": shared[
                "mean_absolute_percentile_rank_disagreement"
            ]
            - baseline["mean_absolute_percentile_rank_disagreement"],
        }
        for domain in DOMAINS:
            result[f"delta_own_label_spearman:{domain}"] = (
                shared["head_to_own_label_spearman"][domain]
                - baseline["head_to_own_label_spearman"][domain]
            )
            result[f"delta_own_label_pearson:{domain}"] = (
                shared["head_to_own_label_pearson"][domain]
                - baseline["head_to_own_label_pearson"][domain]
            )
        contrasts[str(seed)] = result
    return contrasts


def _run_training_matrix(
    payload: dict, data: LockedDevelopmentData, run_dir: Path
):
    expected_fingerprints = payload["label_fingerprints"]
    if data.label_fingerprints() != expected_fingerprints:
        raise RuntimeError("Pre-conversion label fingerprint mismatch")
    train = _partition_cache(data, "train")
    validation = _partition_cache(data, "validation")
    cnn_train = _cnn_features(train)
    cnn_validation = _cnn_features(validation)
    tabular_train, feature_names = _tabular_features(train)
    if data.label_fingerprints() != expected_fingerprints:
        raise RuntimeError("Post-conversion label fingerprint mismatch")
    statistics = {domain: data.statistics(domain) for domain in DOMAINS}
    trained = []
    model_dir = run_dir / "models"
    model_dir.mkdir()
    log_dir = run_dir / "training_logs"
    log_dir.mkdir()
    for specification in payload["experiment_matrix"]:
        family = specification["model_family"]
        domains = tuple(specification["training_domains"])
        seed = specification["seed"]
        experiment_id = specification["experiment_id"]
        progress = _progress_logger(log_dir / f"{experiment_id}.jsonl")
        if family == "cnn":
            model = DomainAwareCNN(
                {domain: statistics[domain] for domain in domains}, seed
            )
            history = train_cnn(
                model,
                {domain: cnn_train[domain] for domain in domains},
                {domain: train[domain]["labels"] for domain in domains},
                {domain: train[domain]["ids"] for domain in domains},
                {domain: cnn_validation[domain] for domain in domains},
                {domain: validation[domain]["labels"] for domain in domains},
                {domain: statistics[domain] for domain in domains},
                seed,
                steps_per_epoch=467,
                max_epochs=100,
                progress_callback=progress,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "domains": domains,
                    "statistics": {
                        domain: statistics[domain].__dict__
                        for domain in domains
                    },
                    "history": history,
                    "specification": specification,
                },
                model_dir / f"{experiment_id}.pt",
            )
        else:
            domain = domains[0]
            model, _ = _tree_model(family, seed)
            tracemalloc.start()
            fit_start = time.perf_counter()
            try:
                history = model.fit(
                    tabular_train[domain],
                    train[domain]["labels"],
                    feature_names=feature_names,
                    **(
                        {"early_stopping_rounds": 0}
                        if family == "xgboost"
                        else {}
                    ),
                )
            except Exception as error:
                tracemalloc.stop()
                exposures = {domain: len(train[domain]["labels"])}
                message = f"{family} fit failed: {error}"
                progress(
                    {
                        "status": "UNSTABLE_RESULT",
                        "reason": message,
                        "domain_exposures": exposures,
                    }
                )
                raise TrainingInstability(message, [], exposures) from error
            history["wall_clock_fit_seconds"] = time.perf_counter() - fit_start
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            history["python_tracemalloc_peak_bytes"] = peak_memory
            history["process_peak_working_set_bytes"] = (
                process_peak_memory_bytes()
            )
            progress({"status": "FIT_COMPLETE", "history": history})
            model.save_model(model_dir / f"{experiment_id}.pkl")
        trained.append((specification, model, history))
    return trained


def _freeze_evaluation(payload, data, trained, run_dir, frozen_hashes):
    expected = [
        specification["experiment_id"]
        for specification in payload["experiment_matrix"]
    ]
    completed = [record[0]["experiment_id"] for record in trained]
    model_dir = run_dir / "models"
    checkpoint_paths = {
        specification["experiment_id"]: model_dir
        / (
            f"{specification['experiment_id']}.pt"
            if specification["model_family"] == "cnn"
            else f"{specification['experiment_id']}.pkl"
        )
        for specification in payload["experiment_matrix"]
    }
    observed_files = set(model_dir.iterdir())
    if observed_files != set(checkpoint_paths.values()) or not all(
        path.is_file() for path in checkpoint_paths.values()
    ):
        raise RuntimeError("Checkpoint file inventory mismatch")
    if implementation_hashes() != frozen_hashes:
        raise RuntimeError("Implementation changed before checkpoint freeze")
    if set(completed) != set(expected) or len(completed) != len(expected):
        raise RuntimeError("Completed checkpoint inventory mismatch")
    return data.freeze_evaluation(checkpoint_paths)


def _evaluate_frozen(payload, data, trained, run_dir, evaluation_freeze):
    validation = _partition_cache(data, "validation")
    internal = _partition_cache(
        data, "internal_test", evaluation_freeze=evaluation_freeze
    )
    bridge = _partition_cache(
        data, "bridge", evaluation_freeze=evaluation_freeze
    )
    cnn_validation = _cnn_features(validation)
    cnn_internal = _cnn_features(internal)
    cnn_bridge = _cnn_features(bridge)
    tabular_validation, _ = _tabular_features(validation)
    tabular_internal, _ = _tabular_features(internal)
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir()
    results = {}
    predictions = {}
    for specification, model, history in trained:
        experiment_id = specification["experiment_id"]
        family = specification["model_family"]
        results[experiment_id] = {"training": history, "domains": {}}
        predictions[experiment_id] = {}
        archive = {}
        for domain in specification["prediction_heads"]:
            if family == "cnn":
                validation_prediction = _predict_cnn(
                    model, cnn_validation[domain], domain
                )
                test_prediction = _predict_cnn(
                    model, cnn_internal[domain], domain
                )
                bridge_prediction = _predict_cnn(
                    model, cnn_bridge[domain], domain
                )
            else:
                validation_prediction = model.predict(
                    tabular_validation[domain]
                )
                test_prediction = model.predict(tabular_internal[domain])
                bridge_prediction = None
            try:
                validation_metrics = {
                    "status": "COMPUTED",
                    **regression_metrics(
                        validation[domain]["labels"], validation_prediction
                    ),
                }
            except ValueError as error:
                validation_metrics = {
                    "status": "UNSTABLE_RESULT",
                    "reason": str(error),
                }
            try:
                internal_metrics = {
                    "status": "COMPUTED",
                    **regression_metrics(
                        internal[domain]["labels"], test_prediction
                    ),
                }
            except ValueError as error:
                internal_metrics = {
                    "status": "UNSTABLE_RESULT",
                    "reason": str(error),
                }
            results[experiment_id]["domains"][domain] = {
                "validation": validation_metrics,
                "internal_test": internal_metrics,
            }
            predictions[experiment_id][domain] = {
                "validation": validation_prediction,
                "internal_test": test_prediction,
            }
            if bridge_prediction is not None:
                predictions[experiment_id][domain][
                    "bridge"
                ] = bridge_prediction
            archive[f"{domain}_internal_ids"] = internal[domain]["ids"]
            archive[f"{domain}_internal_predictions"] = test_prediction
            archive[f"{domain}_validation_ids"] = validation[domain]["ids"]
            archive[f"{domain}_validation_predictions"] = validation_prediction
            if bridge_prediction is not None:
                archive[f"{domain}_bridge_ids"] = bridge[domain]["ids"]
                archive[f"{domain}_bridge_predictions"] = bridge_prediction
        np.savez(prediction_dir / f"{experiment_id}.npz", **archive)
    comparisons = {}
    for domain_index, domain in enumerate(DOMAINS):
        baseline_arm = "A" if domain_index == 0 else "B"
        baseline = {
            seed: predictions[f"18C_{baseline_arm}_s{seed}"][domain][
                "internal_test"
            ]
            for seed in SEEDS
        }
        shared = {
            seed: predictions[f"18C_D_s{seed}"][domain]["internal_test"]
            for seed in SEEDS
        }
        group_ids = np.asarray(
            [sequence_id for sequence_id in internal[domain]["sequences"]]
        )
        group_ids = np.asarray([sequence[4:24] for sequence in group_ids])
        comparisons[domain] = DomainComparison(
            group_ids=group_ids,
            labels=internal[domain]["labels"],
            baseline_predictions=baseline,
            multidomain_predictions=shared,
        )
    inference = hierarchical_paired_bootstrap(comparisons)
    bridge_results = {}
    for seed in SEEDS:
        bridge_results[str(seed)] = {}
        for name, arm_a, arm_b in (
            ("single_domain", "A", "B"),
            ("multidomain", "D", "D"),
        ):
            try:
                bridge_results[str(seed)][name] = {
                    "status": "COMPUTED",
                    **bridge_metrics(
                        bridge[DOMAINS[0]]["labels"],
                        bridge[DOMAINS[1]]["labels"],
                        predictions[f"18C_{arm_a}_s{seed}"][DOMAINS[0]][
                            "bridge"
                        ],
                        predictions[f"18C_{arm_b}_s{seed}"][DOMAINS[1]][
                            "bridge"
                        ],
                    ),
                }
            except ValueError as error:
                bridge_results[str(seed)][name] = {
                    "status": "UNSTABLE_RESULT",
                    "reason": str(error),
                }
    return {
        "runs": results,
        "seed_aggregates": _aggregate_seed_metrics(payload, results),
        "primary_inference": inference,
        "bridge": bridge_results,
        "bridge_ranges": _bridge_ranges(bridge_results),
        "bridge_d_vs_baseline": _bridge_contrasts(bridge_results),
    }


def run_campaign(
    protocol_confirmation: str,
    implementation_confirmation: str,
) -> Path:
    """Execute all 35 locked fits only after the explicit digest interlock."""
    with phase18c_access_guard():
        require_execution_confirmation(
            True, protocol_confirmation, implementation_confirmation
        )
        root = output_root(APPROVED_PROTOCOL_SHA256)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"campaign_{stamp}"
        run_dir.mkdir()
        try:
            payload = load_locked_protocol()
            verify_locked_inputs(payload, APPROVED_PROTOCOL_SHA256)
            preflight = run_preflight()
            frozen_hashes = implementation_hashes()
            if fingerprint(frozen_hashes) != implementation_confirmation:
                raise RuntimeError(
                    "Confirmed implementation changed before execution lock"
                )
            lock = {
                "protocol_sha256": payload["protocol_sha256"],
                "split_sha256": payload["split_sha256"],
                "implementation_hashes": frozen_hashes,
                "implementation_sha256": implementation_confirmation,
                "environment": environment_versions(),
                "preflight": preflight,
                "output_directory": run_dir.as_posix(),
                "matrix": payload["experiment_matrix"],
            }
            _write_json_exclusive(run_dir / "execution_lock.json", lock)
            data = load_locked_development_data(
                payload, APPROVED_PROTOCOL_SHA256
            )
            trained = _run_training_matrix(payload, data, run_dir)
            evaluation_freeze = _freeze_evaluation(
                payload, data, trained, run_dir, frozen_hashes
            )
            results = _evaluate_frozen(
                payload, data, trained, run_dir, evaluation_freeze
            )
            verify_locked_inputs(payload, APPROVED_PROTOCOL_SHA256)
            verify_integrity()
            if implementation_hashes() != frozen_hashes:
                raise RuntimeError(
                    "Phase 18C implementation changed during campaign"
                )
            final = {
                "phase": "18C",
                "protocol_sha256": payload["protocol_sha256"],
                "split_sha256": payload["split_sha256"],
                "checkpoint_freeze_complete": True,
                "results": results,
                "canonical_integrity_after": True,
                "implementation_unchanged": True,
            }
            _write_json_exclusive(run_dir / "campaign_results.json", final)
        except TrainingInstability as error:
            unstable = {
                "status": "UNSTABLE_RESULT",
                "reason": str(error),
                "partial_history": error.history,
                "domain_exposures": error.exposures,
                "campaign_halted": True,
                "review_required": True,
            }
            _write_json_exclusive(run_dir / "unstable_result.json", unstable)
            raise
        except Exception as error:
            violation = {
                "status": "PROTOCOL_VIOLATION",
                "error_type": type(error).__name__,
                "error": str(error),
                "campaign_halted": True,
                "review_required": True,
            }
            _write_json_exclusive(run_dir / "violation.json", violation)
            raise
        return run_dir

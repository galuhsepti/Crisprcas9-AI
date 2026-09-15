"""Complete Phase 18F analysis after a post-fit process timeout.

This recovery path is intentionally incapable of fitting models or generating
predictions. It consumes the 18 frozen model artifacts and 18 prediction
bundles created by the digest-confirmed Phase 18F runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.phase18c_statistics import (  # noqa: E402
    regression_metrics,
)
from src.experiment_protocols.multidomain_protocol import (  # noqa: E402
    DOMAINS,
    ROOT,
    verify_integrity,
)
from src.experiment_protocols.phase18c_access import (  # noqa: E402
    phase18c_access_guard,
)
from src.experiment_protocols.phase18e_feature_ablation_protocol import (  # noqa: E402,E501
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_XGBOOST_CONFIGURATION,
    FEATURE_FAMILIES,
    SPLIT_SHA256,
    fingerprint,
)
from src.feature_ablation.phase18f import (  # noqa: E402
    MODELS_ROOT,
    PHASE18E_PROTOCOL_SHA256,
    REPORT_PATH,
    RESULTS_ROOT,
    _effect_record,
    _report,
    _sha256,
    _strongest_family,
    _write_json_exclusive,
    bonferroni_interval_quantiles,
    classify_domain_consistency,
    implementation_hashes,
    load_phase18e_protocol,
    selected_feature_names,
    validate_matrix,
)
from src.models.xgboost_model import XGBoostModel  # noqa: E402


def _rowwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rank = stats.rankdata(left, axis=1, method="average")
    right_rank = stats.rankdata(right, axis=1, method="average")
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    numerator = np.sum(left_rank * right_rank, axis=1)
    denominator = np.sqrt(
        np.sum(left_rank**2, axis=1) * np.sum(right_rank**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(left), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def vectorized_paired_bootstrap(
    labels: Sequence[float],
    group_ids: Sequence[str],
    full_predictions: Sequence[float],
    comparison_predictions: Mapping[str, Sequence[float]],
    experiment_types: Mapping[str, str],
    iterations: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 64,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Calculate the locked bootstrap with chunked, rowwise rank operations."""
    if iterations <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap iteration and chunk size must be positive")
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    full = np.asarray(full_predictions, dtype=np.float64).reshape(-1)
    comparisons = {
        key: np.asarray(values, dtype=np.float64).reshape(-1)
        for key, values in comparison_predictions.items()
    }
    arrays = [truth, groups, full, *comparisons.values()]
    if any(len(array) != len(truth) for array in arrays) or len(truth) < 2:
        raise ValueError("Bootstrap arrays are incomplete")
    if not np.all(np.isfinite(truth)) or not all(
        np.all(np.isfinite(values)) for values in [full, *comparisons.values()]
    ):
        raise ValueError("Bootstrap arrays must be finite")

    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) != len(groups) or not np.all(counts == 1):
        raise RuntimeError(
            "Recovery bootstrap requires the observed one-row-per-group data"
        )
    sorted_rows = np.argsort(groups, kind="stable")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    distributions = {
        experiment_id: {
            "delta_spearman": np.empty(iterations, dtype=np.float64),
            "delta_rmse": np.empty(iterations, dtype=np.float64),
            "relative_rmse_degradation": np.empty(
                iterations, dtype=np.float64
            ),
        }
        for experiment_id in comparisons
    }

    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        draw = generator.choice(
            len(unique), size=(stop - start, len(unique)), replace=True
        )
        rows = sorted_rows[draw]
        sampled_truth = truth[rows]
        sampled_full = full[rows]
        full_rho = _rowwise_spearman(sampled_truth, sampled_full)
        full_rmse = np.sqrt(
            np.mean((sampled_truth - sampled_full) ** 2, axis=1)
        )
        if not np.all(np.isfinite(full_rho)) or np.any(full_rmse <= 0):
            raise RuntimeError("Undefined bootstrap full-control metric")
        for experiment_id, predictions in comparisons.items():
            sampled = predictions[rows]
            rho = _rowwise_spearman(sampled_truth, sampled)
            rmse = np.sqrt(np.mean((sampled_truth - sampled) ** 2, axis=1))
            if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(rmse)):
                raise RuntimeError("Undefined bootstrap comparison metric")
            distributions[experiment_id]["delta_spearman"][start:stop] = (
                rho - full_rho
            )
            distributions[experiment_id]["delta_rmse"][start:stop] = (
                rmse - full_rmse
            )
            distributions[experiment_id]["relative_rmse_degradation"][
                start:stop
            ] = (rmse - full_rmse) / full_rmse

    nominal_quantiles = (0.025, 0.975)
    adjusted_quantiles = bonferroni_interval_quantiles()
    summary = {}
    for experiment_id, values in distributions.items():
        record = {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": iterations,
            "invalid_bootstrap_n": 0,
            "nominal_95_ci": {
                key: np.quantile(
                    array, nominal_quantiles, method="linear"
                ).tolist()
                for key, array in values.items()
            },
        }
        if experiment_types[experiment_id] == "DROP_ONE_FAMILY":
            record["adjusted_99_7916667_ci"] = {
                key: np.quantile(
                    array, adjusted_quantiles, method="linear"
                ).tolist()
                for key, array in values.items()
                if key in {"delta_spearman", "delta_rmse"}
            }
        summary[experiment_id] = record
    flattened = {
        f"{experiment_id}__{key}": array
        for experiment_id, values in distributions.items()
        for key, array in values.items()
    }
    return summary, flattened


def _scalar_text(bundle: np.lib.npyio.NpzFile, key: str) -> str:
    values = np.asarray(bundle[key]).reshape(-1)
    if len(values) != 1:
        raise RuntimeError(f"Prediction bundle field is not scalar: {key}")
    return str(values[0])


def _load_frozen_runs(
    matrix: list[dict], results_dir: Path, models_dir: Path
) -> tuple[dict, dict, list[dict], list[dict]]:
    prediction_dir = results_dir / "predictions"
    expected_ids = {row["experiment_id"] for row in matrix}
    if {path.stem for path in models_dir.glob("*.pkl")} != expected_ids:
        raise RuntimeError("Frozen model inventory is not exactly 18")
    if {path.stem for path in prediction_dir.glob("*.npz")} != expected_ids:
        raise RuntimeError("Frozen prediction inventory is not exactly 18")

    runs = {}
    domain_arrays = {}
    artifact_models = []
    artifact_predictions = []
    for specification in matrix:
        experiment_id = specification["experiment_id"]
        domain = specification["domain"]
        model_path = models_dir / f"{experiment_id}.pkl"
        prediction_path = prediction_dir / f"{experiment_id}.npz"
        with np.load(prediction_path, allow_pickle=False) as bundle:
            if _scalar_text(bundle, "experiment_id") != experiment_id:
                raise RuntimeError("Prediction experiment identity mismatch")
            if _scalar_text(bundle, "domain") != domain:
                raise RuntimeError("Prediction domain identity mismatch")
            labels = np.asarray(bundle["labels"], dtype=np.float64).reshape(-1)
            predictions = np.asarray(
                bundle["predictions"], dtype=np.float64
            ).reshape(-1)
            sequence_ids = np.asarray(bundle["sequence_ids"]).astype(str)
            groups = np.asarray(bundle["spacer_groups"]).astype(str)
            feature_names = tuple(
                np.asarray(bundle["feature_names"]).astype(str).tolist()
            )
        arrays = [labels, predictions, sequence_ids, groups]
        if any(len(array) != len(labels) for array in arrays):
            raise RuntimeError("Prediction bundle arrays are incomplete")
        if not np.all(np.isfinite(labels)) or not np.all(
            np.isfinite(predictions)
        ):
            raise RuntimeError("Prediction bundle contains nonfinite values")
        expected_features = selected_feature_names(specification)
        if feature_names != expected_features:
            raise RuntimeError("Frozen feature selection mismatch")

        if domain not in domain_arrays:
            domain_arrays[domain] = {
                "labels": labels,
                "ids": sequence_ids,
                "groups": groups,
                "predictions": {},
            }
        else:
            reference = domain_arrays[domain]
            if not (
                np.array_equal(reference["labels"], labels)
                and np.array_equal(reference["ids"], sequence_ids)
                and np.array_equal(reference["groups"], groups)
            ):
                raise RuntimeError("Within-domain prediction pairing mismatch")
        domain_arrays[domain]["predictions"][experiment_id] = predictions

        model = XGBoostModel.load_model(model_path)
        if tuple(model.feature_names) != feature_names:
            raise RuntimeError("Model and prediction feature names differ")
        training = dict(model.training_history)
        if training.get("n_features") != len(feature_names):
            raise RuntimeError("Saved model training feature count mismatch")
        training.update(
            {
                "wall_clock_fit_seconds": None,
                "python_tracemalloc_peak_bytes": None,
                "recovery_measurement_note": (
                    "The tool timeout occurred after prediction persistence; "
                    "process-local wall-clock and tracemalloc values were "
                    "lost. "
                    "The model-retained XGBoost training_time remains exact."
                ),
            }
        )
        model_sha = _sha256(model_path)
        prediction_sha = _sha256(prediction_path)
        relative_model = model_path.relative_to(ROOT).as_posix()
        relative_prediction = prediction_path.relative_to(ROOT).as_posix()
        runs[experiment_id] = {
            "experiment_id": experiment_id,
            "domain": domain,
            "experiment_type": specification["experiment_type"],
            "feature_family_removed": specification["feature_family_removed"],
            "feature_families_retained": specification[
                "feature_families_retained"
            ],
            "feature_count": len(feature_names),
            "feature_name_sha256": fingerprint(list(feature_names)),
            "seed": BOOTSTRAP_SEED,
            "split_sha256": SPLIT_SHA256,
            "xgboost_configuration": EXPECTED_XGBOOST_CONFIGURATION,
            "metrics": regression_metrics(labels, predictions),
            "training": training,
            "model_artifact": relative_model,
            "prediction_artifact": relative_prediction,
            "model_sha256": model_sha,
            "prediction_sha256": prediction_sha,
        }
        artifact_models.append(
            {
                "experiment_id": experiment_id,
                "path": relative_model,
                "sha256": model_sha,
                "bytes": model_path.stat().st_size,
            }
        )
        artifact_predictions.append(
            {
                "experiment_id": experiment_id,
                "path": relative_prediction,
                "sha256": prediction_sha,
                "bytes": prediction_path.stat().st_size,
            }
        )
    return runs, domain_arrays, artifact_models, artifact_predictions


def complete_campaign(
    campaign: str,
    protocol_confirmation: str,
    implementation_confirmation: str,
    recovery_confirmation: str,
) -> Path:
    """Finalize one interrupted campaign without fitting or prediction."""
    recovery_path = Path(__file__).resolve()
    recovery_sha = _sha256(recovery_path)
    if protocol_confirmation != PHASE18E_PROTOCOL_SHA256:
        raise PermissionError("Recovery protocol confirmation mismatch")
    if recovery_confirmation != recovery_sha:
        raise PermissionError("Recovery script confirmation mismatch")
    if not campaign.startswith("campaign_") or Path(campaign).name != campaign:
        raise ValueError("Campaign must be one isolated campaign basename")

    results_dir = RESULTS_ROOT / campaign
    models_dir = MODELS_ROOT / campaign
    if not results_dir.is_dir() or not models_dir.is_dir():
        raise FileNotFoundError("Interrupted campaign directories are missing")
    for forbidden in (
        results_dir / "summary.json",
        results_dir / "artifact_manifest.json",
        REPORT_PATH,
    ):
        if forbidden.exists():
            raise FileExistsError(forbidden)

    with phase18c_access_guard():
        lock_path = results_dir / "execution_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("implementation_sha256") != implementation_confirmation:
            raise PermissionError("Execution-lock implementation mismatch")
        if implementation_hashes() != lock.get("implementation_hashes"):
            raise RuntimeError("Locked training implementation changed")
        protocol = load_phase18e_protocol()
        matrix = validate_matrix(protocol)
        if matrix != lock.get("matrix"):
            raise RuntimeError("Execution-lock matrix mismatch")

        integrity_before = verify_integrity()
        runs, domain_arrays, artifact_models, artifact_predictions = (
            _load_frozen_runs(matrix, results_dir, models_dir)
        )
        bootstrap_summaries = {}
        bootstrap_arrays = {}
        for domain in DOMAINS:
            domain_specs = [row for row in matrix if row["domain"] == domain]
            full_spec = next(
                row
                for row in domain_specs
                if row["experiment_type"] == "FULL_CONTROL"
            )
            comparisons = {
                row["experiment_id"]: domain_arrays[domain]["predictions"][
                    row["experiment_id"]
                ]
                for row in domain_specs
                if row["experiment_type"] != "FULL_CONTROL"
            }
            types = {
                row["experiment_id"]: row["experiment_type"]
                for row in domain_specs
                if row["experiment_type"] != "FULL_CONTROL"
            }
            bootstrap_summaries[domain], arrays = vectorized_paired_bootstrap(
                domain_arrays[domain]["labels"],
                domain_arrays[domain]["groups"],
                domain_arrays[domain]["predictions"][
                    full_spec["experiment_id"]
                ],
                comparisons,
                types,
            )
            bootstrap_arrays.update(
                {f"{domain}__{key}": value for key, value in arrays.items()}
            )

        bootstrap_path = (
            results_dir / "bootstrap" / ("paired_effect_distributions.npz")
        )
        if bootstrap_path.exists():
            raise FileExistsError(bootstrap_path)
        np.savez_compressed(bootstrap_path, **bootstrap_arrays)

        drop_effects = {domain: {} for domain in DOMAINS}
        family_only = {domain: {} for domain in DOMAINS}
        full_controls = {}
        for domain in DOMAINS:
            domain_specs = [row for row in matrix if row["domain"] == domain]
            full_spec = next(
                row
                for row in domain_specs
                if row["experiment_type"] == "FULL_CONTROL"
            )
            full_metrics = runs[full_spec["experiment_id"]]["metrics"]
            full_controls[domain] = runs[full_spec["experiment_id"]]
            for row in domain_specs:
                if row["experiment_type"] == "FULL_CONTROL":
                    continue
                experiment_id = row["experiment_id"]
                metrics = runs[experiment_id]["metrics"]
                effect = _effect_record(
                    full_metrics,
                    metrics,
                    bootstrap_summaries[domain][experiment_id],
                )
                if row["experiment_type"] == "DROP_ONE_FAMILY":
                    drop_effects[domain][
                        row["feature_family_removed"]
                    ] = effect
                else:
                    family = row["feature_families_retained"][0]
                    family_only[domain][family] = {
                        "metrics": metrics,
                        "comparison_to_full": effect,
                        "spearman_fraction_of_full": (
                            metrics["spearman"] / full_metrics["spearman"]
                        ),
                        "rmse_ratio_to_full": (
                            metrics["rmse"] / full_metrics["rmse"]
                        ),
                    }
        consistency = {
            family: classify_domain_consistency(
                drop_effects[DOMAINS[0]][family][
                    "contribution_classification"
                ],
                drop_effects[DOMAINS[1]][family][
                    "contribution_classification"
                ],
            )
            for family in FEATURE_FAMILIES
        }

        integrity_after = verify_integrity()
        if integrity_before != integrity_after:
            raise RuntimeError("Canonical artifacts changed during recovery")
        if implementation_hashes() != lock["implementation_hashes"]:
            raise RuntimeError("Locked training implementation changed")

        recovery = {
            "reason": "Tool timeout during post-fit bootstrap analysis",
            "additional_fits": 0,
            "additional_predictions": 0,
            "script": recovery_path.relative_to(ROOT).as_posix(),
            "script_sha256": recovery_sha,
            "bootstrap_equivalence": (
                "Same sorted group order, PCG64(42) stream, paired draws, "
                "rank-based Spearman, RMSE, and percentile quantiles; chunked "
                "vectorization changes execution strategy only."
            ),
        }
        artifact_manifest = {
            "phase": "18F",
            "models": artifact_models,
            "predictions": artifact_predictions,
            "bootstrap": {
                "path": bootstrap_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(bootstrap_path),
                "bytes": bootstrap_path.stat().st_size,
            },
            "model_count": len(artifact_models),
            "prediction_count": len(artifact_predictions),
            "analysis_recovery": recovery,
        }
        summary = {
            "phase": "18F",
            "completion_status": "COMPLETED",
            "phase18e_protocol_sha256": PHASE18E_PROTOCOL_SHA256,
            "feature_space_sha256": lock["feature_space_sha256"],
            "split_sha256": SPLIT_SHA256,
            "planned_fits": 18,
            "completed_fits": 18,
            "failed_fits": 0,
            "skipped_fits": 0,
            "runs": runs,
            "full_controls": full_controls,
            "drop_one_effects": drop_effects,
            "family_only": family_only,
            "domain_consistency": consistency,
            "strongest_unique_contributor": {
                domain: _strongest_family(drop_effects[domain])
                for domain in DOMAINS
            },
            "bootstrap_policy": protocol["bootstrap_policy"],
            "multiple_comparison_policy": protocol[
                "multiple_comparison_policy"
            ],
            "artifact_manifest": artifact_manifest,
            "canonical_integrity": {
                "before": integrity_before,
                "after": integrity_after,
                "unchanged": True,
            },
            "training_implementation_sha256": implementation_confirmation,
            "implementation_unchanged": True,
            "analysis_recovery": recovery,
            "bridge_excluded": True,
            "quarantine_excluded": True,
            "locked_external_accessed": False,
            "biological_causality_claimed": False,
        }
        _write_json_exclusive(
            results_dir / "artifact_manifest.json", artifact_manifest
        )
        _write_json_exclusive(results_dir / "summary.json", summary)
        recovery_report = (
            _report(summary)
            + "\n"
            + "\n".join(
                [
                    "## Execution Recovery",
                    "",
                    "The 18 fits and deterministic prediction writes "
                    "completed "
                    "under the confirmed training implementation. The "
                    "orchestration process then exceeded its tool time limit "
                    "during bootstrap analysis. No model was refit and no "
                    "prediction was regenerated. The bootstrap was completed "
                    "from the frozen prediction bundles using the same paired "
                    "PCG64(42) draws and statistical definitions. "
                    "Process-local "
                    "wall-clock and tracemalloc measurements were unavailable "
                    "after termination; each saved model retains its exact "
                    "XGBoost-reported training time.",
                    "",
                ]
            )
        )
        with REPORT_PATH.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(recovery_report)
        return results_dir


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--campaign", default="campaign_20260914_231905")
    parser.add_argument("--confirm-protocol")
    parser.add_argument("--confirm-implementation")
    parser.add_argument("--confirm-recovery")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    recovery_sha = _sha256(Path(__file__).resolve())
    preflight = {
        "status": "RECOVERY_PREFLIGHT_NO_FIT_NO_PREDICTION",
        "campaign": arguments.campaign,
        "protocol_sha256": PHASE18E_PROTOCOL_SHA256,
        "recovery_script_sha256": recovery_sha,
        "model_fit_capability": False,
        "prediction_generation_capability": False,
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if not arguments.execute:
        return
    result = complete_campaign(
        arguments.campaign,
        arguments.confirm_protocol,
        arguments.confirm_implementation,
        arguments.confirm_recovery,
    )
    print(f"PHASE18F_RECOVERY_COMPLETE {result}")


if __name__ == "__main__":
    main()

"""Execute the locked Phase 18F XGBoost feature-ablation experiment."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy
import sklearn
import xgboost
from scipy import stats

from src.dataset_integration.label_compatibility import load_phase18a_inputs
from src.evaluation.phase18c_statistics import regression_metrics
from src.experiment_protocols.multidomain_protocol import (
    DOMAINS,
    ROOT,
    validate_locked_artifact,
    verify_integrity,
)
from src.experiment_protocols.phase18c_access import phase18c_access_guard
from src.experiment_protocols.phase18e_feature_ablation_protocol import (
    BONFERRONI_ALPHA,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CANONICAL_FEATURE_COUNT,
    CANONICAL_FEATURE_SHA256,
    EXPECTED_XGBOOST_CONFIGURATION,
    FEATURE_FAMILIES,
    PHASE18B_PROTOCOL_PATH,
    SIMULTANEOUS_QUANTILES,
    SPLIT_SHA256,
    canonical_feature_names,
    feature_family_registry,
    fingerprint,
    serialize,
    validate_locked_protocol,
    verify_feature_source_integrity,
)
from src.models.xgboost_model import XGBoostModel
from src.multidomain.campaign import _tree_model
from src.multidomain.data import (
    LABEL_COLUMNS,
    SEQUENCE_COLUMNS,
    DomainRows,
    _domain_rows,
    label_fingerprint,
    materialize_tabular,
)

PHASE18E_PROTOCOL_SHA256 = (
    "79d56d051ffbc91b0a096483aa1be64a95f577cd57b94148d8734bbacf92f8df"
)
PHASE18E_PROTOCOL_PATH = (
    ROOT / "results/phase18e_feature_ablation_protocol_20260914_220640.json"
)
RESULTS_ROOT = ROOT / "results" / "phase18f" / PHASE18E_PROTOCOL_SHA256
MODELS_ROOT = ROOT / "models" / "phase18f" / PHASE18E_PROTOCOL_SHA256
REPORT_PATH = ROOT / "docs" / "phase18f_xgboost_feature_ablation_report.md"
IMPLEMENTATION_PATHS = (
    "src/feature_ablation/__init__.py",
    "src/feature_ablation/phase18f.py",
    "scripts/run_phase18f_feature_ablation.py",
    "src/experiment_protocols/phase18e_feature_ablation_protocol.py",
    "src/experiment_protocols/phase18c_access.py",
    "src/multidomain/data.py",
    "src/multidomain/campaign.py",
    "src/models/xgboost_model.py",
    "src/evaluation/phase18c_statistics.py",
    "requirements.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialize(payload))


def implementation_hashes() -> dict[str, str]:
    """Hash an explicit source allowlist without repository traversal."""
    return {
        relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_PATHS
    }


def implementation_sha256() -> str:
    return fingerprint(implementation_hashes())


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "platform": platform.platform(),
    }


def load_phase18e_protocol() -> dict:
    """Load and validate the exact authoritative Phase 18E artifact."""
    payload = json.loads(PHASE18E_PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_locked_protocol(payload)
    if payload.get("protocol_sha256") != PHASE18E_PROTOCOL_SHA256:
        raise RuntimeError("Phase 18E protocol SHA-256 mismatch")
    if payload["feature_space"]["canonical_feature_sha256"] != (
        CANONICAL_FEATURE_SHA256
    ):
        raise RuntimeError("Phase 18E feature-space SHA-256 mismatch")
    if payload["split_policy"]["split_sha256"] != SPLIT_SHA256:
        raise RuntimeError("Phase 18E split SHA-256 mismatch")
    if len(payload["experiment_matrix"]) != 18:
        raise RuntimeError("Phase 18E matrix must contain exactly 18 rows")
    return payload


def load_phase18b_protocol() -> dict:
    payload = json.loads(PHASE18B_PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_locked_artifact(payload)
    if payload.get("split_sha256") != SPLIT_SHA256:
        raise RuntimeError("Phase 18C split SHA-256 mismatch")
    return payload


def validate_matrix(payload: Mapping[str, object]) -> list[dict]:
    """Validate, then return the matrix loaded from the locked artifact."""
    matrix = payload.get("experiment_matrix")
    if not isinstance(matrix, list) or len(matrix) != 18:
        raise RuntimeError("Locked Phase 18F matrix size changed")
    required = {
        "experiment_id",
        "domain",
        "experiment_type",
        "feature_families_retained",
        "feature_family_removed",
        "number_of_features_remaining",
        "canonical_xgboost_configuration",
        "split_hash",
        "metric_policy",
        "bootstrap_policy",
        "fit_seed",
    }
    expected_types = {
        "FULL_CONTROL": 2,
        "DROP_ONE_FAMILY": 12,
        "FAMILY_ONLY": 4,
    }
    observed_types = {key: 0 for key in expected_types}
    identifiers = []
    for row in matrix:
        if not required <= set(row):
            raise RuntimeError("Locked Phase 18F matrix field missing")
        if row["domain"] not in DOMAINS or row["fit_seed"] != 42:
            raise RuntimeError("Unregistered Phase 18F domain or seed")
        if row["split_hash"] != SPLIT_SHA256:
            raise RuntimeError("Phase 18F matrix split drift")
        if row["canonical_xgboost_configuration"] != (
            EXPECTED_XGBOOST_CONFIGURATION
        ):
            raise RuntimeError("Phase 18F matrix XGBoost drift")
        if row["experiment_type"] not in observed_types:
            raise RuntimeError("Unregistered Phase 18F experiment type")
        observed_types[row["experiment_type"]] += 1
        identifiers.append(row["experiment_id"])
    if observed_types != expected_types or len(set(identifiers)) != 18:
        raise RuntimeError("Phase 18F matrix inventory drift")
    return matrix


def selected_feature_names(
    specification: Mapping[str, object],
    names: Sequence[str] | None = None,
    registry: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Apply only the feature-family selection in a locked matrix row."""
    names = tuple(canonical_feature_names() if names is None else names)
    registry = feature_family_registry() if registry is None else registry
    retained = specification["feature_families_retained"]
    if not isinstance(retained, list) or not retained:
        raise RuntimeError("Feature-family retention list is invalid")
    if len(retained) != len(set(retained)) or not set(retained) <= set(
        registry
    ):
        raise RuntimeError("Unregistered feature-family retention")
    retained_names = {name for family in retained for name in registry[family]}
    selected = tuple(name for name in names if name in retained_names)
    if len(selected) != specification["number_of_features_remaining"]:
        raise RuntimeError("Unexpected remaining feature count")
    experiment_type = specification["experiment_type"]
    removed = specification["feature_family_removed"]
    if experiment_type == "DROP_ONE_FAMILY":
        if removed not in FEATURE_FAMILIES or removed in retained:
            raise RuntimeError("Drop-one feature manipulation drift")
        if set(retained) != set(FEATURE_FAMILIES) - {removed}:
            raise RuntimeError("Drop-one retained-family inventory drift")
    elif experiment_type == "FULL_CONTROL":
        if removed is not None or set(retained) != set(FEATURE_FAMILIES):
            raise RuntimeError("Full-control feature manipulation drift")
    elif experiment_type == "FAMILY_ONLY":
        if removed is not None or len(retained) != 1:
            raise RuntimeError("Family-only feature manipulation drift")
    else:
        raise RuntimeError("Unregistered experiment type")
    return selected


class _EvaluationFreeze:
    __slots__ = ("owner", "model_hashes")

    def __init__(self, owner: object, model_hashes: Mapping[str, str]):
        self.owner = owner
        self.model_hashes = dict(model_hashes)


@dataclass(frozen=True)
class Phase18FData:
    """Development rows with internal-test access closed until model freeze."""

    rows: Mapping[str, DomainRows]
    expected_experiment_ids: frozenset[str]
    _owner: object

    def partition(
        self,
        domain: str,
        partition: str,
        freeze: _EvaluationFreeze | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if domain not in DOMAINS:
            raise ValueError("Unknown Phase 18F domain")
        if partition in {"bridge", "quarantine", "validation"}:
            raise PermissionError(
                f"Phase 18F does not materialize the {partition} partition"
            )
        if partition == "internal_test":
            valid = (
                isinstance(freeze, _EvaluationFreeze)
                and freeze.owner is self._owner
                and set(freeze.model_hashes) == self.expected_experiment_ids
            )
            if not valid:
                raise PermissionError(
                    "Internal test remains closed until all 18 models "
                    "are frozen"
                )
        elif partition != "train":
            raise ValueError("Unknown Phase 18F partition")
        rows = self.rows[domain]
        indices = rows.partition_indices[partition]
        return (
            rows.sequences[indices].copy(),
            rows.labels[indices].copy(),
            rows.sequence_ids[indices].copy(),
        )

    def freeze(self, model_paths: Mapping[str, Path]) -> _EvaluationFreeze:
        if set(model_paths) != self.expected_experiment_ids:
            raise RuntimeError("Model freeze inventory is incomplete")
        resolved = [path.resolve() for path in model_paths.values()]
        if len(set(resolved)) != 18 or not all(
            path.is_file() for path in resolved
        ):
            raise RuntimeError("Model freeze path inventory is invalid")
        if not all(path.is_relative_to(MODELS_ROOT) for path in resolved):
            raise PermissionError("Model artifact escaped Phase 18F isolation")
        hashes = {
            experiment_id: _sha256(path)
            for experiment_id, path in model_paths.items()
        }
        return _EvaluationFreeze(self._owner, hashes)


def load_phase18f_data(
    phase18b: Mapping[str, object], matrix: Sequence[Mapping[str, object]]
) -> Phase18FData:
    """Reuse exact Phase 18C identities and split assignments."""
    _, deep, xiang = load_phase18a_inputs()
    frames = {DOMAINS[0]: deep, DOMAINS[1]: xiang}
    rows = {
        domain: _domain_rows(
            frames[domain], domain, phase18b["split_manifest"]
        )
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        frame = frames[domain]
        observed = label_fingerprint(
            frame[SEQUENCE_COLUMNS[domain]].tolist(),
            frame[LABEL_COLUMNS[domain]].to_numpy(dtype=np.float64),
        )
        if observed != phase18b["label_fingerprints"][domain]:
            raise RuntimeError("Locked raw-label fingerprint mismatch")
        indices = rows[domain].partition_indices
        development = set(indices["train"]) | set(indices["internal_test"])
        held = set(indices["bridge"]) | set(indices["quarantine"])
        if development & held:
            raise RuntimeError("Bridge/quarantine leakage detected")
    identifiers = frozenset(row["experiment_id"] for row in matrix)
    return Phase18FData(rows, identifiers, object())


def _resolved_model() -> tuple[XGBoostModel, dict]:
    model, resolved = _tree_model("xgboost", 42)
    for key, expected in EXPECTED_XGBOOST_CONFIGURATION.items():
        if key in {
            "early_stopping",
            "hyperparameter_tuning",
            "eval_set",
            "sample_weight",
        }:
            continue
        if resolved.get(key) != expected:
            raise RuntimeError(f"Resolved XGBoost parameter drift: {key}")
    if model.model.get_params().get("early_stopping_rounds") is not None:
        raise RuntimeError("Early stopping must remain disabled")
    return model, resolved


def validate_prediction_repeatability(
    first: Sequence[float], second: Sequence[float]
) -> None:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise RuntimeError("Nonfinite Phase 18F prediction")
    if left.shape != right.shape or not np.array_equal(left, right):
        raise RuntimeError("Deterministic prediction repeatability failed")


def bonferroni_interval_quantiles() -> tuple[float, float]:
    expected_alpha = 0.05 / 24
    if not math.isclose(BONFERRONI_ALPHA, expected_alpha):
        raise RuntimeError("Bonferroni alpha drift")
    expected = (expected_alpha / 2, 1 - expected_alpha / 2)
    if not all(
        math.isclose(observed, target)
        for observed, target in zip(SIMULTANEOUS_QUANTILES, expected)
    ):
        raise RuntimeError("Bonferroni interval quantile drift")
    return expected


def classify_contribution(record: Mapping[str, object]) -> str:
    """Apply the exact Phase 18E CI-direction contribution rules."""
    required = (
        "delta_spearman",
        "delta_rmse",
        "adjusted_delta_spearman_ci",
        "adjusted_delta_rmse_ci",
    )
    try:
        delta_spearman = float(record[required[0]])
        delta_rmse = float(record[required[1]])
        spearman_ci = tuple(float(value) for value in record[required[2]])
        rmse_ci = tuple(float(value) for value in record[required[3]])
    except (KeyError, TypeError, ValueError):
        return "UNSTABLE_OR_INCONCLUSIVE"
    values = (delta_spearman, delta_rmse, *spearman_ci, *rmse_ci)
    if (
        record.get("invalid_bootstrap_n", 0)
        or len(spearman_ci) != 2
        or len(rmse_ci) != 2
        or not all(math.isfinite(value) for value in values)
        or spearman_ci[0] > spearman_ci[1]
        or rmse_ci[0] > rmse_ci[1]
    ):
        return "UNSTABLE_OR_INCONCLUSIVE"
    degradation = (spearman_ci[1] < 0, rmse_ci[0] > 0)
    improvement = (spearman_ci[0] > 0, rmse_ci[1] < 0)
    if any(degradation) and any(improvement):
        return "UNSTABLE_OR_INCONCLUSIVE"
    if all(degradation):
        return "ESSENTIAL_OR_STRONG_CONTRIBUTOR"
    concordant = (degradation[0] and delta_rmse > 0) or (
        degradation[1] and delta_spearman < 0
    )
    if sum(degradation) == 1 and concordant and not any(improvement):
        return "MODERATE_CONTRIBUTOR"
    if any(improvement) and not any(degradation):
        return "POTENTIALLY_REDUNDANT"
    if not any(improvement) and not any(degradation):
        return "LITTLE_UNIQUE_CONTRIBUTION"
    return "UNSTABLE_OR_INCONCLUSIVE"


def classify_domain_consistency(domain_a: str, domain_b: str) -> str:
    contributors = {
        "ESSENTIAL_OR_STRONG_CONTRIBUTOR",
        "MODERATE_CONTRIBUTOR",
    }
    weak = {"LITTLE_UNIQUE_CONTRIBUTION", "POTENTIALLY_REDUNDANT"}
    if domain_a in contributors and domain_b in contributors:
        return "CROSS_DOMAIN_CONSISTENT"
    if domain_a in contributors and domain_b in weak:
        return "DOMAIN_A_ENRICHED"
    if domain_b in contributors and domain_a in weak:
        return "DOMAIN_B_ENRICHED"
    if domain_a in weak and domain_b in weak:
        return "WEAK_IN_BOTH"
    return "UNSTABLE_OR_INCONCLUSIVE"


def phase18f_output_paths(stamp: str) -> tuple[Path, Path]:
    if not stamp or any(character not in "0123456789_" for character in stamp):
        raise ValueError("Invalid Phase 18F campaign stamp")
    results = (RESULTS_ROOT / f"campaign_{stamp}").resolve()
    models = (MODELS_ROOT / f"campaign_{stamp}").resolve()
    if not results.is_relative_to(RESULTS_ROOT.resolve()):
        raise PermissionError("Results path escaped Phase 18F isolation")
    if not models.is_relative_to(MODELS_ROOT.resolve()):
        raise PermissionError("Model path escaped Phase 18F isolation")
    return results, models


def run_preflight() -> dict:
    """Verify all locks and data exclusions without fitting or predicting."""
    with phase18c_access_guard():
        phase18e = load_phase18e_protocol()
        matrix = validate_matrix(phase18e)
        phase18b = load_phase18b_protocol()
        names = canonical_feature_names()
        registry = feature_family_registry()
        selections = {
            row["experiment_id"]: {
                "feature_count": len(
                    selected_feature_names(row, names, registry)
                ),
                "feature_name_sha256": fingerprint(
                    list(selected_feature_names(row, names, registry))
                ),
            }
            for row in matrix
        }
        data = load_phase18f_data(phase18b, matrix)
        for domain in DOMAINS:
            data.partition(domain, "train")
            for held in ("internal_test", "bridge", "quarantine"):
                try:
                    data.partition(domain, held)
                except PermissionError:
                    pass
                else:
                    raise RuntimeError(
                        f"Preflight opened held partition: {held}"
                    )
        _, resolved = _resolved_model()
        integrity = verify_integrity()
        if not all(
            record["matches_expected"] for record in integrity.values()
        ):
            raise RuntimeError("Canonical integrity mismatch")
        source_hashes = verify_feature_source_integrity()
        quantiles = bonferroni_interval_quantiles()
        return {
            "status": "PREFLIGHT_PASSED_NO_TRAINING",
            "phase18e_protocol_sha256": phase18e["protocol_sha256"],
            "feature_space_sha256": CANONICAL_FEATURE_SHA256,
            "feature_count": len(names),
            "family_counts": {
                family: len(registry[family]) for family in FEATURE_FAMILIES
            },
            "split_sha256": phase18b["split_sha256"],
            "matrix_rows": len(matrix),
            "fit_seed": 42,
            "selections": selections,
            "resolved_xgboost": resolved,
            "bonferroni_quantiles": list(quantiles),
            "source_hashes": source_hashes,
            "canonical_integrity": integrity,
            "implementation_hashes": implementation_hashes(),
            "implementation_sha256": implementation_sha256(),
            "output_created": False,
            "model_training_occurred": False,
            "predictions_generated": False,
            "bridge_materialized": False,
            "quarantine_materialized": False,
        }


def require_execution_confirmation(
    execute: bool,
    protocol_confirmation: str | None,
    implementation_confirmation: str | None,
) -> None:
    if not execute:
        raise PermissionError("Phase 18F requires the --execute interlock")
    if protocol_confirmation != PHASE18E_PROTOCOL_SHA256:
        raise PermissionError("Phase 18F protocol confirmation mismatch")
    if implementation_confirmation != implementation_sha256():
        raise PermissionError("Phase 18F implementation confirmation mismatch")


def _feature_matrix(
    sequences: Sequence[str], expected_names: Sequence[str]
) -> np.ndarray:
    values, names = materialize_tabular(sequences)
    if tuple(names) != tuple(expected_names):
        raise RuntimeError(
            "Canonical feature ordering changed during execution"
        )
    if values.shape[1] != CANONICAL_FEATURE_COUNT:
        raise RuntimeError("Canonical feature count changed during execution")
    return values


def _selected_columns(
    full: np.ndarray,
    canonical_names_value: Sequence[str],
    selected: Sequence[str],
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(canonical_names_value)}
    columns = [lookup[name] for name in selected]
    result = full[:, columns]
    if result.shape[1] != len(selected):
        raise RuntimeError("Selected feature matrix count mismatch")
    return result


def _group_draw_indices(group_ids: Sequence[str], generator) -> np.ndarray:
    identifiers = np.asarray(group_ids).astype(str)
    unique = np.asarray(sorted(set(identifiers.tolist())))
    draw = generator.choice(unique, size=len(unique), replace=True)
    positions = []
    for identifier in draw:
        positions.extend(np.flatnonzero(identifiers == identifier).tolist())
    return np.asarray(positions, dtype=np.int64)


def _spearman_rmse(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    rho = float(stats.spearmanr(labels, predictions).statistic)
    rmse = float(np.sqrt(np.mean((labels - predictions) ** 2)))
    if not math.isfinite(rho) or not math.isfinite(rmse):
        raise RuntimeError("Undefined bootstrap metric")
    return rho, rmse


def paired_bootstrap(
    labels: Sequence[float],
    group_ids: Sequence[str],
    full_predictions: Sequence[float],
    comparison_predictions: Mapping[str, Sequence[float]],
    experiment_types: Mapping[str, str],
    iterations: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Run one paired PCG64 stream shared by all arms in one domain."""
    if iterations <= 0:
        raise ValueError("Bootstrap iteration count must be positive")
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
    for replicate in range(iterations):
        rows = _group_draw_indices(groups, generator)
        full_rho, full_rmse = _spearman_rmse(truth[rows], full[rows])
        if full_rmse <= 0:
            raise RuntimeError("Bootstrap full RMSE is nonpositive")
        for experiment_id, predictions in comparisons.items():
            rho, rmse = _spearman_rmse(truth[rows], predictions[rows])
            distributions[experiment_id]["delta_spearman"][replicate] = (
                rho - full_rho
            )
            distributions[experiment_id]["delta_rmse"][replicate] = (
                rmse - full_rmse
            )
            distributions[experiment_id]["relative_rmse_degradation"][
                replicate
            ] = (rmse - full_rmse) / full_rmse
    summary = {}
    nominal_quantiles = (0.025, 0.975)
    adjusted_quantiles = bonferroni_interval_quantiles()
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


def _effect_record(
    full_metrics: Mapping[str, float],
    arm_metrics: Mapping[str, float],
    bootstrap: Mapping[str, object],
) -> dict:
    delta_spearman = arm_metrics["spearman"] - full_metrics["spearman"]
    delta_rmse = arm_metrics["rmse"] - full_metrics["rmse"]
    relative = delta_rmse / full_metrics["rmse"]
    result = {
        "delta_spearman": delta_spearman,
        "delta_rmse": delta_rmse,
        "relative_rmse_degradation": relative,
        **bootstrap,
    }
    adjusted = bootstrap.get("adjusted_99_7916667_ci")
    if adjusted is not None:
        classification_input = {
            "delta_spearman": delta_spearman,
            "delta_rmse": delta_rmse,
            "adjusted_delta_spearman_ci": adjusted["delta_spearman"],
            "adjusted_delta_rmse_ci": adjusted["delta_rmse"],
            "invalid_bootstrap_n": bootstrap["invalid_bootstrap_n"],
        }
        result["contribution_classification"] = classify_contribution(
            classification_input
        )
    return result


def _strongest_family(effects: Mapping[str, Mapping[str, float]]) -> str:
    """Descriptively rank by primary effect, then RMSE as a tie-breaker."""
    return min(
        effects,
        key=lambda family: (
            effects[family]["delta_spearman"],
            -effects[family]["delta_rmse"],
            family,
        ),
    )


def _report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Phase 18F - Locked XGBoost Feature-Ablation Experiment",
        "",
        "## Completion",
        "",
        f"Status: **{summary['completion_status']}**",
        "",
        "This experiment measures predictive information contribution, not "
        "biological causality, molecular necessity, or Cas9 mechanism.",
        "",
        "## Full Controls",
        "",
        "| Domain | Spearman | RMSE | Pearson | R2 | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        metrics = summary["full_controls"][domain]["metrics"]
        lines.append(
            f"| {domain} | {metrics['spearman']:.8f} | {metrics['rmse']:.8f} "
            f"| {metrics['pearson']:.8f} | {metrics['r2']:.8f} "
            f"| {metrics['mae']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Drop-One Results",
            "",
            "| Domain | Family removed | Delta Spearman | Delta RMSE "
            "| Relative "
            "RMSE degradation | Classification |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for domain in DOMAINS:
        for family in FEATURE_FAMILIES:
            effect = summary["drop_one_effects"][domain][family]
            lines.append(
                f"| {domain} | {family} | {effect['delta_spearman']:.8f} "
                f"| {effect['delta_rmse']:.8f} "
                f"| {effect['relative_rmse_degradation']:.8f} "
                f"| {effect['contribution_classification']} |"
            )
    lines.extend(
        [
            "",
            "## Domain Consistency",
            "",
            "| Family | Classification |",
            "|---|---|",
        ]
    )
    for family in FEATURE_FAMILIES:
        lines.append(f"| {family} | {summary['domain_consistency'][family]} |")
    lines.extend(
        [
            "",
            "## Family-Only Models",
            "",
            "| Domain | Family | Spearman | RMSE | Spearman/full "
            "| RMSE/full |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for domain, records in summary["family_only"].items():
        for family, record in records.items():
            metrics = record["metrics"]
            lines.append(
                f"| {domain} | {family} | {metrics['spearman']:.8f} "
                f"| {metrics['rmse']:.8f} "
                f"| {record['spearman_fraction_of_full']:.8f} "
                f"| {record['rmse_ratio_to_full']:.8f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Small drop-one effects indicate little unique contribution "
            "conditional on retained representations; they do not show that a "
            "feature family is useless. GC, composition, dinucleotide, k-mer, "
            "complexity, and positional representations overlap.",
            "",
            "Domains retain native label scales and are not pooled. Bridge "
            "and quarantine observations were excluded. The locked external "
            "dataset "
            "was not accessed.",
            "",
            "## Limitations",
            "",
            "- Results use one fixed Phase 18C split and two development "
            "domains.",
            "- The deterministic fit policy does not estimate model-fit "
            "variance.",
            "- Family ablations cannot isolate information shared redundantly "
            "by "
            "multiple retained families.",
            "- Internal-test evidence does not establish external "
            "generalization.",
            "- Statistical support is not biological causality.",
            "",
        ]
    )
    return "\n".join(lines)


def run_campaign(
    protocol_confirmation: str,
    implementation_confirmation: str,
) -> tuple[Path, Path]:
    """Execute exactly 18 locked fits after explicit digest confirmation."""
    with phase18c_access_guard():
        require_execution_confirmation(
            True, protocol_confirmation, implementation_confirmation
        )
        phase18e = load_phase18e_protocol()
        matrix = validate_matrix(phase18e)
        phase18b = load_phase18b_protocol()
        preflight = run_preflight()
        frozen_implementation = implementation_hashes()
        if fingerprint(frozen_implementation) != implementation_confirmation:
            raise RuntimeError("Implementation changed after confirmation")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_dir, models_dir = phase18f_output_paths(stamp)
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir()
        models_dir.mkdir()
        predictions_dir = results_dir / "predictions"
        bootstrap_dir = results_dir / "bootstrap"
        predictions_dir.mkdir()
        bootstrap_dir.mkdir()
        completed_fits = 0
        model_paths = {}
        try:
            integrity_before = verify_integrity()
            execution_lock = {
                "phase": "18F",
                "phase18e_protocol_sha256": PHASE18E_PROTOCOL_SHA256,
                "feature_space_sha256": CANONICAL_FEATURE_SHA256,
                "split_sha256": SPLIT_SHA256,
                "matrix": matrix,
                "implementation_hashes": frozen_implementation,
                "implementation_sha256": implementation_confirmation,
                "environment": environment_versions(),
                "preflight": preflight,
                "results_directory": results_dir.relative_to(ROOT).as_posix(),
                "models_directory": models_dir.relative_to(ROOT).as_posix(),
            }
            _write_json_exclusive(
                results_dir / "execution_lock.json", execution_lock
            )
            data = load_phase18f_data(phase18b, matrix)
            names = canonical_feature_names()
            registry = feature_family_registry()
            train_cache = {}
            train_features = {}
            for domain in DOMAINS:
                sequences, labels, identifiers = data.partition(
                    domain, "train"
                )
                train_cache[domain] = {
                    "sequences": sequences,
                    "labels": labels,
                    "ids": identifiers,
                }
                train_features[domain] = _feature_matrix(
                    sequences.tolist(), names
                )
            training_records = {}
            selection_records = {}
            for specification in matrix:
                experiment_id = specification["experiment_id"]
                domain = specification["domain"]
                selected = selected_feature_names(
                    specification, names, registry
                )
                selected_matrix = _selected_columns(
                    train_features[domain], names, selected
                )
                model, resolved = _resolved_model()
                tracemalloc.start()
                started = time.perf_counter()
                history = model.fit(
                    selected_matrix,
                    train_cache[domain]["labels"],
                    feature_names=list(selected),
                    early_stopping_rounds=0,
                )
                fit_seconds = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                path = models_dir / f"{experiment_id}.pkl"
                model.save_model(path)
                if not path.is_file():
                    raise RuntimeError(
                        "XGBoost model artifact was not created"
                    )
                model_paths[experiment_id] = path
                completed_fits += 1
                training_records[experiment_id] = {
                    **history,
                    "wall_clock_fit_seconds": fit_seconds,
                    "python_tracemalloc_peak_bytes": peak,
                    "resolved_configuration": resolved,
                }
                selection_records[experiment_id] = {
                    "feature_count": len(selected),
                    "feature_names": list(selected),
                    "feature_name_sha256": fingerprint(list(selected)),
                }
            if completed_fits != 18 or set(model_paths) != {
                row["experiment_id"] for row in matrix
            }:
                raise RuntimeError("Completed fit inventory is not exactly 18")
            freeze = data.freeze(model_paths)
            if implementation_hashes() != frozen_implementation:
                raise RuntimeError(
                    "Implementation changed before evaluation freeze"
                )
            internal_cache = {}
            internal_features = {}
            for domain in DOMAINS:
                sequences, labels, identifiers = data.partition(
                    domain, "internal_test", freeze
                )
                internal_cache[domain] = {
                    "sequences": sequences,
                    "labels": labels,
                    "ids": identifiers,
                    "groups": np.asarray(
                        [sequence[4:24] for sequence in sequences]
                    ),
                }
                internal_features[domain] = _feature_matrix(
                    sequences.tolist(), names
                )
            run_records = {}
            prediction_values = {}
            artifact_models = []
            artifact_predictions = []
            for specification in matrix:
                experiment_id = specification["experiment_id"]
                domain = specification["domain"]
                selected = selection_records[experiment_id]["feature_names"]
                values = _selected_columns(
                    internal_features[domain], names, selected
                )
                model = XGBoostModel.load_model(model_paths[experiment_id])
                first = model.predict(values)
                second = model.predict(values)
                validate_prediction_repeatability(first, second)
                prediction = np.asarray(first, dtype=np.float64).reshape(-1)
                metrics = regression_metrics(
                    internal_cache[domain]["labels"], prediction
                )
                prediction_path = predictions_dir / f"{experiment_id}.npz"
                if prediction_path.exists():
                    raise FileExistsError(prediction_path)
                np.savez_compressed(
                    prediction_path,
                    experiment_id=np.asarray([experiment_id]),
                    domain=np.asarray([domain]),
                    sequence_ids=internal_cache[domain]["ids"],
                    spacer_groups=internal_cache[domain]["groups"],
                    labels=internal_cache[domain]["labels"],
                    predictions=prediction,
                    feature_names=np.asarray(selected),
                )
                model_sha = freeze.model_hashes[experiment_id]
                prediction_sha = _sha256(prediction_path)
                relative_model = (
                    model_paths[experiment_id].relative_to(ROOT).as_posix()
                )
                relative_prediction = prediction_path.relative_to(
                    ROOT
                ).as_posix()
                run_records[experiment_id] = {
                    "experiment_id": experiment_id,
                    "domain": domain,
                    "experiment_type": specification["experiment_type"],
                    "feature_family_removed": specification[
                        "feature_family_removed"
                    ],
                    "feature_families_retained": specification[
                        "feature_families_retained"
                    ],
                    "feature_count": len(selected),
                    "feature_name_sha256": selection_records[experiment_id][
                        "feature_name_sha256"
                    ],
                    "seed": 42,
                    "split_sha256": SPLIT_SHA256,
                    "xgboost_configuration": EXPECTED_XGBOOST_CONFIGURATION,
                    "metrics": metrics,
                    "training": training_records[experiment_id],
                    "model_artifact": relative_model,
                    "prediction_artifact": relative_prediction,
                    "model_sha256": model_sha,
                    "prediction_sha256": prediction_sha,
                }
                prediction_values[experiment_id] = prediction
                artifact_models.append(
                    {
                        "experiment_id": experiment_id,
                        "path": relative_model,
                        "sha256": model_sha,
                        "bytes": model_paths[experiment_id].stat().st_size,
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
            bootstrap_summaries = {}
            bootstrap_arrays = {}
            for domain in DOMAINS:
                domain_specs = [
                    row for row in matrix if row["domain"] == domain
                ]
                full_spec = next(
                    row
                    for row in domain_specs
                    if row["experiment_type"] == "FULL_CONTROL"
                )
                comparisons = {
                    row["experiment_id"]: prediction_values[
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
                bootstrap_summaries[domain], arrays = paired_bootstrap(
                    internal_cache[domain]["labels"],
                    internal_cache[domain]["groups"],
                    prediction_values[full_spec["experiment_id"]],
                    comparisons,
                    types,
                )
                bootstrap_arrays.update(
                    {
                        f"{domain}__{key}": value
                        for key, value in arrays.items()
                    }
                )
            bootstrap_path = bootstrap_dir / "paired_effect_distributions.npz"
            np.savez_compressed(bootstrap_path, **bootstrap_arrays)
            drop_effects = {domain: {} for domain in DOMAINS}
            family_only = {domain: {} for domain in DOMAINS}
            full_controls = {}
            for domain in DOMAINS:
                domain_specs = [
                    row for row in matrix if row["domain"] == domain
                ]
                full_spec = next(
                    row
                    for row in domain_specs
                    if row["experiment_type"] == "FULL_CONTROL"
                )
                full_metrics = run_records[full_spec["experiment_id"]][
                    "metrics"
                ]
                full_controls[domain] = run_records[full_spec["experiment_id"]]
                for row in domain_specs:
                    if row["experiment_type"] == "FULL_CONTROL":
                        continue
                    experiment_id = row["experiment_id"]
                    metrics = run_records[experiment_id]["metrics"]
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
            consistency = {}
            for family in FEATURE_FAMILIES:
                consistency[family] = classify_domain_consistency(
                    drop_effects[DOMAINS[0]][family][
                        "contribution_classification"
                    ],
                    drop_effects[DOMAINS[1]][family][
                        "contribution_classification"
                    ],
                )
            integrity_after = verify_integrity()
            if integrity_before != integrity_after:
                raise RuntimeError(
                    "Canonical artifacts changed during Phase 18F"
                )
            if implementation_hashes() != frozen_implementation:
                raise RuntimeError(
                    "Phase 18F implementation changed during run"
                )
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
            }
            _write_json_exclusive(
                results_dir / "artifact_manifest.json", artifact_manifest
            )
            summary = {
                "phase": "18F",
                "completion_status": "COMPLETED",
                "phase18e_protocol_sha256": PHASE18E_PROTOCOL_SHA256,
                "feature_space_sha256": CANONICAL_FEATURE_SHA256,
                "split_sha256": SPLIT_SHA256,
                "planned_fits": 18,
                "completed_fits": completed_fits,
                "failed_fits": 0,
                "skipped_fits": 0,
                "runs": run_records,
                "full_controls": full_controls,
                "drop_one_effects": drop_effects,
                "family_only": family_only,
                "domain_consistency": consistency,
                "strongest_unique_contributor": {
                    domain: _strongest_family(drop_effects[domain])
                    for domain in DOMAINS
                },
                "bootstrap_policy": phase18e["bootstrap_policy"],
                "multiple_comparison_policy": phase18e[
                    "multiple_comparison_policy"
                ],
                "artifact_manifest": artifact_manifest,
                "canonical_integrity": {
                    "before": integrity_before,
                    "after": integrity_after,
                    "unchanged": True,
                },
                "implementation_unchanged": True,
                "bridge_excluded": True,
                "quarantine_excluded": True,
                "locked_external_accessed": False,
                "biological_causality_claimed": False,
            }
            _write_json_exclusive(results_dir / "summary.json", summary)
            if REPORT_PATH.exists():
                raise FileExistsError(REPORT_PATH)
            with REPORT_PATH.open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(_report(summary))
        except Exception as error:
            violation = {
                "phase": "18F",
                "status": "HALTED",
                "completed_fits": completed_fits,
                "error_type": type(error).__name__,
                "error": str(error),
                "review_required": True,
            }
            violation_path = results_dir / "violation.json"
            if not violation_path.exists():
                _write_json_exclusive(violation_path, violation)
            raise
        return results_dir, models_dir

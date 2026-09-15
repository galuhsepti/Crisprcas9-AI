"""Execute the locked Phase 18H positional-region XGBoost ablation."""

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
    verify_integrity,
)
from src.experiment_protocols.phase18c_access import phase18c_access_guard
from src.experiment_protocols.phase18e_feature_ablation_protocol import (
    BOOTSTRAP_SEED,
    CANONICAL_FEATURE_COUNT,
    CANONICAL_FEATURE_SHA256,
    EXPECTED_XGBOOST_CONFIGURATION,
    SPLIT_SHA256,
    canonical_feature_names,
    fingerprint,
    serialize,
    verify_feature_source_integrity,
)
from src.experiment_protocols.phase18g_positional_region_protocol import (
    BOOTSTRAP_REPLICATES,
    REGION_DEFINITIONS,
    SIMULTANEOUS_QUANTILES,
    classify_cross_domain_region,
    classify_regional_contribution,
    validate_locked_protocol,
)
from src.feature_ablation.phase18f import load_phase18b_protocol
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

PHASE18G_PROTOCOL_SHA256 = (
    "05bee9cd3719824f7314366c1a47d1b9b0b3e1a1a8e651982da42a4943715e6c"
)
PHASE18G_PROTOCOL_FILE_SHA256 = (
    "533267e5b6f49f4321d949f5ee62f6b040b331021773b2b25cb054d3fe715be9"
)
PHASE18G_PROTOCOL_PATH = (
    ROOT / "results/phase18g_positional_region_protocol_20260915_093504.json"
)
RESULTS_ROOT = ROOT / "results" / "phase18h" / PHASE18G_PROTOCOL_SHA256
MODELS_ROOT = ROOT / "models" / "phase18h" / PHASE18G_PROTOCOL_SHA256
REPORT_PATH = ROOT / "docs" / "phase18h_positional_region_ablation_report.md"
IMPLEMENTATION_PATHS = (
    "src/feature_ablation/__init__.py",
    "src/feature_ablation/phase18f.py",
    "src/feature_ablation/phase18h.py",
    "scripts/run_phase18h_positional_region_ablation.py",
    "src/experiment_protocols/phase18g_positional_region_protocol.py",
    "src/experiment_protocols/phase18e_feature_ablation_protocol.py",
    "src/experiment_protocols/phase18c_access.py",
    "src/experiment_protocols/multidomain_protocol.py",
    "src/dataset_integration/label_compatibility.py",
    "src/multidomain/data.py",
    "src/multidomain/campaign.py",
    "src/models/xgboost_model.py",
    "src/evaluation/phase18c_statistics.py",
    "src/evaluation/metrics.py",
    "src/bioinformatics/gc_content.py",
    "src/bioinformatics/nucleotide_composition.py",
    "src/bioinformatics/kmer.py",
    "src/bioinformatics/positional_features.py",
    "src/bioinformatics/sequence_features.py",
    "requirements.txt",
)
REGION_NAMES = tuple(name for name, _ in REGION_DEFINITIONS)
DOMAIN_CODES = {DOMAINS[0]: "A", DOMAINS[1]: "B"}


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
    """Hash the explicit Phase 18H execution-source allowlist."""
    return {
        relative: _sha256(ROOT / relative) for relative in IMPLEMENTATION_PATHS
    }


def implementation_sha256() -> str:
    return fingerprint(implementation_hashes())


def _environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "platform": platform.platform(),
    }


def load_phase18g_protocol() -> dict:
    """Load and validate only the authoritative Phase 18G artifact."""
    if _sha256(PHASE18G_PROTOCOL_PATH) != PHASE18G_PROTOCOL_FILE_SHA256:
        raise RuntimeError("Phase 18G artifact file SHA-256 mismatch")
    payload = json.loads(PHASE18G_PROTOCOL_PATH.read_text(encoding="utf-8"))
    try:
        validate_locked_protocol(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Phase 18G locked protocol validation failed"
        ) from error
    if payload.get("protocol_sha256") != PHASE18G_PROTOCOL_SHA256:
        raise RuntimeError("Phase 18G internal protocol SHA-256 mismatch")
    return payload


def _expected_matrix_ids() -> list[str]:
    identifiers = []
    for domain in DOMAINS:
        code = DOMAIN_CODES[domain]
        identifiers.append(f"18H_REF_{code}_FULL_PHASE18F")
        identifiers.extend(
            f"18H_XGB_{code}_DROP_{region}" for region in REGION_NAMES
        )
        identifiers.append(
            f"18H_REF_{code}_DROP_ALL_POSITION_SPECIFIC_PHASE18F"
        )
    return identifiers


def validate_matrix(payload: Mapping[str, object]) -> list[dict]:
    """Validate and return the exact matrix stored in the locked artifact."""
    if payload.get("protocol_sha256") != PHASE18G_PROTOCOL_SHA256:
        raise RuntimeError("Phase 18H matrix protocol SHA-256 changed")
    try:
        validate_locked_protocol(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Phase 18H matrix lock validation failed"
        ) from error
    matrix = payload.get("experiment_matrix")
    if not isinstance(matrix, list) or len(matrix) != 12:
        raise RuntimeError("Phase 18H matrix must contain exactly 12 rows")
    if [row.get("experiment_id") for row in matrix] != _expected_matrix_ids():
        raise RuntimeError("Phase 18H matrix experiment identities changed")
    trainable = [row for row in matrix if row.get("requires_new_fit") is True]
    references = [
        row for row in matrix if row.get("requires_new_fit") is False
    ]
    if len(trainable) != 8 or len(references) != 4:
        raise RuntimeError("Phase 18H matrix fit budget changed")
    if any(
        row.get("experiment_type") != "DROP_POSITIONAL_REGION"
        or row.get("remaining_feature_count") != 177
        or row.get("seed") != 42
        or row.get("split_hash") != SPLIT_SHA256
        or row.get("xgboost_configuration")
        != EXPECTED_XGBOOST_CONFIGURATION
        for row in trainable
    ):
        raise RuntimeError("Phase 18H trainable matrix row changed")
    reference_types = [row["experiment_type"] for row in references]
    if reference_types.count("REUSED_FULL_REFERENCE") != 2 or (
        reference_types.count("REUSED_WHOLE_POSITIONAL_REFERENCE") != 2
    ):
        raise RuntimeError("Phase 18H reused-reference inventory changed")
    return matrix


def selected_feature_names(
    specification: Mapping[str, object],
    names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Subtract exactly the matrix-provided region from canonical order."""
    if (
        specification.get("requires_new_fit") is not True
        or specification.get("experiment_type") != "DROP_POSITIONAL_REGION"
    ):
        raise RuntimeError("Only locked drop-region rows are trainable")
    canonical = tuple(canonical_feature_names() if names is None else names)
    if len(canonical) != CANONICAL_FEATURE_COUNT or fingerprint(
        list(canonical)
    ) != CANONICAL_FEATURE_SHA256:
        raise RuntimeError("Canonical feature identity changed")
    removed = specification.get("feature_names_removed")
    if (
        not isinstance(removed, list)
        or len(removed) != 20
        or len(set(removed)) != 20
        or not set(removed) <= set(canonical)
    ):
        raise RuntimeError(
            "Matrix-provided regional feature subtraction changed"
        )
    selected = tuple(name for name in canonical if name not in set(removed))
    if len(selected) != 177 or len(selected) != specification.get(
        "remaining_feature_count"
    ):
        raise RuntimeError("Regional feature subtraction did not retain 177")
    return selected


class _EvaluationFreeze:
    __slots__ = ("owner", "model_hashes")

    def __init__(self, owner: object, model_hashes: Mapping[str, str]):
        self.owner = owner
        self.model_hashes = dict(model_hashes)


@dataclass(frozen=True)
class Phase18HData:
    """Phase 18C rows with evaluation closed until eight-model freeze."""

    rows: Mapping[str, DomainRows]
    expected_experiment_ids: frozenset[str]
    _owner: object
    model_root: Path = MODELS_ROOT

    def partition(
        self,
        domain: str,
        partition: str,
        freeze: _EvaluationFreeze | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if domain not in DOMAINS:
            raise ValueError("Unknown Phase 18H domain")
        if partition in {"bridge", "quarantine", "validation"}:
            raise PermissionError(
                f"Phase 18H never materializes the {partition} partition"
            )
        if partition == "internal_test":
            valid = (
                isinstance(freeze, _EvaluationFreeze)
                and freeze.owner is self._owner
                and set(freeze.model_hashes) == self.expected_experiment_ids
                and len(freeze.model_hashes) == 8
            )
            if not valid:
                raise PermissionError(
                    "Internal test remains closed until all 8 new models "
                    "are frozen"
                )
        elif partition != "train":
            raise ValueError("Unknown Phase 18H partition")
        rows = self.rows[domain]
        indices = rows.partition_indices[partition]
        return (
            rows.sequences[indices].copy(),
            rows.labels[indices].copy(),
            rows.sequence_ids[indices].copy(),
        )

    def freeze(self, model_paths: Mapping[str, Path]) -> _EvaluationFreeze:
        if (
            len(self.expected_experiment_ids) != 8
            or set(model_paths) != self.expected_experiment_ids
        ):
            raise RuntimeError("Phase 18H model freeze inventory is not eight")
        root = self.model_root.resolve()
        resolved = [Path(path).resolve() for path in model_paths.values()]
        if (
            len(set(resolved)) != 8
            or not all(path.is_file() for path in resolved)
            or not all(path.parent == root for path in resolved)
            or {path.stem for path in resolved} != self.expected_experiment_ids
        ):
            raise RuntimeError(
                "Phase 18H model freeze path inventory is invalid"
            )
        return _EvaluationFreeze(
            self._owner,
            {
                experiment_id: _sha256(Path(path))
                for experiment_id, path in model_paths.items()
            },
        )


def _load_data(
    phase18b: Mapping[str, object], matrix: Sequence[Mapping[str, object]],
    model_root: Path = MODELS_ROOT,
) -> Phase18HData:
    _, deep, xiang = load_phase18a_inputs()
    frames = {DOMAINS[0]: deep, DOMAINS[1]: xiang}
    rows = {
        domain: _domain_rows(
            frames[domain], domain, phase18b["split_manifest"]
        )
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        observed = label_fingerprint(
            frames[domain][SEQUENCE_COLUMNS[domain]].tolist(),
            frames[domain][LABEL_COLUMNS[domain]].to_numpy(dtype=np.float64),
        )
        if observed != phase18b["label_fingerprints"][domain]:
            raise RuntimeError("Locked raw-label fingerprint mismatch")
        indices = rows[domain].partition_indices
        development = set(indices["train"]) | set(indices["internal_test"])
        excluded = set(indices["bridge"]) | set(indices["quarantine"])
        if development & excluded:
            raise RuntimeError("Bridge/quarantine leakage detected")
    expected = frozenset(
        row["experiment_id"] for row in matrix if row["requires_new_fit"]
    )
    if len(expected) != 8:
        raise RuntimeError("Phase 18H data gate requires exactly eight models")
    return Phase18HData(rows, expected, object(), model_root)


def _verify_reference_artifacts(protocol: Mapping[str, object]) -> list[dict]:
    """Hash all four references without opening any prediction bundle."""
    context = protocol["phase18f_context"]
    summary_path = (ROOT / context["summary_path"]).resolve()
    if _sha256(summary_path) != context["summary_sha256"]:
        raise RuntimeError("Phase 18F reference summary hash mismatch")
    records = {}
    for row in protocol["experiment_matrix"]:
        reference = row["source_reference"]
        source_id = reference["source_experiment_id"]
        records[source_id] = reference
    if len(records) != 4:
        raise RuntimeError("Phase 18F reference artifact inventory changed")
    verified = []
    for source_id, reference in records.items():
        model_path = (ROOT / reference["model_artifact"]).resolve()
        prediction_path = (ROOT / reference["prediction_artifact"]).resolve()
        if not model_path.is_relative_to((ROOT / "models/phase18f").resolve()):
            raise PermissionError("Reference model escaped Phase 18F root")
        phase18f_results = (ROOT / "results/phase18f").resolve()
        if not prediction_path.is_relative_to(phase18f_results):
            raise PermissionError(
                "Reference prediction escaped Phase 18F root"
            )
        if _sha256(model_path) != reference["model_sha256"]:
            raise RuntimeError("Phase 18F reference model hash mismatch")
        if _sha256(prediction_path) != reference["prediction_sha256"]:
            raise RuntimeError("Phase 18F reference prediction hash mismatch")
        verified.append(
            {
                "source_experiment_id": source_id,
                "model_artifact": reference["model_artifact"],
                "model_sha256": reference["model_sha256"],
                "prediction_artifact": reference["prediction_artifact"],
                "prediction_sha256": reference["prediction_sha256"],
                "prediction_bundle_loaded": False,
            }
        )
    return verified


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
    result = full[:, [lookup[name] for name in selected]]
    if result.shape[1] != len(selected):
        raise RuntimeError("Selected feature matrix count mismatch")
    return result


def validate_prediction_repeatability(
    first: Sequence[float], second: Sequence[float]
) -> None:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise RuntimeError("Nonfinite Phase 18H prediction")
    if left.shape != right.shape or not np.array_equal(left, right):
        raise RuntimeError("Deterministic prediction repeatability failed")


def bonferroni_interval_quantiles() -> tuple[float, float]:
    expected = (0.05 / 16 / 2, 1 - 0.05 / 16 / 2)
    if not all(
        math.isclose(observed, target)
        for observed, target in zip(SIMULTANEOUS_QUANTILES, expected)
    ):
        raise RuntimeError("Phase 18H simultaneous quantile drift")
    return expected


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


def _empty_distributions(
    identifiers: Sequence[str], iterations: int
) -> dict[str, dict[str, np.ndarray]]:
    return {
        identifier: {
            "delta_spearman": np.full(iterations, np.nan),
            "delta_rmse": np.full(iterations, np.nan),
            "relative_rmse_degradation": np.full(iterations, np.nan),
        }
        for identifier in identifiers
    }


def _bootstrap_unique_groups(
    truth: np.ndarray,
    groups: np.ndarray,
    full: np.ndarray,
    comparisons: Mapping[str, np.ndarray],
    distributions: Mapping[str, Mapping[str, np.ndarray]],
    generator: np.random.Generator,
    iterations: int,
    chunk_size: int,
) -> None:
    sorted_rows = np.argsort(groups, kind="stable")
    n = len(groups)
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        draw = generator.choice(n, size=(stop - start, n), replace=True)
        rows = sorted_rows[draw]
        sampled_truth = truth[rows]
        sampled_full = full[rows]
        full_rho = _rowwise_spearman(sampled_truth, sampled_full)
        full_rmse = np.sqrt(
            np.mean((sampled_truth - sampled_full) ** 2, axis=1)
        )
        full_valid = np.isfinite(full_rho) & np.isfinite(full_rmse) & (
            full_rmse > 0
        )
        for identifier, predictions in comparisons.items():
            sampled = predictions[rows]
            rho = _rowwise_spearman(sampled_truth, sampled)
            rmse = np.sqrt(np.mean((sampled_truth - sampled) ** 2, axis=1))
            valid = full_valid & np.isfinite(rho) & np.isfinite(rmse)
            values = distributions[identifier]
            values["delta_spearman"][start:stop][valid] = (
                rho[valid] - full_rho[valid]
            )
            values["delta_rmse"][start:stop][valid] = (
                rmse[valid] - full_rmse[valid]
            )
            values["relative_rmse_degradation"][start:stop][valid] = (
                (rmse[valid] - full_rmse[valid]) / full_rmse[valid]
            )


def _group_draw_indices(
    groups: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    unique = np.asarray(sorted(set(groups.tolist())))
    draw = generator.choice(unique, size=len(unique), replace=True)
    positions = []
    for identifier in draw:
        positions.extend(np.flatnonzero(groups == identifier).tolist())
    return np.asarray(positions, dtype=np.int64)


def _bootstrap_repeated_groups(
    truth: np.ndarray,
    groups: np.ndarray,
    full: np.ndarray,
    comparisons: Mapping[str, np.ndarray],
    distributions: Mapping[str, Mapping[str, np.ndarray]],
    generator: np.random.Generator,
    iterations: int,
) -> None:
    for replicate in range(iterations):
        rows = _group_draw_indices(groups, generator)
        with np.errstate(all="ignore"):
            full_rho = float(
                stats.spearmanr(truth[rows], full[rows]).statistic
            )
            full_rmse = float(
                np.sqrt(np.mean((truth[rows] - full[rows]) ** 2))
            )
        for identifier, predictions in comparisons.items():
            with np.errstate(all="ignore"):
                rho = float(
                    stats.spearmanr(truth[rows], predictions[rows]).statistic
                )
                rmse = float(
                    np.sqrt(np.mean((truth[rows] - predictions[rows]) ** 2))
                )
            if not all(map(math.isfinite, (full_rho, full_rmse, rho, rmse))):
                continue
            if full_rmse <= 0:
                continue
            values = distributions[identifier]
            values["delta_spearman"][replicate] = rho - full_rho
            values["delta_rmse"][replicate] = rmse - full_rmse
            values["relative_rmse_degradation"][replicate] = (
                rmse - full_rmse
            ) / full_rmse


def paired_bootstrap(
    labels: Sequence[float],
    group_ids: Sequence[str],
    full_predictions: Sequence[float],
    comparison_predictions: Mapping[str, Sequence[float]],
    iterations: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 64,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Use one deterministic paired group-draw stream for one domain."""
    if iterations <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap iteration and chunk size must be positive")
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    full = np.asarray(full_predictions, dtype=np.float64).reshape(-1)
    comparisons = {
        key: np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in comparison_predictions.items()
    }
    arrays = [truth, groups, full, *comparisons.values()]
    if (
        not comparisons
        or len(truth) < 2
        or any(len(array) != len(truth) for array in arrays)
    ):
        raise ValueError("Bootstrap arrays are incomplete")
    if not np.all(np.isfinite(truth)) or not all(
        np.all(np.isfinite(value)) for value in [full, *comparisons.values()]
    ):
        raise ValueError("Bootstrap arrays must be finite")
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    distributions = _empty_distributions(list(comparisons), iterations)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) == len(groups) and np.all(counts == 1):
        _bootstrap_unique_groups(
            truth,
            groups,
            full,
            comparisons,
            distributions,
            generator,
            iterations,
            chunk_size,
        )
    else:
        _bootstrap_repeated_groups(
            truth,
            groups,
            full,
            comparisons,
            distributions,
            generator,
            iterations,
        )
    summary = {}
    for identifier, values in distributions.items():
        invalid = int(np.count_nonzero(~np.isfinite(values["delta_spearman"])))
        if any(
            np.count_nonzero(~np.isfinite(array)) != invalid
            for array in values.values()
        ):
            raise RuntimeError("Bootstrap invalid-replicate accounting drift")
        if invalid:
            nominal = {key: [None, None] for key in values}
            adjusted = {
                key: [None, None]
                for key in ("delta_spearman", "delta_rmse")
            }
        else:
            nominal = {
                key: np.quantile(
                    array, (0.025, 0.975), method="linear"
                ).tolist()
                for key, array in values.items()
            }
            adjusted = {
                key: np.quantile(
                    values[key],
                    bonferroni_interval_quantiles(),
                    method="linear",
                ).tolist()
                for key in ("delta_spearman", "delta_rmse")
            }
        summary[identifier] = {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": iterations,
            "invalid_bootstrap_n": invalid,
            "nominal_95_ci": nominal,
            "adjusted_99_6875_ci": adjusted,
        }
    flattened = {
        f"{identifier}__{key}": array
        for identifier, values in distributions.items()
        for key, array in values.items()
    }
    return summary, flattened


def classify_distribution(classes: Sequence[str]) -> str:
    """Apply the exact Phase 18G within-domain distribution rule."""
    contributors = {
        "STRONG_REGIONAL_CONTRIBUTOR",
        "MODERATE_REGIONAL_CONTRIBUTOR",
    }
    weak = {
        "LITTLE_UNIQUE_REGIONAL_CONTRIBUTION",
        "POTENTIALLY_REDUNDANT_REGION",
    }
    values = list(classes)
    known = contributors | weak | {"UNSTABLE_OR_INCONCLUSIVE"}
    if len(values) != 4 or not set(values) <= known:
        raise ValueError("Distribution classification requires four regions")
    if "UNSTABLE_OR_INCONCLUSIVE" in values:
        return "NO_CLEAR_REGIONAL_LOCALIZATION"
    count = sum(value in contributors for value in values)
    if count >= 2:
        return "BROADLY_DISTRIBUTED"
    if count == 1 and all(value in contributors | weak for value in values):
        return "CONCENTRATED_SINGLE_REGION"
    return "NO_CLEAR_REGIONAL_LOCALIZATION"


def phase18h_output_paths(stamp: str) -> tuple[Path, Path]:
    if not stamp or any(character not in "0123456789_" for character in stamp):
        raise ValueError("Invalid Phase 18H campaign stamp")
    results = (RESULTS_ROOT / f"campaign_{stamp}").resolve()
    models = (MODELS_ROOT / f"campaign_{stamp}").resolve()
    if not results.is_relative_to(RESULTS_ROOT.resolve()):
        raise PermissionError("Results path escaped Phase 18H isolation")
    if not models.is_relative_to(MODELS_ROOT.resolve()):
        raise PermissionError("Model path escaped Phase 18H isolation")
    return results, models


def run_preflight() -> dict:
    """Verify locks, references, and closed partitions without execution."""
    with phase18c_access_guard():
        protocol = load_phase18g_protocol()
        matrix = validate_matrix(protocol)
        phase18b = load_phase18b_protocol()
        names = canonical_feature_names()
        trainable = [row for row in matrix if row["requires_new_fit"]]
        selections = {
            row["experiment_id"]: {
                "feature_count": len(selected_feature_names(row, names)),
                "feature_name_sha256": fingerprint(
                    list(selected_feature_names(row, names))
                ),
            }
            for row in trainable
        }
        data = _load_data(phase18b, matrix)
        for domain in DOMAINS:
            data.partition(domain, "train")
            for held in (
                "internal_test",
                "bridge",
                "quarantine",
                "validation",
            ):
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
        if not all(item["matches_expected"] for item in integrity.values()):
            raise RuntimeError("Canonical integrity mismatch")
        references = _verify_reference_artifacts(protocol)
        return {
            "status": "PREFLIGHT_PASSED_NO_TRAINING",
            "phase18g_protocol_sha256": PHASE18G_PROTOCOL_SHA256,
            "phase18g_artifact_file_sha256": PHASE18G_PROTOCOL_FILE_SHA256,
            "feature_space_sha256": CANONICAL_FEATURE_SHA256,
            "feature_count": len(names),
            "split_sha256": phase18b["split_sha256"],
            "matrix_rows": len(matrix),
            "planned_new_fits": len(trainable),
            "reused_references": len(matrix) - len(trainable),
            "selections": selections,
            "resolved_xgboost": resolved,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bonferroni_quantiles": list(
                bonferroni_interval_quantiles()
            ),
            "reference_artifacts": references,
            "reference_prediction_bundles_loaded": 0,
            "source_hashes": verify_feature_source_integrity(),
            "canonical_integrity": integrity,
            "implementation_hashes": implementation_hashes(),
            "implementation_sha256": implementation_sha256(),
            "output_created": False,
            "model_training_occurred": False,
            "predictions_generated": False,
            "bridge_materialized": False,
            "quarantine_materialized": False,
            "validation_materialized": False,
            "moreno_accessed": False,
        }


def require_execution_confirmation(
    execute: bool,
    protocol_confirmation: str | None,
    implementation_confirmation: str | None,
) -> None:
    if not execute:
        raise PermissionError("Phase 18H requires the --execute interlock")
    if protocol_confirmation != PHASE18G_PROTOCOL_SHA256:
        raise PermissionError("Phase 18H protocol confirmation mismatch")
    if implementation_confirmation != implementation_sha256():
        raise PermissionError("Phase 18H implementation confirmation mismatch")


def _scalar_text(bundle: np.lib.npyio.NpzFile, key: str) -> str:
    values = np.asarray(bundle[key]).reshape(-1)
    if len(values) != 1:
        raise RuntimeError(f"Prediction bundle field is not scalar: {key}")
    return str(values[0])


def _load_full_reference(
    specification: Mapping[str, object],
    expected: Mapping[str, np.ndarray],
) -> dict:
    """Load one FULL bundle only after the new-model evaluation freeze."""
    reference = specification["source_reference"]
    path = (ROOT / reference["prediction_artifact"]).resolve()
    if _sha256(path) != reference["prediction_sha256"]:
        raise RuntimeError("Reused FULL prediction hash changed after freeze")
    with np.load(path, allow_pickle=False) as bundle:
        if _scalar_text(bundle, "experiment_id") != reference[
            "source_experiment_id"
        ]:
            raise RuntimeError("Reused FULL experiment identity mismatch")
        if _scalar_text(bundle, "domain") != specification["domain"]:
            raise RuntimeError("Reused FULL domain identity mismatch")
        labels = np.asarray(bundle["labels"], dtype=np.float64).reshape(-1)
        predictions = np.asarray(
            bundle["predictions"], dtype=np.float64
        ).reshape(-1)
        sequence_ids = np.asarray(bundle["sequence_ids"]).astype(str)
        groups = np.asarray(bundle["spacer_groups"]).astype(str)
        feature_names = tuple(
            np.asarray(bundle["feature_names"]).astype(str).tolist()
        )
    if not (
        np.array_equal(sequence_ids, expected["ids"])
        and np.array_equal(groups, expected["groups"])
        and np.array_equal(labels, expected["labels"])
    ):
        raise RuntimeError("Reused FULL pairing arrays do not align exactly")
    if (
        feature_names != tuple(canonical_feature_names())
        or fingerprint(list(feature_names)) != reference["feature_name_sha256"]
        or reference["feature_name_sha256"] != CANONICAL_FEATURE_SHA256
    ):
        raise RuntimeError("Reused FULL feature identity/hash mismatch")
    validate_prediction_repeatability(predictions, predictions.copy())
    metrics = regression_metrics(labels, predictions)
    if any(
        not math.isclose(
            metrics[key], reference["metrics"][key], abs_tol=1e-12
        )
        for key in metrics
    ):
        raise RuntimeError("Reused FULL metadata and bundle metrics differ")
    return {
        "experiment_id": specification["experiment_id"],
        "source_experiment_id": reference["source_experiment_id"],
        "domain": specification["domain"],
        "feature_count": CANONICAL_FEATURE_COUNT,
        "feature_name_sha256": CANONICAL_FEATURE_SHA256,
        "metrics": metrics,
        "prediction_artifact": reference["prediction_artifact"],
        "prediction_sha256": reference["prediction_sha256"],
        "predictions": predictions,
    }


def _effect_record(
    full_metrics: Mapping[str, float],
    arm_metrics: Mapping[str, float],
    bootstrap: Mapping[str, object],
) -> dict:
    delta_spearman = arm_metrics["spearman"] - full_metrics["spearman"]
    delta_rmse = arm_metrics["rmse"] - full_metrics["rmse"]
    adjusted = bootstrap["adjusted_99_6875_ci"]
    classification = classify_regional_contribution(
        {
            "delta_spearman": delta_spearman,
            "delta_rmse": delta_rmse,
            "adjusted_delta_spearman_ci": adjusted["delta_spearman"],
            "adjusted_delta_rmse_ci": adjusted["delta_rmse"],
            "invalid_bootstrap_n": bootstrap["invalid_bootstrap_n"],
        }
    )
    return {
        "delta_spearman": delta_spearman,
        "delta_rmse": delta_rmse,
        "relative_rmse_degradation": delta_rmse / full_metrics["rmse"],
        **bootstrap,
        "contribution_classification": classification,
    }


def _region_order(effects: Mapping[str, Mapping[str, float]]) -> list[str]:
    return sorted(
        effects,
        key=lambda region: (
            effects[region]["delta_spearman"],
            -effects[region]["delta_rmse"],
            region,
        ),
    )


def _report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Phase 18H - Locked Positional-Region Ablation",
        "",
        f"Status: **{summary['completion_status']}**",
        "",
        "This experiment measures predictive contribution and associated "
        "positional information. It makes no biological-causality or "
        "mechanistic claim.",
        "",
        "## Reused Full Controls",
        "",
        "| Domain | Spearman | RMSE | Pearson | R2 | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        metrics = summary["full_references"][domain]["metrics"]
        lines.append(
            f"| {domain} | {metrics['spearman']:.8f} | "
            f"{metrics['rmse']:.8f} | {metrics['pearson']:.8f} | "
            f"{metrics['r2']:.8f} | {metrics['mae']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Regional Results",
            "",
            "| Domain | Region | Spearman | RMSE | Pearson | R2 | MAE | "
            "Delta Spearman | Delta RMSE | Relative RMSE degradation | "
            "Adjusted Spearman CI | Adjusted RMSE CI | Class |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for domain in DOMAINS:
        for region in REGION_NAMES:
            record = summary["regional_results"][domain][region]
            metrics = record["metrics"]
            effect = record["effect"]
            spearman_ci = effect["adjusted_99_6875_ci"]["delta_spearman"]
            rmse_ci = effect["adjusted_99_6875_ci"]["delta_rmse"]
            lines.append(
                f"| {domain} | {region} | {metrics['spearman']:.8f} | "
                f"{metrics['rmse']:.8f} | {metrics['pearson']:.8f} | "
                f"{metrics['r2']:.8f} | {metrics['mae']:.8f} | "
                f"{effect['delta_spearman']:.8f} | "
                f"{effect['delta_rmse']:.8f} | "
                f"{effect['relative_rmse_degradation']:.8f} | "
                f"{spearman_ci} | {rmse_ci} | "
                f"{effect['contribution_classification']} |"
            )
    lines.extend(["", "## Preregistered Questions", ""])
    for domain in DOMAINS:
        strongest = summary["strongest_region"][domain]
        order = ", ".join(summary["descriptive_region_order"][domain])
        comparison = summary["pam_proximal_vs_distal"][domain]
        lines.extend(
            [
                f"- {domain}: strongest descriptive region is {strongest}; "
                f"primary delta-Spearman then delta-RMSE order is {order}.",
                f"- {domain}: PAM-proximal versus PAM-distal descriptive "
                f"comparison is {comparison['conclusion']}.",
                f"- {domain}: positional information classification is "
                f"{summary['distribution'][domain]}.",
            ]
        )
    for region in REGION_NAMES:
        lines.append(
            f"- {region}: cross-domain class is "
            f"{summary['cross_domain_consistency'][region]}."
        )
    for domain in DOMAINS:
        region4 = summary["region_4_significance"][domain]
        lines.append(
            f"- {domain}: REGION_4 confirmatory result is "
            f"{region4['classification']}; contributor support is "
            f"{region4['confirmatory_contributor']}."
        )
        lines.append(
            f"- {domain}: REGION_4-versus-broad localization answer is "
            f"{summary['region_4_vs_broad_distribution'][domain]}."
        )
    lines.extend(["", "## Whole-Family Context", ""])
    for domain in DOMAINS:
        context = summary["whole_family_context"][domain]
        lines.append(
            f"- {domain}: reused Phase 18F drop-all class is "
            f"{context['classification']} with delta Spearman "
            f"{context['effect']['delta_spearman']:.8f} and delta RMSE "
            f"{context['effect']['delta_rmse']:.8f}. "
            f"{summary['whole_family_signal_assessment'][domain]}"
        )
    lines.extend(
        [
            "",
            "Regional effects are qualitative, conditional, and non-additive; "
            "they are not summed to reconstruct the whole-family effect.",
            "",
            "## Execution And Inference Boundaries",
            "",
            "- Exactly 8 new fits and 8 new prediction bundles were produced; "
            "4 Phase 18F references were reused and none was refit.",
            "- XGBoost used seed 42 with no tuning, early stopping, sample "
            "weights, validation fit, or validation refit.",
            "- Paired bootstrap used 10,000 replicates per domain, sorted "
            "unique forward spacer20 groups, all observations in sampled "
            "groups, one PCG64(42) stream shared by four comparisons, paired "
            "draws, and linear percentile quantiles.",
            "- Nominal 95% intervals cover all three effects. Bonferroni "
            "99.6875% intervals at quantiles (0.0015625, 0.9984375) cover "
            "delta Spearman and delta RMSE across 16 confirmatory intervals.",
            "- Bridge, quarantine, and validation were excluded. "
            "Moreno-Mateos "
            "was not accessed. Domains retain native label scales.",
            "- Cross-region rankings and PAM-proximal versus PAM-distal "
            "comparisons are descriptive, not superiority tests.",
            "- Recovery additional fits: 0; recovery additional "
            "predictions: 0.",
            "",
        ]
    )
    return "\n".join(lines)


def run_campaign(
    protocol_confirmation: str,
    implementation_confirmation: str,
) -> tuple[Path, Path]:
    """Execute exactly eight locked fits after explicit digest confirmation."""
    with phase18c_access_guard():
        require_execution_confirmation(
            True, protocol_confirmation, implementation_confirmation
        )
        protocol = load_phase18g_protocol()
        matrix = validate_matrix(protocol)
        trainable = [row for row in matrix if row["requires_new_fit"]]
        phase18b = load_phase18b_protocol()
        preflight = run_preflight()
        frozen_implementation = implementation_hashes()
        if fingerprint(frozen_implementation) != implementation_confirmation:
            raise RuntimeError("Implementation changed after confirmation")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_dir, models_dir = phase18h_output_paths(stamp)
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        if results_dir.exists() or models_dir.exists():
            raise FileExistsError("Phase 18H campaign timestamp collision")
        results_dir.mkdir()
        models_dir.mkdir()
        predictions_dir = results_dir / "predictions"
        bootstrap_dir = results_dir / "bootstrap"
        predictions_dir.mkdir()
        bootstrap_dir.mkdir()
        completed_fits = 0
        completed_predictions = 0
        model_paths = {}
        try:
            if REPORT_PATH.exists():
                raise FileExistsError(REPORT_PATH)
            integrity_before = verify_integrity()
            execution_lock = {
                "phase": "18H",
                "phase18g_protocol_sha256": PHASE18G_PROTOCOL_SHA256,
                "phase18g_protocol_file_sha256": PHASE18G_PROTOCOL_FILE_SHA256,
                "feature_space_sha256": CANONICAL_FEATURE_SHA256,
                "split_sha256": SPLIT_SHA256,
                "matrix": matrix,
                "trainable_experiment_ids": [
                    row["experiment_id"] for row in trainable
                ],
                "implementation_hashes": frozen_implementation,
                "implementation_sha256": implementation_confirmation,
                "environment": _environment_versions(),
                "preflight": preflight,
                "results_directory": results_dir.relative_to(ROOT).as_posix(),
                "models_directory": models_dir.relative_to(ROOT).as_posix(),
            }
            _write_json_exclusive(
                results_dir / "execution_lock.json", execution_lock
            )
            data = _load_data(phase18b, matrix, models_dir)
            names = canonical_feature_names()
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
            for specification in trainable:
                experiment_id = specification["experiment_id"]
                domain = specification["domain"]
                selected = selected_feature_names(specification, names)
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
                    eval_set=None,
                    early_stopping_rounds=0,
                    sample_weight=None,
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
                    "tuning": False,
                    "validation_refit": False,
                    "sample_weight": None,
                }
                selection_records[experiment_id] = {
                    "feature_count": len(selected),
                    "feature_names": list(selected),
                    "feature_name_sha256": fingerprint(list(selected)),
                    "feature_names_removed": specification[
                        "feature_names_removed"
                    ],
                }
            if completed_fits != 8 or set(model_paths) != {
                row["experiment_id"] for row in trainable
            }:
                raise RuntimeError(
                    "Completed fit inventory is not exactly eight"
                )
            if {path.stem for path in models_dir.glob("*.pkl")} != set(
                model_paths
            ):
                raise RuntimeError(
                    "Phase 18H model directory inventory changed"
                )
            freeze = data.freeze(model_paths)
            if implementation_hashes() != frozen_implementation:
                raise RuntimeError(
                    "Implementation changed before model freeze"
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
                    "ids": identifiers.astype(str),
                    "groups": np.asarray(
                        [sequence[4:24] for sequence in sequences]
                    ).astype(str),
                }
                internal_features[domain] = _feature_matrix(
                    sequences.tolist(), names
                )
            run_records = {}
            prediction_values = {}
            artifact_models = []
            artifact_predictions = []
            for specification in trainable:
                experiment_id = specification["experiment_id"]
                domain = specification["domain"]
                selected = selection_records[experiment_id]["feature_names"]
                values = _selected_columns(
                    internal_features[domain], names, selected
                )
                model = XGBoostModel.load_model(model_paths[experiment_id])
                if tuple(model.feature_names) != tuple(selected):
                    raise RuntimeError(
                        "Frozen model feature identity mismatch"
                    )
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
                    feature_name_sha256=np.asarray(
                        [
                            selection_records[experiment_id][
                                "feature_name_sha256"
                            ]
                        ]
                    ),
                )
                completed_predictions += 1
                prediction_values[experiment_id] = prediction
                model_sha = freeze.model_hashes[experiment_id]
                prediction_sha = _sha256(prediction_path)
                relative_model = model_paths[experiment_id].relative_to(
                    ROOT
                ).as_posix()
                relative_prediction = prediction_path.relative_to(
                    ROOT
                ).as_posix()
                run_records[experiment_id] = {
                    "experiment_id": experiment_id,
                    "domain": domain,
                    "experiment_type": "DROP_POSITIONAL_REGION",
                    "region_removed": specification["region_removed"],
                    "positions_removed": specification["positions_removed"],
                    **selection_records[experiment_id],
                    "seed": 42,
                    "split_sha256": SPLIT_SHA256,
                    "xgboost_configuration": EXPECTED_XGBOOST_CONFIGURATION,
                    "metrics": metrics,
                    "training": training_records[experiment_id],
                    "model_artifact": relative_model,
                    "model_sha256": model_sha,
                    "prediction_artifact": relative_prediction,
                    "prediction_sha256": prediction_sha,
                }
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
            if completed_predictions != 8:
                raise RuntimeError("Prediction inventory is not exactly eight")
            full_references = {}
            for domain in DOMAINS:
                specification = next(
                    row
                    for row in matrix
                    if row["domain"] == domain
                    and row["experiment_type"] == "REUSED_FULL_REFERENCE"
                )
                full_references[domain] = _load_full_reference(
                    specification, internal_cache[domain]
                )
            bootstrap_summaries = {}
            bootstrap_arrays = {}
            for domain in DOMAINS:
                comparisons = {
                    row["experiment_id"]: prediction_values[
                        row["experiment_id"]
                    ]
                    for row in trainable
                    if row["domain"] == domain
                }
                bootstrap_summaries[domain], arrays = paired_bootstrap(
                    internal_cache[domain]["labels"],
                    internal_cache[domain]["groups"],
                    full_references[domain]["predictions"],
                    comparisons,
                )
                bootstrap_arrays.update(
                    {
                        f"{domain}__{key}": value
                        for key, value in arrays.items()
                    }
                )
            bootstrap_path = bootstrap_dir / "paired_region_effects.npz"
            np.savez_compressed(bootstrap_path, **bootstrap_arrays)
            regional_results = {domain: {} for domain in DOMAINS}
            for specification in trainable:
                experiment_id = specification["experiment_id"]
                domain = specification["domain"]
                region = specification["region_removed"]
                effect = _effect_record(
                    full_references[domain]["metrics"],
                    run_records[experiment_id]["metrics"],
                    bootstrap_summaries[domain][experiment_id],
                )
                regional_results[domain][region] = {
                    "experiment_id": experiment_id,
                    "metrics": run_records[experiment_id]["metrics"],
                    "effect": effect,
                }
            distribution = {
                domain: classify_distribution(
                    [
                        regional_results[domain][region]["effect"][
                            "contribution_classification"
                        ]
                        for region in REGION_NAMES
                    ]
                )
                for domain in DOMAINS
            }
            cross_domain = {
                region: classify_cross_domain_region(
                    regional_results[DOMAINS[0]][region]["effect"][
                        "contribution_classification"
                    ],
                    regional_results[DOMAINS[1]][region]["effect"][
                        "contribution_classification"
                    ],
                )
                for region in REGION_NAMES
            }
            orders = {
                domain: _region_order(
                    {
                        region: regional_results[domain][region]["effect"]
                        for region in REGION_NAMES
                    }
                )
                for domain in DOMAINS
            }
            pam_comparison = {}
            for domain in DOMAINS:
                proximal = regional_results[domain][REGION_NAMES[-1]]["effect"]
                distal = regional_results[domain][REGION_NAMES[0]]["effect"]
                stronger = (
                    proximal["delta_spearman"], -proximal["delta_rmse"]
                ) < (distal["delta_spearman"], -distal["delta_rmse"])
                pam_comparison[domain] = {
                    "pam_proximal_region": REGION_NAMES[-1],
                    "pam_distal_region": REGION_NAMES[0],
                    "pam_proximal_stronger_descriptively": stronger,
                    "conclusion": (
                        "PAM-proximal removal is descriptively stronger"
                        if stronger
                        else (
                            "PAM-distal removal is equal or stronger "
                            "descriptively"
                        )
                    ),
                }
            whole_context = {}
            whole_signal_assessment = {}
            for domain in DOMAINS:
                reference = protocol["phase18f_references"][domain]
                whole_context[domain] = {
                    "classification": reference[
                        "whole_family_classification"
                    ],
                    "effect": reference["whole_family_effect"],
                    "metrics": reference["drop_all_position_specific"][
                        "metrics"
                    ],
                    "prediction_bundle_loaded": False,
                    "model_refit": False,
                }
                if distribution[domain] == "BROADLY_DISTRIBUTED":
                    assessment = (
                        "Contributor evidence appears across multiple regions"
                    )
                elif distribution[domain] == "CONCENTRATED_SINGLE_REGION":
                    assessment = (
                        "Contributor evidence appears localized to one region"
                    )
                else:
                    assessment = "Regional results do not clearly localize"
                whole_signal_assessment[domain] = (
                    f"{assessment} within the whole-family signal. Whether "
                    "regional effects account for most of that signal is not "
                    "identifiable from these conditional, non-additive "
                    "ablations, and the effects are not summed."
                )
            region4_significance = {}
            region4_vs_broad = {}
            for domain in DOMAINS:
                effect = regional_results[domain][REGION_NAMES[-1]]["effect"]
                classification = effect["contribution_classification"]
                region4_significance[domain] = {
                    "classification": classification,
                    "confirmatory_contributor": classification
                    in {
                        "STRONG_REGIONAL_CONTRIBUTOR",
                        "MODERATE_REGIONAL_CONTRIBUTOR",
                    },
                    "adjusted_99_6875_ci": effect["adjusted_99_6875_ci"],
                }
                if distribution[domain] == "BROADLY_DISTRIBUTED":
                    localization = "NOT_CONCENTRATED_IN_REGION_4_BROAD"
                elif distribution[domain] == "CONCENTRATED_SINGLE_REGION":
                    localization = (
                        "CONCENTRATED_IN_REGION_4"
                        if region4_significance[domain][
                            "confirmatory_contributor"
                        ]
                        else "CONCENTRATED_OUTSIDE_REGION_4"
                    )
                else:
                    localization = "NO_CLEAR_REGION_4_LOCALIZATION"
                region4_vs_broad[domain] = localization
            integrity_after = verify_integrity()
            if integrity_before != integrity_after:
                raise RuntimeError(
                    "Canonical artifacts changed during Phase 18H"
                )
            if implementation_hashes() != frozen_implementation:
                raise RuntimeError(
                    "Phase 18H implementation changed during run"
                )
            artifact_manifest = {
                "phase": "18H",
                "models": artifact_models,
                "predictions": artifact_predictions,
                "bootstrap": {
                    "path": bootstrap_path.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(bootstrap_path),
                    "bytes": bootstrap_path.stat().st_size,
                },
                "references": [
                    {
                        **record,
                        "prediction_bundle_loaded": record[
                            "source_experiment_id"
                        ]
                        in {"18F_XGB_A_FULL", "18F_XGB_B_FULL"},
                    }
                    for record in preflight["reference_artifacts"]
                ],
                "model_count": 8,
                "prediction_count": 8,
                "new_fit_count": 8,
                "reused_reference_count": 4,
                "loaded_full_prediction_reference_count": 2,
                "loaded_whole_positional_prediction_reference_count": 0,
            }
            _write_json_exclusive(
                results_dir / "artifact_manifest.json", artifact_manifest
            )
            summary = {
                "phase": "18H",
                "completion_status": "COMPLETED",
                "phase18g_protocol_sha256": PHASE18G_PROTOCOL_SHA256,
                "feature_space_sha256": CANONICAL_FEATURE_SHA256,
                "split_sha256": SPLIT_SHA256,
                "planned_new_fits": 8,
                "completed_new_fits": completed_fits,
                "new_prediction_count": completed_predictions,
                "reused_full_reference_count": 2,
                "reused_whole_positional_reference_count": 2,
                "reference_refits": 0,
                "runs": run_records,
                "full_references": {
                    domain: {
                        key: value
                        for key, value in record.items()
                        if key != "predictions"
                    }
                    for domain, record in full_references.items()
                },
                "regional_results": regional_results,
                "strongest_region": {
                    domain: orders[domain][0] for domain in DOMAINS
                },
                "descriptive_region_order": orders,
                "pam_proximal_vs_distal": pam_comparison,
                "distribution": distribution,
                "cross_domain_consistency": cross_domain,
                "region_4_significance": region4_significance,
                "region_4_vs_broad_distribution": region4_vs_broad,
                "whole_family_context": whole_context,
                "whole_family_signal_assessment": whole_signal_assessment,
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
                "implementation_unchanged": True,
                "fit_policy": {
                    "seed": 42,
                    "early_stopping": False,
                    "hyperparameter_tuning": False,
                    "sample_weights": False,
                    "validation_refit": False,
                },
                "bridge_excluded": True,
                "quarantine_excluded": True,
                "validation_excluded": True,
                "moreno_accessed": False,
                "locked_external_accessed": False,
                "biological_causality_claimed": False,
                "analysis_recovery": {
                    "additional_fits": 0,
                    "additional_predictions": 0,
                },
            }
            _write_json_exclusive(results_dir / "summary.json", summary)
            if REPORT_PATH.exists():
                raise FileExistsError(REPORT_PATH)
            with REPORT_PATH.open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(_report(summary))
        except Exception as error:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            violation = {
                "phase": "18H",
                "status": "HALTED",
                "completed_new_fits": completed_fits,
                "completed_new_predictions": completed_predictions,
                "error_type": type(error).__name__,
                "error": str(error),
                "review_required": True,
            }
            violation_path = results_dir / "violation.json"
            if not violation_path.exists():
                _write_json_exclusive(violation_path, violation)
            raise
        return results_dir, models_dir

"""Read-only exploratory interpretation of the official Phase 18C campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from src.bioinformatics.kmer import (
    calculate_kmer_complexity,
    calculate_kmer_entropy,
)
from src.bioinformatics.nucleotide_composition import calculate_heterogeneity
from src.experiment_protocols.multidomain_protocol import DOMAINS, SEEDS
from src.experiment_protocols.phase18c_access import phase18c_access_guard

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SHA256 = "79ec4f03fe7314bc54a337c0d235c4f40e93b441aa40633828e1a50a3ee4d49b"
IMPLEMENTATION_SHA256 = (
    "cfacd0cefae25e897b08d363bbdb68f1ff072977317a2e7230ef39ad04a5a093"
)
SPLIT_SHA256 = "b7e043b30b3d070e1a41c461ad83774aec21a6abbfe9948cf1387a4ae49ab4a8"
CODE_COMMIT = "b8323b671050256cf1a85c3b1b9eec27ae1ed833"
CAMPAIGN_ID = "campaign_20260914_141742"
CAMPAIGN_RESULTS_SHA256 = (
    "5a2ec302fc73acbd2f121dfa34be7641dc9fbe27d6a8b283d3f32426141b2d78"
)
EXECUTION_LOCK_SHA256 = (
    "3ff5465c030a710bcff9b1e97755c8f832d6cf83a366b23138696b4b520bda89"
)
ARTIFACT_MANIFEST_SHA256 = (
    "a17be67fe2ebbc0ae92c3b07131f8402ffc1995e8ce783dfcfdbbf0d9cf81d06"
)
SOURCE_CLASSIFICATION = "NO_CLEAR_BENEFIT"
OFFICIAL_CAMPAIGN = ROOT / "results" / "phase18c" / PROTOCOL_SHA256 / CAMPAIGN_ID
OUTPUT_ROOT = ROOT / "results" / "phase18d"
SOURCE_IDENTITIES = {
    "campaign_results.json": CAMPAIGN_RESULTS_SHA256,
    "execution_lock.json": EXECUTION_LOCK_SHA256,
    "artifact_manifest.json": ARTIFACT_MANIFEST_SHA256,
}
TRAINING_NAMES = {
    "backward",
    "calibrate",
    "cross_validate",
    "fit",
    "hyperparameter_tuning",
    "optimizer_step",
    "partial_fit",
    "step",
    "train",
    "train_cnn",
    "zero_grad",
}
MODEL_COLUMNS = {
    "single_cnn": "prediction_single_cnn",
    "cnn_d": "prediction_cnn_d",
    "random_forest": "prediction_random_forest",
    "xgboost": "prediction_xgboost",
}
DOMAIN_LABELS = {
    DOMAINS[0]: "Domain A (DeepSpCas9)",
    DOMAINS[1]: "Domain B (CRISPRon Xiang/Luo)",
}
PROTOCOL_PATH = ROOT / "results" / "phase18b_multidomain_protocol_20260913_180731.json"
MIN_CORRELATION_N = 10
MIN_POSITION_N = 30


def sha256_file(path: Path) -> str:
    """Hash one explicit file without repository traversal."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_campaign_path(candidate: str | Path | None = None) -> Path:
    """Accept only the complete, immutable official Phase 18C campaign."""
    path = OFFICIAL_CAMPAIGN if candidate is None else Path(candidate)
    resolved = path.resolve()
    if resolved != OFFICIAL_CAMPAIGN.resolve():
        raise PermissionError("Phase 18D accepts only the official campaign")
    required = set(SOURCE_IDENTITIES) | {"models", "predictions"}
    if not resolved.is_dir() or any(
        not (resolved / name).exists() for name in required
    ):
        raise FileNotFoundError("Official Phase 18C campaign is incomplete")
    if (resolved / "unstable_result.json").exists() or (
        resolved / "violation.json"
    ).exists():
        raise RuntimeError("Partial or failed Phase 18C campaign prohibited")
    return resolved


def _validated_artifact_path(campaign: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError("Artifact manifest path escaped the campaign")
    resolved = (campaign / candidate).resolve()
    if not resolved.is_relative_to(campaign.resolve()):
        raise PermissionError("Artifact manifest alias escaped the campaign")
    return resolved


def verify_official_artifacts(
    candidate: str | Path | None = None,
) -> dict[str, object]:
    """Verify campaign identities and every registered model/prediction hash."""
    campaign = official_campaign_path(candidate)
    source_hashes = {name: sha256_file(campaign / name) for name in SOURCE_IDENTITIES}
    if source_hashes != SOURCE_IDENTITIES:
        raise RuntimeError("Official Phase 18C campaign identity mismatch")
    manifest = json.loads(
        (campaign / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    expected_metadata = {
        "campaign_id": CAMPAIGN_ID,
        "code_commit": CODE_COMMIT,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "phase": "18C",
        "protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "campaign_results_sha256": CAMPAIGN_RESULTS_SHA256,
        "execution_lock_sha256": EXECUTION_LOCK_SHA256,
    }
    if any(manifest.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("Phase 18C artifact manifest metadata mismatch")
    artifacts = manifest.get("artifacts", [])
    if manifest.get("artifact_count") != 70 or len(artifacts) != 70:
        raise RuntimeError("Phase 18C artifact inventory is incomplete")
    seen = set()
    verified = []
    for record in artifacts:
        relative = record.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise RuntimeError("Invalid or duplicate artifact manifest path")
        seen.add(relative)
        if any(
            record.get(key) != value
            for key, value in expected_metadata.items()
            if key
            in {
                "code_commit",
                "implementation_sha256",
                "protocol_sha256",
                "split_sha256",
            }
        ):
            raise RuntimeError(f"Artifact metadata mismatch: {relative}")
        path = _validated_artifact_path(campaign, relative)
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"Artifact missing or size-mismatched: {relative}")
        if sha256_file(path) != record.get("artifact_sha256"):
            raise RuntimeError(f"Artifact hash mismatch: {relative}")
        verified.append(relative)
    expected_types = {"model": 35, "prediction": 35}
    observed_types = {
        kind: sum(row.get("artifact_type") == kind for row in artifacts)
        for kind in expected_types
    }
    if observed_types != expected_types:
        raise RuntimeError("Phase 18C model/prediction inventory mismatch")
    return {
        "campaign": campaign.relative_to(ROOT).as_posix(),
        "campaign_id": CAMPAIGN_ID,
        "source_hashes": source_hashes,
        "artifact_count": len(verified),
        "artifact_types": observed_types,
        "all_hashes_match": True,
        "manifest": manifest,
    }


def artifact_snapshot(campaign: Path, manifest: dict) -> dict[str, str]:
    """Capture all immutable Phase 18C source identities."""
    paths = list(SOURCE_IDENTITIES) + [row["path"] for row in manifest["artifacts"]]
    return {relative: sha256_file(campaign / relative) for relative in paths}


@contextmanager
def phase18d_read_only_guard():
    """Block Phase 18C mutation and common fitting/gradient entry points."""
    state = {"active": True}
    protected = (ROOT / "results" / "phase18c").resolve()
    previous_profile = sys.getprofile()

    def on_audit(event, args):
        if not state["active"] or event != "open" or not args:
            return
        name = args[0]
        if not isinstance(name, (str, bytes, Path)):
            return
        path = Path(os.fsdecode(name)).resolve()
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        writing = bool(flags & write_flags) or bool(
            isinstance(mode, str) and any(character in mode for character in "wax+")
        )
        if writing and path.is_relative_to(protected):
            raise PermissionError("Phase 18D cannot mutate Phase 18C artifacts")

    def no_training(frame, event, arg):
        if event == "call" and frame.f_code.co_name in TRAINING_NAMES:
            module = str(frame.f_globals.get("__name__", ""))
            if module.startswith(
                (
                    "torch",
                    "sklearn",
                    "xgboost",
                    "src.models",
                    "src.multidomain",
                    "src.calibration",
                )
            ):
                raise RuntimeError("Phase 18D model fitting/training prohibited")
        if previous_profile is not None:
            previous_profile(frame, event, arg)

    sys.addaudithook(on_audit)
    sys.setprofile(no_training)
    try:
        with phase18c_access_guard():
            yield
    finally:
        state["active"] = False
        sys.setprofile(previous_profile)


def deterministic_json(payload: object) -> str:
    """Serialize machine-readable output deterministically."""
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def calculate_residuals(truth, prediction) -> dict[str, np.ndarray]:
    """Calculate prediction-minus-observation residual quantities."""
    observed = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if observed.ndim != 1 or predicted.shape != observed.shape:
        raise ValueError("Truth and prediction must be aligned 1-D arrays")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Residual inputs must be finite")
    residual = predicted - observed
    return {
        "residual": residual,
        "absolute_error": np.abs(residual),
        "squared_error": residual**2,
    }


def spacer_gc_fraction(sequence: str) -> float:
    """Return GC fraction for canonical spacer positions [4:24]."""
    _validate_sequence(sequence)
    spacer = sequence[4:24]
    return (spacer.count("G") + spacer.count("C")) / 20


def full_gc_fraction(sequence: str) -> float:
    """Return GC fraction for the complete canonical 30-mer."""
    _validate_sequence(sequence)
    return (sequence.count("G") + sequence.count("C")) / 30


def extract_pam(sequence: str) -> str:
    """Extract the canonical PAM at positions [24:27]."""
    _validate_sequence(sequence)
    return sequence[24:27]


def _validate_sequence(sequence: str) -> None:
    if (
        not isinstance(sequence, str)
        or len(sequence) != 30
        or set(sequence) - set("ACGT")
    ):
        raise ValueError("Expected an uppercase unambiguous 30-mer")


def deterministic_rank_bins(
    values,
    identifiers,
    labels: tuple[str, ...],
) -> np.ndarray:
    """Assign balanced rank bins using identifier order to resolve ties."""
    values = np.asarray(values, dtype=np.float64)
    identifiers = np.asarray(identifiers, dtype=str)
    if values.ndim != 1 or identifiers.shape != values.shape or values.size == 0:
        raise ValueError("Binning inputs must be aligned nonempty vectors")
    if not labels or not np.all(np.isfinite(values)):
        raise ValueError("Binning requires finite values and labels")
    order = np.lexsort((identifiers, values))
    result = np.empty(values.size, dtype=f"U{max(map(len, labels))}")
    for rank, index in enumerate(order):
        bin_index = min(len(labels) - 1, rank * len(labels) // values.size)
        result[index] = labels[bin_index]
    return result


def assign_activity_bins(values, identifiers) -> np.ndarray:
    """Assign predeclared within-domain balanced activity quartiles."""
    return deterministic_rank_bins(values, identifiers, ("Q1", "Q2", "Q3", "Q4"))


def assign_gc_bins(values) -> np.ndarray:
    """Assign fixed GC strata: below 0.40, 0.40-0.60, and above 0.60."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("GC values must be a finite vector")
    return np.where(
        values < 0.40,
        "low_lt_0.40",
        np.where(values <= 0.60, "mid_0.40_to_0.60", "high_gt_0.60"),
    )


def assign_error_category(
    primary_error: float,
    comparator_error: float,
    target_iqr: float,
    primary_name: str,
    comparator_name: str,
) -> str:
    """Assign a predeclared native-scale disagreement category."""
    if (
        not all(
            np.isfinite(value)
            for value in (primary_error, comparator_error, target_iqr)
        )
        or target_iqr <= 0
    ):
        raise ValueError("Error categorization requires finite positive scale")
    difference_margin = 0.10 * target_iqr
    if comparator_error - primary_error >= difference_margin:
        return f"{primary_name}_substantially_better"
    if primary_error - comparator_error >= difference_margin:
        return f"{comparator_name}_substantially_better"
    if max(primary_error, comparator_error) <= 0.10 * target_iqr:
        return "both_accurate"
    if min(primary_error, comparator_error) >= 0.25 * target_iqr:
        return "both_inaccurate"
    return "similar_or_mixed"


def align_predictions(
    expected_ids,
    observed_ids,
    predictions,
) -> np.ndarray:
    """Align one prediction vector to exact registered identifiers."""
    expected = np.asarray(expected_ids, dtype=str)
    observed = np.asarray(observed_ids, dtype=str)
    values = np.asarray(predictions, dtype=np.float64)
    if observed.shape != values.shape or expected.ndim != 1:
        raise ValueError("Prediction identifiers and values are not aligned")
    if len(set(observed.tolist())) != observed.size:
        raise ValueError("Duplicate prediction identifier")
    if set(observed.tolist()) != set(expected.tolist()):
        raise ValueError("Prediction identifiers do not match the locked split")
    lookup = dict(zip(observed.tolist(), values.tolist()))
    aligned = np.asarray([lookup[value] for value in expected], dtype=np.float64)
    if not np.all(np.isfinite(aligned)):
        raise ValueError("Predictions must be finite")
    return aligned


def seed_help_counts(
    single_predictions: dict[int, np.ndarray],
    shared_predictions: dict[int, np.ndarray],
    truth,
) -> np.ndarray:
    """Count seeds where CNN_D has strictly lower observation error."""
    if set(single_predictions) != set(SEEDS) or set(shared_predictions) != set(SEEDS):
        raise ValueError("Seed alignment requires all five registered seeds")
    truth = np.asarray(truth, dtype=np.float64)
    outcomes = []
    for seed in SEEDS:
        single = np.asarray(single_predictions[seed], dtype=np.float64)
        shared = np.asarray(shared_predictions[seed], dtype=np.float64)
        if single.shape != truth.shape or shared.shape != truth.shape:
            raise ValueError("Seed predictions are not observation-aligned")
        outcomes.append(np.abs(shared - truth) < np.abs(single - truth))
    return np.sum(np.vstack(outcomes), axis=0)


def select_hard_cases(
    identifiers,
    difficulty,
    fraction: float = 0.10,
) -> np.ndarray:
    """Select the highest consensus-difficulty fraction deterministically."""
    identifiers = np.asarray(identifiers, dtype=str)
    difficulty = np.asarray(difficulty, dtype=np.float64)
    if (
        identifiers.shape != difficulty.shape
        or identifiers.ndim != 1
        or identifiers.size == 0
        or not np.all(np.isfinite(difficulty))
        or not 0 < fraction <= 1
    ):
        raise ValueError("Hard-case inputs are invalid")
    count = max(1, math.ceil(fraction * identifiers.size))
    order = np.lexsort((identifiers, -difficulty))
    selected = np.zeros(identifiers.size, dtype=bool)
    selected[order[:count]] = True
    return selected


def align_bridge_pairs(left_ids, right_ids) -> np.ndarray:
    """Return right-side indices for exact one-to-one bridge alignment."""
    left = np.asarray(left_ids, dtype=str)
    right = np.asarray(right_ids, dtype=str)
    if left.size != 41 or right.size != 41:
        raise ValueError("Bridge analysis requires exactly 41 pairs")
    if len(set(left.tolist())) != 41 or len(set(right.tolist())) != 41:
        raise ValueError("Bridge identifiers must be unique")
    if set(left.tolist()) != set(right.tolist()):
        raise ValueError("Bridge pair identities do not align")
    lookup = {identifier: index for index, identifier in enumerate(right)}
    return np.asarray([lookup[identifier] for identifier in left], dtype=np.int64)


def _prediction_archive(experiment_id: str) -> dict[str, np.ndarray]:
    path = OFFICIAL_CAMPAIGN / "predictions" / f"{experiment_id}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _prediction(
    archive: dict[str, np.ndarray],
    domain: str,
    partition: str,
    expected_ids,
) -> np.ndarray:
    ids = archive[f"{domain}_{partition}_ids"]
    values = archive[f"{domain}_{partition}_predictions"]
    return align_predictions(expected_ids, ids, values)


def _arm_id(domain: str, family: str, seed: int) -> str:
    suffix = "A" if domain == DOMAINS[0] else "B"
    if family == "single_cnn":
        return f"18C_{suffix}_s{seed}"
    if family == "cnn_d":
        return f"18C_D_s{seed}"
    if family == "random_forest":
        return f"18C_random_forest_{suffix}_s{seed}"
    if family == "xgboost":
        return f"18C_xgboost_{suffix}_s{seed}"
    raise ValueError(f"Unknown model family: {family}")


def _sequence_properties(sequences, identifiers, labels) -> pd.DataFrame:
    records = []
    for sequence, identifier, label in zip(sequences, identifiers, labels):
        spacer = sequence[4:24]
        records.append(
            {
                "sequence_id": identifier,
                "sequence_30mer": sequence,
                "spacer_20mer": spacer,
                "pam": extract_pam(sequence),
                "true_activity": float(label),
                "spacer_gc_fraction": spacer_gc_fraction(sequence),
                "full_gc_fraction": full_gc_fraction(sequence),
                "nucleotide_entropy": calculate_heterogeneity(sequence),
                "k3_entropy": calculate_kmer_entropy(sequence, 3),
                "k3_complexity": calculate_kmer_complexity(sequence, 3),
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["activity_bin"] = assign_activity_bins(
        frame["true_activity"], frame["sequence_id"]
    )
    frame["spacer_gc_bin"] = assign_gc_bins(frame["spacer_gc_fraction"])
    frame["full_gc_bin"] = assign_gc_bins(frame["full_gc_fraction"])
    frame["complexity_bin"] = deterministic_rank_bins(
        frame["nucleotide_entropy"],
        frame["sequence_id"],
        ("low", "middle", "high"),
    )
    return frame


def _load_locked_rows(manifest: dict) -> dict[str, dict[str, np.ndarray]]:
    from src.experiment_protocols.multidomain_protocol import (
        validate_locked_artifact,
    )
    from src.multidomain.data import load_locked_development_data

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_locked_artifact(protocol, PROTOCOL_SHA256)
    if protocol["split_sha256"] != SPLIT_SHA256:
        raise RuntimeError("Registered split identity mismatch")
    data = load_locked_development_data(protocol, PROTOCOL_SHA256)
    model_paths = {
        record["experiment_id"]: OFFICIAL_CAMPAIGN / record["path"]
        for record in manifest["artifacts"]
        if record["artifact_type"] == "model"
    }
    freeze = data.freeze_evaluation(model_paths)
    result = {}
    for domain in DOMAINS:
        result[domain] = {}
        for partition in ("internal_test", "bridge"):
            sequences, labels, identifiers = data.partition(
                domain,
                partition,
                evaluation_freeze=freeze,
            )
            result[domain][partition] = {
                "sequences": sequences,
                "labels": labels,
                "ids": identifiers,
            }
    return result


def build_analysis_tables(
    locked_rows: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build immutable, seed-aligned internal-test analysis tables."""
    long_tables = {}
    observation_tables = {}
    for domain in DOMAINS:
        rows = locked_rows[domain]["internal_test"]
        properties = _sequence_properties(
            rows["sequences"], rows["ids"], rows["labels"]
        )
        seed_frames = []
        for seed in SEEDS:
            frame = properties.copy()
            frame.insert(0, "domain", domain)
            frame.insert(1, "seed", seed)
            frame.insert(2, "split_identity", SPLIT_SHA256)
            for family, column in MODEL_COLUMNS.items():
                experiment_id = _arm_id(domain, family, seed)
                archive = _prediction_archive(experiment_id)
                frame[column] = _prediction(archive, domain, "internal", rows["ids"])
                quantities = calculate_residuals(frame["true_activity"], frame[column])
                for quantity, values in quantities.items():
                    frame[f"{quantity}_{family}"] = values
            seed_frames.append(frame)
        long_frame = pd.concat(seed_frames, ignore_index=True)
        long_tables[domain] = long_frame
        static = properties.copy()
        static.insert(0, "domain", domain)
        for family, column in MODEL_COLUMNS.items():
            grouped = long_frame.groupby("sequence_id", sort=False)
            static[f"mean_prediction_{family}"] = static["sequence_id"].map(
                grouped[column].mean()
            )
            for quantity in ("residual", "absolute_error", "squared_error"):
                static[f"{quantity}_{family}"] = static["sequence_id"].map(
                    grouped[f"{quantity}_{family}"].mean()
                )
        observation_tables[domain] = static
    return long_tables, observation_tables


def _safe_spearman(truth, prediction, minimum_n=MIN_CORRELATION_N):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        truth.size < minimum_n
        or np.unique(truth).size < 2
        or np.unique(prediction).size < 2
    ):
        return None
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else None


def _safe_correlation(left, right, method: str):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.size < MIN_CORRELATION_N
        or np.unique(left).size < 2
        or np.unique(right).size < 2
    ):
        return None
    if method == "spearman":
        value = spearmanr(left, right).statistic
    elif method == "pearson":
        value = pearsonr(left, right).statistic
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(value) if np.isfinite(value) else None


def _error_metrics(frame: pd.DataFrame, family: str) -> dict[str, object]:
    residual = frame[f"residual_{family}"].to_numpy(dtype=np.float64)
    absolute = np.abs(residual)
    prediction = frame[MODEL_COLUMNS[family]].to_numpy(dtype=np.float64)
    truth = frame["true_activity"].to_numpy(dtype=np.float64)
    quantiles = np.quantile(absolute, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "n": int(len(frame)),
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": _safe_spearman(truth, prediction),
        "absolute_error_quantiles": {
            label: float(value)
            for label, value in zip(
                ("q10", "q25", "q50", "q75", "q90", "q95"), quantiles
            )
        },
    }


def _seed_aggregated_error_metrics(
    frame: pd.DataFrame, family: str
) -> dict[str, object]:
    """Aggregate only after each registered seed metric is computed."""
    seed_metrics = {
        int(seed): _error_metrics(group, family)
        for seed, group in frame.groupby("seed", sort=True)
    }
    if set(seed_metrics) != set(SEEDS):
        raise ValueError("Error summary requires all five registered seeds")
    residual = frame[f"residual_{family}"].to_numpy(dtype=np.float64)
    absolute = frame[f"absolute_error_{family}"].to_numpy(dtype=np.float64)
    quantiles = np.quantile(absolute, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    scalar_keys = ("mean_residual", "median_residual", "mae", "rmse", "spearman")
    return {
        "n": int(frame["sequence_id"].nunique()),
        "prediction_observation_n": int(len(frame)),
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "mae": float(np.mean([record["mae"] for record in seed_metrics.values()])),
        "rmse": float(np.mean([record["rmse"] for record in seed_metrics.values()])),
        "spearman": float(
            np.mean([record["spearman"] for record in seed_metrics.values()])
        ),
        "values_by_seed": {
            str(seed): {key: record[key] for key in scalar_keys}
            for seed, record in seed_metrics.items()
        },
        "absolute_error_quantiles": {
            label: float(value)
            for label, value in zip(
                ("q10", "q25", "q50", "q75", "q90", "q95"), quantiles
            )
        },
    }


def residual_and_strata_analysis(
    long_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Summarize residuals and predeclared activity/sequence strata."""
    domain_summaries = {}
    error_records = []
    strata_records = []
    stratifiers = (
        ("activity", "activity_bin"),
        ("spacer_gc", "spacer_gc_bin"),
        ("full_gc", "full_gc_bin"),
        ("pam", "pam"),
        ("sequence_complexity", "complexity_bin"),
    )
    for domain, frame in long_tables.items():
        domain_summaries[domain] = {"model_error_summary": {}}
        for family in MODEL_COLUMNS:
            metrics = _seed_aggregated_error_metrics(frame, family)
            domain_summaries[domain]["model_error_summary"][family] = metrics
            error_records.append(
                {
                    "domain": domain,
                    "model": family,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"absolute_error_quantiles", "values_by_seed"}
                    },
                    **metrics["absolute_error_quantiles"],
                }
            )
        for name, column in stratifiers:
            nested = {}
            for stratum, group in frame.groupby(column, sort=True):
                nested[str(stratum)] = {}
                for family in MODEL_COLUMNS:
                    metrics = _seed_aggregated_error_metrics(group, family)
                    compact = {
                        key: metrics[key]
                        for key in (
                            "n",
                            "prediction_observation_n",
                            "mean_residual",
                            "mae",
                            "rmse",
                            "spearman",
                        )
                    }
                    nested[str(stratum)][family] = compact
                    strata_records.append(
                        {
                            "domain": domain,
                            "stratifier": name,
                            "stratum": str(stratum),
                            "model": family,
                            **compact,
                        }
                    )
            domain_summaries[domain][f"{name}_strata"] = nested
    return (
        domain_summaries,
        pd.DataFrame.from_records(error_records),
        pd.DataFrame.from_records(strata_records),
    )


def model_disagreement_analysis(
    observation_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Categorize model disagreements using a locked native-IQR margin."""
    comparisons = (
        ("xgboost_vs_single_cnn", "xgboost", "single_cnn"),
        ("xgboost_vs_cnn_d", "xgboost", "cnn_d"),
        ("cnn_d_vs_single_cnn", "cnn_d", "single_cnn"),
    )
    assignments = []
    distributions = []
    summary = {}
    for domain, frame in observation_tables.items():
        target_iqr = float(
            np.quantile(frame["true_activity"], 0.75)
            - np.quantile(frame["true_activity"], 0.25)
        )
        if target_iqr <= 0:
            raise RuntimeError("Domain target IQR is degenerate")
        summary[domain] = {
            "target_iqr": target_iqr,
            "substantial_difference_margin": 0.10 * target_iqr,
            "both_accurate_max_error": 0.10 * target_iqr,
            "both_inaccurate_min_error": 0.25 * target_iqr,
            "comparisons": {},
        }
        for comparison, primary, comparator in comparisons:
            categories = []
            for _, row in frame.iterrows():
                primary_error = float(row[f"absolute_error_{primary}"])
                comparator_error = float(row[f"absolute_error_{comparator}"])
                category = assign_error_category(
                    primary_error,
                    comparator_error,
                    target_iqr,
                    primary,
                    comparator,
                )
                categories.append(category)
                assignments.append(
                    {
                        "domain": domain,
                        "comparison": comparison,
                        "sequence_id": row["sequence_id"],
                        "sequence_30mer": row["sequence_30mer"],
                        "true_activity": row["true_activity"],
                        "spacer_gc_fraction": row["spacer_gc_fraction"],
                        "full_gc_fraction": row["full_gc_fraction"],
                        "pam": row["pam"],
                        "nucleotide_entropy": row["nucleotide_entropy"],
                        "primary_model": primary,
                        "comparator_model": comparator,
                        "primary_absolute_error": primary_error,
                        "comparator_absolute_error": comparator_error,
                        "absolute_error_difference_comparator_minus_primary": (
                            comparator_error - primary_error
                        ),
                        "category": category,
                    }
                )
            categorized = frame.assign(category=categories)
            counts = Counter(categories)
            summary[domain]["comparisons"][comparison] = {
                category: {
                    "n": int(count),
                    "fraction": float(count / len(frame)),
                }
                for category, count in sorted(counts.items())
            }
            for category, group in categorized.groupby("category", sort=True):
                record = {
                    "domain": domain,
                    "comparison": comparison,
                    "category": category,
                    "n": int(len(group)),
                    "fraction": float(len(group) / len(frame)),
                    "mean_activity": float(group["true_activity"].mean()),
                    "mean_spacer_gc": float(group["spacer_gc_fraction"].mean()),
                    "mean_full_gc": float(group["full_gc_fraction"].mean()),
                    "mean_nucleotide_entropy": float(
                        group["nucleotide_entropy"].mean()
                    ),
                    "primary_mae": float(group[f"absolute_error_{primary}"].mean()),
                    "comparator_mae": float(
                        group[f"absolute_error_{comparator}"].mean()
                    ),
                    "primary_error_q90": float(
                        group[f"absolute_error_{primary}"].quantile(0.90)
                    ),
                    "comparator_error_q90": float(
                        group[f"absolute_error_{comparator}"].quantile(0.90)
                    ),
                }
                distributions.append(record)
    return (
        summary,
        pd.DataFrame.from_records(assignments),
        pd.DataFrame.from_records(distributions),
    )


def cnn_help_analysis(
    long_tables: dict[str, pd.DataFrame],
    observation_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    """Summarize CNN_D per-observation effects and seed consistency."""
    help_summary = {}
    pattern_records = []
    seed_records = []
    stratifiers = (
        ("activity", "activity_bin"),
        ("spacer_gc", "spacer_gc_bin"),
        ("full_gc", "full_gc_bin"),
        ("pam", "pam"),
        ("sequence_complexity", "complexity_bin"),
    )
    for domain, frame in observation_tables.items():
        delta = frame["absolute_error_single_cnn"] - frame["absolute_error_cnn_d"]
        enriched = frame.assign(delta_absolute_error=delta)
        help_summary[domain] = {
            "definition": "abs_error_single_cnn - abs_error_cnn_d",
            "mean_delta_absolute_error": float(delta.mean()),
            "median_delta_absolute_error": float(delta.median()),
            "fraction_improved": float((delta > 0).mean()),
            "patterns": {},
        }
        for name, column in stratifiers:
            help_summary[domain]["patterns"][name] = {}
            for stratum, group in enriched.groupby(column, sort=True):
                values = group["delta_absolute_error"]
                record = {
                    "n": int(len(group)),
                    "mean_delta_absolute_error": float(values.mean()),
                    "median_delta_absolute_error": float(values.median()),
                    "fraction_improved": float((values > 0).mean()),
                }
                help_summary[domain]["patterns"][name][str(stratum)] = record
                pattern_records.append(
                    {
                        "domain": domain,
                        "stratifier": name,
                        "stratum": str(stratum),
                        **record,
                    }
                )
        long_frame = long_tables[domain]
        per_seed = (
            long_frame["absolute_error_cnn_d"] < long_frame["absolute_error_single_cnn"]
        ).astype(int)
        counts = (
            long_frame.assign(helped=per_seed)
            .groupby("sequence_id", sort=False)["helped"]
            .sum()
        )
        for _, row in frame.iterrows():
            seed_records.append(
                {
                    "domain": domain,
                    "sequence_id": row["sequence_id"],
                    "sequence_30mer": row["sequence_30mer"],
                    "true_activity": row["true_activity"],
                    "spacer_gc_fraction": row["spacer_gc_fraction"],
                    "pam": row["pam"],
                    "n_seeds_cnn_d_better": int(counts[row["sequence_id"]]),
                }
            )
        distribution = Counter(int(value) for value in counts)
        help_summary[domain]["seed_consistency"] = {
            f"{count}/5": {
                "n": int(distribution.get(count, 0)),
                "fraction": float(distribution.get(count, 0) / len(frame)),
            }
            for count in range(5, -1, -1)
        }
    return (
        help_summary,
        pd.DataFrame.from_records(pattern_records),
        pd.DataFrame.from_records(seed_records),
    )


def position_specific_analysis(
    observation_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize associative nucleotide/error patterns at all 30 positions."""
    records = []
    for domain, frame in observation_tables.items():
        sequences = frame["sequence_30mer"].tolist()
        for position in range(30):
            region = (
                "5prime_context"
                if position < 4
                else (
                    "spacer"
                    if position < 24
                    else "pam" if position < 27 else "3prime_context"
                )
            )
            for nucleotide in "ACGT":
                mask = np.asarray(
                    [sequence[position] == nucleotide for sequence in sequences]
                )
                count = int(mask.sum())
                if count < MIN_POSITION_N:
                    continue
                group = frame.loc[mask]
                for family in MODEL_COLUMNS:
                    errors = group[f"absolute_error_{family}"]
                    records.append(
                        {
                            "domain": domain,
                            "position_zero_based": position,
                            "region": region,
                            "nucleotide": nucleotide,
                            "model": family,
                            "n": count,
                            "mean_absolute_error": float(errors.mean()),
                            "median_absolute_error": float(errors.median()),
                        }
                    )
    return pd.DataFrame.from_records(records)


def hard_case_analysis(
    long_tables: dict[str, pd.DataFrame],
    observation_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Identify top-10% consensus errors across four families and five seeds."""
    records = []
    summary = {}
    for domain, long_frame in long_tables.items():
        frame = observation_tables[domain].copy()
        target_iqr = float(
            np.quantile(frame["true_activity"], 0.75)
            - np.quantile(frame["true_activity"], 0.25)
        )
        error_columns = [f"absolute_error_{family}" for family in MODEL_COLUMNS]
        difficulty = (
            long_frame.assign(
                consensus_error=long_frame[error_columns].mean(axis=1) / target_iqr
            )
            .groupby("sequence_id", sort=False)["consensus_error"]
            .mean()
        )
        frame["consensus_difficulty"] = frame["sequence_id"].map(difficulty)
        frame["hard_case"] = select_hard_cases(
            frame["sequence_id"], frame["consensus_difficulty"]
        )
        hard = frame.loc[frame["hard_case"]].copy()
        for _, row in hard.iterrows():
            records.append(
                {
                    "domain": domain,
                    **row.to_dict(),
                    "cross_model_absolute_error_sd": float(
                        np.std([row[column] for column in error_columns], ddof=0)
                    ),
                }
            )
        summary[domain] = {
            "definition": (
                "top 10% within-domain mean absolute error across four "
                "model families and five seeds, divided by target IQR"
            ),
            "n": int(len(hard)),
            "fraction": float(len(hard) / len(frame)),
            "mean_activity": float(hard["true_activity"].mean()),
            "activity_quantile_counts": {
                str(key): int(value)
                for key, value in sorted(Counter(hard["activity_bin"]).items())
            },
            "mean_spacer_gc": float(hard["spacer_gc_fraction"].mean()),
            "mean_full_gc": float(hard["full_gc_fraction"].mean()),
            "mean_nucleotide_entropy": float(hard["nucleotide_entropy"].mean()),
            "pam_counts": {
                str(key): int(value)
                for key, value in sorted(Counter(hard["pam"]).items())
            },
            "mean_consensus_difficulty": float(hard["consensus_difficulty"].mean()),
        }
    return summary, pd.DataFrame.from_records(records)


def error_complementarity_analysis(
    long_tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Measure seed-wise XGBoost/CNN residual and absolute-error association."""
    records = []
    summary = {}
    for domain, frame in long_tables.items():
        summary[domain] = {}
        for comparison, family in (
            ("xgboost_vs_single_cnn", "single_cnn"),
            ("xgboost_vs_cnn_d", "cnn_d"),
        ):
            for seed, group in frame.groupby("seed", sort=True):
                record = {
                    "domain": domain,
                    "comparison": comparison,
                    "seed": int(seed),
                    "n": int(len(group)),
                    "residual_pearson": _safe_correlation(
                        group["residual_xgboost"],
                        group[f"residual_{family}"],
                        "pearson",
                    ),
                    "residual_spearman": _safe_correlation(
                        group["residual_xgboost"],
                        group[f"residual_{family}"],
                        "spearman",
                    ),
                    "absolute_error_pearson": _safe_correlation(
                        group["absolute_error_xgboost"],
                        group[f"absolute_error_{family}"],
                        "pearson",
                    ),
                    "absolute_error_spearman": _safe_correlation(
                        group["absolute_error_xgboost"],
                        group[f"absolute_error_{family}"],
                        "spearman",
                    ),
                }
                records.append(record)
            selected = [
                row
                for row in records
                if row["domain"] == domain and row["comparison"] == comparison
            ]
            summary[domain][comparison] = {}
            for metric in (
                "residual_pearson",
                "residual_spearman",
                "absolute_error_pearson",
                "absolute_error_spearman",
            ):
                values = [row[metric] for row in selected]
                summary[domain][comparison][metric] = {
                    "mean": float(np.mean(values)),
                    "range": [float(min(values)), float(max(values))],
                    "values_by_seed": {
                        str(row["seed"]): row[metric] for row in selected
                    },
                }
    return summary, pd.DataFrame.from_records(records)


def _feature_class(name: str) -> str:
    if name.startswith("guide_pos_"):
        return "spacer_position_one_hot"
    if name.startswith(("k2_", "k3_")):
        if name.endswith(("_entropy", "_complexity")):
            return "global_kmer_entropy_complexity"
        return "global_kmer_frequency"
    if name.startswith("dinuc_"):
        return "global_dinucleotide_frequency"
    if name.startswith("gc_") or name in {
        "gc_content",
        "at_skew",
        "is_optimal_gc",
    }:
        return "gc_regional_and_skew"
    if name.startswith("freq_") or name in {
        "purine_content",
        "pyrimidine_content",
        "heterogeneity",
    }:
        return "global_nucleotide_composition"
    raise ValueError(f"Unclassified registered feature: {name}")


def xgboost_interpretability(
    manifest: dict,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Load verified XGBoost models read-only and aggregate built-in importance."""
    from src.models.xgboost_model import XGBoostModel

    artifact_lookup = {
        row["experiment_id"]: OFFICIAL_CAMPAIGN / row["path"]
        for row in manifest["artifacts"]
        if row["artifact_type"] == "model"
    }
    feature_names = None
    records = []
    for domain in DOMAINS:
        suffix = "A" if domain == DOMAINS[0] else "B"
        for seed in SEEDS:
            experiment_id = f"18C_xgboost_{suffix}_s{seed}"
            model = XGBoostModel.load_model(str(artifact_lookup[experiment_id]))
            names = list(model.feature_names)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("XGBoost feature ordering differs across runs")
            if len(names) != 197 or len(set(names)) != 197:
                raise RuntimeError("Registered XGBoost feature inventory changed")
            booster = model.model.get_booster()
            scores = {
                kind: booster.get_score(importance_type=kind)
                for kind in ("gain", "weight", "cover")
            }
            for kind, values in scores.items():
                mapped = {
                    names[int(key[1:])] if key.startswith("f") else key: value
                    for key, value in values.items()
                }
                total = float(sum(mapped.values()))
                for name in names:
                    raw = float(mapped.get(name, 0.0))
                    records.append(
                        {
                            "domain": domain,
                            "seed": seed,
                            "experiment_id": experiment_id,
                            "importance_type": kind,
                            "feature": name,
                            "feature_class": _feature_class(name),
                            "importance": raw,
                            "normalized_importance": (
                                raw / total if total > 0 else 0.0
                            ),
                        }
                    )
    importance = pd.DataFrame.from_records(records)
    gains = importance.loc[importance["importance_type"] == "gain"]
    aggregate = (
        gains.groupby(["domain", "feature", "feature_class"], sort=True)[
            "normalized_importance"
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_normalized_gain",
                "std": "sd_normalized_gain",
                "min": "min_normalized_gain",
                "max": "max_normalized_gain",
            }
        )
    )
    aggregate["rank"] = (
        aggregate.groupby("domain")["mean_normalized_gain"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    aggregate = aggregate.sort_values(["domain", "rank"])
    names_frame = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names)),
            "feature": feature_names,
            "feature_class": [_feature_class(name) for name in feature_names],
        }
    )
    class_counts = {
        key: int(value)
        for key, value in sorted(Counter(names_frame["feature_class"]).items())
    }
    summary = {
        "representation": {
            "feature_count": 197,
            "geometry": "canonical 30-mer; spacer positions [4:24]",
            "feature_class_counts": class_counts,
            "exact_feature_names_table": (
                "results/phase18d/tables/xgboost_feature_names.csv"
            ),
        },
        "importance_method": (
            "read-only XGBoost built-in normalized gain; mean and SD across "
            "five registered seed models per domain"
        ),
        "top_features": {},
        "shap_used": False,
        "causality_warning": (
            "Model feature importance is not evidence of biological causality."
        ),
    }
    for domain in DOMAINS:
        top = aggregate.loc[aggregate["domain"] == domain].head(15)
        summary["top_features"][domain] = [
            {
                "rank": int(row["rank"]),
                "feature": row["feature"],
                "feature_class": row["feature_class"],
                "mean_normalized_gain": float(row["mean_normalized_gain"]),
                "sd_normalized_gain": float(row["sd_normalized_gain"]),
            }
            for _, row in top.iterrows()
        ]
    return summary, aggregate, names_frame


def bridge_interpretation(
    locked_rows: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Re-analyze only the 41 already-generated paired bridge predictions."""
    left_rows = locked_rows[DOMAINS[0]]["bridge"]
    right_rows = locked_rows[DOMAINS[1]]["bridge"]
    right_order = align_bridge_pairs(left_rows["ids"], right_rows["ids"])
    if not np.array_equal(left_rows["sequences"], right_rows["sequences"][right_order]):
        raise RuntimeError("Bridge sequence geometry does not align exactly")
    pair_records = []
    seed_records = []
    for seed in SEEDS:
        single_a_archive = _prediction_archive(f"18C_A_s{seed}")
        single_b_archive = _prediction_archive(f"18C_B_s{seed}")
        shared_archive = _prediction_archive(f"18C_D_s{seed}")
        expected = left_rows["ids"]
        single_a = _prediction(single_a_archive, DOMAINS[0], "bridge", expected)
        single_b = _prediction(single_b_archive, DOMAINS[1], "bridge", expected)
        shared_a = _prediction(shared_archive, DOMAINS[0], "bridge", expected)
        shared_b = _prediction(shared_archive, DOMAINS[1], "bridge", expected)
        ranks = {
            "single_a": (rankdata(single_a, method="average") - 1) / 40,
            "single_b": (rankdata(single_b, method="average") - 1) / 40,
            "shared_a": (rankdata(shared_a, method="average") - 1) / 40,
            "shared_b": (rankdata(shared_b, method="average") - 1) / 40,
        }
        baseline_disagreement = np.abs(ranks["single_a"] - ranks["single_b"])
        shared_disagreement = np.abs(ranks["shared_a"] - ranks["shared_b"])
        improvement = baseline_disagreement - shared_disagreement
        full_baseline = float(spearmanr(single_a, single_b).statistic)
        full_shared = float(spearmanr(shared_a, shared_b).statistic)
        loo_delta = []
        for index in range(41):
            keep = np.arange(41) != index
            loo_delta.append(
                float(spearmanr(shared_a[keep], shared_b[keep]).statistic)
                - float(spearmanr(single_a[keep], single_b[keep]).statistic)
            )
        seed_records.append(
            {
                "seed": seed,
                "n_pairs": 41,
                "single_head_spearman": full_baseline,
                "cnn_d_head_spearman": full_shared,
                "delta_spearman": full_shared - full_baseline,
                "single_mean_rank_disagreement": float(baseline_disagreement.mean()),
                "cnn_d_mean_rank_disagreement": float(shared_disagreement.mean()),
                "rank_disagreement_reduction": float(improvement.mean()),
                "pairs_reduced_disagreement": int((improvement > 0).sum()),
                "pairs_unchanged_disagreement": int((improvement == 0).sum()),
                "pairs_increased_disagreement": int((improvement < 0).sum()),
                "loo_delta_spearman_min": float(min(loo_delta)),
                "loo_delta_spearman_max": float(max(loo_delta)),
            }
        )
        for index, identifier in enumerate(expected):
            pair_records.append(
                {
                    "seed": seed,
                    "sequence_id": identifier,
                    "sequence_30mer": left_rows["sequences"][index],
                    "domain_a_true_activity": float(left_rows["labels"][index]),
                    "domain_b_true_activity": float(
                        right_rows["labels"][right_order[index]]
                    ),
                    "single_a_percentile_rank": ranks["single_a"][index],
                    "single_b_percentile_rank": ranks["single_b"][index],
                    "cnn_d_a_percentile_rank": ranks["shared_a"][index],
                    "cnn_d_b_percentile_rank": ranks["shared_b"][index],
                    "domain_a_rank_shift_cnn_d_minus_single": (
                        ranks["shared_a"][index] - ranks["single_a"][index]
                    ),
                    "domain_b_rank_shift_cnn_d_minus_single": (
                        ranks["shared_b"][index] - ranks["single_b"][index]
                    ),
                    "single_rank_disagreement": baseline_disagreement[index],
                    "cnn_d_rank_disagreement": shared_disagreement[index],
                    "rank_disagreement_reduction": improvement[index],
                }
            )
    pairs = pd.DataFrame.from_records(pair_records)
    seeds = pd.DataFrame.from_records(seed_records)
    per_pair = pairs.groupby("sequence_id", sort=False)[
        "rank_disagreement_reduction"
    ].mean()
    summary = {
        "n_pairs": 41,
        "limitation": (
            "Only 41 exact bridge pairs were available; this is a diagnostic, "
            "not a confirmatory benefit result."
        ),
        "source": "already-generated Phase 18C bridge predictions only",
        "classification_unchanged": SOURCE_CLASSIFICATION,
        "per_seed": {
            str(int(row["seed"])): {
                key: (
                    int(value)
                    if key.startswith("pairs_") or key == "n_pairs"
                    else float(value)
                )
                for key, value in row.items()
                if key != "seed"
            }
            for row in seed_records
        },
        "distribution": {
            "mean_pairwise_rank_disagreement_reduction": float(per_pair.mean()),
            "median_pairwise_rank_disagreement_reduction": float(per_pair.median()),
            "pairs_improved_on_average": int((per_pair > 0).sum()),
            "pairs_unchanged_on_average": int((per_pair == 0).sum()),
            "pairs_worsened_on_average": int((per_pair < 0).sum()),
            "fraction_pairs_improved_on_average": float((per_pair > 0).mean()),
        },
        "outlier_influence": {
            "minimum_leave_one_out_delta_spearman": float(
                seeds["loo_delta_spearman_min"].min()
            ),
            "maximum_leave_one_out_delta_spearman": float(
                seeds["loo_delta_spearman_max"].max()
            ),
            "all_leave_one_out_deltas_positive": bool(
                (seeds["loo_delta_spearman_min"] > 0).all()
            ),
        },
    }
    return summary, pairs, seeds


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )


def _position_summary(position_table: pd.DataFrame, domain: str) -> dict:
    result = {}
    selected = position_table.loc[position_table["domain"] == domain]
    for family in MODEL_COLUMNS:
        rows = selected.loc[selected["model"] == family]
        highest = rows.nlargest(5, "mean_absolute_error")
        result[family] = [
            {
                "position_zero_based": int(row["position_zero_based"]),
                "region": row["region"],
                "nucleotide": row["nucleotide"],
                "n": int(row["n"]),
                "mean_absolute_error": float(row["mean_absolute_error"]),
                "median_absolute_error": float(row["median_absolute_error"]),
            }
            for _, row in highest.iterrows()
        ]
    return result


def _make_figures(
    error_table: pd.DataFrame,
    strata_table: pd.DataFrame,
    observation_tables: dict[str, pd.DataFrame],
    importance: pd.DataFrame,
    bridge_pairs: pd.DataFrame,
    figure_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "single_cnn": "#3B5B92",
        "cnn_d": "#6C4BA1",
        "random_forest": "#C9792B",
        "xgboost": "#287D6B",
    }
    figures = []

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for axis, domain in zip(axes, DOMAINS):
        rows = error_table.loc[error_table["domain"] == domain]
        axis.bar(
            rows["model"],
            rows["rmse"],
            color=[colors[value] for value in rows["model"]],
        )
        axis.set_title(DOMAIN_LABELS[domain])
        axis.set_ylabel("RMSE (native units)")
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Exploratory Phase 18D model error comparison")
    fig.tight_layout()
    name = "model_performance_comparison.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for axis, domain in zip(axes, DOMAINS):
        frame = observation_tables[domain]
        axis.boxplot(
            [frame[f"residual_{family}"] for family in MODEL_COLUMNS],
            tick_labels=list(MODEL_COLUMNS),
            showfliers=False,
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(DOMAIN_LABELS[domain])
        axis.set_ylabel("Prediction - observed")
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Exploratory residual distributions")
    fig.tight_layout()
    name = "residual_distributions.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for axis, domain in zip(axes, DOMAINS):
        frame = observation_tables[domain]
        for family in ("single_cnn", "cnn_d", "xgboost"):
            axis.scatter(
                frame["true_activity"],
                frame[f"absolute_error_{family}"],
                s=5,
                alpha=0.25,
                label=family,
                color=colors[family],
            )
        axis.set_title(DOMAIN_LABELS[domain])
        axis.set_xlabel("Observed activity (native units)")
        axis.set_ylabel("Absolute error")
    axes[1].legend(frameon=False, markerscale=2)
    fig.suptitle("Exploratory activity-range error patterns")
    fig.tight_layout()
    name = "activity_vs_absolute_error.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)

    gc = strata_table.loc[strata_table["stratifier"] == "spacer_gc"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for axis, domain in zip(axes, DOMAINS):
        rows = gc.loc[gc["domain"] == domain]
        pivot = rows.pivot(index="stratum", columns="model", values="mae")
        pivot[list(MODEL_COLUMNS)].plot.bar(
            ax=axis,
            color=[colors[value] for value in MODEL_COLUMNS],
            legend=False,
        )
        axis.set_title(DOMAIN_LABELS[domain])
        axis.set_xlabel("Spacer GC stratum")
        axis.set_ylabel("MAE (native units)")
        axis.tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Exploratory error by fixed spacer-GC bin")
    fig.tight_layout()
    name = "error_by_spacer_gc.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)

    pam = strata_table.loc[
        (strata_table["stratifier"] == "pam") & (strata_table["n"] >= MIN_POSITION_N)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for axis, domain in zip(axes, DOMAINS):
        rows = pam.loc[pam["domain"] == domain]
        order = rows.groupby("stratum")["n"].max().nlargest(8).index.tolist()
        pivot = rows.loc[rows["stratum"].isin(order)].pivot(
            index="stratum", columns="model", values="mae"
        )
        pivot = pivot.reindex(order)
        pivot[list(MODEL_COLUMNS)].plot.bar(
            ax=axis,
            color=[colors[value] for value in MODEL_COLUMNS],
            legend=False,
        )
        axis.set_title(DOMAIN_LABELS[domain])
        axis.set_xlabel("Observed PAM (n >= 30)")
        axis.set_ylabel("MAE (native units)")
        axis.tick_params(axis="x", rotation=0)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Exploratory associative PAM error analysis")
    fig.tight_layout()
    name = "error_by_pam.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for axis, domain in zip(axes, DOMAINS):
        frame = observation_tables[domain]
        delta = frame["absolute_error_single_cnn"] - frame["absolute_error_cnn_d"]
        axis.hist(delta, bins=35, color=colors["cnn_d"], alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(DOMAIN_LABELS[domain])
        axis.set_xlabel("Positive means CNN_D improved")
        axis.set_ylabel("Observations")
    fig.suptitle("Exploratory CNN_D absolute-error change")
    fig.tight_layout()
    name = "cnn_d_improvement_distribution.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for axis, domain in zip(axes, DOMAINS):
        rows = importance.loc[importance["domain"] == domain].head(12)
        axis.barh(
            rows["feature"][::-1],
            rows["mean_normalized_gain"][::-1],
            color=colors["xgboost"],
        )
        axis.set_title(DOMAIN_LABELS[domain])
        axis.set_xlabel("Mean normalized gain")
    fig.suptitle("Exploratory read-only XGBoost feature importance")
    fig.tight_layout()
    name = "xgboost_feature_importance.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)

    bridge_mean = bridge_pairs.groupby("sequence_id", sort=False)[
        ["single_rank_disagreement", "cnn_d_rank_disagreement"]
    ].mean()
    fig, axis = plt.subplots(figsize=(4.5, 4))
    axis.scatter(
        bridge_mean["single_rank_disagreement"],
        bridge_mean["cnn_d_rank_disagreement"],
        color=colors["cnn_d"],
        s=20,
        alpha=0.8,
    )
    maximum = float(bridge_mean.to_numpy().max())
    axis.plot([0, maximum], [0, maximum], color="black", linewidth=0.8)
    axis.set_xlabel("Single-domain head rank disagreement")
    axis.set_ylabel("CNN_D head rank disagreement")
    axis.set_title("41 bridge pairs, mean across five seeds")
    fig.tight_layout()
    name = "bridge_rank_consistency.png"
    fig.savefig(figure_dir / name, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)
    return figures


def _domain_key(domain: str) -> str:
    return "domain_A" if domain == DOMAINS[0] else "domain_B"


def _observed_findings(
    domain_summary: dict,
    cnn_help: dict,
    hard_cases: dict,
    bridge: dict,
    complementarity: dict,
) -> list[str]:
    findings = []
    for domain in DOMAINS:
        errors = domain_summary[domain]["model_error_summary"]
        lowest = min(errors, key=lambda family: errors[family]["rmse"])
        findings.append(
            f"{DOMAIN_LABELS[domain]}: {lowest} had the lowest exploratory "
            f"mean per-seed RMSE ({errors[lowest]['rmse']:.6g} native units)."
        )
        q1 = domain_summary[domain]["activity_strata"]["Q1"]
        q4 = domain_summary[domain]["activity_strata"]["Q4"]
        findings.append(
            f"{DOMAIN_LABELS[domain]}: mean residuals shifted from "
            f"{q1['xgboost']['mean_residual']:.6g} in Q1 to "
            f"{q4['xgboost']['mean_residual']:.6g} in Q4 for XGBoost."
        )
        findings.append(
            f"{DOMAIN_LABELS[domain]}: CNN_D reduced absolute error for "
            f"{cnn_help[domain]['fraction_improved']:.1%} of observations "
            "when comparing each observation's mean error across the five "
            "registered seeds."
        )
        findings.append(
            f"{DOMAIN_LABELS[domain]}: the algorithmic hard-case set contained "
            f"{hard_cases[domain]['n']} observations."
        )
        correlation = complementarity[domain]["xgboost_vs_cnn_d"][
            "absolute_error_spearman"
        ]["mean"]
        findings.append(
            f"{DOMAIN_LABELS[domain]}: XGBoost/CNN_D absolute-error Spearman "
            f"correlation averaged {correlation:.4f} across seeds."
        )
    distribution = bridge["distribution"]
    findings.append(
        "Across the 41-pair bridge, CNN_D reduced mean rank disagreement for "
        f"{distribution['pairs_improved_on_average']}/41 pairs on average across seeds."
    )
    return findings


def _markdown_rows(frame: pd.DataFrame, columns: list[str], digits=5) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _render_report(
    summary: dict,
    error_table: pd.DataFrame,
    help_table: pd.DataFrame,
    position_table: pd.DataFrame,
    importance: pd.DataFrame,
    bridge_seeds: pd.DataFrame,
    figures: list[str],
    tables: list[str],
) -> str:
    sections = [
        "# Phase 18D Post-Experiment Interpretation Report",
        "",
        "## Status and scope",
        "",
        "Phase 18D is complete as an **EXPLORATORY POST-HOC** analysis of only "
        "the official Phase 18C campaign. The immutable source classification "
        "remains **NO_CLEAR_BENEFIT**. No Phase 18C classifier was rerun, no "
        "predictions were generated or changed, and no model was fitted, tuned, "
        "calibrated, or updated.",
        "",
        "Domains retain native label scales throughout; raw activities are never "
        "pooled across domains.",
        "",
        "## Deterministic policy",
        "",
        "- Residual is prediction minus observed activity.",
        "- Activity bins are balanced within-domain rank quartiles fixed by true "
        "activity, with SHA-256 sequence identity as the tie-breaker.",
        "- GC bins are fixed at `<0.40`, `0.40-0.60`, and `>0.60`.",
        "- Substantial model disagreement is an absolute-error difference of at "
        "least `0.10 x within-domain target IQR`, set before examples are extracted.",
        "- Hard cases are the top 10% within each domain by mean absolute error "
        "across four model families and five seeds, normalized by target IQR.",
        "- Position-specific rows require `n >= 30`; stratum Spearman requires "
        "`n >= 10` and nonconstant values.",
        "- Performance metrics are computed separately for each registered seed "
        "and then arithmetically averaged; seed-averaged predictions are not "
        "evaluated as an ensemble.",
        "",
        "## Residual results",
        "",
        _markdown_rows(
            error_table,
            [
                "domain",
                "model",
                "n",
                "mean_residual",
                "median_residual",
                "mae",
                "rmse",
                "spearman",
            ],
        ),
        "",
        "The activity-, GC-, PAM-, and complexity-stratified estimates are "
        "associative and exploratory. Sample sizes are reported in the tables. "
        "Tiny PAM groups are not interpreted strongly.",
        "",
        "## CNN_D error signature",
        "",
        _markdown_rows(
            help_table,
            [
                "domain",
                "stratifier",
                "stratum",
                "n",
                "mean_delta_absolute_error",
                "fraction_improved",
            ],
        ),
        "",
        "Positive delta absolute error means CNN_D improved over the corresponding "
        "single-domain CNN. These patterns do not alter the Phase 18C outcome.",
        "",
        "## Position-specific associations",
        "",
        f"{len(position_table)} nucleotide-position/model strata met the `n >= 30` "
        "threshold. The reported associations are not causal sequence determinants.",
        "",
        "## XGBoost representation and importance",
        "",
        "The registered XGBoost models received exactly 197 deterministic features: "
        "full/regional GC and skew features; global nucleotide composition and "
        "dinucleotide frequencies; nucleotide heterogeneity; global 2-mer and "
        "3-mer frequencies plus entropy/complexity; and one-hot nucleotide identity "
        "at each of the 20 spacer positions. The exact ordered names are archived.",
        "",
        _markdown_rows(
            importance.groupby("domain", sort=False).head(10),
            [
                "domain",
                "rank",
                "feature",
                "feature_class",
                "mean_normalized_gain",
                "sd_normalized_gain",
            ],
        ),
        "",
        "Built-in gain importance describes model split utility, not biological "
        "causality. SHAP was not used, avoiding a new dependency and reproducibility "
        "risk.",
        "",
        "## Bridge interpretation",
        "",
        "**Limitation: the bridge contains only 41 exact pairs.** It remains a "
        "diagnostic and cannot alter **NO_CLEAR_BENEFIT**.",
        "",
        _markdown_rows(
            bridge_seeds,
            [
                "seed",
                "single_head_spearman",
                "cnn_d_head_spearman",
                "delta_spearman",
                "single_mean_rank_disagreement",
                "cnn_d_mean_rank_disagreement",
                "pairs_reduced_disagreement",
            ],
        ),
        "",
        "Leave-one-out and pairwise rank-disagreement summaries are in "
        "`summary.json` and `bridge_pairs.csv`; they assess whether the consistency "
        "shift is broad or outlier-dominated.",
        "",
        "## Scientific interpretation",
        "",
        "### Observed results",
        "",
        *[f"- {item}" for item in summary["observed_findings"]],
        "",
        "### Plausible interpretations",
        "",
        *[f"- {item}" for item in summary["plausible_interpretations"]],
        "",
        "### Untested hypotheses",
        "",
        *[f"- {item}" for item in summary["untested_hypotheses"]],
        "",
        "### Recommended next questions",
        "",
        *[f"- {item}" for item in summary["recommended_next_questions"]],
        "",
        "No ensemble was built or evaluated. Any ensemble, alternative feature set, "
        "architecture, or external evaluation requires a later preregistered phase.",
        "",
        "## Outputs",
        "",
        *[f"- Figure: `results/phase18d/figures/{name}`" for name in figures],
        *[f"- Table: `results/phase18d/tables/{name}`" for name in tables],
        "",
        "## Integrity",
        "",
        "All 70 Phase 18C prediction/model artifacts matched the official manifest "
        "before analysis and all Phase 18C source identities remained unchanged "
        "after analysis. The locked external dataset remained prohibited and "
        "untouched.",
        "",
    ]
    return "\n".join(sections)


def run_phase18d() -> dict[str, object]:
    """Execute the complete read-only Phase 18D analysis."""
    from src.experiment_protocols.multidomain_protocol import verify_integrity

    with phase18d_read_only_guard():
        verification = verify_official_artifacts()
        manifest = verification["manifest"]
        campaign = official_campaign_path()
        source_before = artifact_snapshot(campaign, manifest)
        canonical_before = verify_integrity()

        campaign_results = json.loads(
            (campaign / "campaign_results.json").read_text(encoding="utf-8")
        )
        inference = campaign_results["results"]["primary_inference"]
        if inference.get("outcome") != SOURCE_CLASSIFICATION:
            raise RuntimeError("Immutable Phase 18C classification mismatch")
        runs = campaign_results["results"].get("runs", {})
        if len(runs) != 35:
            raise RuntimeError("Official Phase 18C run inventory is incomplete")
        if any(
            record["domains"][domain]["internal_test"]["status"] != "COMPUTED"
            for record in runs.values()
            for domain in record["domains"]
        ):
            raise RuntimeError("Official Phase 18C contains an incomplete run")

        locked_rows = _load_locked_rows(manifest)
        long_tables, observation_tables = build_analysis_tables(locked_rows)
        domain_summary, error_table, strata_table = residual_and_strata_analysis(
            long_tables
        )
        disagreement, category_table, category_summary_table = (
            model_disagreement_analysis(observation_tables)
        )
        cnn_help, help_table, seed_table = cnn_help_analysis(
            long_tables, observation_tables
        )
        position_table = position_specific_analysis(observation_tables)
        hard_summary, hard_table = hard_case_analysis(long_tables, observation_tables)
        complementarity, complementarity_table = error_complementarity_analysis(
            long_tables
        )
        xgboost_summary, importance_table, feature_names_table = (
            xgboost_interpretability(manifest)
        )
        bridge_summary, bridge_pairs, bridge_seeds = bridge_interpretation(locked_rows)

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        table_dir = OUTPUT_ROOT / "tables"
        figure_dir = OUTPUT_ROOT / "figures"
        table_dir.mkdir(exist_ok=True)
        figure_dir.mkdir(exist_ok=True)
        table_frames = {
            "analysis_rows_domain_A.csv": long_tables[DOMAINS[0]],
            "analysis_rows_domain_B.csv": long_tables[DOMAINS[1]],
            "observation_summary_domain_A.csv": observation_tables[DOMAINS[0]],
            "observation_summary_domain_B.csv": observation_tables[DOMAINS[1]],
            "model_error_summary.csv": error_table,
            "strata_summary.csv": strata_table,
            "model_disagreement_assignments.csv": category_table,
            "model_disagreement_summary.csv": category_summary_table,
            "cnn_d_help_patterns.csv": help_table,
            "cnn_d_seed_consistency.csv": seed_table,
            "position_specific_error_associations.csv": position_table,
            "hard_cases.csv": hard_table,
            "error_complementarity.csv": complementarity_table,
            "xgboost_feature_importance.csv": importance_table,
            "xgboost_feature_names.csv": feature_names_table,
            "bridge_pairs.csv": bridge_pairs,
            "bridge_seed_summary.csv": bridge_seeds,
        }
        for name, frame in table_frames.items():
            _write_csv(frame, table_dir / name)
        table_names = list(table_frames)
        figure_names = _make_figures(
            error_table,
            strata_table,
            observation_tables,
            importance_table,
            bridge_pairs,
            figure_dir,
        )

        observed = _observed_findings(
            domain_summary,
            cnn_help,
            hard_summary,
            bridge_summary,
            complementarity,
        )
        plausible = [
            (
                "The registered 197-feature representation exposes regional GC, "
                "composition, k-mer, entropy/complexity, and spacer-position "
                "information directly; its superior internal metrics are "
                "consistent with efficient tabular representation for short 30-mers."
            ),
            (
                "With 7,040 Domain A and 7,465 Domain B training observations, "
                "the fixed tree ensemble may use the engineered representation more "
                "statistically efficiently than the 17,985-parameter single-domain "
                "CNNs; Phase 18D does not isolate dataset size as the cause."
            ),
            (
                "The increased bridge head agreement is consistent with CNN_D "
                "learning a more shared cross-domain rank representation, while "
                "separate heads retain native assay scales."
            ),
        ]
        untested = [
            "Engineered features causally explain XGBoost's advantage.",
            "A larger dataset or different CNN architecture would reverse the ranking.",
            "XGBoost and CNN predictions would improve performance in an ensemble.",
            "Any reported sequence association is a causal biological determinant.",
            "Internal findings generalize to an external dataset or production use.",
        ]
        next_questions = [
            (
                "Preregister a controlled representation-ablation study to separate "
                "positional, k-mer, composition, and GC contributions."
            ),
            (
                "Preregister an ensemble experiment only if the observed residual "
                "correlations justify testing complementarity."
            ),
            (
                "Test whether bridge consistency predicts cross-domain utility in a "
                "larger independent paired-sequence set."
            ),
            (
                "Evaluate learning-curve and architecture questions in a later phase "
                "with fixed hypotheses and no post-hoc model selection."
            ),
        ]

        source_after = artifact_snapshot(campaign, manifest)
        canonical_after = verify_integrity()
        source_unchanged = source_before == source_after
        canonical_unchanged = canonical_before == canonical_after
        if not source_unchanged or not canonical_unchanged:
            raise RuntimeError("Canonical Phase 18C source integrity changed")

        summary = {
            "phase": "18D",
            "source_phase": "18C",
            "source_classification": SOURCE_CLASSIFICATION,
            "analysis_type": "EXPLORATORY_POST_HOC",
            "no_training": True,
            "no_moreno_access": True,
            "official_campaign": verification["campaign"],
            "domain_scale_policy": (
                "separate native activity scales; no cross-domain raw pooling"
            ),
            "residual_definition": "prediction_minus_observed",
            "seed_aggregation_policy": (
                "compute metrics per registered seed, then arithmetic mean; no "
                "seed-averaged prediction evaluation"
            ),
            "deterministic_policies": {
                "activity_bins": (
                    "within-domain balanced rank quartiles; sequence SHA-256 "
                    "tie-break"
                ),
                "gc_bins": ["<0.40", "0.40-0.60", ">0.60"],
                "error_difference_margin": "0.10 * within-domain target IQR",
                "both_accurate": "both absolute errors <= 0.10 * target IQR",
                "both_inaccurate": "both absolute errors >= 0.25 * target IQR",
                "hard_case": (
                    "top 10% within-domain consensus absolute error across four "
                    "families and five seeds"
                ),
                "minimum_position_group_n": MIN_POSITION_N,
                "minimum_correlation_n": MIN_CORRELATION_N,
            },
            "xgboost_interpretation": xgboost_summary,
            "cnn_interpretation": {
                "delta_absolute_error_definition": (
                    "abs_error_single_cnn - abs_error_cnn_d"
                ),
                "domain_patterns": cnn_help,
                "classification_unchanged": SOURCE_CLASSIFICATION,
            },
            "bridge_interpretation": bridge_summary,
            "error_complementarity": complementarity,
            "model_disagreement": disagreement,
            "hard_cases": hard_summary,
            "observed_findings": observed,
            "plausible_interpretations": plausible,
            "untested_hypotheses": untested,
            "recommended_next_questions": next_questions,
            "artifacts": {
                "figures": [f"figures/{name}" for name in figure_names],
                "tables": [f"tables/{name}" for name in table_names],
            },
            "canonical_integrity": {
                "official_campaign_identities_match": True,
                "verified_model_artifact_n": 35,
                "verified_prediction_artifact_n": 35,
                "source_artifacts_unchanged": source_unchanged,
                "canonical_inputs_unchanged": canonical_unchanged,
                "phase18c_outcome_read_not_reclassified": True,
                "phase18c_completed_run_n": 35,
                "protocol_sha256": PROTOCOL_SHA256,
                "implementation_sha256": IMPLEMENTATION_SHA256,
                "split_sha256": SPLIT_SHA256,
                "campaign_results_sha256": CAMPAIGN_RESULTS_SHA256,
                "execution_lock_sha256": EXECUTION_LOCK_SHA256,
                "artifact_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
            },
        }
        for domain in DOMAINS:
            domain_record = domain_summary[domain]
            summary[_domain_key(domain)] = {
                "model_error_summary": domain_record["model_error_summary"],
                "activity_strata": domain_record["activity_strata"],
                "gc_strata": {
                    "spacer": domain_record["spacer_gc_strata"],
                    "full_30mer": domain_record["full_gc_strata"],
                },
                "pam_strata": domain_record["pam_strata"],
                "sequence_complexity_strata": domain_record[
                    "sequence_complexity_strata"
                ],
                "position_specific_high_error_associations": (
                    _position_summary(position_table, domain)
                ),
                "cnn_d_help_patterns": cnn_help[domain],
                "hard_cases": hard_summary[domain],
            }

        (OUTPUT_ROOT / "summary.json").write_text(
            deterministic_json(summary), encoding="utf-8", newline="\n"
        )
        report = _render_report(
            summary,
            error_table,
            help_table,
            position_table,
            importance_table,
            bridge_seeds,
            figure_names,
            table_names,
        )
        (
            ROOT / "docs" / "phase18d_post_experiment_interpretation_report.md"
        ).write_text(report, encoding="utf-8", newline="\n")
        return summary

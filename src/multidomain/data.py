"""Locked data reconstruction and feature materialization for Phase 18C."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.bioinformatics.sequence_features import (
    SequenceFeatureExtractor,
    extract_one_hot_for_cnn,
)
from src.dataset_integration.label_compatibility import load_phase18a_inputs
from src.experiment_protocols.multidomain_protocol import (
    DOMAINS,
    EXPECTED_COUNTS,
    fingerprint,
    validate_locked_artifact,
)

SEQUENCE_COLUMNS = {
    DOMAINS[0]: "sequence_30mer",
    DOMAINS[1]: "30mer_gRNA",
}
LABEL_COLUMNS = {
    DOMAINS[0]: "activity",
    DOMAINS[1]: "HEK293T_indel_freq_avg_d8_d10",
}
DEVELOPMENT_PARTITIONS = ("train", "validation")
FROZEN_PARTITIONS = ("internal_test", "bridge")


@dataclass(frozen=True)
class TrainingStatistics:
    """Train-only native-unit moments used by output and loss conditioning."""

    mean: float
    sd: float
    variance: float
    n: int


class _EvaluationFreeze:
    """Opaque proof that every registered model checkpoint was frozen."""

    __slots__ = ("_owner", "checkpoint_ids", "protocol_sha256")

    def __init__(self, owner, checkpoint_ids, protocol_sha256):
        self._owner = owner
        self.checkpoint_ids = frozenset(checkpoint_ids)
        self.protocol_sha256 = protocol_sha256


def training_statistics(labels: Sequence[float]) -> TrainingStatistics:
    """Compute locked float64 population moments without changing labels."""
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Training labels must be a nonempty one-dimensional array"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Training labels must be finite")
    mean = float(np.mean(values, dtype=np.float64))
    variance = float(np.var(values, dtype=np.float64, ddof=0))
    sd = float(np.sqrt(variance))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError("Training-label SD is degenerate")
    return TrainingStatistics(
        mean=mean, sd=sd, variance=variance, n=values.size
    )


def validate_geometry(sequences: Sequence[str]) -> None:
    """Require the already-approved uppercase, unambiguous 30-mer geometry."""
    for sequence in sequences:
        if (
            not isinstance(sequence, str)
            or len(sequence) != 30
            or not set(sequence) <= set("ACGT")
            or sequence != sequence.upper()
        ):
            raise ValueError(
                "Phase 18C requires exact uppercase A/C/G/T 30-mers"
            )


def label_fingerprint(
    sequences: Sequence[str], labels: Sequence[float]
) -> str:
    """Reproduce the Phase 18B sorted raw float64 label fingerprint."""
    values = np.asarray(labels, dtype=np.float64)
    if len(sequences) != values.size or not np.all(np.isfinite(values)):
        raise ValueError("Cannot fingerprint mismatched or nonfinite labels")
    pairs = sorted(zip(sequences, (float(value).hex() for value in values)))
    return fingerprint(pairs)


@dataclass(frozen=True)
class DomainRows:
    sequences: np.ndarray
    labels: np.ndarray
    sequence_ids: np.ndarray
    partition_indices: Mapping[str, np.ndarray]


class LockedDevelopmentData:
    """Verified raw rows with an explicit internal-test/bridge freeze gate."""

    def __init__(
        self,
        domains: Mapping[str, DomainRows],
        protocol_sha256: str,
        expected_run_ids: Sequence[str],
    ):
        if set(domains) != set(DOMAINS):
            raise ValueError("Exactly the two locked domains are required")
        if (
            len(protocol_sha256) != 64
            or not expected_run_ids
            or len(expected_run_ids) != len(set(expected_run_ids))
        ):
            raise ValueError("Locked protocol/run inventory is invalid")
        self._domains = dict(domains)
        self._protocol_sha256 = protocol_sha256
        self._expected_run_ids = frozenset(expected_run_ids)
        self._freeze_owner = object()

    def freeze_evaluation(
        self, checkpoint_paths: Mapping[str, object]
    ) -> _EvaluationFreeze:
        """Issue a data-bound token after every expected checkpoint exists."""
        from pathlib import Path

        if set(checkpoint_paths) != self._expected_run_ids:
            raise RuntimeError(
                "Checkpoint inventory is incomplete or duplicated"
            )
        paths = [Path(path).resolve() for path in checkpoint_paths.values()]
        if len(set(paths)) != len(paths) or not all(
            path.is_file() for path in paths
        ):
            raise RuntimeError("Checkpoint file inventory mismatch")
        return _EvaluationFreeze(
            self._freeze_owner,
            self._expected_run_ids,
            self._protocol_sha256,
        )

    def partition(
        self,
        domain: str,
        partition: str,
        *,
        evaluation_freeze: _EvaluationFreeze | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if domain not in self._domains:
            raise ValueError(f"Unknown domain: {domain}")
        if partition == "quarantine":
            raise PermissionError("Quarantine rows cannot be materialized")
        if partition in FROZEN_PARTITIONS:
            valid_freeze = (
                isinstance(evaluation_freeze, _EvaluationFreeze)
                and evaluation_freeze._owner is self._freeze_owner
                and evaluation_freeze.protocol_sha256 == self._protocol_sha256
                and evaluation_freeze.checkpoint_ids == self._expected_run_ids
            )
            if not valid_freeze:
                raise PermissionError(
                    "Evaluation rows remain closed until checkpoint freeze"
                )
        if partition not in DEVELOPMENT_PARTITIONS + FROZEN_PARTITIONS:
            raise ValueError(f"Unknown partition: {partition}")
        rows = self._domains[domain]
        indices = rows.partition_indices[partition]
        return (
            rows.sequences[indices].copy(),
            rows.labels[indices].copy(),
            rows.sequence_ids[indices].copy(),
        )

    def statistics(self, domain: str) -> TrainingStatistics:
        _, labels, _ = self.partition(domain, "train")
        return training_statistics(labels)

    def label_fingerprints(self) -> dict[str, str]:
        """Fingerprint immutable raw labels across every locked row."""
        return {
            domain: label_fingerprint(rows.sequences.tolist(), rows.labels)
            for domain, rows in self._domains.items()
        }


def _domain_rows(frame, domain: str, manifest: list[dict]) -> DomainRows:
    sequence_column = SEQUENCE_COLUMNS[domain]
    label_column = LABEL_COLUMNS[domain]
    sequences = frame[sequence_column].tolist()
    validate_geometry(sequences)
    labels = frame[label_column].to_numpy(dtype=np.float64, copy=True)
    sequence_ids = np.asarray(
        [
            hashlib.sha256(sequence.encode("ascii")).hexdigest()
            for sequence in sequences
        ],
        dtype="U64",
    )
    if len(set(sequence_ids.tolist())) != len(sequence_ids):
        raise ValueError("Unexpected duplicate 30-mer identity")
    lookup = {
        identifier: index for index, identifier in enumerate(sequence_ids)
    }
    domain_manifest = [row for row in manifest if row["domain"] == domain]
    if len(domain_manifest) != len(sequences):
        raise ValueError("Locked manifest row count mismatch")
    manifest_ids = {row["sequence_sha256"] for row in domain_manifest}
    if manifest_ids != set(lookup):
        raise ValueError("Locked manifest sequence identity mismatch")
    partition_indices = {}
    for partition in EXPECTED_COUNTS[domain]:
        ids = sorted(
            row["sequence_sha256"]
            for row in domain_manifest
            if row["split"] == partition
        )
        partition_indices[partition] = np.asarray(
            [lookup[identifier] for identifier in ids], dtype=np.int64
        )
        if len(ids) != EXPECTED_COUNTS[domain][partition]:
            raise ValueError("Locked partition count mismatch")
    labels.setflags(write=False)
    sequences_array = np.asarray(sequences, dtype="U30")
    sequences_array.setflags(write=False)
    sequence_ids.setflags(write=False)
    return DomainRows(
        sequences=sequences_array,
        labels=labels,
        sequence_ids=sequence_ids,
        partition_indices=partition_indices,
    )


def load_locked_development_data(
    payload: dict, approved_digest: str
) -> LockedDevelopmentData:
    """Load approved A/B inputs and reconstruct every locked row identity."""
    validate_locked_artifact(payload, approved_digest)
    _, deep, xiang = load_phase18a_inputs()
    frames = {DOMAINS[0]: deep, DOMAINS[1]: xiang}
    domains = {
        domain: _domain_rows(frames[domain], domain, payload["split_manifest"])
        for domain in DOMAINS
    }
    run_ids = [
        specification["experiment_id"]
        for specification in payload["experiment_matrix"]
    ]
    data = LockedDevelopmentData(domains, payload["protocol_sha256"], run_ids)
    if data.label_fingerprints() != payload["label_fingerprints"]:
        raise RuntimeError("Locked raw-label fingerprint mismatch")
    return data


def materialize_cnn(sequences: Sequence[str]) -> np.ndarray:
    """Apply the canonical one-hot encoder and enforce its locked shape."""
    validate_geometry(sequences)
    features = extract_one_hot_for_cnn(list(sequences), context_length=30)
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (len(sequences), 30, 4):
        raise ValueError("Canonical CNN feature geometry changed")
    return features


def materialize_tabular(
    sequences: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Apply the canonical unfitted extractor and enforce 197 features."""
    validate_geometry(sequences)
    frame = SequenceFeatureExtractor().extract_features_batch(list(sequences))
    if frame.shape != (len(sequences), 197):
        raise ValueError("Canonical tabular feature count changed")
    values = frame.to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("Canonical tabular features must be finite")
    return values, frame.columns.tolist()

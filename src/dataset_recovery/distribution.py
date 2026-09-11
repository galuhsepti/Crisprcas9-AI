"""Phase 17E distribution and potential information-value audit.

Descriptive sequence composition only unless Phase 17C established a compatible
continuous activity label. No modeling, pooling, label transformation, sequence
transformation, Moreno access, or dataset acceptance.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.bioinformatics.kmer import extract_kmer_matrix, generate_kmers
from .independence import PHASE17C_RESULT, candidate_records_from_phase17b, load_json
from .phase17a import protected_hashes


PHASE17B_RESULT = Path("results/phase17b_primary_file_verification_20260911_143300.json")
PHASE17D_RESULT = Path("results/phase17d_independence_overlap_20260911_145754.json")

DATASET_TYPES = {
    "COMPATIBLE_CONTINUOUS_ACTIVITY",
    "SEQUENCE_ONLY_DESCRIPTIVE",
    "INCOMPATIBLE_LABEL_DATA",
    "INSUFFICIENT_DATA",
}

INFO_VALUE_STATUSES = {
    "POTENTIAL_INFORMATION_VALUE",
    "LIMITED_INFORMATION_VALUE",
    "NO_INFORMATION_VALUE_ESTABLISHED",
    "INSUFFICIENT_EVIDENCE",
}

NO_ACTIVITY = "NO_CANDIDATE_AVAILABLE_FOR_ACTIVITY_DISTRIBUTION_ANALYSIS"


def _is_dna(value: str) -> bool:
    return bool(value) and set(value.upper()) <= set("ACGT")


def gc_fraction(sequence: str) -> float:
    seq = sequence.upper()
    if not seq:
        return float("nan")
    return (seq.count("G") + seq.count("C")) / len(seq)


def descriptive(values: Sequence[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def sequence_distribution(sequences: Sequence[object], kmer: bool = True) -> Dict[str, object]:
    observed = ["" if seq is None else str(seq).strip().upper() for seq in sequences]
    missing = sum(1 for seq in observed if not seq or seq == "NAN")
    raw = [seq for seq in observed if seq and seq != "NAN"]
    valid = [seq for seq in raw if _is_dna(seq)]
    lengths = Counter(len(seq) for seq in raw)
    if not raw:
        return {
            "status": "NO_SEQUENCE_AVAILABLE",
            "total_records": len(observed),
            "records_with_valid_acgt_sequence": 0,
            "unique_sequences": 0,
            "missing_sequence_count": missing,
        }
    nuc_counts = Counter(base for seq in valid for base in seq)
    nuc_total = sum(nuc_counts.values())
    gc_values = [gc_fraction(seq) for seq in valid]
    out: Dict[str, object] = {
        "total_records": len(observed),
        "records_with_valid_acgt_sequence": len(valid),
        "unique_sequences": len(set(valid)),
        "missing_sequence_count": missing,
        "length_distribution": dict(sorted(lengths.items())),
        "mean_length": float(np.mean([len(seq) for seq in raw])) if raw else None,
        "median_length": float(np.median([len(seq) for seq in raw])) if raw else None,
        "gc": descriptive(gc_values),
        "nucleotide_composition": {
            base: float(nuc_counts.get(base, 0) / nuc_total) if nuc_total else 0.0
            for base in "ACGT"
        },
        "non_acgt_records": len(raw) - len(valid),
        "analysis_note": "Observed sequence geometry only; no 30-mer construction or sequence transformation.",
    }
    if kmer and valid:
        out["kmer_2"] = kmer_summary(valid, 2)
        out["kmer_3"] = kmer_summary(valid, 3)
    return out


def jensen_shannon_divergence(left: Dict[str, float], right: Dict[str, float]) -> float:
    """Descriptive Jensen-Shannon divergence for aligned frequency dictionaries."""
    keys = sorted(set(left) | set(right))
    p = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    q = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def kmer_summary(sequences: Sequence[str], k: int) -> Dict[str, object]:
    valid = [seq for seq in sequences if _is_dna(seq) and len(seq) >= k]
    if not valid:
        return {"status": "NOT_ASSESSED_NO_VALID_SEQUENCE", "k": k}
    matrix = extract_kmer_matrix(valid, k)
    means = matrix.mean(axis=0)
    kmers = generate_kmers(k)
    top = np.argsort(means)[::-1][:10]
    return {
        "k": k,
        "top_kmers": [{"kmer": kmers[i], "mean_frequency": float(means[i])} for i in top],
        "mean_frequencies": {kmers[i]: float(means[i]) for i in range(len(kmers))},
    }


def compare_gc(candidate_gc: Dict[str, object], canonical_gc: Dict[str, object]) -> Dict[str, object]:
    if not candidate_gc or candidate_gc.get("n", 0) == 0:
        return {"status": "NOT_ASSESSED"}
    return {
        "status": "DESCRIPTIVE_ONLY",
        "reference": "DeepSpCas9 Phase 16E distribution population (n=12,832)",
        "mean_gc_delta_candidate_minus_reference": float(candidate_gc["mean"] - canonical_gc["mean"]),
        "median_gc_delta_candidate_minus_reference": float(candidate_gc["median"] - canonical_gc["median"]),
    }


def sequence_comparison(candidate_sequence: Dict[str, object], canonical_sequence: Dict[str, object]) -> Dict[str, object]:
    if not candidate_sequence or candidate_sequence.get("status") == "NO_SEQUENCE_AVAILABLE":
        return {"status": "NOT_ASSESSED"}
    comparison = {
        "status": "DESCRIPTIVE_ONLY",
        "reference": "DeepSpCas9 Phase 16E distribution population (n=12,832)",
        "gc": compare_gc(candidate_sequence.get("gc", {}), canonical_sequence.get("gc", {})),
        "nucleotide_composition_jsd": jensen_shannon_divergence(
            candidate_sequence.get("nucleotide_composition", {}),
            canonical_sequence.get("nucleotide_composition", {}),
        ),
    }
    for key in ["kmer_2", "kmer_3"]:
        left = candidate_sequence.get(key, {}).get("mean_frequencies", {})
        right = canonical_sequence.get(key, {}).get("mean_frequencies", {})
        comparison[f"{key}_jsd"] = jensen_shannon_divergence(left, right) if left and right else "NOT_ASSESSED"
    return comparison


def canonical_deepspcas9_distribution(path: str = "data/raw/DeepSpCas9.csv") -> Dict[str, object]:
    df = pd.read_csv(path, usecols=["sequence_30mer", "activity"])
    seqs = [str(seq).strip().upper() for seq in df["sequence_30mer"]]
    return {
        "reference": "DeepSpCas9 Phase 16E distribution population",
        "n": 12832,
        "canonical_modeling_n": 10117,
        "train_n": 8599,
        "validation_n": 1518,
        "sequence": sequence_distribution(seqs, kmer=True),
        "activity": {
            "status": "CANONICAL_REFERENCE_ONLY",
            "descriptive": descriptive(df["activity"].astype(float).tolist()),
        },
    }


def classify_candidate(candidate_id: str, phase17c_record: Dict[str, object], has_sequences: bool) -> Dict[str, object]:
    label_status = phase17c_record["label_status"]
    seq_status = phase17c_record["sequence_status"]
    if not has_sequences:
        dataset_type = "INSUFFICIENT_DATA"
        info = "INSUFFICIENT_EVIDENCE"
    elif label_status == "LABEL_COMPATIBLE":
        dataset_type = "COMPATIBLE_CONTINUOUS_ACTIVITY"
        info = "POTENTIAL_INFORMATION_VALUE"
    elif label_status == "LABEL_INCOMPATIBLE":
        dataset_type = "INCOMPATIBLE_LABEL_DATA"
        info = "LIMITED_INFORMATION_VALUE" if seq_status != "SEQUENCE_INSUFFICIENT_EVIDENCE" else "INSUFFICIENT_EVIDENCE"
    elif has_sequences:
        dataset_type = "SEQUENCE_ONLY_DESCRIPTIVE"
        info = "LIMITED_INFORMATION_VALUE"
    else:
        dataset_type = "INSUFFICIENT_DATA"
        info = "INSUFFICIENT_EVIDENCE"
    return {
        "dataset_type": dataset_type,
        "information_value_status": info,
    }


def activity_distribution_for_candidate(phase17c_record: Dict[str, object]) -> Dict[str, object]:
    if phase17c_record["label_status"] != "LABEL_COMPATIBLE":
        return {
            "status": "NOT_ANALYZED_LABEL_NOT_COMPATIBLE",
            "reason": "Phase 17C did not establish LABEL_COMPATIBLE.",
        }
    return {
        "status": "NOT_IMPLEMENTED_NO_PHASE17C_LABEL_COMPATIBLE_CANDIDATES",
    }


def _activity_global_status(candidates: Sequence[Dict[str, object]]) -> str:
    if not any(record.get("label_status") == "LABEL_COMPATIBLE" for record in candidates):
        return NO_ACTIVITY
    return "ACTIVITY_DISTRIBUTION_AVAILABLE_FOR_COMPATIBLE_CANDIDATE"


def build_phase17e_distribution(
    phase17b_path: Path = PHASE17B_RESULT,
    phase17c_path: Path = PHASE17C_RESULT,
) -> Dict[str, object]:
    before = protected_hashes()
    phase17b = load_json(phase17b_path)
    phase17c = load_json(phase17c_path)
    records_by_candidate = candidate_records_from_phase17b(phase17b)
    c_by_id = {record["candidate_id"]: record for record in phase17c["candidates"]}
    canonical = canonical_deepspcas9_distribution()
    canonical_sequence = canonical["sequence"]
    candidates = []
    for candidate_id in sorted(c_by_id):
        records = records_by_candidate.get(candidate_id, [])
        seqs = [record["sequence"] for record in records if record.get("sequence")]
        classification = classify_candidate(candidate_id, c_by_id[candidate_id], bool(seqs))
        sequence_report = sequence_distribution(seqs, kmer=bool(seqs)) if seqs else {"status": "NO_SEQUENCE_AVAILABLE"}
        activity_report = activity_distribution_for_candidate(c_by_id[candidate_id])
        candidates.append({
            "candidate_id": candidate_id,
            "dataset_type": classification["dataset_type"],
            "verified_sequence_available": bool(seqs),
            "verified_continuous_activity_available": c_by_id[candidate_id]["label_status"] == "LABEL_COMPATIBLE",
            "analysis_performed": {
                "sequence_distribution": bool(seqs),
                "activity_distribution": c_by_id[candidate_id]["label_status"] == "LABEL_COMPATIBLE",
            },
            "sequence_distribution": sequence_report,
            "sequence_comparison_to_canonical": sequence_comparison(sequence_report, canonical_sequence) if bool(seqs) else {"status": "NOT_ASSESSED"},
            "activity_distribution": activity_report,
            "information_value_status": classification["information_value_status"],
            "information_value_reason": information_value_reason(classification["information_value_status"], classification["dataset_type"]),
        })
    after = protected_hashes()
    return {
        "phase": "17E",
        "protocol_version": "phase17e-distribution-information-value-v1",
        "scope": "distribution_and_potential_information_value_audit_only",
        "phase17b_source": str(phase17b_path).replace("\\", "/"),
        "phase17c_source": str(phase17c_path).replace("\\", "/"),
        "canonical_reference": {
            "distribution_population_n": 12832,
            "canonical_modeling_population_n": 10117,
            "train_n": 8599,
            "validation_n": 1518,
            "distribution_reference": canonical,
        },
        "candidates": candidates,
        "activity_distribution_global_status": _activity_global_status(list(c_by_id.values())),
        "information_value_statuses": sorted(INFO_VALUE_STATUSES),
        "dataset_types": sorted(DATASET_TYPES),
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_hyperparameter_tuning": True,
            "no_moreno_raw_data_accessed": True,
            "no_dataset_pooling": True,
            "no_combined_training_dataset": True,
            "no_retraining": True,
            "no_calibration": True,
            "no_adaptation_or_domain_weighting": True,
            "no_label_transformation": True,
            "no_label_harmonization": True,
            "no_sequence_transformation": True,
            "no_30mer_construction": True,
            "no_reverse_complement": True,
            "no_near_duplicate_analysis": True,
            "no_continual_learning": True,
            "no_dataset_acceptance": True,
        },
        "safety_guards": {
            "binary_labels_not_activity": True,
            "rank_enrichment_logfc_not_activity": True,
            "guide_or_23mer_not_converted_to_30mer": True,
            "kmer_not_overlap_criterion": True,
            "no_generalization_claim": True,
        },
    }


def information_value_reason(status: str, dataset_type: str) -> str:
    if status == "LIMITED_INFORMATION_VALUE":
        return "Sequence composition can be described, but labels are incompatible or not usable as thesis activity."
    if status == "INSUFFICIENT_EVIDENCE":
        return "Verified local sequence/activity evidence is insufficient."
    if status == "POTENTIAL_INFORMATION_VALUE":
        return "Potential only; no modeling evidence. Requires later gates."
    return "No additional sequence/activity coverage established."


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

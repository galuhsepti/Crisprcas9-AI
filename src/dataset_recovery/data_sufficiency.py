"""Phase 17G data sufficiency and readiness assessment.

This module consumes Phase 17 evidence artifacts only. It defines configurable
project-specific GO/NO-GO thresholds and performs no model training, pooling,
label rescue, sequence construction, Moreno access, or Phase 18 work.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from .acceptance import PHASE17B_RESULT, PHASE17C_RESULT, PHASE17D_RESULT, PHASE17E_RESULT
from .phase17a import protected_hashes


DEFAULT_THRESHOLDS = {
    "minimum_candidate_unique_n": 1000,
    "recommended_candidate_unique_n": 2000,
    "minimum_total_new_unique_n": 2000,
    "minimum_continuous_label_unique_values": 20,
}

FINAL_DECISIONS = {"GO", "CONDITIONAL_GO", "NO_GO", "INSUFFICIENT_EVIDENCE"}
PHASE17F_RESULT = Path("results/phase17f_dataset_acceptance_gate_20260911_203808.json")


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_data_readiness_thresholds(path: Path = Path("config.yaml")) -> Dict[str, int]:
    """Read the small config block without introducing a YAML dependency."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    if not path.exists():
        return thresholds
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False
    for raw in lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("data_readiness:"):
            in_block = True
            continue
        if in_block and not raw.startswith(" "):
            break
        if in_block and ":" in line:
            key, value = line.strip().split(":", 1)
            if key in thresholds:
                thresholds[key] = int(value.strip())
    return thresholds


def is_valid_dna(value: object) -> bool:
    seq = str(value).strip().upper()
    return bool(seq) and set(seq) <= set("ACGT")


def sequence_usability(sequences: Sequence[object]) -> Dict[str, object]:
    observed = ["" if seq is None else str(seq).strip().upper() for seq in sequences]
    nonmissing = [seq for seq in observed if seq and seq != "NAN"]
    valid = [seq for seq in nonmissing if is_valid_dna(seq)]
    malformed = len(nonmissing) - len(valid)
    lengths = Counter(len(seq) for seq in nonmissing)
    return {
        "raw_n": len(observed),
        "valid_sequence_n": len(valid),
        "unique_sequence_n": len(set(valid)),
        "duplicate_sequence_n": len(valid) - len(set(valid)),
        "length_distribution": dict(sorted(lengths.items())),
        "malformed_sequence_n": malformed,
        "ambiguous_base_n": malformed,
        "has_usable_sequence_evidence": bool(valid),
    }


def label_quality(
    labels: Sequence[object],
    label_status: str,
    thresholds: Dict[str, int] | None = None,
    canonical_scale_established: bool = False,
) -> Dict[str, object]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if label_status != "LABEL_COMPATIBLE":
        return {
            "status": "INCOMPATIBLE_TARGET" if label_status == "LABEL_INCOMPATIBLE" else "UNKNOWN_SCALE",
            "compatible_label_n": 0,
            "unique_label_values": 0,
            "canonical_0_1_scale_directly_established": False,
            "reason": "Phase evidence did not establish compatible continuous activity labels.",
        }
    values = np.asarray([float(v) for v in labels if v is not None and np.isfinite(float(v))], dtype=float)
    missing = len(labels) - len(values)
    if values.size == 0:
        return {"status": "UNKNOWN_SCALE", "compatible_label_n": 0, "missing_labels": missing, "unique_label_values": 0}
    unique_values = int(np.unique(values).size)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if unique_values == 1:
        status = "CONSTANT_LABEL"
    elif unique_values < thresholds["minimum_continuous_label_unique_values"]:
        status = "DISCRETE_LOW_CARDINALITY"
    elif float(np.std(values, ddof=0)) < 1e-9:
        status = "NEAR_CONSTANT_LABEL"
    elif not canonical_scale_established:
        status = "UNKNOWN_SCALE"
    else:
        status = "CONTINUOUS_USABLE"
    return {
        "status": status,
        "n_labels": len(labels),
        "missing_labels": missing,
        "finite_labels": int(values.size),
        "compatible_label_n": int(values.size) if status == "CONTINUOUS_USABLE" else 0,
        "min": min_value,
        "max": max_value,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "unique_label_values": unique_values,
        "fraction_at_minimum": float(np.mean(values == min_value)),
        "fraction_at_maximum": float(np.mean(values == max_value)),
        "quantiles": {str(q): float(np.quantile(values, q)) for q in [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]},
        "canonical_0_1_scale_directly_established": canonical_scale_established,
    }


def quantity_gate(unique_compatible_n: int, thresholds: Dict[str, int] | None = None) -> Dict[str, object]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if unique_compatible_n < thresholds["minimum_candidate_unique_n"]:
        status = "INSUFFICIENT_QUANTITY"
    elif unique_compatible_n < thresholds["recommended_candidate_unique_n"]:
        status = "MINIMUM_ONLY"
    else:
        status = "RECOMMENDED_QUANTITY_MET"
    return {
        "status": status,
        "unique_compatible_observations": int(unique_compatible_n),
        "minimum_candidate_unique_n": thresholds["minimum_candidate_unique_n"],
        "recommended_candidate_unique_n": thresholds["recommended_candidate_unique_n"],
        "threshold_type": "project_specific_policy_not_universal_crispr_rule",
    }


def exact_overlap_summary(candidate: Iterable[str], canonical: Iterable[str]) -> Dict[str, object]:
    left = {str(seq).strip().upper() for seq in candidate if is_valid_dna(seq)}
    right = {str(seq).strip().upper() for seq in canonical if is_valid_dna(seq)}
    shared = left & right
    return {
        "candidate_unique_n": len(left),
        "canonical_unique_n": len(right),
        "exact_overlap_n": len(shared),
        "exact_overlap_fraction": float(len(shared) / len(left)) if left else 0.0,
    }


def final_readiness_decision(candidates: Sequence[Dict[str, object]], thresholds: Dict[str, int] | None = None) -> Dict[str, object]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    accepted = [c for c in candidates if c["final_candidate_status"] == "ACCEPTED_READY_CANDIDATE"]
    total = sum(int(c["unique_sequence_n"]) for c in accepted)
    if not accepted:
        return {
            "final_decision": "NO_GO",
            "accepted_candidate_count": 0,
            "total_new_unique_compatible_observations": 0,
            "decision_reasons": ["No Phase 17 candidate satisfies provenance, compatibility, label, independence, quantity, and information-value gates."],
        }
    if total < thresholds["minimum_total_new_unique_n"]:
        return {
            "final_decision": "NO_GO",
            "accepted_candidate_count": len(accepted),
            "total_new_unique_compatible_observations": total,
            "decision_reasons": [f"Total accepted unique compatible observations {total} is below project threshold {thresholds['minimum_total_new_unique_n']}."],
        }
    return {
        "final_decision": "GO",
        "accepted_candidate_count": len(accepted),
        "total_new_unique_compatible_observations": total,
        "decision_reasons": ["Project-specific data-readiness thresholds are met."],
    }


def _gate_status(candidate_id: str, c: Dict[str, object], d: Dict[str, object], f: Dict[str, object]) -> Dict[str, object]:
    primary = f["final_acceptance_status"] not in {"INSUFFICIENT_EVIDENCE", "ALREADY_ANALYZED_NOT_NEW"}
    if candidate_id in {"crispron_2021_rth_tools", "sgdesigner_2020_public_data_listing", "corsi_2022_free_energy_pam_context"}:
        provenance = "INSUFFICIENT_EVIDENCE"
    elif candidate_id == "deep_hf_2019_public_data_listing":
        provenance = "FAIL"
    else:
        provenance = "PASS" if primary else "INSUFFICIENT_EVIDENCE"
    task = "COMPATIBLE" if c["sequence_status"] == "SEQUENCE_COMPATIBLE" and c["label_status"] == "LABEL_COMPATIBLE" else (
        "INSUFFICIENT_EVIDENCE" if "INSUFFICIENT" in c["sequence_status"] or "INSUFFICIENT" in c["label_status"] else "INCOMPATIBLE"
    )
    independence_status = str(d.get("provenance_relationship", {}).get("status", "INSUFFICIENT_PROVENANCE"))
    independence = "PASS" if independence_status == "INDEPENDENCE_ESTABLISHED" else (
        "FAIL" if independence_status in {"DERIVED_OR_PROCESSED_RELATIONSHIP", "SHARED_SOURCE_STUDY"} else "INSUFFICIENT_EVIDENCE"
    )
    return {
        "provenance_gate": {"status": provenance},
        "task_compatibility_gate": {"status": task},
        "independence_gate": {"status": independence, "evidence_status": independence_status},
    }


def _candidate_record(
    candidate_id: str,
    c: Dict[str, object],
    d: Dict[str, object],
    e: Dict[str, object],
    f: Dict[str, object],
    thresholds: Dict[str, int],
) -> Dict[str, object]:
    gates = _gate_status(candidate_id, c, d, f)
    seq = e.get("sequence_distribution", {})
    raw_n = int(seq.get("total_records", 0) or 0) if isinstance(seq, dict) else 0
    valid_n = int(seq.get("records_with_valid_acgt_sequence", 0) or 0) if isinstance(seq, dict) else 0
    unique_n = int(seq.get("unique_sequences", 0) or 0) if isinstance(seq, dict) else 0
    length_distribution = seq.get("length_distribution", {}) if isinstance(seq, dict) else {}
    exact_30 = d.get("exact_sequence_overlap_with_canonical_valid_30mer", {})
    overlap_n = int(exact_30.get("shared_unique", 0)) if isinstance(exact_30, dict) else 0
    overlap_fraction = float(overlap_n / unique_n) if unique_n else 0.0
    label = label_quality([], str(c.get("label_status")), thresholds)
    compatible_unique = unique_n if label["status"] == "CONTINUOUS_USABLE" and c["sequence_status"] == "SEQUENCE_COMPATIBLE" else 0
    quantity = quantity_gate(compatible_unique, thresholds)
    information = e.get("information_value_status", "INSUFFICIENT_EVIDENCE")
    if information == "LIMITED_INFORMATION_VALUE":
        information_status = "LOW"
    elif information == "POTENTIAL_INFORMATION_VALUE":
        information_status = "MODERATE"
    else:
        information_status = "INSUFFICIENT_EVIDENCE"
    final_status = "ACCEPTED_READY_CANDIDATE" if (
        gates["provenance_gate"]["status"] == "PASS"
        and gates["task_compatibility_gate"]["status"] == "COMPATIBLE"
        and gates["independence_gate"]["status"] == "PASS"
        and label["status"] == "CONTINUOUS_USABLE"
        and quantity["status"] in {"MINIMUM_ONLY", "RECOMMENDED_QUANTITY_MET"}
        and information_status in {"MODERATE", "HIGH"}
    ) else "NOT_READY"
    reasons = []
    if gates["provenance_gate"]["status"] != "PASS":
        reasons.append("Provenance gate did not pass.")
    if gates["task_compatibility_gate"]["status"] != "COMPATIBLE":
        reasons.append("Biological/task compatibility gate did not pass.")
    if label["status"] != "CONTINUOUS_USABLE":
        reasons.append("Compatible continuous label quality was not established.")
    if gates["independence_gate"]["status"] != "PASS":
        reasons.append("Independence/leakage gate did not pass.")
    if quantity["status"] == "INSUFFICIENT_QUANTITY":
        reasons.append("Project-specific unique compatible observation threshold was not met.")
    return {
        "candidate_id": candidate_id,
        **gates,
        "sequence_gate": {
            "status": "PASS" if valid_n else "INSUFFICIENT_EVIDENCE",
            "raw_n": raw_n,
            "valid_sequence_n": valid_n,
            "unique_sequence_n": unique_n,
            "sequence_length_distribution": length_distribution,
            "no_30mer_construction": True,
        },
        "label_gate": label,
        "quantity_gate": quantity,
        "information_value_gate": {
            "status": information_status,
            "phase17e_status": information,
            "reason": "Sequence-only descriptive value is not enough without compatible labels and provenance.",
        },
        "raw_n": raw_n,
        "valid_sequence_n": valid_n,
        "unique_sequence_n": unique_n,
        "compatible_label_n": label.get("compatible_label_n", 0),
        "unique_label_values": label.get("unique_label_values", 0),
        "exact_deepsp_overlap_n": overlap_n,
        "exact_deepsp_overlap_fraction": overlap_fraction,
        "metadata_coverage": {
            "distinct_genes": "NOT_ESTABLISHED",
            "distinct_loci": "NOT_ESTABLISHED",
            "cell_types": "NOT_ESTABLISHED_OR_FILE_LEVEL_ONLY",
            "experiments": "NOT_ESTABLISHED_OR_FILE_LEVEL_ONLY",
            "publications_or_source_screens": 1 if raw_n else 0,
        },
        "final_candidate_status": final_status,
        "phase17f_acceptance_status": f["final_acceptance_status"],
        "reasons": reasons,
    }


def build_phase17g_data_sufficiency(
    thresholds: Dict[str, int] | None = None,
    timestamp: str | None = None,
    phase17c_path: Path = PHASE17C_RESULT,
    phase17d_path: Path = PHASE17D_RESULT,
    phase17e_path: Path = PHASE17E_RESULT,
    phase17f_path: Path = PHASE17F_RESULT,
) -> Dict[str, object]:
    thresholds = thresholds or load_data_readiness_thresholds()
    before = protected_hashes()
    phase17c = load_json(phase17c_path)
    phase17d = load_json(phase17d_path)
    phase17e = load_json(phase17e_path)
    phase17f = load_json(phase17f_path)
    c_by_id = {r["candidate_id"]: r for r in phase17c["candidates"]}
    d_by_id = {r["candidate_id"]: r for r in phase17d["candidate_reports"]}
    e_by_id = {r["candidate_id"]: r for r in phase17e["candidates"]}
    f_by_id = {r["candidate_id"]: r for r in phase17f["candidate_decisions"]}
    candidates = [
        _candidate_record(cid, c_by_id[cid], d_by_id[cid], e_by_id[cid], f_by_id[cid], thresholds)
        for cid in sorted(c_by_id)
    ]
    aggregate = final_readiness_decision(candidates, thresholds)
    after = protected_hashes()
    return {
        "phase": "17G",
        "timestamp": timestamp,
        "scope": {
            "type": "data_sufficiency_and_readiness_assessment_only",
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_moreno_access": True,
            "no_pooling": True,
            "no_label_transformation": True,
            "no_sequence_transformation": True,
            "no_phase18": True,
        },
        "thresholds": thresholds,
        "canonical_reference": {
            "dataset": "DeepSpCas9",
            "phase16e_distribution_population_n": 12832,
            "canonical_modeling_population_n": 10117,
            "train_n": 8599,
            "validation_n": 1518,
            "locked_external_dataset": "LOCKED_EXTERNAL_DATASET_NOT_ACCESSED",
        },
        "candidates": candidates,
        "aggregate": aggregate,
        "final_decision": aggregate["final_decision"],
        "decision_reasons": aggregate["decision_reasons"],
        "additional_compatible_unique_observations_required": max(0, thresholds["minimum_total_new_unique_n"] - aggregate["total_new_unique_compatible_observations"]),
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
    }


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

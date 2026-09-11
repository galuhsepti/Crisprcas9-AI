"""Phase 17C sequence and label compatibility audit.

This module consumes Phase 17B primary-file verification evidence. It records
compatibility against the thesis target without constructing 30-mers, changing
labels, pooling datasets, overlap analysis, Moreno access, or model work.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from .phase17a import protected_hashes


PHASE17B_RESULT = Path("results/phase17b_primary_file_verification_20260911_143300.json")

SEQUENCE_STATUSES = {
    "SEQUENCE_COMPATIBLE",
    "SEQUENCE_PARTIALLY_COMPATIBLE",
    "SEQUENCE_INCOMPATIBLE",
    "SEQUENCE_INSUFFICIENT_EVIDENCE",
}

LABEL_STATUSES = {
    "LABEL_COMPATIBLE",
    "LABEL_POTENTIALLY_COMPATIBLE_NEEDS_SEMANTIC_VERIFICATION",
    "LABEL_INCOMPATIBLE",
    "LABEL_INSUFFICIENT_EVIDENCE",
}

OVERALL_STATUSES = {
    "COMPATIBLE_FOR_FUTURE_AUDIT",
    "PARTIALLY_COMPATIBLE",
    "INCOMPATIBLE_FOR_CURRENT_TARGET",
    "INSUFFICIENT_EVIDENCE",
}


def load_phase17b_payload(path: Path = PHASE17B_RESULT) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _length_distribution_from_summary(summary: Dict[str, object]) -> Dict[int, int]:
    lengths = summary.get("observed_lengths", {})
    return {int(k): int(v) for k, v in lengths.items()}


def characterize_sequence_column(column_name: str, summary: Dict[str, object]) -> Dict[str, object]:
    """Describe sequence structure only, without transformation."""
    lengths = _length_distribution_from_summary(summary)
    row_count = int(summary.get("observed_record_count", 0) or 0)
    non_acgt = int(summary.get("non_acgt_count", 0) or 0)
    min_len = min(lengths) if lengths else None
    max_len = max(lengths) if lengths else None
    fraction_acgt = ((row_count - non_acgt) / row_count) if row_count else None
    if lengths == {30: row_count} and non_acgt == 0:
        geometry = "OBSERVED_30MER_UNVERIFIED_COORDINATES"
        status = "SEQUENCE_PARTIALLY_COMPATIBLE"
        reason = "30 nt sequences observed, but guide/PAM/flank coordinates and orientation are not verified."
    elif lengths and set(lengths) <= {20, 21}:
        geometry = "GUIDE_ONLY_OR_SHORT_GUIDE"
        status = "SEQUENCE_INCOMPATIBLE"
        reason = "Guide-only/short-guide sequence lacks canonical flanking context and PAM."
    elif lengths and set(lengths) == {23}:
        geometry = "GUIDE_PLUS_PAM_23MER_OR_UNVERIFIED_23MER"
        status = "SEQUENCE_INCOMPATIBLE"
        reason = "23-mer sequence cannot be mapped to canonical 30-mer without adding flanking context."
    elif not lengths:
        geometry = "NO_SEQUENCE_COLUMN_VALUES"
        status = "SEQUENCE_INSUFFICIENT_EVIDENCE"
        reason = "No sequence values available."
    else:
        geometry = "NONCANONICAL_OR_MIXED_LENGTH"
        status = "SEQUENCE_INCOMPATIBLE"
        reason = "Observed lengths do not establish canonical 30-mer geometry."
    return {
        "column_name": column_name,
        "rows_with_sequence": row_count,
        "missing_sequence_count": summary.get("missing_sequence_count"),
        "unique_sequence_count": "NOT_COMPUTED_NO_OVERLAP_OR_DEDUP_GATE",
        "min_length": min_len,
        "max_length": max_len,
        "length_distribution": lengths,
        "fraction_acgt_only": fraction_acgt,
        "non_acgt_count": non_acgt,
        "orientation_documented": summary.get("orientation_documented", "UNKNOWN"),
        "includes_pam": summary.get("includes_pam", "UNKNOWN"),
        "includes_flanking_nucleotides": summary.get("includes_flanking_nucleotides", "UNKNOWN"),
        "observed_geometry": geometry,
        "sequence_status": status,
        "reason": reason,
        "evidence_source": "file contents and Phase 17B structure inspection",
    }


def characterize_label_column(column_name: str, summary: Dict[str, object]) -> Dict[str, object]:
    """Describe label structure only, without rescaling or harmonization."""
    label_type = str(summary.get("label_type_observed", "unknown"))
    documented = str(summary.get("documented_biological_meaning", "UNKNOWN"))
    numeric = summary.get("numeric_summary", {})
    if label_type == "binary":
        status = "LABEL_INCOMPATIBLE"
        reason = "Binary classification label is not a continuous experimentally measured activity label."
    elif any(token in label_type for token in ["rank", "enrichment", "fold", "statistic"]):
        status = "LABEL_INCOMPATIBLE"
        reason = "Rank/enrichment/log-fold-change/screen statistics are not canonical continuous activity labels."
    elif documented in {"UNKNOWN", "UNKNOWN_FROM_FILE_ONLY"}:
        status = "LABEL_INSUFFICIENT_EVIDENCE"
        reason = "Numeric or categorical label semantics are not established by source evidence."
    else:
        status = "LABEL_POTENTIALLY_COMPATIBLE_NEEDS_SEMANTIC_VERIFICATION"
        reason = "Source semantics require later compatibility review; no acceptance in Phase 17C."
    return {
        "column_name": column_name,
        "dtype": summary.get("dtype"),
        "missing_count": summary.get("missing_count"),
        "unique_summary": summary.get("unique_summary"),
        "numeric_summary": numeric,
        "documented_unit": summary.get("documented_unit"),
        "documented_biological_meaning": documented,
        "experimentally_measured_explicit": summary.get("experimentally_measured_explicit"),
        "processed_or_derived": "NOT_ESTABLISHED_OR_PROCESSED_CONTEXT_RETAINED",
        "model_generated": False,
        "label_status": status,
        "reason": reason,
        "evidence_source": "file contents and Phase 17B structure inspection",
    }


def _collect_file_sequence_results(file_record: Dict[str, object]) -> List[Dict[str, object]]:
    inspection = file_record["inspection"]
    file_path = file_record["exact_local_filename"]
    out: List[Dict[str, object]] = []
    if inspection["file_format"] == "csv":
        for column, summary in inspection.get("sequence_columns", {}).items():
            result = characterize_sequence_column(column, summary)
            result["file"] = file_path
            out.append(result)
    elif inspection["file_format"] == "xlsx":
        for sheet in inspection.get("sheets", []):
            for column, summary in sheet.get("sequence_columns", {}).items():
                result = characterize_sequence_column(column, summary)
                result["file"] = file_path
                result["sheet_name"] = sheet["sheet_name"]
                out.append(result)
    elif inspection["file_format"] == "zip":
        out.append({
            "file": file_path,
            "observed_geometry": "ARCHIVE_NO_VERIFIED_EXPERIMENTAL_SEQUENCE_TABLE",
            "sequence_status": "SEQUENCE_INSUFFICIENT_EVIDENCE",
            "reason": "CRISPRon archive contains code/models/test outputs; primary experimental sequence table not identified.",
            "data_like_files": inspection.get("data_like_files", []),
            "candidate_sequence_tables": inspection.get("candidate_sequence_tables", []),
        })
    return out


def _collect_file_label_results(file_record: Dict[str, object]) -> List[Dict[str, object]]:
    inspection = file_record["inspection"]
    file_path = file_record["exact_local_filename"]
    out: List[Dict[str, object]] = []
    if inspection["file_format"] == "csv":
        for column, summary in inspection.get("label_columns", {}).items():
            result = characterize_label_column(column, summary)
            result["file"] = file_path
            out.append(result)
    elif inspection["file_format"] == "xlsx":
        for sheet in inspection.get("sheets", []):
            for column, summary in sheet.get("label_columns", {}).items():
                result = characterize_label_column(column, summary)
                result["file"] = file_path
                result["sheet_name"] = sheet["sheet_name"]
                out.append(result)
    elif inspection["file_format"] == "zip":
        out.append({
            "file": file_path,
            "label_status": "LABEL_INSUFFICIENT_EVIDENCE",
            "reason": "CRISPRon archive does not identify a primary experimental activity label table.",
            "data_like_files": inspection.get("data_like_files", []),
            "candidate_label_tables": inspection.get("candidate_label_tables", []),
        })
    return out


def _aggregate_status(statuses: Iterable[str], allowed: set[str], insufficient: str) -> str:
    observed = [s for s in statuses if s in allowed]
    if not observed:
        return insufficient
    if any("INCOMPATIBLE" in s for s in observed):
        return next(s for s in observed if "INCOMPATIBLE" in s)
    if any("INSUFFICIENT" in s for s in observed):
        return next(s for s in observed if "INSUFFICIENT" in s)
    if any("PARTIALLY" in s or "POTENTIALLY" in s for s in observed):
        return next(s for s in observed if "PARTIALLY" in s or "POTENTIALLY" in s)
    return observed[0]


def _overall(sequence_status: str, label_status: str) -> str:
    if sequence_status == "SEQUENCE_INSUFFICIENT_EVIDENCE" or label_status == "LABEL_INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE"
    if sequence_status == "SEQUENCE_INCOMPATIBLE" or label_status == "LABEL_INCOMPATIBLE":
        return "INCOMPATIBLE_FOR_CURRENT_TARGET"
    if sequence_status == "SEQUENCE_COMPATIBLE" and label_status == "LABEL_COMPATIBLE":
        return "COMPATIBLE_FOR_FUTURE_AUDIT"
    return "PARTIALLY_COMPATIBLE"


def _candidate_overrides(candidate_id: str, sequence_status: str, label_status: str) -> tuple[str, str]:
    if candidate_id == "crispron_2021_rth_tools":
        return "SEQUENCE_INSUFFICIENT_EVIDENCE", "LABEL_INSUFFICIENT_EVIDENCE"
    if candidate_id in {"deep_hf_2019_public_data_listing", "sgdesigner_2020_public_data_listing", "corsi_2022_free_energy_pam_context"}:
        return "SEQUENCE_INSUFFICIENT_EVIDENCE", "LABEL_INSUFFICIENT_EVIDENCE"
    return sequence_status, label_status


def build_phase17c_compatibility(phase17b_path: Path = PHASE17B_RESULT) -> Dict[str, object]:
    phase17b = load_phase17b_payload(phase17b_path)
    before = protected_hashes()
    records = []
    for candidate in phase17b["candidates"]:
        seq_results: List[Dict[str, object]] = []
        label_results: List[Dict[str, object]] = []
        for file_record in candidate.get("files", []):
            seq_results.extend(_collect_file_sequence_results(file_record))
            label_results.extend(_collect_file_label_results(file_record))
        sequence_status = _aggregate_status(
            [r.get("sequence_status", "") for r in seq_results],
            SEQUENCE_STATUSES,
            "SEQUENCE_INSUFFICIENT_EVIDENCE",
        )
        label_status = _aggregate_status(
            [r.get("label_status", "") for r in label_results],
            LABEL_STATUSES,
            "LABEL_INSUFFICIENT_EVIDENCE",
        )
        sequence_status, label_status = _candidate_overrides(str(candidate["candidate_id"]), sequence_status, label_status)
        overall = _overall(sequence_status, label_status)
        records.append({
            "candidate_id": candidate["candidate_id"],
            "publication": candidate["publication"],
            "repository_accession": candidate["repository_accession"],
            "sequence_status": sequence_status,
            "observed_sequence_geometry": _summarize_geometry(seq_results),
            "sequence_file_results": seq_results,
            "label_status": label_status,
            "label_semantics": _summarize_label_semantics(label_results),
            "label_file_results": label_results,
            "overall_status": overall,
            "key_evidence": _key_evidence(candidate, seq_results, label_results),
            "unresolved_issues": sorted(set(candidate.get("unresolved_issues", []))),
        })
    after = protected_hashes()
    return {
        "phase": "17C",
        "protocol_version": "phase17c-sequence-label-compatibility-v1",
        "scope": "sequence_and_label_compatibility_audit_only_no_acceptance",
        "phase17b_source": str(phase17b_path).replace("\\", "/"),
        "candidate_count": len(records),
        "candidates": records,
        "status_vocabularies": {
            "sequence_statuses": sorted(SEQUENCE_STATUSES),
            "label_statuses": sorted(LABEL_STATUSES),
            "overall_statuses": sorted(OVERALL_STATUSES),
        },
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
        "safety_guards": {
            "numeric_range_not_activity_without_semantics": True,
            "binary_not_converted_to_continuous": True,
            "guide_only_not_converted_to_30mer": True,
            "pam_or_flanks_not_invented": True,
            "rank_enrichment_logfc_not_activity": True,
            "model_output_not_experimental_measurement": True,
            "processed_benchmark_not_independent_primary": True,
        },
        "non_modeling_guards": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_hyperparameter_tuning": True,
            "no_moreno_raw_data_accessed": True,
            "no_dataset_pooling": True,
            "no_combined_training_dataset": True,
            "no_label_transformation": True,
            "no_label_harmonization": True,
            "no_sequence_transformation": True,
            "no_30mer_construction": True,
            "no_near_duplicate_analysis": True,
            "no_rc_overlap": True,
            "no_continual_learning": True,
            "no_dataset_acceptance": True,
        },
    }


def _summarize_geometry(results: List[Dict[str, object]]) -> str:
    if not results:
        return "NO_RECOVERED_SEQUENCE_EVIDENCE"
    geometries = sorted({str(r.get("observed_geometry", "UNKNOWN")) for r in results})
    return "; ".join(geometries)


def _summarize_label_semantics(results: List[Dict[str, object]]) -> str:
    if not results:
        return "NO_RECOVERED_LABEL_EVIDENCE"
    reasons = sorted({str(r.get("reason", "UNKNOWN")) for r in results})
    return "; ".join(reasons[:5])


def _key_evidence(candidate: Dict[str, object], seq: List[Dict[str, object]], label: List[Dict[str, object]]) -> List[str]:
    evidence = [
        "Phase 17B primary-file verification JSON",
        str(candidate.get("primary_vs_derived_processed_status")),
        str(candidate.get("dataset_relationship")),
    ]
    if seq:
        evidence.append("sequence evidence from file contents")
    if label:
        evidence.append("label evidence from file contents and source semantics recorded in Phase 17B")
    return evidence


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

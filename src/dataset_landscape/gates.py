"""Phase 16F final dataset-readiness gate.

This module consumes frozen Phase 16B-16E JSON artifacts only. It makes no
new sequence, label, overlap, distribution, or modeling calculation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


PROTOCOL_VERSION = "phase16f-dataset-gate-v1"
ALLOWED_STATUSES = {
    "ACCEPTED_FOR_FUTURE_PIPELINE",
    "CONDITIONAL_NEEDS_PROVENANCE",
    "CONDITIONAL_NEEDS_COMPATIBILITY",
    "CONDITIONAL_NEEDS_INDEPENDENCE",
    "REJECTED_FOR_CURRENT_TARGET",
    "INSUFFICIENT_EVIDENCE",
    "ALREADY_ANALYZED_NOT_NEW",
}

DEFAULT_ARTIFACTS = {
    "phase16b": Path("results/phase16b_primary_file_verification_20260908_151501.json"),
    "phase16c": Path("results/phase16c_compatibility_audit_20260908_152340.json"),
    "phase16d": Path("results/phase16d_overlap_independence_audit_20260908_153958.json"),
    "phase16e": Path("results/phase16e_distribution_audit_20260908_170415.json"),
}

DECISIONS = {
    "doench_2016_orcs_publication_screens": {
        "final_decision": "REJECTED_FOR_CURRENT_TARGET",
        "reason": (
            "Phase 16C identifies screen rank/enrichment/log-fold-change label semantics "
            "and guide-only geometry; accepting it would require prohibited label "
            "harmonization and sequence transformation."
        ),
        "unresolved_questions": [
            "Phase 16D reports 214,820 sequence records while Phase 16E reports 194,653 usable sequence records; the discrepancy was not reconciled in Phase 16F.",
            "PAM and orientation remain unverified; reverse-complement overlap was not assessed.",
        ],
    },
    "crisprpredseq_2020_bmc_additional_files": {
        "final_decision": "REJECTED_FOR_CURRENT_TARGET",
        "reason": (
            "Phase 16C identifies binary labels and 23-mer guide+PAM geometry, "
            "which are not the canonical continuous 30-mer activity label; no rescue "
            "or transformation is permitted."
        ),
        "unresolved_questions": [
            "Primary parent/processed relationships remain incompletely established.",
            "Orientation remains unverified; reverse-complement overlap was not assessed.",
        ],
    },
    "crispron_2021_rth_tools": {
        "final_decision": "INSUFFICIENT_EVIDENCE",
        "reason": "No primary experimental table or verified sequence file was available in Phase 16B-16E.",
        "unresolved_questions": [
            "Primary file, SHA-256, row schema, sequence geometry, label units, and parent relationships require verification.",
            "Overlap and distribution evidence cannot be assessed without verified primary data.",
        ],
    },
    "deep_hf_2019_public_data_listing": {
        "final_decision": "ALREADY_ANALYZED_NOT_NEW",
        "reason": "DeepHF was already analyzed in Phase 10-12 and is not a new independent Phase 16 discovery.",
        "unresolved_questions": [
            "Any future reuse would require an explicit protocol decision and reproduction of the prior Phase 10-12 evidence, not rediscovery as an independent candidate.",
        ],
    },
    "sgdesigner_2020_public_data_listing": {
        "final_decision": "INSUFFICIENT_EVIDENCE",
        "reason": "No official primary file or confirmed repository data file was verified in Phase 16B-16E.",
        "unresolved_questions": [
            "Primary file, SHA-256, experimental label definition, sequence geometry, and independence require verification.",
        ],
    },
}

CANDIDATE_ORDER = [
    "crispron_2021_rth_tools",
    "crisprpredseq_2020_bmc_additional_files",
    "deep_hf_2019_public_data_listing",
    "doench_2016_orcs_publication_screens",
    "sgdesigner_2020_public_data_listing",
]


def _first_file_value(record: Dict[str, object], key: str, default: str = "UNKNOWN") -> object:
    files = record.get("files", [])
    if not files:
        return default
    values = [file_record.get(key, default) for file_record in files]
    return values[0] if len(set(map(str, values))) == 1 else values


def _file_summary(record: Dict[str, object] | None) -> Dict[str, object]:
    if not record:
        return {
            "status": "NOT_AVAILABLE",
            "files": [],
            "sha256_available": False,
            "source_urls": [],
            "access_date": "UNKNOWN",
        }
    files = record.get("files", [])
    return {
        "status": record.get("primary_file_status", "UNKNOWN"),
        "files": [
            {
                "original_filename": file_record.get("original_filename"),
                "local_raw_path": file_record.get("local_raw_path"),
                "sha256": file_record.get("sha256"),
                "source_url": file_record.get("source_url"),
                "provenance_role": file_record.get("provenance_role"),
            }
            for file_record in files
        ],
        "sha256_available": any(file_record.get("sha256") for file_record in files),
        "source_urls": [file_record.get("source_url") for file_record in files if file_record.get("source_url")],
        "access_date": record.get("access_date", "2026-09-08"),
    }


def _sequence_status(comp: Dict[str, object], ov: Dict[str, object]) -> Dict[str, object]:
    sequence_schema = _first_file_value(comp, "sequence_schema", "NOT_VERIFIED")
    sequence_length = _first_file_value(comp, "sequence_length", "UNKNOWN")
    can_use_30mer = sequence_schema == "canonical_30mer" and sequence_length == 30
    return {
        "status": "COMPATIBLE_CANONICAL_30MER" if can_use_30mer else "NOT_COMPATIBLE_OR_NOT_VERIFIED",
        "verified_sequence_available": bool(ov.get("sequence_available", False)),
        "sequence_record_count": ov.get("sequence_record_count", 0),
        "sequence_kinds": ov.get("sequence_kinds", []),
        "sequence_alphabet": "A/C/G/T where sequence records were parsed",
        "sequence_geometry": sequence_schema,
        "sequence_length": sequence_length,
        "guide_definition": _first_file_value(comp, "guide_definition", "UNKNOWN"),
        "pam_definition": _first_file_value(comp, "PAM_definition", "UNKNOWN"),
        "orientation_or_strand": _first_file_value(comp, "orientation", "UNKNOWN"),
        "canonical_30mer_without_unapproved_transformation": bool(can_use_30mer),
    }


def _label_status(comp: Dict[str, object]) -> Dict[str, object]:
    status = comp.get("compatibility_status", "NOT_VERIFIED")
    return {
        "status": "TARGET_COMPATIBLE_CONTINUOUS_ACTIVITY" if status == "COMPATIBLE" else status,
        "experimentally_measured_status": _first_file_value(comp, "experimental_status", "UNKNOWN"),
        "activity_definition": _first_file_value(comp, "activity_definition", "UNKNOWN"),
        "activity_units": _first_file_value(comp, "activity_units", "UNKNOWN"),
        "continuous_status": _first_file_value(comp, "continuous_status", "UNKNOWN"),
        "substantive_compatibility_with_deepspcas9_target": status == "COMPATIBLE",
        "unit_conversion": "NOT_PERFORMED",
        "label_harmonization": "NOT_PERFORMED",
    }


def _distribution_summary(dist: Dict[str, object]) -> Dict[str, object]:
    sequence = dist.get("sequence", {})
    gc = sequence.get("gc") if isinstance(sequence, dict) else None
    activity = dist.get("activity", {})
    return {
        "sequence_composition": sequence if sequence else "NOT_ASSESSED",
        "gc_summary": gc or "NOT_AVAILABLE",
        "activity_distribution": activity if activity else "NOT_ANALYZED_OR_NOT_TARGET_COMPATIBLE",
        "potential_information_value": (
            "Potential sequence-composition differences are descriptive only; admissible information value is not established unless provenance, sequence, label, compatibility, and independence are all sufficient."
        ),
    }


def _short_overlap(ov: Dict[str, object]) -> Dict[str, object]:
    return {
        "exact_30mer": ov.get("exact_30mer_vs_deepspcas9", "NOT_ASSESSED"),
        "exact_guide": ov.get("exact_guide_vs_deepspcas9", "NOT_ASSESSED"),
        "duplicates": ov.get("duplicate_sequences", "NOT_ASSESSED"),
        "identical_sequence_label_conflicts": ov.get("identical_sequence_label_conflicts", "NOT_ASSESSED"),
        "rc_status": ov.get("rc_overlap_status", {"status": "RC_OVERLAP_NOT_ASSESSED"}),
    }


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(record["candidate_id"]): record for record in records}


def _provenance(record: Dict[str, object] | None) -> Dict[str, object]:
    if record is None:
        return {"status": "NOT_AVAILABLE", "evidence": None}
    return {
        "status": record.get("candidate_status_after_16b", "NOT_VERIFIED"),
        "evidence": record,
    }


def build_phase16f_dataset_gate(
    artifact_paths: Dict[str, Path] | None = None,
    timestamp: str | None = None,
) -> Dict[str, object]:
    """Assemble the final gate from frozen Phase 16B-16E evidence."""
    paths = artifact_paths or DEFAULT_ARTIFACTS
    artifacts = {name: _load(path) for name, path in paths.items()}
    provenance = _by_id(artifacts["phase16b"].get("records", []))
    compatibility = _by_id(artifacts["phase16c"].get("records", []))
    overlap = _by_id(artifacts["phase16d"].get("candidate_reports", []))
    distribution = _by_id(artifacts["phase16e"].get("candidate_reports", []))

    candidates = []
    for candidate_id in CANDIDATE_ORDER:
        decision = DECISIONS[candidate_id]
        comp = compatibility.get(candidate_id, {})
        ov = overlap.get(candidate_id, {})
        dist = distribution.get(candidate_id, {})
        prov = provenance.get(candidate_id)
        primary_file = _file_summary(prov)
        sequence = _sequence_status(comp, ov)
        label = _label_status(comp)
        overlap_summary = _short_overlap(ov)
        independence = ov.get("provenance_relationship", {"independence_assessment": "NOT_ASSESSED"})
        distribution_summary = _distribution_summary(dist)
        candidate = {
            "candidate_id": candidate_id,
            "candidate_name": comp.get("candidate_name", prov.get("candidate_name", candidate_id) if prov else candidate_id),
            "provenance_status": prov.get("candidate_status_after_16b", "NOT_AVAILABLE") if prov else "NOT_AVAILABLE",
            "provenance": _provenance(prov),
            "primary_file_status": primary_file["status"],
            "primary_file": primary_file,
            "sequence_status": sequence["status"],
            "sequence": sequence,
            "label_status": label["status"],
            "label": label,
            "compatibility_status": comp.get("compatibility_status", "NOT_VERIFIED"),
            "compatibility": {
                "status": comp.get("compatibility_status", "NOT_VERIFIED"),
                "reason": comp.get("compatibility_reason", "No Phase 16C evidence."),
                "phase16c_decision_carried_forward": True,
            },
            "overlap_status": overlap_summary,
            "overlap": overlap_summary,
            "independence_status": independence,
            "independence": independence,
            "distribution_information_value": distribution_summary,
            "final_decision": decision["final_decision"],
            "final_status": decision["final_decision"],
            "reason_for_non_acceptance": decision["reason"],
            "reason": decision["reason"],
            "unresolved_questions": decision["unresolved_questions"],
        }
        if candidate["final_decision"] not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported Phase 16F status: {candidate['final_decision']}")
        candidates.append(candidate)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": timestamp,
        "phase": "16F",
        "scope": "final_data_readiness_and_dataset_acceptance_gate_only",
        "source_artifacts": {name: str(path) for name, path in paths.items()},
        "candidate_ids": CANDIDATE_ORDER,
        "candidate_evidence_matrix": candidates,
        "provenance_status": {record["candidate_id"]: record["provenance_status"] for record in candidates},
        "primary_file_status": {record["candidate_id"]: record["primary_file_status"] for record in candidates},
        "sequence_status": {record["candidate_id"]: record["sequence_status"] for record in candidates},
        "compatibility_status": {record["candidate_id"]: record["compatibility_status"] for record in candidates},
        "label_status": {record["candidate_id"]: record["label_status"] for record in candidates},
        "overlap_status": {record["candidate_id"]: record["overlap_status"] for record in candidates},
        "independence_status": {record["candidate_id"]: record["independence_status"] for record in candidates},
        "distribution_information_value_summary": {
            record["candidate_id"]: record["distribution_information_value"] for record in candidates
        },
        "final_decision": {
            "accepted_for_future_pipeline": [],
            "no_candidate_accepted": True,
            "reason": "No candidate satisfies every critical provenance, sequence, label, compatibility, and independence requirement without prohibited transformation.",
        },
        "unresolved_questions": sorted({
            question
            for candidate in candidates
            for question in candidate["unresolved_questions"]
        }),
        "clarification_checks": {
            "deepspcas9_distribution_population_n": 12832,
            "deepspcas9_canonical_valid_modeling_population_n": 10117,
            "deepspcas9_train_n": 8599,
            "deepspcas9_validation_n": 1518,
            "distribution_population_distinct_from_modeling_population": True,
            "phase16f_report_preserves_distribution_vs_modeling_distinction": True,
            "overlap_parser_excludes_non_dna_workbook_metadata_rank_stars": True,
            "overlap_parser_limitation": "Selected fields use a DNA-character filter; DNA-looking metadata strings could still pass.",
        },
        "prohibited_transformations_not_performed": [
            "label_harmonization",
            "sequence_reconstruction",
            "canonical_30mer_rescue_from_guide_only_or_guide_plus_pam",
            "rank_to_activity_conversion",
            "enrichment_or_log_fold_change_to_activity_conversion",
            "binary_to_continuous_activity_conversion",
            "reverse_complement_overlap_without_verified_orientation",
        ],
        "locks": {
            "no_modeling_performed": True,
            "no_moreno_raw_data_accessed": True,
            "moreno_remained_locked": True,
            "no_pooling_occurred": True,
            "no_label_transformation_occurred": True,
            "no_sequence_transformation_occurred": True,
            "no_generalization_claim_was_made": True,
            "no_automatic_dataset_promotion": True,
            "no_dataset_promoted_into_training": True,
        },
    }

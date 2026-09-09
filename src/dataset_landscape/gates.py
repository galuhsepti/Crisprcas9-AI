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
    for candidate_id in sorted(DECISIONS):
        decision = DECISIONS[candidate_id]
        comp = compatibility.get(candidate_id, {})
        ov = overlap.get(candidate_id, {})
        dist = distribution.get(candidate_id, {})
        candidate = {
            "candidate_id": candidate_id,
            "candidate_name": comp.get("candidate_name", provenance.get(candidate_id, {}).get("candidate_name", candidate_id)),
            "provenance": _provenance(provenance.get(candidate_id)),
            "sequence": {
                "status": comp.get("files", [{}])[0].get("sequence_schema", "NOT_VERIFIED") if comp.get("files") else "NOT_VERIFIED",
                "evidence": [f.get("sequence_schema") for f in comp.get("files", [])],
            },
            "label": {
                "status": comp.get("compatibility_status", "NOT_VERIFIED"),
                "evidence": [f.get("activity_definition") for f in comp.get("files", [])],
                "unit_conversion": "NOT_PERFORMED",
                "label_harmonization": "NOT_PERFORMED",
            },
            "compatibility": {
                "status": comp.get("compatibility_status", "NOT_VERIFIED"),
                "reason": comp.get("compatibility_reason", "No Phase 16C evidence."),
            },
            "overlap": {
                "exact_30mer": ov.get("exact_30mer_vs_deepspcas9", "NOT_ASSESSED"),
                "exact_guide": ov.get("exact_guide_vs_deepspcas9", "NOT_ASSESSED"),
                "duplicate_evidence": ov.get("duplicate_sequences", ov.get("duplicate_records", "NOT_ASSESSED")),
                "identical_sequence_label_conflicts": ov.get("identical_sequence_label_conflicts", "NOT_ASSESSED"),
                "rc_status": ov.get("rc_overlap_status", {"status": "NOT_ASSESSED"}),
            },
            "independence": ov.get("provenance_relationship", {"independence_assessment": "NOT_ASSESSED"}),
            "distribution_information_value": {
                "sequence": dist.get("sequence", "NOT_ASSESSED"),
                "activity": dist.get("activity", "NOT_ANALYZED"),
                "potential_information_value": (
                    "Not established as admissible information value because the candidate fails or lacks a critical readiness requirement."
                ),
            },
            "final_decision": decision["final_decision"],
            "reason_for_non_acceptance": decision["reason"],
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
        "candidate_evidence_matrix": candidates,
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
            "distribution_population_distinct_from_modeling_population": True,
            "phase16e_report_explicitly_states_distinction": False,
            "overlap_parser_excludes_non_dna_workbook_metadata_rank_stars": True,
            "overlap_parser_limitation": "Selected fields use a DNA-character filter; DNA-looking metadata strings could still pass.",
        },
        "locks": {
            "no_modeling_performed": True,
            "no_moreno_raw_data_accessed": True,
            "moreno_remained_locked": True,
            "no_pooling_occurred": True,
            "no_transformation_occurred": True,
            "no_generalization_claim_was_made": True,
            "no_automatic_dataset_promotion": True,
        },
    }
"""Phase 17F final dataset acceptance gate.

Consumes Phase 17A-17E evidence only. It performs no modeling, pooling,
sequence rescue, label conversion, overlap expansion, or dataset integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .phase17a import protected_hashes


PHASE17A_RESULT = Path("results/phase17a_dataset_recovery_20260911_140154.json")
PHASE17B_RESULT = Path("results/phase17b_primary_file_verification_20260911_143300.json")
PHASE17C_RESULT = Path("results/phase17c_sequence_label_compatibility_20260911_144924.json")
PHASE17D_RESULT = Path("results/phase17d_independence_overlap_20260911_145754.json")
PHASE17E_RESULT = Path("results/phase17e_distribution_information_value_20260911_190523.json")

ACCEPTANCE_STATUSES = {
    "ACCEPTED_FOR_FUTURE_PIPELINE",
    "CONDITIONAL_NEEDS_PROVENANCE",
    "CONDITIONAL_NEEDS_COMPATIBILITY",
    "CONDITIONAL_NEEDS_INDEPENDENCE",
    "REJECTED_FOR_CURRENT_TARGET",
    "ALREADY_ANALYZED_NOT_NEW",
    "INSUFFICIENT_EVIDENCE",
}

CANDIDATE_ORDER = [
    "crispron_2021_rth_tools",
    "crisprpredseq_2020_bmc_additional_files",
    "deep_hf_2019_public_data_listing",
    "doench_2016_orcs_publication_screens",
    "sgdesigner_2020_public_data_listing",
    "corsi_2022_free_energy_pam_context",
]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def by_id(records: Iterable[Dict[str, object]], key: str = "candidate_id") -> Dict[str, Dict[str, object]]:
    return {str(record[key]): record for record in records}


def decide_acceptance(evidence: Dict[str, object]) -> Dict[str, object]:
    """Apply the required Phase 17F decision order to normalized evidence."""
    if evidence.get("already_analyzed_not_new"):
        return {
            "final_acceptance_status": "ALREADY_ANALYZED_NOT_NEW",
            "decision_step": "already_analyzed",
            "exact_reason": "Candidate was previously analyzed and is not new evidence from the Phase 17 recovery cycle.",
        }
    if not evidence.get("primary_evidence"):
        return {
            "final_acceptance_status": "INSUFFICIENT_EVIDENCE",
            "decision_step": "primary_evidence",
            "exact_reason": "Primary dataset/file evidence is not sufficiently established.",
        }
    if not evidence.get("experimental_measurement"):
        return {
            "final_acceptance_status": evidence.get("experimental_failure_status", "INSUFFICIENT_EVIDENCE"),
            "decision_step": "experimental_measurement",
            "exact_reason": "Experimental observation status is not sufficiently established for a compatible training dataset.",
        }
    if not evidence.get("sequence_compatible"):
        return {
            "final_acceptance_status": evidence.get("sequence_failure_status", "REJECTED_FOR_CURRENT_TARGET"),
            "decision_step": "sequence_compatibility",
            "exact_reason": "Sequence evidence is not compatible with the thesis target without prohibited sequence rescue.",
        }
    if not evidence.get("label_compatible"):
        return {
            "final_acceptance_status": "REJECTED_FOR_CURRENT_TARGET",
            "decision_step": "label_compatibility",
            "exact_reason": "Label evidence is not an explicitly documented experimentally measured continuous activity quantity.",
        }
    if not evidence.get("independence_established"):
        return {
            "final_acceptance_status": "CONDITIONAL_NEEDS_INDEPENDENCE",
            "decision_step": "independence",
            "exact_reason": "Independence/provenance relationship is not sufficiently established.",
        }
    if evidence.get("requires_prohibited_transformation"):
        return {
            "final_acceptance_status": "REJECTED_FOR_CURRENT_TARGET",
            "decision_step": "prohibited_transformation",
            "exact_reason": "Use would require a prohibited label or sequence transformation.",
        }
    return {
        "final_acceptance_status": "ACCEPTED_FOR_FUTURE_PIPELINE",
        "decision_step": "accepted",
        "exact_reason": "All Phase 17F acceptance requirements are sufficiently established.",
    }


def _primary_evidence(candidate_id: str, phase17b_record: Dict[str, object]) -> bool:
    if candidate_id in {"deep_hf_2019_public_data_listing", "sgdesigner_2020_public_data_listing", "corsi_2022_free_energy_pam_context"}:
        return False
    status = str(phase17b_record.get("verification_status", ""))
    if status in {"PRIMARY_FILE_NOT_RECOVERED", "DISCOVERY_ONLY", "PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED"}:
        return False
    return bool(phase17b_record.get("files"))


def _experimental_measurement(candidate_id: str, phase17b_record: Dict[str, object], phase17c_record: Dict[str, object]) -> bool:
    if candidate_id == "crisprpredseq_2020_bmc_additional_files":
        return True
    if candidate_id == "doench_2016_orcs_publication_screens":
        return True
    return phase17c_record.get("label_status") == "LABEL_COMPATIBLE"


def _requires_prohibited_transformation(phase17c_record: Dict[str, object]) -> bool:
    seq = str(phase17c_record.get("sequence_status"))
    label = str(phase17c_record.get("label_status"))
    return seq == "SEQUENCE_INCOMPATIBLE" or label == "LABEL_INCOMPATIBLE"


def _independence_established(phase17d_record: Dict[str, object]) -> bool:
    status = str(phase17d_record.get("provenance_relationship", {}).get("status", ""))
    return status == "INDEPENDENCE_ESTABLISHED"


def _normalized_evidence(
    candidate_id: str,
    phase17a_record: Dict[str, object],
    phase17b_record: Dict[str, object],
    phase17c_record: Dict[str, object],
    phase17d_record: Dict[str, object],
    phase17e_record: Dict[str, object],
) -> Dict[str, object]:
    primary = _primary_evidence(candidate_id, phase17b_record)
    already = candidate_id == "deep_hf_2019_public_data_listing"
    sequence_compatible = phase17c_record.get("sequence_status") == "SEQUENCE_COMPATIBLE"
    label_compatible = phase17c_record.get("label_status") == "LABEL_COMPATIBLE"
    return {
        "candidate_id": candidate_id,
        "already_analyzed_not_new": already,
        "primary_evidence": primary,
        "experimental_measurement": _experimental_measurement(candidate_id, phase17b_record, phase17c_record),
        "sequence_compatible": sequence_compatible,
        "sequence_failure_status": "REJECTED_FOR_CURRENT_TARGET",
        "label_compatible": label_compatible,
        "independence_established": _independence_established(phase17d_record),
        "requires_prohibited_transformation": _requires_prohibited_transformation(phase17c_record),
        "phase17a_status": phase17a_record.get("planned_status"),
        "phase17b_status": phase17b_record.get("verification_status"),
        "phase17c_sequence_status": phase17c_record.get("sequence_status"),
        "phase17c_label_status": phase17c_record.get("label_status"),
        "phase17d_independence_status": phase17d_record.get("provenance_relationship", {}).get("status", "INSUFFICIENT_PROVENANCE"),
        "phase17e_information_value_status": phase17e_record.get("information_value_status", "INSUFFICIENT_EVIDENCE"),
    }


def _specific_reason(candidate_id: str, decision: Dict[str, object], evidence: Dict[str, object]) -> str:
    if candidate_id == "crispron_2021_rth_tools":
        return "Primary experimental training table was not identified; sequence and label evidence remain insufficient."
    if candidate_id == "crisprpredseq_2020_bmc_additional_files":
        return "Observed 23-mer guide+PAM records have binary labels and processed/derived relationship; accepting would require prohibited label and sequence rescue."
    if candidate_id == "doench_2016_orcs_publication_screens":
        return "Guide-only screen files expose STARS/rank/enrichment/log-fold-change style values, not compatible continuous activity labels; same-study relationships remain unresolved."
    if candidate_id == "deep_hf_2019_public_data_listing":
        return "DeepHF was already analyzed in Phases 10-12 and no new Phase 17 primary local file established it as a new independent dataset."
    if candidate_id == "sgdesigner_2020_public_data_listing":
        return "No verified primary file, sequence evidence, label evidence, or independence evidence was recovered."
    if candidate_id == "corsi_2022_free_energy_pam_context":
        return "Discovery-only candidate with no recovered primary file and no verified sequence or label evidence."
    return str(decision["exact_reason"])


def _missing_evidence(status: str, candidate_id: str) -> List[str]:
    if status == "INSUFFICIENT_EVIDENCE":
        return ["primary file identity", "experimental status", "sequence geometry", "label semantics", "independence"]
    if status == "CONDITIONAL_NEEDS_INDEPENDENCE":
        return ["independence/provenance relationship"]
    if status == "CONDITIONAL_NEEDS_COMPATIBILITY":
        return ["sequence compatibility", "label compatibility"]
    if status == "CONDITIONAL_NEEDS_PROVENANCE":
        return ["primary provenance"]
    if candidate_id == "deep_hf_2019_public_data_listing":
        return ["new Phase 17 primary evidence, if future reuse is proposed"]
    return []


def build_phase17f_acceptance_gate(
    phase17a_path: Path = PHASE17A_RESULT,
    phase17b_path: Path = PHASE17B_RESULT,
    phase17c_path: Path = PHASE17C_RESULT,
    phase17d_path: Path = PHASE17D_RESULT,
    phase17e_path: Path = PHASE17E_RESULT,
    timestamp: str | None = None,
) -> Dict[str, object]:
    before = protected_hashes()
    phase17a = load_json(phase17a_path)
    phase17b = load_json(phase17b_path)
    phase17c = load_json(phase17c_path)
    phase17d = load_json(phase17d_path)
    phase17e = load_json(phase17e_path)
    a_by_id = by_id(phase17a["candidates"])
    b_by_id = by_id(phase17b["candidates"])
    c_by_id = by_id(phase17c["candidates"])
    d_by_id = by_id(phase17d["candidate_reports"])
    e_by_id = by_id(phase17e["candidates"])

    candidates = []
    for candidate_id in CANDIDATE_ORDER:
        evidence = _normalized_evidence(
            candidate_id,
            a_by_id.get(candidate_id, {}),
            b_by_id.get(candidate_id, {}),
            c_by_id.get(candidate_id, {}),
            d_by_id.get(candidate_id, {}),
            e_by_id.get(candidate_id, {}),
        )
        decision = decide_acceptance(evidence)
        status = decision["final_acceptance_status"]
        if status not in ACCEPTANCE_STATUSES:
            raise ValueError(f"Unsupported Phase 17F status: {status}")
        reason = _specific_reason(candidate_id, decision, evidence)
        candidates.append({
            "candidate_id": candidate_id,
            "provenance_status": evidence["phase17b_status"] or evidence["phase17a_status"],
            "experimental_status": "ESTABLISHED_SCREEN_OR_OBSERVATION_NOT_NECESSARILY_COMPATIBLE" if evidence["experimental_measurement"] else "NOT_ESTABLISHED",
            "sequence_status": evidence["phase17c_sequence_status"],
            "label_status": evidence["phase17c_label_status"],
            "independence_status": evidence["phase17d_independence_status"],
            "information_value_status": evidence["phase17e_information_value_status"],
            "decision_step": decision["decision_step"],
            "final_acceptance_status": status,
            "exact_reason": reason,
            "missing_future_evidence": _missing_evidence(status, candidate_id),
            "normalized_evidence": evidence,
        })

    accepted = [r["candidate_id"] for r in candidates if r["final_acceptance_status"] == "ACCEPTED_FOR_FUTURE_PIPELINE"]
    after = protected_hashes()
    return {
        "phase": "17F",
        "protocol_version": "phase17f-final-dataset-acceptance-gate-v1",
        "timestamp": timestamp,
        "scope": "final_dataset_acceptance_gate_only_no_integration",
        "source_artifacts": {
            "phase17a": str(phase17a_path).replace("\\", "/"),
            "phase17b": str(phase17b_path).replace("\\", "/"),
            "phase17c": str(phase17c_path).replace("\\", "/"),
            "phase17d": str(phase17d_path).replace("\\", "/"),
            "phase17e": str(phase17e_path).replace("\\", "/"),
        },
        "candidate_decisions": candidates,
        "accepted_for_future_pipeline": accepted,
        "accepted_candidate_count": len(accepted),
        "acceptance_status_vocabulary": sorted(ACCEPTANCE_STATUSES),
        "canonical_population_distinction": {
            "deepsp_phase16e_distribution_n": 12832,
            "deepsp_canonical_modeling_n": 10117,
            "train_n": 8599,
            "validation_n": 1518,
        },
        "phase17e_information_value_interpretation": {
            "no_candidate_with_verified_compatible_continuous_activity": phase17e["activity_distribution_global_status"],
            "potential_information_value_does_not_justify_acceptance": True,
            "sequence_diversity_without_compatible_activity_not_sufficient": True,
        },
        "canonical_integrity": {"before": before, "after": after, "unchanged": before == after},
        "scope_confirmation": {
            "no_model_training": True,
            "no_model_evaluation": True,
            "no_hyperparameter_tuning": True,
            "no_moreno_raw_data_accessed": True,
            "no_dataset_pooling": True,
            "no_combined_training_dataset": True,
            "no_new_training_split": True,
            "no_label_transformation": True,
            "no_label_harmonization": True,
            "no_sequence_transformation": True,
            "no_30mer_construction": True,
            "no_reverse_complement_assumption": True,
            "no_near_duplicate_or_similarity_threshold_analysis": True,
            "no_dataset_construction": True,
            "no_accepted_dataset_integrated": True,
            "no_continual_learning": True,
            "no_phase18": True,
            "no_canonical_artifact_modifications": True,
            "no_model_improvement_claim": True,
            "no_generalization_claim": True,
        },
    }


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

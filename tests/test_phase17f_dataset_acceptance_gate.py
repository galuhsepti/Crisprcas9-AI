import ast
import json
from pathlib import Path

from src.dataset_recovery.acceptance import (
    ACCEPTANCE_STATUSES,
    build_phase17f_acceptance_gate,
    decide_acceptance,
)


def _base_evidence(**overrides):
    evidence = {
        "primary_evidence": True,
        "experimental_measurement": True,
        "sequence_compatible": True,
        "label_compatible": True,
        "independence_established": True,
        "requires_prohibited_transformation": False,
        "already_analyzed_not_new": False,
    }
    evidence.update(overrides)
    return evidence


def test_decision_order_accepts_synthetic_compatible_primary_independent_case():
    decision = decide_acceptance(_base_evidence())
    assert decision["final_acceptance_status"] == "ACCEPTED_FOR_FUTURE_PIPELINE"
    assert decision["decision_step"] == "accepted"


def test_binary_label_rejected_synthetic_case():
    decision = decide_acceptance(_base_evidence(label_compatible=False))
    assert decision["final_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET"
    assert decision["decision_step"] == "label_compatibility"


def test_rank_enrichment_logfc_rejected_synthetic_case():
    decision = decide_acceptance(_base_evidence(label_compatible=False))
    assert decision["final_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET"


def test_guide_only_without_canonical_geometry_rejected_synthetic_case():
    decision = decide_acceptance(_base_evidence(sequence_compatible=False))
    assert decision["final_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET"
    assert decision["decision_step"] == "sequence_compatibility"


def test_primary_provenance_unresolved_is_insufficient_synthetic_case():
    decision = decide_acceptance(_base_evidence(primary_evidence=False))
    assert decision["final_acceptance_status"] == "INSUFFICIENT_EVIDENCE"
    assert decision["decision_step"] == "primary_evidence"


def test_independence_unresolved_is_conditional_synthetic_case():
    decision = decide_acceptance(_base_evidence(independence_established=False))
    assert decision["final_acceptance_status"] == "CONDITIONAL_NEEDS_INDEPENDENCE"
    assert decision["decision_step"] == "independence"


def test_prior_analyzed_dataset_is_already_analyzed_synthetic_case():
    decision = decide_acceptance(_base_evidence(already_analyzed_not_new=True))
    assert decision["final_acceptance_status"] == "ALREADY_ANALYZED_NOT_NEW"
    assert decision["decision_step"] == "already_analyzed"


def test_phase17f_real_candidate_decisions_are_expected_and_no_false_acceptance():
    payload = build_phase17f_acceptance_gate(timestamp="2026-09-11T00:00:00")
    decisions = {r["candidate_id"]: r["final_acceptance_status"] for r in payload["candidate_decisions"]}
    assert decisions == {
        "crispron_2021_rth_tools": "INSUFFICIENT_EVIDENCE",
        "crisprpredseq_2020_bmc_additional_files": "REJECTED_FOR_CURRENT_TARGET",
        "deep_hf_2019_public_data_listing": "ALREADY_ANALYZED_NOT_NEW",
        "doench_2016_orcs_publication_screens": "REJECTED_FOR_CURRENT_TARGET",
        "sgdesigner_2020_public_data_listing": "INSUFFICIENT_EVIDENCE",
        "corsi_2022_free_energy_pam_context": "INSUFFICIENT_EVIDENCE",
    }
    assert payload["accepted_for_future_pipeline"] == []
    assert payload["accepted_candidate_count"] == 0


def test_status_vocabulary_and_required_decision_fields():
    payload = build_phase17f_acceptance_gate(timestamp="2026-09-11T00:00:00")
    assert set(payload["acceptance_status_vocabulary"]) == ACCEPTANCE_STATUSES
    required = {
        "candidate_id",
        "provenance_status",
        "experimental_status",
        "sequence_status",
        "label_status",
        "independence_status",
        "information_value_status",
        "final_acceptance_status",
        "exact_reason",
    }
    for record in payload["candidate_decisions"]:
        assert required <= set(record)
        assert record["final_acceptance_status"] in ACCEPTANCE_STATUSES
        assert record["exact_reason"]


def test_incompatible_label_and_geometry_rejections_real_candidates():
    payload = build_phase17f_acceptance_gate(timestamp="2026-09-11T00:00:00")
    by_id = {r["candidate_id"]: r for r in payload["candidate_decisions"]}
    crisprpred = by_id["crisprpredseq_2020_bmc_additional_files"]
    doench = by_id["doench_2016_orcs_publication_screens"]
    assert crisprpred["sequence_status"] == "SEQUENCE_INCOMPATIBLE"
    assert crisprpred["label_status"] == "LABEL_INCOMPATIBLE"
    assert crisprpred["final_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET"
    assert doench["sequence_status"] == "SEQUENCE_INCOMPATIBLE"
    assert doench["label_status"] == "LABEL_INCOMPATIBLE"
    assert doench["final_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET"


def test_unresolved_provenance_and_already_analyzed_real_candidates():
    payload = build_phase17f_acceptance_gate(timestamp="2026-09-11T00:00:00")
    by_id = {r["candidate_id"]: r for r in payload["candidate_decisions"]}
    assert by_id["crispron_2021_rth_tools"]["final_acceptance_status"] == "INSUFFICIENT_EVIDENCE"
    assert by_id["sgdesigner_2020_public_data_listing"]["final_acceptance_status"] == "INSUFFICIENT_EVIDENCE"
    assert by_id["corsi_2022_free_energy_pam_context"]["final_acceptance_status"] == "INSUFFICIENT_EVIDENCE"
    assert by_id["deep_hf_2019_public_data_listing"]["final_acceptance_status"] == "ALREADY_ANALYZED_NOT_NEW"


def test_population_information_value_and_scope_guards():
    payload = build_phase17f_acceptance_gate(timestamp="2026-09-11T00:00:00")
    pop = payload["canonical_population_distinction"]
    assert pop["deepsp_phase16e_distribution_n"] == 12832
    assert pop["deepsp_canonical_modeling_n"] == 10117
    assert pop["train_n"] == 8599
    assert pop["validation_n"] == 1518
    assert payload["phase17e_information_value_interpretation"]["potential_information_value_does_not_justify_acceptance"] is True
    guards = payload["scope_confirmation"]
    assert guards["no_dataset_pooling"] is True
    assert guards["no_label_transformation"] is True
    assert guards["no_sequence_transformation"] is True
    assert guards["no_moreno_raw_data_accessed"] is True
    assert guards["no_dataset_construction"] is True
    assert guards["no_accepted_dataset_integrated"] is True


def test_canonical_protection_and_deterministic_output():
    first = build_phase17f_acceptance_gate(timestamp="2026-09-11T00:00:00")
    second = build_phase17f_acceptance_gate(timestamp="2026-09-11T00:00:00")
    assert first == second
    assert first["canonical_integrity"]["unchanged"] is True
    json.dumps(first)


def test_phase17f_sources_do_not_import_models_or_forbidden_logic():
    paths = [
        Path("src/dataset_recovery/acceptance.py"),
        Path("scripts/run_phase17f_dataset_acceptance_gate.py"),
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    forbidden_text = [
        "Moreno-Mateos.csv",
        "load_moreno_mateos",
        "build_training_pool",
        ".fit(",
        ".predict(",
        "hamming",
        "edit-distance",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(phrase in text for phrase in forbidden_text)
        tree = ast.parse(text)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert forbidden_imports.isdisjoint(imports)

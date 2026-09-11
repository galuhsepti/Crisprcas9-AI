import ast
import json
from pathlib import Path

from src.dataset_landscape.gates import (
    ALLOWED_STATUSES,
    build_phase16f_dataset_gate,
)
from src.dataset_landscape.overlap import extract_doench_records


def test_phase16f_decisions_are_conservative_and_complete():
    payload = build_phase16f_dataset_gate(timestamp="2026-09-08T00:00:00")
    decisions = {
        record["candidate_id"]: record["final_decision"]
        for record in payload["candidate_evidence_matrix"]
    }
    assert decisions == {
        "crispron_2021_rth_tools": "INSUFFICIENT_EVIDENCE",
        "crisprpredseq_2020_bmc_additional_files": "REJECTED_FOR_CURRENT_TARGET",
        "deep_hf_2019_public_data_listing": "ALREADY_ANALYZED_NOT_NEW",
        "doench_2016_orcs_publication_screens": "REJECTED_FOR_CURRENT_TARGET",
        "sgdesigner_2020_public_data_listing": "INSUFFICIENT_EVIDENCE",
    }
    assert set(decisions.values()) <= ALLOWED_STATUSES
    assert payload["final_decision"]["accepted_for_future_pipeline"] == []


def test_phase16f_preserves_required_evidence_and_locks():
    payload = build_phase16f_dataset_gate(timestamp="2026-09-08T00:00:00")
    assert payload["clarification_checks"]["deepspcas9_distribution_population_n"] == 12832
    assert payload["clarification_checks"]["deepspcas9_canonical_valid_modeling_population_n"] == 10117
    assert payload["clarification_checks"]["deepspcas9_train_n"] == 8599
    assert payload["clarification_checks"]["deepspcas9_validation_n"] == 1518
    assert payload["clarification_checks"]["phase16f_report_preserves_distribution_vs_modeling_distinction"] is True
    assert payload["clarification_checks"]["overlap_parser_excludes_non_dna_workbook_metadata_rank_stars"] is True
    assert all(payload["locks"].values())
    for record in payload["candidate_evidence_matrix"]:
        assert record["final_decision"] == "ALREADY_ANALYZED_NOT_NEW" or record["reason_for_non_acceptance"]


def test_phase16f_json_contains_required_matrix_fields():
    payload = build_phase16f_dataset_gate(timestamp="2026-09-08T00:00:00")
    assert payload["candidate_ids"] == [
        "crispron_2021_rth_tools",
        "crisprpredseq_2020_bmc_additional_files",
        "deep_hf_2019_public_data_listing",
        "doench_2016_orcs_publication_screens",
        "sgdesigner_2020_public_data_listing",
    ]
    required_top_level = {
        "provenance_status",
        "primary_file_status",
        "sequence_status",
        "compatibility_status",
        "label_status",
        "overlap_status",
        "independence_status",
        "distribution_information_value_summary",
        "prohibited_transformations_not_performed",
    }
    assert required_top_level <= set(payload)
    required_record_fields = {
        "candidate_id",
        "provenance",
        "primary_file",
        "sequence",
        "compatibility",
        "label",
        "overlap",
        "independence",
        "distribution_information_value",
        "final_status",
        "reason",
    }
    for record in payload["candidate_evidence_matrix"]:
        assert required_record_fields <= set(record)


def test_phase16f_preserves_deephf_prior_relationship():
    payload = build_phase16f_dataset_gate(timestamp="2026-09-08T00:00:00")
    deephf = next(r for r in payload["candidate_evidence_matrix"] if r["candidate_id"].startswith("deep_hf"))
    assert deephf["final_decision"] == "ALREADY_ANALYZED_NOT_NEW"
    assert "Phase 10-12" in deephf["reason_for_non_acceptance"]


def test_doench_parser_excludes_non_dna_metadata_and_stars_values():
    paths = sorted(Path("data/phase16/raw").glob("doench2016_orcs_*.xlsx"))
    records = [record for path in paths for record in extract_doench_records(str(path))]
    assert records
    assert all(set(record["sequence"]) <= set("ACGT") for record in records)
    assert not any("STARS" in record["sequence"] for record in records)


def test_phase16f_has_no_forbidden_modeling_or_moreno_sources():
    paths = [Path("src/dataset_landscape/gates.py"), Path("scripts/run_phase16f_dataset_gate.py")]
    forbidden_text = ["load_moreno_mateos", "Moreno-Mateos.csv", "build_training_pool", "torch", "xgboost", "sklearn"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(phrase in text for phrase in forbidden_text)
        tree = ast.parse(text)
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not imports.intersection({"src.models", "torch", "xgboost", "sklearn"})


def test_phase16f_report_contains_required_sections_and_clarifications():
    text = Path("docs/phase16f_dataset_gate_report.md").read_text(encoding="utf-8")
    required_headings = [
        "# Phase 16F - Final Dataset Landscape Gate",
        "## 1. Objective",
        "## 2. Locked Constraints",
        "## 3. Evidence Sources",
        "## 4. Candidate-by-Candidate Assessment",
        "## 5. Final Evidence Matrix",
        "## 6. Dataset Acceptance Decisions",
        "## 7. Unresolved Scientific Questions",
        "## 8. What This Phase Does NOT Establish",
        "## 9. Reproducibility",
        "## 10. Final Phase 16 Status",
    ]
    for heading in required_headings:
        assert heading in text
    assert "DeepSpCas9 n=12,832 is the Phase 16E distribution-audit population" in text
    assert "canonical modeling population remains valid n=10,117, train n=8,599, and validation n=1,518" in text
    assert "overlap parsing excludes ordinary non-DNA workbook metadata, rank values, and STARS values" in text


def test_phase16f_output_is_json_serializable_and_deterministic():
    first = build_phase16f_dataset_gate(timestamp="2026-09-08T00:00:00")
    second = build_phase16f_dataset_gate(timestamp="2026-09-08T00:00:00")
    assert first == second
    json.dumps(first)

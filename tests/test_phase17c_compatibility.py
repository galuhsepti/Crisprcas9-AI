import ast
import json
from pathlib import Path

from src.dataset_recovery.compatibility import (
    LABEL_STATUSES,
    OVERALL_STATUSES,
    PHASE17B_RESULT,
    SEQUENCE_STATUSES,
    build_phase17c_compatibility,
    characterize_label_column,
    characterize_sequence_column,
)


def test_phase17c_status_vocabularies_are_enforced():
    payload = build_phase17c_compatibility(PHASE17B_RESULT)
    for record in payload["candidates"]:
        assert record["sequence_status"] in SEQUENCE_STATUSES
        assert record["label_status"] in LABEL_STATUSES
        assert record["overall_status"] in OVERALL_STATUSES
    assert all(record["overall_status"] != "COMPATIBLE_FOR_FUTURE_AUDIT" for record in payload["candidates"])


def test_sequence_characterization_does_not_rescue_noncanonical_lengths():
    guide = characterize_sequence_column("sgRNA", {"observed_record_count": 2, "observed_lengths": {20: 2}, "non_acgt_count": 0})
    guide_pam = characterize_sequence_column("sgRNA", {"observed_record_count": 2, "observed_lengths": {23: 2}, "non_acgt_count": 0})
    thirty = characterize_sequence_column("sequence_30mer", {"observed_record_count": 2, "observed_lengths": {30: 2}, "non_acgt_count": 0})
    assert guide["sequence_status"] == "SEQUENCE_INCOMPATIBLE"
    assert guide_pam["sequence_status"] == "SEQUENCE_INCOMPATIBLE"
    assert thirty["sequence_status"] == "SEQUENCE_PARTIALLY_COMPATIBLE"
    assert "30-mer" in guide_pam["reason"] or "30mer" in guide_pam["reason"]


def test_label_characterization_rejects_binary_and_unknown_numeric_range():
    binary = characterize_label_column("label", {
        "label_type_observed": "binary",
        "documented_biological_meaning": "UNKNOWN_FROM_FILE_ONLY",
        "numeric_summary": {"numeric": True, "min": 0.0, "max": 1.0},
    })
    unknown = characterize_label_column("score", {
        "label_type_observed": "unknown_or_continuous_numeric",
        "documented_biological_meaning": "UNKNOWN_FROM_FILE_ONLY",
        "numeric_summary": {"numeric": True, "min": 0.0, "max": 1.0},
    })
    assert binary["label_status"] == "LABEL_INCOMPATIBLE"
    assert unknown["label_status"] == "LABEL_INSUFFICIENT_EVIDENCE"


def test_phase17c_candidate_aggregation_logic():
    payload = build_phase17c_compatibility(PHASE17B_RESULT)
    by_id = {record["candidate_id"]: record for record in payload["candidates"]}
    assert by_id["crispron_2021_rth_tools"]["sequence_status"] == "SEQUENCE_INSUFFICIENT_EVIDENCE"
    assert by_id["crispron_2021_rth_tools"]["label_status"] == "LABEL_INSUFFICIENT_EVIDENCE"
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["sequence_status"] == "SEQUENCE_INCOMPATIBLE"
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["label_status"] == "LABEL_INCOMPATIBLE"
    assert by_id["doench_2016_orcs_publication_screens"]["overall_status"] in {
        "INCOMPATIBLE_FOR_CURRENT_TARGET",
        "INSUFFICIENT_EVIDENCE",
    }
    assert by_id["deep_hf_2019_public_data_listing"]["overall_status"] == "INSUFFICIENT_EVIDENCE"
    assert by_id["corsi_2022_free_energy_pam_context"]["overall_status"] == "INSUFFICIENT_EVIDENCE"


def test_phase17c_crisprpred_keeps_four_files_separate():
    payload = build_phase17c_compatibility(PHASE17B_RESULT)
    record = next(r for r in payload["candidates"] if r["candidate_id"] == "crisprpredseq_2020_bmc_additional_files")
    files = {result["file"] for result in record["sequence_file_results"]}
    assert len(files) == 4
    assert all(result["length_distribution"] == {23: result["rows_with_sequence"]} for result in record["sequence_file_results"])
    assert all(result["label_status"] == "LABEL_INCOMPATIBLE" for result in record["label_file_results"])


def test_phase17c_safety_and_canonical_guards():
    payload = build_phase17c_compatibility(PHASE17B_RESULT)
    assert payload["canonical_integrity"]["unchanged"] is True
    assert all(payload["safety_guards"].values())
    guards = payload["non_modeling_guards"]
    assert guards["no_moreno_raw_data_accessed"] is True
    assert guards["no_dataset_pooling"] is True
    assert guards["no_label_transformation"] is True
    assert guards["no_sequence_transformation"] is True
    assert guards["no_30mer_construction"] is True
    assert guards["no_dataset_acceptance"] is True


def test_phase17c_deterministic_and_json_serializable():
    first = build_phase17c_compatibility(PHASE17B_RESULT)
    second = build_phase17c_compatibility(PHASE17B_RESULT)
    assert first == second
    json.dumps(first)


def test_phase17c_sources_do_not_import_models_or_forbidden_logic():
    paths = [
        Path("src/dataset_recovery/compatibility.py"),
        Path("scripts/run_phase17c_compatibility_audit.py"),
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    forbidden_text = [
        "Moreno-Mateos.csv",
        "load_moreno_mateos",
        "build_training_pool",
        "fit(",
        "predict(",
        "hamming",
        "edit-distance",
        "reverse_complement",
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

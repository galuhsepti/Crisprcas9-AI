import ast
import json
from pathlib import Path

from src.dataset_recovery.verification import (
    ALLOWED_VERIFICATION_STATUSES,
    PHASE17A_RESULT,
    load_phase17a_payload,
    verify_phase17b,
)


def test_phase17b_loads_phase17a_recovery_metadata():
    payload17a = load_phase17a_payload(PHASE17A_RESULT)
    payload17b = verify_phase17b(PHASE17A_RESULT)
    assert payload17b["phase17a_source"] == str(PHASE17A_RESULT).replace("\\", "/")
    assert payload17b["candidate_count"] == payload17a["candidate_count"]


def test_phase17b_statuses_are_allowed_and_no_acceptance():
    payload = verify_phase17b(PHASE17A_RESULT)
    statuses = {record["verification_status"] for record in payload["candidates"]}
    assert statuses <= ALLOWED_VERIFICATION_STATUSES
    assert "ACCEPTED_FOR_FUTURE_PIPELINE" not in statuses
    assert payload["non_modeling_guards"]["no_dataset_acceptance"] is True


def test_phase17b_sha256_integrity_for_recovered_files():
    payload = verify_phase17b(PHASE17A_RESULT)
    recovered_files = [
        file_record
        for record in payload["candidates"]
        for file_record in record["files"]
    ]
    assert recovered_files
    for file_record in recovered_files:
        assert file_record["sha256_matches_phase17a"] is True
        assert len(file_record["sha256"]) == 64
        assert Path(file_record["exact_local_filename"]).exists()


def test_phase17b_crispron_archive_preserves_unidentified_training_table():
    payload = verify_phase17b(PHASE17A_RESULT)
    crispron = next(r for r in payload["candidates"] if r["candidate_id"] == "crispron_2021_rth_tools")
    assert crispron["verification_status"] == "PRIMARY_EXPERIMENTAL_TABLE_NOT_IDENTIFIED"
    inspection = crispron["files"][0]["inspection"]
    assert inspection["code_files"]
    assert inspection["model_artifact_like_files"]
    assert inspection["experimental_training_table_identified"] is False


def test_phase17b_crisprpred_sequence_and_label_inspection():
    payload = verify_phase17b(PHASE17A_RESULT)
    record = next(r for r in payload["candidates"] if r["candidate_id"] == "crisprpredseq_2020_bmc_additional_files")
    assert record["verification_status"] == "DERIVED_OR_PROCESSED"
    assert len(record["files"]) == 4
    for file_record in record["files"]:
        inspection = file_record["inspection"]
        assert inspection["row_count"] > 0
        assert "sgRNA" in inspection["sequence_columns"]
        assert "label" in inspection["label_columns"]
        assert inspection["label_columns"]["label"]["label_type_observed"] == "binary"


def test_phase17b_doench_workbook_inspection():
    payload = verify_phase17b(PHASE17A_RESULT)
    record = next(r for r in payload["candidates"] if r["candidate_id"] == "doench_2016_orcs_publication_screens")
    assert record["verification_status"] == "PRIMARY_VERIFIED_WITH_METADATA_GAPS"
    assert len(record["files"]) == 4
    for file_record in record["files"]:
        inspection = file_record["inspection"]
        assert inspection["sheet_count"] >= 1
        assert inspection["sheets"]


def test_phase17b_non_recovered_candidates_remain_unaccepted():
    payload = verify_phase17b(PHASE17A_RESULT)
    by_id = {record["candidate_id"]: record for record in payload["candidates"]}
    assert by_id["deep_hf_2019_public_data_listing"]["verification_status"] == "PRIMARY_FILE_NOT_RECOVERED"
    assert by_id["sgdesigner_2020_public_data_listing"]["verification_status"] == "PRIMARY_FILE_NOT_RECOVERED"
    assert by_id["corsi_2022_free_energy_pam_context"]["verification_status"] == "DISCOVERY_ONLY"
    assert not by_id["deep_hf_2019_public_data_listing"]["files"]


def test_phase17b_canonical_integrity_and_guards():
    payload = verify_phase17b(PHASE17A_RESULT)
    assert payload["canonical_integrity"]["unchanged"] is True
    guards = payload["non_modeling_guards"]
    assert guards["no_moreno_raw_data_accessed"] is True
    assert guards["no_dataset_pooling"] is True
    assert guards["no_label_transformation"] is True
    assert guards["no_sequence_transformation"] is True
    assert guards["no_rc_overlap"] is True


def test_phase17b_deterministic_and_json_serializable():
    first = verify_phase17b(PHASE17A_RESULT)
    second = verify_phase17b(PHASE17A_RESULT)
    assert first == second
    json.dumps(first)


def test_phase17b_sources_do_not_import_models_or_use_forbidden_logic():
    paths = [
        Path("src/dataset_recovery/verification.py"),
        Path("scripts/run_phase17b_primary_file_verification.py"),
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

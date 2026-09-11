import ast
import json
from pathlib import Path

from src.dataset_recovery.phase17a import (
    ALLOWED_STATUSES,
    RAW_DIR,
    protected_hashes,
    provenance_payload,
    recover_phase17a,
    sha256_file,
)


def test_phase17a_uses_allowed_statuses_and_no_acceptance():
    payload = recover_phase17a("2026-09-11")
    statuses = {record["acquisition_status"] for record in payload["candidates"]}
    assert statuses <= ALLOWED_STATUSES
    assert "ACCEPTED_FOR_FUTURE_PIPELINE" not in statuses
    assert payload["non_modeling_guards"]["no_dataset_acceptance"] is True


def test_phase17a_records_required_provenance_fields():
    payload = recover_phase17a("2026-09-11")
    required = {
        "candidate_id",
        "source_study_id",
        "publication",
        "doi_or_pubmed",
        "repository",
        "accession",
        "official_dataset_url",
        "parent_dataset",
        "subset_id",
        "experimental_condition",
        "organism_cell_line",
        "nuclease",
        "assay_type",
        "derived_or_processed_status",
        "provenance_notes",
        "unresolved_items",
    }
    for record in payload["candidates"]:
        assert required <= set(record)
        assert record["unresolved_items"]


def test_phase17a_preserves_original_files_and_hashes():
    payload = recover_phase17a("2026-09-11")
    recovered = payload["primary_files_recovered"]
    assert recovered
    for item in recovered:
        path = Path(item["local_raw_path"])
        assert path.exists()
        assert str(path).replace("\\", "/").startswith(str(RAW_DIR).replace("\\", "/"))
        assert sha256_file(path) == item["sha256"]
        assert len(item["sha256"]) == 64


def test_phase17a_canonical_integrity_guard():
    before = protected_hashes()
    payload = recover_phase17a("2026-09-11")
    after = protected_hashes()
    assert payload["canonical_integrity"]["unchanged"] is True
    assert before == after
    assert after["data/raw/Moreno-Mateos.csv"]["git_diff_clean"] is True
    assert all(record["exists"] for record in after.values())
    expected_model_checks = [
        record["matches_expected"]
        for path, record in after.items()
        if path.startswith("models/") and record["matches_expected"] is not None
    ]
    assert expected_model_checks and all(expected_model_checks)


def test_phase17a_candidate_status_logic():
    payload = recover_phase17a("2026-09-11")
    by_id = {record["candidate_id"]: record for record in payload["candidates"]}
    assert by_id["crispron_2021_rth_tools"]["acquisition_status"] == "RECOVERED_PRIMARY_WITH_UNRESOLVED_METADATA"
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["primary_file_located"] is True
    assert by_id["doench_2016_orcs_publication_screens"]["primary_file_located"] is True
    assert by_id["deep_hf_2019_public_data_listing"]["acquisition_status"] == "PRIMARY_FILE_NOT_FOUND"
    assert by_id["sgdesigner_2020_public_data_listing"]["acquisition_status"] == "INSUFFICIENT_PROVENANCE"
    assert by_id["corsi_2022_free_energy_pam_context"]["acquisition_status"] == "DISCOVERY_ONLY"


def test_phase17a_structure_inspection_only():
    payload = recover_phase17a("2026-09-11")
    by_id = {record["candidate_id"]: record for record in payload["candidates"]}
    crisprpred_file = by_id["crisprpredseq_2020_bmc_additional_files"]["files"][0]
    assert crisprpred_file["structure"]["row_count"] > 0
    assert "column_names" in crisprpred_file["structure"]
    doench_file = by_id["doench_2016_orcs_publication_screens"]["files"][0]
    assert "sheet_names" in doench_file["structure"]
    assert payload["non_modeling_guards"]["no_label_transformation"] is True


def test_phase17a_provenance_payload_is_deterministic_and_json_serializable():
    first = provenance_payload(recover_phase17a("2026-09-11"))
    second = provenance_payload(recover_phase17a("2026-09-11"))
    assert first == second
    json.dumps(first)


def test_phase17a_sources_do_not_import_models_or_moreno():
    paths = [
        Path("src/dataset_recovery/phase17a.py"),
        Path("scripts/run_phase17a_dataset_recovery.py"),
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

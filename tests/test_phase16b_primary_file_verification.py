import ast
from pathlib import Path

from src.dataset_landscape.verification import phase16b_file_verification


def test_phase16b_keeps_no_compatibility_acceptance():
    payload = phase16b_file_verification("2026-09-08")
    statuses = {record["candidate_status_after_16b"] for record in payload["records"]}
    assert "COMPATIBLE" not in statuses
    assert "CONDITIONALLY_COMPATIBLE" not in statuses


def test_verified_files_have_hashes_and_paths_when_present():
    payload = phase16b_file_verification("2026-09-08")
    verified = [
        item for record in payload["records"]
        for item in record["files"]
        if item["verified_local_file"]
    ]
    assert verified
    for item in verified:
        assert item["sha256"] is not None
        assert len(item["sha256"]) == 64
        assert item["local_raw_path"].startswith("data/phase16/raw/")
        assert item["source_url"].startswith("https://")


def test_deephf_prior_evidence_not_new_independent():
    payload = phase16b_file_verification("2026-09-08")
    deep_hf = next(
        record for record in payload["records"]
        if record["candidate_id"] == "deep_hf_2019_public_data_listing"
    )
    assert "already analyzed in Phases 10-12" in deep_hf["provenance_relationship"]
    assert deep_hf["primary_file_status"] == (
        "PRIMARY_FILE_NOT_OBTAINED_IN_16B_PRIOR_EVIDENCE_EXISTS"
    )


def test_no_moreno_model_or_training_imports_in_phase16b_sources():
    paths = list(Path("src/dataset_landscape").glob("*.py")) + [
        Path("scripts/run_phase16b_primary_file_verification.py")
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    for path in paths:
        text = path.read_text()
        assert "load_moreno_mateos" not in text
        assert "Moreno-Mateos.csv" not in text
        tree = ast.parse(text)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert forbidden_imports.isdisjoint(set(imports))


def test_no_downstream_phase16b_analyses():
    payload = phase16b_file_verification("2026-09-08")
    guards = payload["non_modeling_guards"]
    assert guards["no_label_compatibility_decision"] is True
    assert guards["no_sequence_transformation"] is True
    assert guards["no_overlap_analysis"] is True
    assert guards["no_distribution_analysis"] is True

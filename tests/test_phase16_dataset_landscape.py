import ast
from pathlib import Path

from src.dataset_landscape.discovery import build_phase16a_inventory
from src.dataset_landscape.provenance import (
    PENDING_PRIMARY_VERIFICATION,
    REQUIRED_IDENTITY_FIELDS,
    REQUIRED_PROVENANCE_FIELDS,
    UNKNOWN,
    validate_record,
)


def test_candidates_initialize_pending_primary_verification():
    inventory = build_phase16a_inventory(access_date="2026-09-08")
    assert inventory["candidate_count"] > 0
    assert {
        c["discovery_status"] for c in inventory["candidates"]
    } == {PENDING_PRIMARY_VERIFICATION}


def test_provenance_records_require_identity_fields():
    candidate = build_phase16a_inventory("2026-09-08")["candidates"][0]
    for field in REQUIRED_IDENTITY_FIELDS:
        assert field in candidate
    validation = validate_record(candidate)
    assert validation["schema_complete"] is True
    assert validation["status_is_pending_primary_verification"] is True


def test_unknown_fields_are_explicit_not_fabricated():
    candidates = build_phase16a_inventory("2026-09-08")["candidates"]
    assert any(c["source_repository"] != UNKNOWN for c in candidates)
    assert any(c["strand_orientation_documentation"] == UNKNOWN for c in candidates)
    for candidate in candidates:
        for field in REQUIRED_PROVENANCE_FIELDS:
            assert field in candidate


def test_identity_not_determined_by_candidate_name_only():
    candidate = {
        field: UNKNOWN for field in REQUIRED_PROVENANCE_FIELDS
    }
    candidate["candidate_id"] = "name_only"
    candidate["candidate_name"] = "HCT116"
    candidate["discovery_status"] = PENDING_PRIMARY_VERIFICATION
    validation = validate_record(candidate)
    assert validation["identity_not_name_only"] is False


def test_no_moreno_raw_path_or_loader_used_in_phase16a_sources():
    phase16_paths = list(Path("src/dataset_landscape").glob("*.py")) + [
        Path("scripts/run_phase16_dataset_landscape.py")
    ]
    text = "\n".join(path.read_text() for path in phase16_paths)
    assert "load_moreno_mateos" not in text
    assert "Moreno-Mateos.csv" not in text


def test_no_model_or_training_imports_in_phase16a_sources():
    phase16_paths = list(Path("src/dataset_landscape").glob("*.py")) + [
        Path("scripts/run_phase16_dataset_landscape.py")
    ]
    forbidden = {"src.models", "torch", "xgboost", "sklearn"}
    for path in phase16_paths:
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert forbidden.isdisjoint(set(imports))


def test_deterministic_output_structure():
    a = build_phase16a_inventory(access_date="2026-09-08")
    b = build_phase16a_inventory(access_date="2026-09-08")
    assert list(a.keys()) == list(b.keys())
    assert [c["candidate_id"] for c in a["candidates"]] == sorted(
        c["candidate_id"] for c in a["candidates"]
    )
    assert a == b


def test_no_near_duplicate_threshold_exists_in_phase16a():
    phase16_paths = list(Path("src/dataset_landscape").glob("*.py")) + [
        Path("scripts/run_phase16_dataset_landscape.py")
    ]
    text = "\n".join(path.read_text().lower() for path in phase16_paths)
    assert "hamming" not in text
    assert "near_duplicate_threshold" not in text

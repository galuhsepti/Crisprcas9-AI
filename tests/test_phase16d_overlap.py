import ast
from pathlib import Path

import pytest

from src.dataset_landscape.overlap import (
    RC_OVERLAP_NOT_ASSESSED,
    build_phase16d_overlap_audit,
    duplicate_counts,
    exact_overlap,
    label_conflicts,
    reverse_complement_overlap,
)


@pytest.fixture(scope="module")
def phase16d_payload():
    """Build the heavy Phase 16D audit once for this test module."""
    return build_phase16d_overlap_audit("2026-09-08")


def test_exact_30mer_overlap():
    a = ["A" * 30, "C" * 30]
    b = ["A" * 30, "G" * 30]
    report = exact_overlap(a, b)
    assert report["shared_unique"] == 1
    assert report["n_a"] == 2
    assert report["n_b"] == 2


def test_exact_guide_overlap():
    a = ["A" * 20, "C" * 20]
    b = ["C" * 20, "G" * 20]
    assert exact_overlap(a, b)["shared_unique"] == 1


def test_duplicate_records_and_sequences():
    report = duplicate_counts(["AAA", "AAA", "CCC"])
    assert report["duplicate_sequence_count"] == 1
    assert report["duplicate_record_excess"] == 1


def test_identical_sequence_label_conflicts():
    records = [
        {"sequence": "AAA", "label": 0},
        {"sequence": "AAA", "label": 1},
        {"sequence": "CCC", "label": 1},
    ]
    report = label_conflicts(records)
    assert report["n_conflicting_sequences"] == 1


def test_rc_overlap_not_assessed_when_orientation_unverified():
    report = reverse_complement_overlap(["AAA"], ["TTT"], orientation_verified=False)
    assert report["status"] == RC_OVERLAP_NOT_ASSESSED


def test_rc_overlap_only_when_orientation_verified():
    report = reverse_complement_overlap(["AAA"], ["TTT"], orientation_verified=True)
    assert report["status"] == "ASSESSED"
    assert report["shared_unique"] == 1


def test_phase16d_provenance_relationship_detection(phase16d_payload):
    payload = phase16d_payload
    by_id = {r["candidate_id"]: r for r in payload["candidate_reports"]}
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["provenance_relationship"][
        "independence_assessment"
    ] == "NOT_ESTABLISHED_DERIVED_OR_PROCESSED_RISK"


def test_phase16d_deterministic_output(phase16d_payload):
    assert phase16d_payload["phase"] == "16D"
    assert phase16d_payload["status"] == "OVERLAP_AUDIT_RECORDED"
    assert phase16d_payload["canonical_reference"]["n_rows"] == 12832
    by_id = {r["candidate_id"]: r for r in phase16d_payload["candidate_reports"]}
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["sequence_record_count"] == 16749
    assert by_id["doench_2016_orcs_publication_screens"]["sequence_record_count"] == 194653
    assert by_id["doench_2016_orcs_publication_screens"]["rc_overlap_status"]["status"] == RC_OVERLAP_NOT_ASSESSED


def test_no_forbidden_phase16d_sources():
    paths = list(Path("src/dataset_landscape").glob("*.py")) + [
        Path("scripts/run_phase16d_overlap_audit.py")
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    forbidden_text = [
        "load_moreno_mateos",
        "Moreno-Mateos.csv",
        "hamming",
        "edit-distance",
        "build_training_pool",
    ]
    for path in paths:
        text = path.read_text()
        for phrase in forbidden_text:
            assert phrase not in text
        tree = ast.parse(text)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert forbidden_imports.isdisjoint(set(imports))

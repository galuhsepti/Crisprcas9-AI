import ast
from pathlib import Path

from src.dataset_landscape.overlap import (
    RC_OVERLAP_NOT_ASSESSED,
    build_phase16d_overlap_audit,
    duplicate_counts,
    exact_overlap,
    label_conflicts,
    reverse_complement_overlap,
)


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


def test_phase16d_provenance_relationship_detection():
    payload = build_phase16d_overlap_audit("2026-09-08")
    by_id = {r["candidate_id"]: r for r in payload["candidate_reports"]}
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["provenance_relationship"][
        "independence_assessment"
    ] == "NOT_ESTABLISHED_DERIVED_OR_PROCESSED_RISK"


def test_phase16d_deterministic_output():
    a = build_phase16d_overlap_audit("2026-09-08")
    b = build_phase16d_overlap_audit("2026-09-08")
    assert a == b


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

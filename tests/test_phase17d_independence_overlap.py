import ast
import json
from pathlib import Path

from src.dataset_recovery.independence import (
    PHASE17B_RESULT,
    PHASE17C_RESULT,
    RC_OVERLAP_NOT_ASSESSED,
    build_phase17d_independence_overlap,
    duplicate_record_summary,
    duplicate_sequence_summary,
    exact_overlap,
    is_dna,
    label_conflicts,
    rc_overlap_status,
)


def test_exact_overlap_synthetic():
    report = exact_overlap(["AAAA", "CCCC"], ["CCCC", "GGGG"])
    assert report["shared_unique"] == 1
    assert report["shared_examples"] == ["CCCC"]


def test_duplicate_sequence_and_row_definitions():
    records = [
        {"sequence": "AAAA", "label_key": "1"},
        {"sequence": "AAAA", "label_key": "1"},
        {"sequence": "CCCC", "label_key": "0"},
    ]
    seq = duplicate_sequence_summary(records)
    rows = duplicate_record_summary(records, ["sequence", "label_key"])
    assert seq["duplicate_sequence_excess"] == 1
    assert rows["duplicate_record_excess"] == 1


def test_conflicting_label_sequence_synthetic():
    records = [
        {"sequence": "AAAA", "label_key": "0"},
        {"sequence": "AAAA", "label_key": "1"},
        {"sequence": "CCCC", "label_key": "1"},
    ]
    conflicts = label_conflicts(records, label_key="label_key")
    assert conflicts["conflict_sequence_count"] == 1


def test_orientation_not_established_guard():
    assert rc_overlap_status(False)["status"] == RC_OVERLAP_NOT_ASSESSED


def test_non_dna_metadata_filtering():
    assert is_dna("ACGTACGT") is True
    assert is_dna("STARS Score") is False
    assert is_dna("Rank 1") is False


def test_phase17d_candidate_reports_and_guards():
    payload = build_phase17d_independence_overlap(PHASE17B_RESULT, PHASE17C_RESULT)
    by_id = {record["candidate_id"]: record for record in payload["candidate_reports"]}
    assert by_id["crispron_2021_rth_tools"]["sequence_availability"] == "INSUFFICIENT_SEQUENCE_INFORMATION"
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["exact_guide_overlap_with_canonical_valid_guides"]["shared_unique"] >= 0
    assert by_id["doench_2016_orcs_publication_screens"]["duplicate_exact_sequences"]["duplicate_sequence_excess"] >= 0
    assert by_id["deep_hf_2019_public_data_listing"]["exact_guide_overlap_with_canonical_valid_guides"] == "INSUFFICIENT_SEQUENCE_INFORMATION"
    assert payload["canonical_integrity"]["unchanged"] is True
    assert all(payload["interpretation_safeguards"].values())
    assert payload["non_modeling_guards"]["no_moreno_raw_data_accessed"] is True
    assert payload["non_modeling_guards"]["no_dataset_pooling"] is True
    assert payload["non_modeling_guards"]["no_distance_based_matching"] is True


def test_phase17d_pairwise_overlap_is_exact_and_conservative():
    payload = build_phase17d_independence_overlap(PHASE17B_RESULT, PHASE17C_RESULT)
    pair = payload["candidate_to_candidate_overlap"][
        "crisprpredseq_2020_bmc_additional_files_vs_doench_2016_orcs_publication_screens"
    ]
    assert "guide_overlap" in pair
    assert pair["provenance_relationship"]["status"] == "INDEPENDENCE_UNRESOLVED"
    assert pair["guide_overlap"]["shared_unique"] >= 0


def test_phase17d_deterministic_and_json_serializable():
    first = build_phase17d_independence_overlap(PHASE17B_RESULT, PHASE17C_RESULT)
    second = build_phase17d_independence_overlap(PHASE17B_RESULT, PHASE17C_RESULT)
    assert first == second
    json.dumps(first)


def test_phase17d_sources_do_not_import_models_or_forbidden_logic():
    paths = [
        Path("src/dataset_recovery/independence.py"),
        Path("scripts/run_phase17d_independence_overlap_audit.py"),
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

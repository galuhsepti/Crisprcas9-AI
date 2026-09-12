import ast
import json
from pathlib import Path

from src.dataset_recovery.data_sufficiency import (
    DEFAULT_THRESHOLDS,
    build_phase17g_data_sufficiency,
    exact_overlap_summary,
    final_readiness_decision,
    label_quality,
    quantity_gate,
    sequence_usability,
)


def test_quantity_threshold_boundaries():
    assert quantity_gate(999)["status"] == "INSUFFICIENT_QUANTITY"
    assert quantity_gate(1000)["status"] == "MINIMUM_ONLY"
    assert quantity_gate(1999)["status"] == "MINIMUM_ONLY"
    assert quantity_gate(2000)["status"] == "RECOMMENDED_QUANTITY_MET"


def test_sequence_duplicate_invalid_and_length_handling():
    report = sequence_usability(["ACGT", "ACGT", "AAAA", "ACGN", "", None, "TTTTTT"])
    assert report["raw_n"] == 7
    assert report["valid_sequence_n"] == 4
    assert report["unique_sequence_n"] == 3
    assert report["duplicate_sequence_n"] == 1
    assert report["malformed_sequence_n"] == 1
    assert report["length_distribution"] == {4: 4, 6: 1}


def test_label_quality_missing_constant_low_cardinality_and_continuous():
    constant = label_quality([0.5, 0.5, None], "LABEL_COMPATIBLE", canonical_scale_established=True)
    assert constant["status"] == "CONSTANT_LABEL"
    assert constant["missing_labels"] == 1
    low = label_quality([0, 1, 0, 1], "LABEL_COMPATIBLE", canonical_scale_established=True)
    assert low["status"] == "DISCRETE_LOW_CARDINALITY"
    continuous = label_quality([i / 25 for i in range(25)], "LABEL_COMPATIBLE", canonical_scale_established=True)
    assert continuous["status"] == "CONTINUOUS_USABLE"
    unknown = label_quality([i / 25 for i in range(25)], "LABEL_COMPATIBLE", canonical_scale_established=False)
    assert unknown["status"] == "UNKNOWN_SCALE"


def test_incompatible_binary_label_is_blocked():
    report = label_quality([0, 1, 1, 0], "LABEL_INCOMPATIBLE")
    assert report["status"] == "INCOMPATIBLE_TARGET"
    assert report["compatible_label_n"] == 0


def test_exact_overlap_summary():
    report = exact_overlap_summary(["AAAA", "CCCC", "NNNN", "AAAA"], ["CCCC", "GGGG"])
    assert report["candidate_unique_n"] == 2
    assert report["exact_overlap_n"] == 1
    assert report["exact_overlap_fraction"] == 0.5


def test_final_go_no_go_logic():
    ready = {
        "final_candidate_status": "ACCEPTED_READY_CANDIDATE",
        "unique_sequence_n": 2000,
    }
    assert final_readiness_decision([ready])["final_decision"] == "GO"
    small = dict(ready, unique_sequence_n=1000)
    assert final_readiness_decision([small])["final_decision"] == "NO_GO"
    assert final_readiness_decision([])["final_decision"] == "NO_GO"


def test_phase17g_real_output_is_no_go_and_preserves_decisions():
    payload = build_phase17g_data_sufficiency(timestamp="2026-09-11T00:00:00")
    assert payload["final_decision"] == "NO_GO"
    assert payload["aggregate"]["total_new_unique_compatible_observations"] == 0
    assert payload["additional_compatible_unique_observations_required"] == 2000
    by_id = {c["candidate_id"]: c for c in payload["candidates"]}
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["phase17f_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET"
    assert by_id["doench_2016_orcs_publication_screens"]["phase17f_acceptance_status"] == "REJECTED_FOR_CURRENT_TARGET"
    assert by_id["deep_hf_2019_public_data_listing"]["phase17f_acceptance_status"] == "ALREADY_ANALYZED_NOT_NEW"


def test_phase17g_population_thresholds_and_scope():
    payload = build_phase17g_data_sufficiency(timestamp="2026-09-11T00:00:00")
    assert payload["thresholds"] == DEFAULT_THRESHOLDS
    ref = payload["canonical_reference"]
    assert ref["phase16e_distribution_population_n"] == 12832
    assert ref["canonical_modeling_population_n"] == 10117
    assert ref["train_n"] == 8599
    assert ref["validation_n"] == 1518
    assert payload["scope"]["no_moreno_access"] is True
    assert payload["scope"]["no_phase18"] is True
    assert payload["canonical_integrity"]["unchanged"] is True


def test_phase17g_deterministic_json_except_timestamp():
    first = build_phase17g_data_sufficiency(timestamp="A")
    second = build_phase17g_data_sufficiency(timestamp="B")
    first["timestamp"] = None
    second["timestamp"] = None
    assert first == second
    json.dumps(first)


def test_phase17g_sources_do_not_import_models_or_forbidden_logic():
    paths = [
        Path("src/dataset_recovery/data_sufficiency.py"),
        Path("scripts/run_phase17g_data_sufficiency_audit.py"),
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    forbidden_text = [
        "Moreno-Mateos.csv",
        "load_moreno_mateos",
        "build_training_pool",
        ".fit(",
        ".predict(",
        "reverse_complement(",
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

import ast
import json
from pathlib import Path

import pytest

from src.dataset_recovery.distribution import (
    DATASET_TYPES,
    INFO_VALUE_STATUSES,
    NO_ACTIVITY,
    activity_distribution_for_candidate,
    build_phase17e_distribution,
    gc_fraction,
    jensen_shannon_divergence,
    kmer_summary,
    sequence_distribution,
)


@pytest.fixture(scope="module")
def phase17e_payload():
    return build_phase17e_distribution()


def test_gc_and_sequence_statistics_are_deterministic():
    report = sequence_distribution(["ACGT", "GGCC", "", None, "ACNT"], kmer=True)
    assert gc_fraction("ACGT") == 0.5
    assert report["total_records"] == 5
    assert report["records_with_valid_acgt_sequence"] == 2
    assert report["missing_sequence_count"] == 2
    assert report["non_acgt_records"] == 1
    assert report["length_distribution"] == {4: 3}
    assert report["gc"]["mean"] == 0.75
    assert report["nucleotide_composition"] == {"A": 0.125, "C": 0.375, "G": 0.375, "T": 0.125}


def test_kmer_and_jsd_are_deterministic_descriptive_statistics():
    kmers = kmer_summary(["AAAA", "AACC"], 2)
    assert kmers["k"] == 2
    assert kmers["mean_frequencies"]["AA"] > 0
    assert jensen_shannon_divergence({"A": 1.0}, {"A": 1.0}) == 0.0
    assert jensen_shannon_divergence({"A": 1.0}, {"C": 1.0}) > 0.0


def test_activity_analysis_blocked_for_incompatible_labels():
    binary = {"label_status": "LABEL_INCOMPATIBLE"}
    report = activity_distribution_for_candidate(binary)
    assert report["status"] == "NOT_ANALYZED_LABEL_NOT_COMPATIBLE"
    assert "Phase 17C" in report["reason"]


def test_phase17e_population_distinction_and_global_activity_block(phase17e_payload):
    payload = phase17e_payload
    ref = payload["canonical_reference"]
    assert ref["distribution_population_n"] == 12832
    assert ref["canonical_modeling_population_n"] == 10117
    assert ref["train_n"] == 8599
    assert ref["validation_n"] == 1518
    assert payload["activity_distribution_global_status"] == NO_ACTIVITY


def test_phase17e_candidate_classification_binding_to_phase17c(phase17e_payload):
    payload = phase17e_payload
    by_id = {record["candidate_id"]: record for record in payload["candidates"]}
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["dataset_type"] == "INCOMPATIBLE_LABEL_DATA"
    assert by_id["doench_2016_orcs_publication_screens"]["dataset_type"] == "INCOMPATIBLE_LABEL_DATA"
    assert by_id["crispron_2021_rth_tools"]["dataset_type"] == "INSUFFICIENT_DATA"
    assert by_id["deep_hf_2019_public_data_listing"]["dataset_type"] == "INSUFFICIENT_DATA"
    assert by_id["sgdesigner_2020_public_data_listing"]["dataset_type"] == "INSUFFICIENT_DATA"
    assert by_id["corsi_2022_free_energy_pam_context"]["dataset_type"] == "INSUFFICIENT_DATA"
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["verified_continuous_activity_available"] is False
    assert by_id["doench_2016_orcs_publication_screens"]["verified_continuous_activity_available"] is False


def test_sequence_descriptive_analysis_without_30mer_conversion(phase17e_payload):
    payload = phase17e_payload
    by_id = {record["candidate_id"]: record for record in payload["candidates"]}
    crisprpred = by_id["crisprpredseq_2020_bmc_additional_files"]["sequence_distribution"]
    doench = by_id["doench_2016_orcs_publication_screens"]["sequence_distribution"]
    assert crisprpred["length_distribution"] == {23: 16749}
    assert 30 not in crisprpred["length_distribution"]
    assert doench["records_with_valid_acgt_sequence"] > 0
    assert "no 30-mer construction" in crisprpred["analysis_note"]
    assert "no 30-mer construction" in doench["analysis_note"]


def test_information_value_vocabularies_and_statuses(phase17e_payload):
    payload = phase17e_payload
    assert set(payload["dataset_types"]) == DATASET_TYPES
    assert set(payload["information_value_statuses"]) == INFO_VALUE_STATUSES
    for record in payload["candidates"]:
        assert record["dataset_type"] in DATASET_TYPES
        assert record["information_value_status"] in INFO_VALUE_STATUSES


def test_no_label_transformation_no_pooling_no_moreno_and_canonical_protection(phase17e_payload):
    payload = phase17e_payload
    guards = payload["non_modeling_guards"]
    assert guards["no_label_transformation"] is True
    assert guards["no_sequence_transformation"] is True
    assert guards["no_dataset_pooling"] is True
    assert guards["no_moreno_raw_data_accessed"] is True
    assert guards["no_model_training"] is True
    assert guards["no_model_evaluation"] is True
    assert guards["no_continual_learning"] is True
    assert payload["canonical_integrity"]["unchanged"] is True
    assert payload["safety_guards"]["binary_labels_not_activity"] is True
    assert payload["safety_guards"]["rank_enrichment_logfc_not_activity"] is True
    assert payload["safety_guards"]["guide_or_23mer_not_converted_to_30mer"] is True


def test_phase17e_deterministic_and_json_serializable():
    first = build_phase17e_distribution()
    second = build_phase17e_distribution()
    assert first == second
    json.dumps(first)


def test_phase17e_sources_do_not_import_models_or_forbidden_logic():
    paths = [
        Path("src/dataset_recovery/distribution.py"),
        Path("scripts/run_phase17e_distribution_audit.py"),
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    forbidden_text = [
        "Moreno-Mateos.csv",
        "load_moreno_mateos",
        "build_training_pool",
        ".fit(",
        ".predict(",
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

import ast
from pathlib import Path

from src.dataset_landscape.compatibility import (
    COMPATIBLE,
    CONDITIONALLY_COMPATIBLE,
    INCOMPATIBLE,
    INSUFFICIENT_PROVENANCE,
    build_phase16c_compatibility,
    classify_label_schema,
    classify_sequence_schema,
)


def test_true_compatible_continuous_experimental_measurement():
    result = classify_label_schema("editing fraction", True, True, True)
    assert result["status"] == COMPATIBLE


def test_disallowed_label_types_rejected():
    assert classify_label_schema("binary", True, False, False, is_binary=True)["status"] == INCOMPATIBLE
    assert classify_label_schema("rank", True, False, False, is_rank=True)["status"] == INCOMPATIBLE
    assert classify_label_schema("enrichment", True, True, False, is_enrichment=True)["status"] == INCOMPATIBLE
    assert classify_label_schema("predicted", False, True, False, is_predicted_score=True)["status"] == INCOMPATIBLE


def test_unit_conversion_only_when_measurement_matches():
    ok = classify_label_schema("editing percent", True, True, True, unit_conversion_only=True)
    bad = classify_label_schema("rank percent", True, False, False, is_rank=True, unit_conversion_only=True)
    assert ok["status"] == COMPATIBLE
    assert ok["unit_conversion"] is True
    assert bad["status"] == INCOMPATIBLE


def test_label_harmonization_not_silently_performed():
    result = classify_label_schema("normalized proxy", True, True, None)
    assert result["status"] == INSUFFICIENT_PROVENANCE
    assert result["label_harmonization_performed"] is False


def test_noncanonical_sequence_schemas_are_conditional():
    assert classify_sequence_schema("guide_only", 20, False, False)["status"] == CONDITIONALLY_COMPATIBLE
    assert classify_sequence_schema("guide_plus_pam", 23, False, False)["status"] == CONDITIONALLY_COMPATIBLE


def test_variable_unknown_orientation_not_transformed():
    result = classify_sequence_schema("variable_length", None, False, False)
    assert result["status"] == INSUFFICIENT_PROVENANCE
    assert result["transformation_status"] == "NOT_PERFORMED"


def test_synthetic_flanks_not_treated_as_genomic():
    result = classify_sequence_schema("canonical_30mer", 30, True, False, synthetic_or_constant_flanks=True)
    assert result["status"] == CONDITIONALLY_COMPATIBLE


def test_phase16c_candidate_decisions_are_conservative():
    payload = build_phase16c_compatibility("2026-09-08")
    by_id = {record["candidate_id"]: record for record in payload["records"]}
    assert by_id["crisprpredseq_2020_bmc_additional_files"]["compatibility_status"] == INCOMPATIBLE
    assert by_id["doench_2016_orcs_publication_screens"]["compatibility_status"] == INCOMPATIBLE
    assert by_id["crispron_2021_rth_tools"]["compatibility_status"] == INSUFFICIENT_PROVENANCE


def test_no_moreno_model_pooling_or_downstream_imports():
    paths = list(Path("src/dataset_landscape").glob("*.py")) + [
        Path("scripts/run_phase16c_compatibility_audit.py")
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    for path in paths:
        text = path.read_text()
        assert "load_moreno_mateos" not in text
        assert "Moreno-Mateos.csv" not in text
        assert "build_training_pool" not in text
        tree = ast.parse(text)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert forbidden_imports.isdisjoint(set(imports))


def test_deterministic_output():
    a = build_phase16c_compatibility("2026-09-08")
    b = build_phase16c_compatibility("2026-09-08")
    assert a == b

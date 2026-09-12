import ast
import json
from pathlib import Path

import pandas as pd

from src.dataset_recovery.primary_acquisition import (
    build_phase17h_primary_acquisition,
    duplicate_source_detection,
    inspect_table,
    label_profile,
    normalize_doi,
    sequence_summary,
    sha256_file,
)
from src.dataset_recovery.provenance_chain import validate_provenance_chain


def test_sha256_calculation(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_doi_normalization_and_duplicate_source_detection():
    assert normalize_doi("https://doi.org/10.1038/S41467-022-30515-0.") == "10.1038/s41467-022-30515-0"
    records = [
        {"doi": "doi:10.1/ABC", "primary_publication": "A"},
        {"doi": "https://doi.org/10.1/abc", "primary_publication": "B"},
        {"doi": "", "primary_publication": "C"},
    ]
    dup = duplicate_source_detection(records)
    assert dup["duplicates"] == ["doi:10.1/abc"]
    assert dup["unique_source_count"] == 2


def test_provenance_chain_validation():
    complete = validate_provenance_chain({
        "primary_publication": "paper",
        "primary_file": "file.xlsx",
        "table_or_sheet": "sheet",
        "sequence_column": "seq",
        "label_column": "label",
        "primary_experimental_measurement": True,
    })
    assert complete["status"] == "VERIFIED_PRIMARY"
    missing = validate_provenance_chain({"primary_publication": "paper"})
    assert missing["status"] == "UNVERIFIED"
    assert "primary_file" in missing["missing"]


def test_missing_sequence_and_label_column_detection():
    frame = pd.DataFrame({"seq": ["ACGT"], "label": [0.5]})
    assert inspect_table(frame, None, "label")["status"] == "MISSING_SEQUENCE_COLUMN"
    assert inspect_table(frame, "seq", None)["status"] == "MISSING_LABEL_COLUMN"


def test_malformed_sequence_reporting_and_length_counting():
    report = sequence_summary(["ACGT", "ACGT", "ACGN", "", None, "AAAAAA"])
    assert report["valid_sequence_n"] == 3
    assert report["malformed_sequence_n"] == 1
    assert report["sequence_length_distribution"] == {4: 3, 6: 1}
    assert report["unique_sequence_n"] == 2


def test_continuous_binary_and_screen_label_detection():
    continuous = label_profile([i / 10 for i in range(25)], "Indel frequency (%)")
    assert continuous["label_status"] == "CONTINUOUS_EXPERIMENTAL_CANDIDATE"
    binary = label_profile([0, 1, 0, 1], "binary label")
    assert binary["label_status"] == "INCOMPATIBLE_LABEL"
    rank = label_profile([1, 2, 3, 4], "STARS rank")
    assert rank["label_status"] == "SCREEN_OR_RANK_LABEL"


def test_table_unique_observation_counting():
    frame = pd.DataFrame({
        "seq": ["ACGT", "ACGT", "AAAA", "CCCC"],
        "label": [0.1, 0.2, 0.3, 0.4],
    })
    report = inspect_table(frame, "seq", "label")
    assert report["status"] == "INSPECTED"
    assert report["sequence"]["unique_sequence_n"] == 3


def test_phase17h_payload_and_manifest_are_deterministic():
    first = build_phase17h_primary_acquisition(retrieval_date="2026-09-11")
    second = build_phase17h_primary_acquisition(retrieval_date="2026-09-11")
    assert first == second
    assert first["phase"] == "17H"
    assert "corsi_2022_pam_context_supplementary_data_1" in first["promising_candidates"]
    assert first["potential_total_unique_observations"] == 1022
    assert first["remaining_deficit_relative_to_2000"] == 978
    manifest = Path("data/phase17h/manifests/phase17h_investigated_sources.json")
    assert manifest.exists()
    json.dumps(first)


def test_no_canonical_path_overwrite_and_no_moreno_access():
    payload = build_phase17h_primary_acquisition(retrieval_date="2026-09-11")
    assert payload["canonical_protection"]["no_moreno_access"] is True
    assert payload["canonical_protection"]["no_model_training"] is True
    assert payload["canonical_integrity"]["unchanged"] is True
    assert all(not f["path"].startswith("data/raw/") for f in payload["files_recovered"])


def test_phase17h_sources_do_not_import_models_or_forbidden_logic():
    paths = [
        Path("src/dataset_recovery/primary_acquisition.py"),
        Path("src/dataset_recovery/provenance_chain.py"),
        Path("scripts/run_phase17h_primary_dataset_acquisition.py"),
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

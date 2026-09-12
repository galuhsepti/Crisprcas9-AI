import ast
from pathlib import Path

import pandas as pd

from src.dataset_recovery.biological_independence_crispron import (
    audit_crispron,
    clean_dna,
    detect_condition_conflicts,
    effective_guide_diversity_ratio,
    extract_spacer20_from_30mer,
    extract_target_pam23_from_30mer,
    load_deepspcas9_sequences,
    overlap_summary,
    separate_crispron_sources,
    sha256_file,
    unique_count,
)


SEQ30_A = "AAAA" + "C" * 20 + "GGG" + "TTT"
SEQ30_B = "TTTT" + "G" * 20 + "AGG" + "AAA"
SEQ30_C = "CCCC" + "A" * 20 + "TGG" + "GGG"


def test_20mer_spacer_extraction_from_canonical_30mer():
    assert extract_spacer20_from_30mer(SEQ30_A) == "CCCCCCCCCCCCCCCCCCCC"
    assert extract_target_pam23_from_30mer(SEQ30_A) == "CCCCCCCCCCCCCCCCCCCCGGG"


def test_unique_spacer_counting():
    spacers = [extract_spacer20_from_30mer(s) for s in [SEQ30_A, SEQ30_A, SEQ30_B]]
    assert unique_count(spacers) == 2


def test_condition_aware_duplicate_and_conflicting_label_detection():
    frame = pd.DataFrame(
        {
            "seq": [SEQ30_A, SEQ30_A, SEQ30_B, SEQ30_B],
            "condition": ["Dox-", "Dox+", "Dox-", "Dox+"],
            "label": [10.0, 20.0, 5.0, 5.0],
        }
    )
    report = detect_condition_conflicts(frame, "seq", "condition", "label")
    assert report["shared_sequence_n"] == 2
    assert report["shared_sequence_with_label_difference_n"] == 1
    assert report["median_absolute_label_difference"] == 10.0
    assert report["pooling_creates_target_ambiguity"] is True


def test_effective_guide_diversity_calculation():
    assert effective_guide_diversity_ratio(10, 100) == 0.1
    assert effective_guide_diversity_ratio(10, 0) == 0.0


def test_crispron_source_component_separation(tmp_path):
    path = tmp_path / "crispron.xlsx"
    frame = pd.DataFrame(
        {
            "Source": ["Xiang/Luo", "Kim2019", "Other"],
            "30mer_gRNA": [SEQ30_A, SEQ30_B, SEQ30_C],
            "Quant_norm_efficiency": [0.1, 0.2, 0.3],
        }
    )
    frame.to_excel(path, index=False)
    report = separate_crispron_sources(path)
    assert report["source_identity_recoverable"] is True
    assert report["xiang_luo"]["raw_n"] == 1
    assert report["kim"]["unique_30mer_n"] == 1
    assert report["other"]["unique_spacer20_n"] == 1


def test_deepspcas9_exact_30mer_and_spacer_overlap():
    canonical = {
        "unique_30mer": {SEQ30_A},
        "unique_spacer20": {extract_spacer20_from_30mer(SEQ30_A), extract_spacer20_from_30mer(SEQ30_B)},
    }
    report = overlap_summary([SEQ30_A, SEQ30_B, "NNNN"], canonical)
    assert report["unique_30mer_n"] == 2
    assert report["exact_deepspcas9_overlap_n"] == 1
    assert report["spacer20_overlap_with_deepspcas9_n"] == 2
    assert report["new_unique_30mer_n"] == 1


def test_malformed_sequence_handling():
    assert clean_dna("ACGN") is None
    assert extract_spacer20_from_30mer("ACGT") is None
    assert clean_dna(" acgt ") == "ACGT"


def test_source_file_hash_validation(tmp_path):
    path = tmp_path / "hash.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_crispron_audit_blocks_without_recovered_file():
    report = audit_crispron(None, {"retrieval_status": "FAILED", "file_sha256": None, "source_url": "x"})
    assert report["preliminary_status"] == "BLOCKED"


def test_deepspcas9_loader_uses_only_deepspcas9():
    seqs = load_deepspcas9_sequences()
    assert seqs["unique_30mer"]
    assert seqs["unique_spacer20"]


def test_no_moreno_access_and_canonical_path_protection():
    paths = [
        Path("src/dataset_recovery/biological_independence_crispron.py"),
        Path("scripts/run_phase17i_biological_independence_crispron_recovery.py"),
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
        "data/raw/Moreno",
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

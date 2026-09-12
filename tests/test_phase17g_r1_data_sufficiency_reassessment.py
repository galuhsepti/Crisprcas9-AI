import ast
from pathlib import Path

import pandas as pd

from src.dataset_recovery.data_sufficiency_reassessment import (
    build_phase17g_r1,
    classify_guide_diversity,
    final_decision_from_candidates,
    label_profile,
    readiness_accounting,
    sequence_profile,
    source_masks,
)


SEQ30_A = "AAAA" + "C" * 20 + "GGG" + "TTT"
SEQ30_B = "TTTT" + "G" * 20 + "AGG" + "AAA"


def test_source_component_separation_masks():
    frame = pd.DataFrame({"Dataset": ["LuoSpCas92020_min200", "Kim2019_Train", "Overlap_Luo_Kim2019"]})
    masks = source_masks(frame)
    assert masks["crispron_xiang_luo"].sum() == 2
    assert masks["crispron_kim"].sum() == 2
    assert masks["crispron_xiang_luo_exclusive"].sum() == 1
    assert masks["crispron_kim_exclusive"].sum() == 1
    assert masks["crispron_overlap_luo_kim"].sum() == 1


def test_corsi_low_guide_diversity_handling():
    report = classify_guide_diversity(4, 1022)
    assert report["status"] == "LOW_GUIDE_DIVERSITY"
    assert report["guide_diversity_ratio"] < 0.01


def test_30mer_vs_spacer_diversity_distinction():
    profile = sequence_profile([SEQ30_A, "TTTT" + "C" * 20 + "AGG" + "AAA"])
    assert profile["unique_30mer_n"] == 2
    assert profile["unique_spacer20_n"] == 1


def test_accepted_new_observation_accounting_and_overlap_removal():
    rows = [
        {"candidate": "a", "accepted_for_future_pipeline": True, "accepted_new_unique_n": 1500},
        {"candidate": "b", "accepted_for_future_pipeline": False, "accepted_new_unique_n": 9000},
    ]
    aggregate = readiness_accounting(rows, {"minimum_total_new_unique_n": 2000})
    assert aggregate["accepted_candidate_count"] == 1
    assert aggregate["total_new_unique_compatible_observations"] == 1500
    assert aggregate["surplus_or_deficit"] == -500


def test_candidate_and_aggregate_boundaries():
    thresholds = {"minimum_candidate_unique_n": 1000, "recommended_candidate_unique_n": 2000, "minimum_total_new_unique_n": 2000}
    low = [{"candidate": "a", "accepted_for_future_pipeline": True, "accepted_new_unique_n": 1999}]
    high = [{"candidate": "a", "accepted_for_future_pipeline": True, "accepted_new_unique_n": 2000}]
    assert final_decision_from_candidates(low, thresholds)["final_decision"] == "NO_GO"
    assert final_decision_from_candidates(high, thresholds)["final_decision"] == "GO"


def test_normalized_primary_continuous_label_classification():
    report = label_profile([float(i) for i in range(25)], "Quant_norm_efficiency", "NORMALIZED_PRIMARY_CONTINUOUS")
    assert report["classification"] == "NORMALIZED_PRIMARY_CONTINUOUS"
    assert report["phase17g_continuous_label_status"] == "CONTINUOUS_USABLE"


def test_nonprimary_derived_label_rejection():
    report = label_profile([float(i) for i in range(25)], "model_score", "DERIVED_NONPRIMARY")
    assert report["phase17g_continuous_label_status"] == "NOT_CONTINUOUS_USABLE"


def test_phase17g_r1_real_output_go_and_exclusions():
    payload = build_phase17g_r1(timestamp="2026-09-11T00:00:00")
    by_id = {c["candidate_id"]: c for c in payload["candidates"]}
    assert by_id["crispron_xiang_luo"]["final_candidate_status"] == "ACCEPTED_READY_CANDIDATE"
    assert by_id["crispron_xiang_luo"]["independence_gate"]["overlap_30mer_n"] == 48
    assert by_id["crispron_xiang_luo"]["accepted_new_unique_n"] == 10544
    assert by_id["corsi_2022"]["final_candidate_status"] == "AUXILIARY_ONLY"
    assert by_id["corsi_2022"]["accepted_new_unique_n"] == 0
    assert by_id["crispron_kim"]["final_candidate_status"] == "ALREADY_REPRESENTED"
    assert by_id["crispron_kim"]["accepted_new_unique_n"] == 0
    assert payload["aggregate"]["accepted_candidate_count"] == 1
    assert payload["aggregate"]["total_new_unique_compatible_observations"] == 10544
    assert payload["final_decision"] == "GO"


def test_final_no_go_logic_without_accepted_candidates():
    thresholds = {"minimum_candidate_unique_n": 1000, "recommended_candidate_unique_n": 2000, "minimum_total_new_unique_n": 2000}
    result = final_decision_from_candidates(
        [{"candidate": "corsi", "accepted_for_future_pipeline": False, "accepted_new_unique_n": 0}],
        thresholds,
    )
    assert result["final_decision"] == "NO_GO"


def test_no_moreno_access():
    paths = [
        Path("src/dataset_recovery/data_sufficiency_reassessment.py"),
        Path("scripts/run_phase17g_r1_data_sufficiency_reassessment.py"),
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

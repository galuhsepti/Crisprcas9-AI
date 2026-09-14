import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_integration.label_compatibility import (
    DEEPSPCAS9_PATH,
    EXPECTED_HASHES,
    build_phase18a_audit,
    decide_integration_strategy,
    extract_shared_sequence_bridge,
    label_statistics,
    prohibit_locked_path,
    require_expected_integrity,
    require_canonical_path,
)
from scripts.run_phase18a_label_compatibility_audit import render_report

SEQ_A = "ACGT" + "ACGT" * 5 + "AGG" + "TTA"
SEQ_B = "TGCA" + "TGCA" * 5 + "CGG" + "AAC"
SEQ_C = "CAGT" + "CAGT" * 5 + "GGG" + "TTC"


def test_label_statistics_and_quantiles():
    result = label_statistics([0, 1, 2, 3, 4])
    assert result["n"] == 5
    assert result["finite_n"] == 5
    assert result["missing_n"] == 0
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["SD"] == pytest.approx(np.std([0, 1, 2, 3, 4], ddof=1))
    assert result["IQR"] == 2.0
    assert result["quantiles_percent"]["0"] == 0.0
    assert result["quantiles_percent"]["50"] == 2.0
    assert result["quantiles_percent"]["100"] == 4.0
    assert result["skewness"] == pytest.approx(0.0)
    assert result["fraction_at_min"] == 0.2
    assert result["fraction_at_max"] == 0.2


def test_label_statistics_handles_missing_and_infinite_values():
    result = label_statistics([1, None, "bad", np.nan, np.inf, 3])
    assert result["n"] == 6
    assert result["finite_n"] == 2
    assert result["missing_n"] == 4
    assert result["min"] == 1.0
    assert result["max"] == 3.0


def test_label_statistics_handles_constant_values():
    result = label_statistics([7, 7, 7])
    assert result["unique_value_n"] == 1
    assert result["SD"] == 0.0
    assert result["IQR"] == 0.0
    assert result["skewness"] == 0.0
    assert result["fraction_at_min"] == 1.0
    assert result["fraction_at_max"] == 1.0


def test_exact_30mer_bridge_and_correlations():
    deep = pd.DataFrame(
        {"sequence_30mer": [SEQ_A, SEQ_B, SEQ_C], "activity": [1.0, 2.0, 3.0]}
    )
    xiang = pd.DataFrame(
        {
            "30mer_gRNA": [SEQ_C, SEQ_A, "AAAA" + "ATGC" * 5 + "TGG" + "CCC", SEQ_B],
            "HEK293T_indel_freq_avg_d8_d10": [30.0, 10.0, 99.0, 20.0],
        }
    )
    result = extract_shared_sequence_bridge(deep, xiang)
    assert result["n"] == 3
    assert result["finite_pair_n"] == 3
    assert result["pearson"]["coefficient"] == pytest.approx(1.0)
    assert result["spearman"]["coefficient"] == pytest.approx(1.0)
    assert result["kendall"]["coefficient"] == pytest.approx(1.0)
    assert result["rank_agreement"]["mean_absolute_percentile_rank_disagreement"] == 0.0
    assert [row["sequence_30mer"] for row in result["rows"]] == sorted(
        [SEQ_A, SEQ_B, SEQ_C]
    )


def test_bridge_rank_disagreement_and_missing_labels():
    deep = pd.DataFrame(
        {"sequence_30mer": [SEQ_A, SEQ_B, SEQ_C], "activity": [1.0, 2.0, np.nan]}
    )
    xiang = pd.DataFrame(
        {
            "30mer_gRNA": [SEQ_A, SEQ_B, SEQ_C],
            "HEK293T_indel_freq_avg_d8_d10": [20.0, 10.0, 30.0],
        }
    )
    result = extract_shared_sequence_bridge(deep, xiang)
    assert result["n"] == 3
    assert result["finite_pair_n"] == 2
    assert result["missing_pair_n"] == 1
    assert result["spearman"]["coefficient"] == pytest.approx(-1.0)
    assert result["rank_agreement"]["mean_absolute_percentile_rank_disagreement"] == 1.0


def test_bridge_constant_label_is_explicitly_undefined():
    deep = pd.DataFrame(
        {"sequence_30mer": [SEQ_A, SEQ_B, SEQ_C], "activity": [1.0, 1.0, 1.0]}
    )
    xiang = pd.DataFrame(
        {
            "30mer_gRNA": [SEQ_A, SEQ_B, SEQ_C],
            "HEK293T_indel_freq_avg_d8_d10": [10.0, 20.0, 30.0],
        }
    )
    result = extract_shared_sequence_bridge(deep, xiang)
    assert result["pearson"]["coefficient"] is None
    assert result["pearson"]["status"] == "UNDEFINED_CONSTANT_LABEL"
    assert result["spearman"]["status"] == "UNDEFINED_CONSTANT_LABEL"
    assert result["linear_regression"]["status"] == "UNDEFINED_CONSTANT_LABEL"


def test_strategy_decision_logic():
    direct = decide_integration_strategy(True, True, True)
    controlled = decide_integration_strategy(True, False, True)
    separate = decide_integration_strategy(True, False, False)
    insufficient = decide_integration_strategy(False, False, False)
    assert direct["recommended_strategy"] == "DIRECT_POOLING_READY"
    assert direct["immediate_pooling_justified"] is True
    assert controlled["recommended_strategy"] == "MULTI_DOMAIN_EXPERIMENT_RECOMMENDED"
    assert controlled["controlled_integration_experiments_justified"] is True
    assert separate["recommended_strategy"] == "KEEP_DATASETS_SEPARATE"
    assert insufficient["recommended_strategy"] == "INSUFFICIENT_EVIDENCE"


def test_locked_path_prohibition():
    with pytest.raises(PermissionError):
        prohibit_locked_path(Path("data/raw") / ("Mor" + "eno-Mateos.csv"))


def test_canonical_path_protection():
    assert require_canonical_path(DEEPSPCAS9_PATH, DEEPSPCAS9_PATH) == DEEPSPCAS9_PATH
    with pytest.raises(ValueError):
        require_canonical_path(Path("data/raw/not-canonical.csv"), DEEPSPCAS9_PATH)


def test_phase18a_source_has_no_training_calls_or_model_imports():
    paths = [
        Path("src/dataset_integration/label_compatibility.py"),
        Path("scripts/run_phase18a_label_compatibility_audit.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not any(
            name == "torch" or name.startswith("src.models") for name in imports
        )
        assert calls.isdisjoint({"fit", "predict"})


def test_expected_hashes_are_valid_sha256_values():
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in EXPECTED_HASHES.values()
    )


def test_integrity_guard_rejects_mismatch():
    with pytest.raises(RuntimeError, match="integrity check failed"):
        require_expected_integrity({"artifact": {"matches_expected": False}})


def test_real_phase18a_build_and_report_are_serializable():
    payload = build_phase18a_audit("test-timestamp")
    encoded = json.dumps(payload, allow_nan=False)
    report = render_report(payload)

    assert encoded
    assert payload["recommended_strategy"] == "MULTI_DOMAIN_EXPERIMENT_RECOMMENDED"
    assert payload["immediate_pooling_justified"] is False
    assert payload["canonical_integrity"]["unchanged"] is True
    assert payload["canonical_integrity"]["all_expected_hashes_match"] is True
    assert payload["shared_sequence_bridge"]["n"] == 48
    sensitivity = payload["shared_sequence_bridge"][
        "canonical_modeling_population_sensitivity"
    ]
    assert sensitivity["shared_30mer_n"] == 41
    assert "Canonical-Population Sensitivity" in report
    assert "MULTI_DOMAIN_EXPERIMENT_RECOMMENDED" in report

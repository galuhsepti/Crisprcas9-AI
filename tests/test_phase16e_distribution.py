import ast
import json
from pathlib import Path

import pytest

from src.dataset_landscape.distribution import (
    activity_distribution,
    build_phase16e_distribution_audit,
    fixed_activity_bins,
    sequence_distribution,
)


@pytest.fixture(scope="module")
def phase16e_payload():
    """Build the heavy Phase 16E distribution audit once for this module."""
    return build_phase16e_distribution_audit("2026-09-08")


def test_sequence_distribution_basic_gc_and_kmers():
    report = sequence_distribution(["ACGTACGT", "GGGGCCCC"], kmer=True)
    assert report["n_sequences"] == 2
    assert report["gc"]["n"] == 2
    assert "kmer_2" in report
    assert "kmer_3" in report


def test_canonical_activity_bins_are_fixed():
    bins = fixed_activity_bins([0.01, 0.21, 0.41, 0.61, 0.81])
    assert [b["n"] for b in bins] == [1, 1, 1, 1, 1]


def test_activity_distribution_only_for_established_continuous_labels():
    report = activity_distribution([0.1, 0.5, 0.9])
    assert report["status"] == "ANALYZED_ESTABLISHED_CONTINUOUS_ACTIVITY"
    assert report["high_activity_>0.8"] == 1


def test_phase16e_candidate_labels_not_analyzed_as_activity(phase16e_payload):
    payload = phase16e_payload
    for record in payload["candidate_reports"]:
        assert record["record_counts"]["activity_label_analyzed"] is False
        assert record["activity"]["status"] == "NOT_ANALYZED_LABEL_SEMANTICS_NOT_ESTABLISHED_AS_CONTINUOUS_ACTIVITY"


def test_phase16e_carries_phase16d_context_without_gate_decision(phase16e_payload):
    payload = phase16e_payload
    assert "phase16d_context" in payload
    assert payload["non_modeling_guards"]["no_gate_decision"] is True


def test_no_forbidden_phase16e_sources():
    paths = list(Path("src/dataset_landscape").glob("*.py")) + [
        Path("scripts/run_phase16e_distribution_audit.py")
    ]
    forbidden_imports = {"src.models", "torch", "xgboost", "sklearn"}
    forbidden_text = [
        "load_moreno_mateos",
        "Moreno-Mateos.csv",
        "build_training_pool",
        "fit(",
        "predict(",
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


def test_phase16e_deterministic_output(phase16e_payload):
    frozen = json.loads(
        Path("results/phase16e_distribution_audit_20260908_170415.json").read_text(
            encoding="utf-8"
        )
    )
    for key in [
        "phase",
        "scope",
        "status",
        "canonical_reference",
        "candidate_reports",
        "phase16d_context",
        "non_modeling_guards",
    ]:
        assert phase16e_payload[key] == frozen[key]

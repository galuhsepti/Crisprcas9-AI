"""Synthetic Phase 18D analysis and integrity tests; no model fitting."""

import hashlib
import json

import numpy as np
import pytest

import src.evaluation.phase18d_interpretation as phase18d
from src.models.xgboost_model import XGBoostModel


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_campaign(tmp_path, monkeypatch):
    campaign = tmp_path / phase18d.CAMPAIGN_ID
    (campaign / "models").mkdir(parents=True)
    (campaign / "predictions").mkdir()
    (campaign / "campaign_results.json").write_text("{}", encoding="utf-8")
    (campaign / "execution_lock.json").write_text("{}", encoding="utf-8")
    artifacts = []
    for kind, folder, extension in (
        ("model", "models", "bin"),
        ("prediction", "predictions", "npz"),
    ):
        for index in range(35):
            path = campaign / folder / f"artifact-{index}.{extension}"
            path.write_bytes(f"{kind}-{index}".encode("ascii"))
            artifacts.append(
                {
                    "artifact_sha256": _sha256(path),
                    "artifact_type": kind,
                    "bytes": path.stat().st_size,
                    "code_commit": phase18d.CODE_COMMIT,
                    "experiment_id": f"{kind}-{index}",
                    "implementation_sha256": phase18d.IMPLEMENTATION_SHA256,
                    "path": path.relative_to(campaign).as_posix(),
                    "protocol_sha256": phase18d.PROTOCOL_SHA256,
                    "seed": 42 + index % 5,
                    "split_sha256": phase18d.SPLIT_SHA256,
                }
            )
    manifest = {
        "artifact_count": 70,
        "artifacts": artifacts,
        "campaign_id": phase18d.CAMPAIGN_ID,
        "campaign_results_sha256": _sha256(campaign / "campaign_results.json"),
        "code_commit": phase18d.CODE_COMMIT,
        "execution_lock_sha256": _sha256(campaign / "execution_lock.json"),
        "implementation_sha256": phase18d.IMPLEMENTATION_SHA256,
        "phase": "18C",
        "protocol_sha256": phase18d.PROTOCOL_SHA256,
        "split_sha256": phase18d.SPLIT_SHA256,
    }
    manifest_path = campaign / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(phase18d, "ROOT", tmp_path)
    monkeypatch.setattr(phase18d, "OFFICIAL_CAMPAIGN", campaign)
    monkeypatch.setattr(
        phase18d,
        "SOURCE_IDENTITIES",
        {
            "campaign_results.json": _sha256(campaign / "campaign_results.json"),
            "execution_lock.json": _sha256(campaign / "execution_lock.json"),
            "artifact_manifest.json": _sha256(manifest_path),
        },
    )
    monkeypatch.setattr(
        phase18d,
        "CAMPAIGN_RESULTS_SHA256",
        manifest["campaign_results_sha256"],
    )
    monkeypatch.setattr(
        phase18d,
        "EXECUTION_LOCK_SHA256",
        manifest["execution_lock_sha256"],
    )
    return campaign


def test_official_campaign_selection_accepts_only_exact_complete_path():
    selected = phase18d.official_campaign_path()
    assert selected == phase18d.OFFICIAL_CAMPAIGN.resolve()
    with pytest.raises(PermissionError, match="official campaign"):
        phase18d.official_campaign_path(selected.parent)


def test_partial_campaign_is_excluded(tmp_path, monkeypatch):
    campaign = _synthetic_campaign(tmp_path, monkeypatch)
    (campaign / "unstable_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Partial or failed"):
        phase18d.official_campaign_path()


def test_prediction_and_model_hash_verification(tmp_path, monkeypatch):
    campaign = _synthetic_campaign(tmp_path, monkeypatch)
    result = phase18d.verify_official_artifacts()
    assert result["artifact_types"] == {"model": 35, "prediction": 35}
    (campaign / "predictions" / "artifact-7.npz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="size-mismatched|hash mismatch"):
        phase18d.verify_official_artifacts()


def test_residual_calculation_uses_prediction_minus_observed():
    result = phase18d.calculate_residuals([1.0, 2.0], [1.5, 1.0])
    assert result["residual"].tolist() == [0.5, -1.0]
    assert result["absolute_error"].tolist() == [0.5, 1.0]
    assert result["squared_error"].tolist() == [0.25, 1.0]


def test_activity_binning_is_balanced_and_deterministic():
    values = np.asarray([2, 1, 4, 3, 1, 4, 2, 3], dtype=float)
    identifiers = np.asarray(list("hgfedcba"))
    first = phase18d.assign_activity_bins(values, identifiers)
    second = phase18d.assign_activity_bins(values, identifiers)
    assert np.array_equal(first, second)
    assert sorted(np.unique(first, return_counts=True)[1].tolist()) == [2] * 4


def test_gc_calculation_uses_spacer_and_full_geometry():
    sequence = "AAAA" + "G" * 10 + "A" * 10 + "AGG" + "AAA"
    assert phase18d.spacer_gc_fraction(sequence) == 0.5
    assert phase18d.full_gc_fraction(sequence) == pytest.approx(12 / 30)


def test_pam_extraction_uses_positions_24_to_26():
    sequence = "A" * 24 + "TGG" + "CCC"
    assert phase18d.extract_pam(sequence) == "TGG"


def test_error_category_assignment_uses_locked_iqr_margin():
    assert phase18d.assign_error_category(0.1, 0.3, 1.0, "xgb", "cnn") == (
        "xgb_substantially_better"
    )
    assert phase18d.assign_error_category(0.05, 0.08, 1.0, "xgb", "cnn") == (
        "both_accurate"
    )
    assert phase18d.assign_error_category(0.3, 0.3, 1.0, "xgb", "cnn") == (
        "both_inaccurate"
    )


def test_seed_alignment_counts_strict_cnn_d_improvements():
    truth = np.asarray([0.0, 1.0])
    single = {seed: np.asarray([1.0, 1.0]) for seed in phase18d.SEEDS}
    shared = {
        seed: np.asarray([0.0 if seed < 45 else 2.0, 1.0]) for seed in phase18d.SEEDS
    }
    assert phase18d.seed_help_counts(single, shared, truth).tolist() == [3, 0]
    with pytest.raises(ValueError, match="five registered"):
        phase18d.seed_help_counts({42: single[42]}, shared, truth)


def test_hard_case_selection_is_top_ten_percent_with_id_tie_break():
    identifiers = np.asarray([f"id-{index:02d}" for index in range(20)])
    difficulty = np.arange(20, dtype=float)
    selected = phase18d.select_hard_cases(identifiers, difficulty)
    assert identifiers[selected].tolist() == ["id-18", "id-19"]


def test_bridge_pair_alignment_requires_exact_41_id_set():
    left = np.asarray([f"id-{index:02d}" for index in range(41)])
    right = left[::-1]
    order = phase18d.align_bridge_pairs(left, right)
    assert np.array_equal(right[order], left)
    with pytest.raises(ValueError, match="exactly 41"):
        phase18d.align_bridge_pairs(left[:-1], right[:-1])


def test_no_training_guard_fails_before_xgboost_fit_body():
    model = XGBoostModel(n_estimators=1)
    with phase18d.phase18d_read_only_guard():
        with pytest.raises(RuntimeError, match="fitting/training prohibited"):
            model.fit(np.zeros((2, 1)), np.zeros(2))
    assert model.is_fitted is False


def test_locked_external_name_remains_prohibited(tmp_path):
    synthetic = tmp_path / ("Moreno" + "-Mateos.csv")
    with phase18d.phase18d_read_only_guard():
        with pytest.raises(PermissionError, match="locked external dataset"):
            synthetic.read_text(encoding="utf-8")


def test_phase18c_artifact_mutation_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(phase18d, "ROOT", tmp_path)
    protected = tmp_path / "results" / "phase18c" / "synthetic.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("before", encoding="utf-8")
    with phase18d.phase18d_read_only_guard():
        with pytest.raises(PermissionError, match="cannot mutate"):
            protected.write_text("after", encoding="utf-8")
    assert protected.read_text(encoding="utf-8") == "before"


def test_deterministic_summary_serialization_is_order_independent():
    first = phase18d.deterministic_json({"b": 2, "a": [1.0]})
    second = phase18d.deterministic_json({"a": [1.0], "b": 2})
    assert first == second
    assert first.endswith("\n")

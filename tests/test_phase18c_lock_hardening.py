from pathlib import Path

import pytest

import src.experiment_protocols.phase18c_access as access
from src.dataset_integration.label_compatibility import EXPECTED_HASHES

LOCKED_RULE = "data/raw/Moreno-Mateos.csv"
CANONICAL_HASHES = {
    "data/raw/DeepSpCas9.csv": (
        "6aeab30f55155ca9f1b80aae3159231ce113815354597ea4ca1d6c84ba0296df"
    ),
    "data/phase17i/raw/crispron/Luo2020_Kim2019.xlsx": (
        "146f211575179fc31669dbd5eaef46434b0866013a2d03767354375a1cc5d61f"
    ),
    "models/rf_baseline_fixed_20260905_001107.pkl": (
        "1f521bed090c091dd7a35e060f1166eaaa8e864ad4be09708a285750740280b6"
    ),
    "models/xgboost_baseline_20260905_002839.pkl": (
        "129aa2c7826bc6ae087ba6b751eb39838a34aafeea27aa88674be8bcbf606dbd"
    ),
    "models/cnn_baseline_20260905_011720.pt": (
        "76773642b59923fa0d4407011c6f656715cb7bf247ac5ab9a437c1d0f3d8f616"
    ),
}


def _active_rules(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_search_exclusions_are_exact_and_narrow():
    assert _active_rules(Path(".rgignore")) == [LOCKED_RULE]
    assert _active_rules(Path(".ignore")) == [LOCKED_RULE]


def test_search_policy_requires_allowlisted_inspection():
    policy = Path("docs/phase18c_search_safety_policy.md").read_text(
        encoding="utf-8"
    )
    assert "allowlisted path inspection" in policy.casefold()
    assert "--glob '!data/raw/Moreno-Mateos.csv'" in policy
    assert "generic delegated repository-content search" in policy


@pytest.mark.parametrize(
    "name", ["Moreno-Mateos.csv", "MORENO.csv", "mateos.xlsx"]
)
def test_locked_synthetic_paths_are_rejected_before_resolution(
    tmp_path, monkeypatch, name
):
    def unexpected_resolution(path):
        raise AssertionError(f"locked path was resolved: {path}")

    monkeypatch.setattr(access, "_resolve_path", unexpected_resolution)
    with pytest.raises(PermissionError, match="locked external dataset"):
        access.resolve_phase18c_path(tmp_path / name)


def test_resolved_synthetic_alias_is_rejected(tmp_path, monkeypatch):
    synthetic_locked = tmp_path / ("Moreno" + "-Mateos.csv")
    monkeypatch.setattr(access, "_resolve_path", lambda path: synthetic_locked)
    with pytest.raises(PermissionError, match="locked external dataset"):
        access.resolve_phase18c_path(tmp_path / "synthetic-alias.csv")


def test_runtime_guard_rejects_nonexistent_synthetic_locked_open(tmp_path):
    synthetic_locked = tmp_path / ("Moreno" + "-Mateos.csv")
    with access.phase18c_access_guard():
        with pytest.raises(PermissionError, match="locked external dataset"):
            synthetic_locked.read_text(encoding="utf-8")
    assert not synthetic_locked.exists()


def test_runtime_guard_allows_non_locked_file(tmp_path):
    allowed = tmp_path / "approved-development-input.txt"
    allowed.write_text("synthetic", encoding="utf-8")
    with access.phase18c_access_guard():
        assert allowed.read_text(encoding="utf-8") == "synthetic"


def test_canonical_hash_manifest_is_unchanged_and_excludes_locked_data():
    assert EXPECTED_HASHES == CANONICAL_HASHES
    assert LOCKED_RULE not in EXPECTED_HASHES

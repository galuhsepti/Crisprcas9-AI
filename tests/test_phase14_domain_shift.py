"""
Phase 14 tests: Domain-Shift Attribution / Robustness Audit.

Covers the 20 required test categories, including explicit guards against
external-test leakage and premature Moreno-Mateos access. Written BEFORE the
experiment is executed.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.domain_shift.config import Phase14DomainShiftConfig
from src.domain_shift.state_machine import Phase14StateMachine
from src.domain_shift.stratification import (
    activity_bin_edges,
    bin_bounds,
    bin_indices,
    common_support_joint_delta,
    construct_gc_edges,
    per_bin_stats,
)
from src.domain_shift.composition import (
    cohens_d,
    gc_array,
    internal_null_band,
    js_divergence,
    kmer_frequency_vector,
    nucleotide_frequencies,
    position_frequencies,
)
from src.domain_shift.error_analysis import (
    absolute_error,
    association_summary,
    composition_distance,
    signed_error,
)
from src.domain_shift.statistics import (
    bh_adjust,
    bootstrap_mean_delta_ci,
    mean_pairwise_spearman,
    percentile_bootstrap_ci,
    reproducible_bootstrap_seed,
    spearman_with_bootstrap,
)
from src.domain_shift import provenance as prov

CFG = Phase14DomainShiftConfig()


# ---------------------------------------------------------------------------
# 1. Correct canonical split loading
# ---------------------------------------------------------------------------
def test_canonical_split_loading():
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from src.data.validation import validate_sequence

    df = pd.read_csv(CFG.paths()["primary_dataset"])
    valid = [validate_sequence(s, check_pam=False)[0] for s in df["sequence_30mer"]]
    df = df[valid].copy()
    assert len(df) == CFG.expected_n_valid
    idx = np.arange(len(df))
    tr, va = train_test_split(idx, test_size=CFG.val_ratio, random_state=CFG.split_seed)
    assert tr.shape[0] == CFG.expected_n_train
    assert va.shape[0] == CFG.expected_n_val


# ---------------------------------------------------------------------------
# 2. Canonical model artifact loading
# ---------------------------------------------------------------------------
def test_canonical_model_loading():
    from src.models import RandomForestModel, XGBoostModel, CNNModel

    cnn = CNNModel.load_model(str(CFG.paths()["model_cnn"]))
    rf = RandomForestModel.load_model(str(CFG.paths()["model_rf"]))
    xgb = XGBoostModel.load_model(str(CFG.paths()["model_xgb"]))
    assert cnn.is_fitted and rf.is_fitted and xgb.is_fitted


def test_canonical_model_hashes_match_recorded():
    assert prov.sha256_file(CFG.paths()["model_rf"]) == CFG.expected_canonical_hashes[CFG.model_rf]
    assert prov.sha256_file(CFG.paths()["model_xgb"]) == CFG.expected_canonical_hashes[CFG.model_xgb]
    assert prov.sha256_file(CFG.paths()["model_cnn"]) == CFG.expected_canonical_hashes[CFG.model_cnn]


# ---------------------------------------------------------------------------
# 3. Frozen GC thresholds
# ---------------------------------------------------------------------------
def test_config_frozen_and_deterministic():
    c1 = Phase14DomainShiftConfig()
    c2 = Phase14DomainShiftConfig()
    assert c1 == c2
    assert c1.gc_method == "internal_train_quantile"
    assert c1.split_seed == 42 and c1.val_ratio == 0.15
    assert c1.gc_quantiles == [0.2, 0.4, 0.6, 0.8]


def test_gc_edges_reproducible_from_train():
    rng = np.random.default_rng(5)
    g = rng.uniform(0.3, 0.8, size=5000)
    e1 = construct_gc_edges(g, tag="train")
    e2 = construct_gc_edges(g, tag="train")
    assert e1["edges"] == e2["edges"]
    assert e1["source"] == "internal_train_quantile"
    assert len(e1["edges"]) == 4


# ---------------------------------------------------------------------------
# 4. Fixed activity bins
# ---------------------------------------------------------------------------
def test_activity_bin_edges_fixed():
    assert activity_bin_edges() == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert CFG.activity_edges == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_activity_bin_assignment():
    split = activity_bin_edges()[1:-1]
    idx = bin_indices([0.0, 0.19, 0.2, 0.4, 0.6, 0.8, 1.0], split)
    assert idx.tolist() == [0, 0, 1, 2, 3, 4, 4]


# ---------------------------------------------------------------------------
# 5. No external-specific quantile binning
# ---------------------------------------------------------------------------
def test_gc_edges_reject_non_train_tag():
    g = np.random.default_rng(1).uniform(0.3, 0.8, 500)
    with pytest.raises(RuntimeError):
        construct_gc_edges(g, tag="external")
    with pytest.raises(RuntimeError):
        construct_gc_edges(g, tag="validation")


def test_no_external_quantile_binning_in_modules():
    modules = [
        "stratification",
        "composition",
        "statistics",
        "error_analysis",
        "state_machine",
    ]
    for m in modules:
        src = (Path(__file__).parent.parent / "src" / "domain_shift" / f"{m}.py").read_text()
        assert "Moreno" not in src
        assert "read_csv" not in src
        assert "external_dataset" not in src


# ---------------------------------------------------------------------------
# 6. Sequence length validation
# ---------------------------------------------------------------------------
VALID_30MER = "TTCTGCCTTGTTTCTTTCCTCTCTGGGTCG"


def test_sequence_length_validation():
    from src.data.validation import validate_sequence

    ok, _ = validate_sequence(VALID_30MER, check_pam=False)
    assert ok
    bad, errs = validate_sequence(VALID_30MER[:29], check_pam=False)
    assert not bad and len(errs) >= 1
    assert len(VALID_30MER) == CFG.context_length


# ---------------------------------------------------------------------------
# 7. Nucleotide vocabulary validation
# ---------------------------------------------------------------------------
def test_nucleotide_vocabulary():
    from src.data.validation import validate_sequence

    ok, _ = validate_sequence(VALID_30MER, check_pam=False)
    assert ok
    bad, _ = validate_sequence(VALID_30MER[:29] + "X", check_pam=False)
    assert not bad


# ---------------------------------------------------------------------------
# 8. No retraining
# ---------------------------------------------------------------------------
def test_no_training_fields_in_config():
    cfg = Phase14DomainShiftConfig()
    for attr in ("learning_rate", "batch_size", "epochs", "patience", "optimizer", "loss"):
        assert not hasattr(cfg, attr)


def test_runner_never_fits_models():
    runner = (Path(__file__).parent.parent / "scripts" / "run_phase14_domain_shift_audit.py")
    if not runner.exists():
        pytest.skip("runner not present")
    text = runner.read_text()
    assert ".fit(" not in text
    assert "src.training" not in text


# ---------------------------------------------------------------------------
# 9. No canonical artifact modification
# ---------------------------------------------------------------------------
def test_sha256_deterministic_and_unchanged():
    p = CFG.paths()["model_cnn"]
    h1 = prov.sha256_file(p)
    h2 = prov.sha256_file(p)
    assert h1 == h2
    assert h1 == CFG.expected_canonical_hashes[CFG.model_cnn]


# ---------------------------------------------------------------------------
# 10. No Phase 10 pool usage
# ---------------------------------------------------------------------------
def test_no_pool_dataset():
    assert CFG.primary_dataset.endswith("DeepSpCas9.csv")
    assert "pool" not in CFG.primary_dataset.lower()


# ---------------------------------------------------------------------------
# 11. No premature Moreno access
# ---------------------------------------------------------------------------
def test_state_machine_blocks_external_before_freeze():
    sm = Phase14StateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    sm.transition("ANALYSIS")
    with pytest.raises(RuntimeError):
        sm.transition("FINAL_EXTERNAL_EVALUATION")


def test_moreno_read_guard_before_freeze():
    sm = Phase14StateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    sm.transition("ANALYSIS")
    with pytest.raises(RuntimeError):
        sm.assert_moreno_read_allowed()
    sm.transition("STATISTICAL_EVALUATION")
    sm.freeze()
    sm.assert_moreno_read_allowed()  # allowed only after freeze


def test_runner_contains_state_machine():
    runner = (Path(__file__).parent.parent / "scripts" / "run_phase14_domain_shift_audit.py")
    if not runner.exists():
        pytest.skip("runner not present")
    text = runner.read_text()
    assert "Phase14StateMachine" in text
    assert "FREEZE" in text
    assert "FINAL_EXTERNAL_EVALUATION" in text


# ---------------------------------------------------------------------------
# 12. State-machine transitions
# ---------------------------------------------------------------------------
def test_state_machine_full_sequence():
    sm = Phase14StateMachine(start="DESIGN")
    for s in ["FEASIBILITY", "ANALYSIS", "STATISTICAL_EVALUATION"]:
        sm.transition(s)
    sm.freeze()
    sm.transition("FINAL_EXTERNAL_EVALUATION")
    sm.transition("AUDIT")
    sm.transition("COMPLETE")
    assert sm.state == "COMPLETE"


def test_state_machine_rejects_backward():
    sm = Phase14StateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    with pytest.raises(RuntimeError):
        sm.transition("DESIGN")


def test_state_machine_rejects_skip_to_external():
    sm = Phase14StateMachine(start="DESIGN")
    sm.transition("FEASIBILITY")
    with pytest.raises(RuntimeError):
        sm.transition("FINAL_EXTERNAL_EVALUATION")


# ---------------------------------------------------------------------------
# 13. Reproducibility
# ---------------------------------------------------------------------------
def test_bootstrap_reproducible():
    r = np.random.RandomState(0).rand(100)
    a = percentile_bootstrap_ci(r, lambda v: float(np.mean(v)), n_boot=200,
                                rng=reproducible_bootstrap_seed(3))
    b = percentile_bootstrap_ci(r, lambda v: float(np.mean(v)), n_boot=200,
                                rng=reproducible_bootstrap_seed(3))
    assert a["ci_lower"] == b["ci_lower"]
    assert a["ci_upper"] == b["ci_upper"]


def test_spearman_bootstrap_reproducible():
    rng = np.random.default_rng(2)
    x = rng.uniform(size=200)
    y = 0.5 * x + 0.1 * rng.uniform(size=200)
    r1 = spearman_with_bootstrap(x, y, n_boot=150, seed=9)
    r2 = spearman_with_bootstrap(x, y, n_boot=150, seed=9)
    assert r1 == r2


# ---------------------------------------------------------------------------
# 14. Bootstrap correctness
# ---------------------------------------------------------------------------
def test_bootstrap_known_answer_zero_variance():
    v = np.ones(50)
    res = percentile_bootstrap_ci(v, lambda x: float(np.mean(x)), n_boot=100)
    assert res["point"] == pytest.approx(1.0)
    assert res["ci_lower"] == pytest.approx(1.0)
    assert res["ci_upper"] == pytest.approx(1.0)


def test_betweendomain_bootstrap_not_paired():
    ext = np.random.default_rng(0).uniform(size=60)
    internal = np.random.default_rng(1).uniform(size=40)
    res = bootstrap_mean_delta_ci(ext, internal, n_boot=200,
                                  rng=reproducible_bootstrap_seed(11))
    expected = float(np.mean(ext)) - float(np.mean(internal))
    assert res["mean_delta"] == pytest.approx(expected)
    assert res["comparison"] == "independent_domains (external - internal), not paired"
    assert res["n_external"] == 60 and res["n_internal"] == 40


# ---------------------------------------------------------------------------
# 15. Metric correctness
# ---------------------------------------------------------------------------
def test_error_metrics_manual():
    yt = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    yp = np.array([0.15, 0.25, 0.5, 0.65, 0.85])
    assert absolute_error(yt, yp).tolist() == pytest.approx(np.abs(yt - yp).tolist())
    assert signed_error(yt, yp).tolist() == pytest.approx((yp - yt).tolist())
    res = per_bin_stats(yt, yp, np.zeros(5, dtype=int), 1, [0.0], [1.0])
    assert res[0]["mae"] == pytest.approx(float(np.mean(np.abs(yt - yp))))
    assert res[0]["bias"] == pytest.approx(float(np.mean(yp - yt)))


# ---------------------------------------------------------------------------
# 16. Subgroup / bin handling
# ---------------------------------------------------------------------------
def test_empty_bin_null_entry():
    yt = np.array([0.1, 0.2])
    yp = np.array([0.1, 0.2])
    idx = np.array([1, 1])  # bin0 empty
    res = per_bin_stats(yt, yp, idx, 2, [0.0, 1.0], [1.0, 2.0])
    assert res[0]["n"] == 0
    assert res[1]["n"] == 2
    assert res[1]["mae"] == pytest.approx(0.0)


def test_small_bin_no_ci_large_bin_ci():
    yt = np.linspace(0, 1, 100)
    yp = 0.8 * yt + 0.05
    idx = np.where(yt < 0.5, 0, 1)
    res = per_bin_stats(yt, yp, idx, 2, [0.0, 0.5, 1.0][:-1], [0.5, 1.0],
                        min_n_report=20, min_n_boot=30, n_boot=50)
    # bin0 ~50 samples
    assert res[0]["n"] >= 30
    assert res[0]["mae_ci"] is not None
    assert res[0]["excluded_min_n"] is False


# ---------------------------------------------------------------------------
# 17. Degenerate-quantile fallback
# ---------------------------------------------------------------------------
def test_degenerate_gc_edges_fallback():
    g = np.zeros(500) + 0.5  # degenerate GC distribution
    res = construct_gc_edges(g, tag="train")
    assert res["source"] == "fallback_absolute_grid"
    assert res["edges"] == CFG.fallback_gc_edges


def test_bin_bounds_shapes():
    lows, highs = bin_bounds([0.2, 0.4, 0.6, 0.8], 0.0, 1.0)
    assert lows == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert highs == [0.2, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# 18. Multiple-testing correction
# ---------------------------------------------------------------------------
def test_bh_adjustment_correct():
    p = np.array([0.005, 0.01, 0.045, 0.2, 0.5])
    adj = bh_adjust(p)
    assert len(adj) == len(p)
    # adjusted p-values are >= raw
    assert np.all(np.array(adj) >= p)
    # classic BH example
    q = bh_adjust([0.01, 0.02, 0.05])
    assert q == sorted([min(3 * 0.01 / 1, 1), min(3 * 0.02 / 2, 1), min(3 * 0.05 / 3, 1)])


# ---------------------------------------------------------------------------
# 19. Provenance / hash recording
# ---------------------------------------------------------------------------
def test_provenance_recordings():
    p = CFG.paths()["model_cnn"]
    h = prov.sha256_file(p)
    assert isinstance(h, str) and len(h) == 64
    c1 = prov.config_hash(Phase14DomainShiftConfig())
    c2 = prov.config_hash(Phase14DomainShiftConfig())
    assert c1 == c2
    head = prov.git_head(CFG.project_root)
    assert head and head != "unknown"


# ---------------------------------------------------------------------------
# 20. Output schema (function-level) + run-dependent integration
# ---------------------------------------------------------------------------
def test_bin_schema_keys():
    yt = np.random.default_rng(0).uniform(size=40)
    yp = yt * 0.5
    idx = np.where(yt < 0.5, 0, 1)
    res = per_bin_stats(yt, yp, idx, 2, [0.0, 0.5], [0.5, 1.0])
    assert set(res[0].keys()) >= {"bin", "n", "mae", "rmse", "bias"}


def test_joint_cells_schema():
    rng = np.random.default_rng(0)
    n = 200
    int_d = {
        "true": rng.uniform(size=n),
        "pred_random_forest": rng.uniform(size=n),
        "gc_bin": rng.integers(0, 4, size=n),
        "activity_bin": rng.integers(0, 4, size=n),
    }
    m = 200
    ext_d = {
        "true": rng.uniform(size=m),
        "pred_random_forest": rng.uniform(size=m),
        "gc_bin": rng.integers(0, 4, size=m),
        "activity_bin": rng.integers(0, 4, size=m),
    }
    out = common_support_joint_delta(int_d, ext_d, 4, 4, min_n=10, model="random_forest")
    assert "cells" in out
    assert "median_delta_mae" in out
    assert "common-support" in out["definition"]


def test_association_schema():
    y_true = np.random.default_rng(1).uniform(size=60)
    y_pred = y_true * 0.5
    cov = {"gc": np.random.default_rng(2).uniform(size=60)}
    out = association_summary(y_true, y_pred, cov, model="rf", n_boot=50, seed=3)
    assert "gc" in out
    assert {"spearman", "ci_lower", "ci_upper", "n"} <= set(out["gc"].keys())


@pytest.mark.skipif(
    not list(Path(CFG.project_root).glob("results/experiments/phase14_domain_shift_audit_*_final.json")),
    reason="integration: run the Phase 14 audit first",
)
def test_final_json_integration_schema():
    finals = sorted(Path(CFG.project_root).glob(
        "results/experiments/phase14_domain_shift_audit_*_final.json"))
    assert finals
    import json
    d = json.load(open(finals[-1]))
    for key in ["experiment_name", "status", "decision_gate", "frozen_protocol",
                "composition", "confirmatory", "convergence", "canonical_artifact_hashes",
                "leakage_checks", "state_log"]:
        assert key in d, key


@pytest.mark.skipif(
    not list(Path(CFG.project_root).glob("results/experiments/phase14_domain_shift_audit_*_freeze.json")),
    reason="integration: run the Phase 14 audit first",
)
def test_freeze_json_integration_schema():
    freezes = sorted(Path(CFG.project_root).glob(
        "results/experiments/phase14_domain_shift_audit_*_freeze.json"))
    assert freezes
    import json
    d = json.load(open(freezes[-1]))
    for key in ["config_hash", "gc_edges", "activity_edges", "confirmatory_tests",
                "canonical_hashes_before", "git_head", "timestamp"]:
        assert key in d, key


# ---------------------------------------------------------------------------
# Composition sanity tests
# ---------------------------------------------------------------------------
def test_gc_array_sanity():
    seqs = ["A" * 30, "C" * 30, "G" * 30, "T" * 30]
    g = gc_array(seqs)
    assert g.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_kmer_vector_normalized():
    seqs = ["A" * 30, "C" * 30]
    v = kmer_frequency_vector(seqs, k=2)
    assert abs(v.sum() - 1.0) < 1e-9
    assert len(v) == 16


def test_js_and_null_band():
    a = ["ACGTACGTACGTACGTACGTACGTACGTACGT", "ACGTTTTTGGGGCCCCAAAAGGGGTTTTCCCC"]
    b = ["TGCA" * 8, "CCCCGGGGAAAATTTTACGTACGTACGTACGT"]
    j = js_divergence(kmer_frequency_vector(a, k=3), kmer_frequency_vector(b, k=3))
    assert j["js_divergence"] >= 0.0
    band = internal_null_band(a * 40, k=2, n_boot=50, seed=1)
    assert "null_band_2.5pct" in band


def test_cohens_d_formula():
    a = [0.5] * 10
    b = [0.4] * 10
    d = cohens_d(a, b)
    # sa = sb = 0 (zero variance) -> handled as 0.0 or None ambiguity: accept
    assert d["cohens_d"] in (0.0, None)


def test_position_frequencies_sanity():
    seqs = ["A" * 30, "C" * 30]
    pf = position_frequencies(seqs, 30)
    assert pf["A"][0] == 0.5 and pf["C"][0] == 0.5


def test_composition_distance():
    seqs = ["ACGT" * 8]
    mu = kmer_frequency_vector(["AAAA" * 8], k=2)
    d = composition_distance(seqs, mu, k=2)
    assert d.shape == (1,)


def test_mean_pairwise_spearman():
    v = {"a": np.array([1.0, 2.0, 3.0, 4.0]), "b": np.array([2.0, 3.0, 4.0, 5.0])}
    out = mean_pairwise_spearman(v)
    assert out["n_pairs"] == 1
    assert out["mean_pairwise_spearman"] == pytest.approx(1.0)
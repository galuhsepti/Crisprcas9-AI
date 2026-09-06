"""Unit tests for the Phase 10 multi-dataset diversification module.
Pure/light tests only - no model fitting, no heavy data reads."""

import numpy as np
import pandas as pd
import pytest

import src.multidataset.config as config_mod
from src.multidataset.config import (
    ARMS,
    PRIMARY_ARM,
    PRIMARY_MODEL,
    EFFECT_THRESHOLD,
    CONTRADICTION_THRESHOLD,
    decide_verdict,
    FLANK3_PADDING,
    FLANK5_PADDING,
)
from src.multidataset.datasets import (
    NORMALIZED_SEQ_COL,
    ACTIVITY_COL,
    SOURCE_COL,
    DATASET_INVENTORY,
    construct_deephf_30mer,
    canonical_validate,
    dataset_inventory_table,
    allowed_additional_datasets,
    sha256_file,
)
from src.multidataset.audit import (
    reverse_complement,
    overlap_report,
    contamination_report,
    duplicate_and_conflict_scan,
    label_distribution_per_source,
)
from src.multidataset.pool import (
    build_training_pool,
    shared_guide_set,
    POOL_COLUMNS,
)
from src.multidataset.stats import (
    mean_pairwise_diff,
    cohens_dz,
    paired_t_test,
    paired_wilcoxon,
    bootstrap_ci_meandiff,
    paired_report,
)


class TestGeometry:
    def test_construct_deephf_30mer_layout(self):
        guide = "ACGT" * 5  # 20 bp
        pam = "NGG"
        s = construct_deephf_30mer(guide, pam)
        assert len(s) == 30
        assert s[:4] == FLANK5_PADDING == "AAAA"
        assert s[4:24] == guide
        assert s[24:27] == pam
        assert s[27:] == FLANK3_PADDING == "AAA"

    def test_construct_rejects_malformed_pam(self):
        assert construct_deephf_30mer("ACGT" * 5, "GG") is None

    def test_construct_rejects_short_guide(self):
        assert construct_deephf_30mer("ACGT" * 4, "NGG") is None

    def test_construct_uppercases_input(self):
        assert construct_deephf_30mer("acgt" * 5, "ngg") == \
            "AAAA" + "ACGT" * 5 + "NGG" + "AAA"


class TestReverseComplement:
    def test_known_value(self):
        assert reverse_complement("AAAACCCCTTTTGGGG") == \
            "CCCCAAAAGGGGTTTT"

    def test_uppercases(self):
        assert reverse_complement("aaaaGGGG") == "CCCCtttt".upper()


class TestOverlapAudit:
    def test_overlap_report_forward_and_rc(self):
        sa = {"AAAA" + "A" * 20 + "GGGAAA"}
        sb = {reverse_complement("AAAA" + "A" * 20 + "GGGAAA")}
        r = overlap_report({"a": sa, "b": sb})
        assert r["a_vs_b"]["intersection_fwd"] == 0
        assert r["a_vs_b"]["intersection_rc"] == 1

    def test_overlap_report_jaccard(self):
        sa = {"A" * 30, "T" * 30, "G" * 30}
        sb = {"A" * 30, "C" * 30}
        r = overlap_report({"a": sa, "b": sb})
        assert r["a_vs_b"]["intersection_fwd"] == 1
        assert r["a_vs_b"]["jaccard"] == pytest.approx(1 / 4)

    def test_contamination_report_zero_overlap_is_required(self):
        ext = {"A" * 30, "T" * 30}
        clean = {"G" * 30}  # RC(G*30)=C*30, not in ext
        dirty = {"G" * 30, "A" * 30}
        rc_dirty = {reverse_complement("G" * 30), "T" * 30}
        r = contamination_report(ext, {"clean": clean, "dirty": dirty,
                                       "rc_dirty": rc_dirty})
        assert r["candidates"]["clean"]["eligible"] is True
        assert r["candidates"]["dirty"]["eligible"] is False
        assert r["candidates"]["rc_dirty"]["eligible"] is False


def make_dsp_frame(sequences, activities):
    return pd.DataFrame({
        NORMALIZED_SEQ_COL: sequences,
        ACTIVITY_COL: activities,
        SOURCE_COL: "deepspcas9",
        "guide_sequence": [s[4:24] for s in sequences],
        "pam": [s[24:27] for s in sequences],
    })


def make_hf_frame(guides, pams, activities):
    return pd.DataFrame({
        NORMALIZED_SEQ_COL: [construct_deephf_30mer(g, p) for g, p
                             in zip(guides, pams)],
        ACTIVITY_COL: activities,
        SOURCE_COL: "deephf",
        "guide_sequence": guides,
        "pam": pams,
    })


def _seq(i, guide_bp="G"):
    """Deterministic 30-mer with guide = guide_bp*20."""
    return "AAAA" + guide_bp * 20 + "GGGAAA"


class TestPool:
    def test_pool_composition_and_columns(self):
        dsp = make_dsp_frame([_seq(1), _seq(2)], [0.5, 0.6])
        hf = make_hf_frame(["A" * 20, "C" * 20], ["NGG", "NGG"], [0.7, 0.8])
        pool = build_training_pool(dsp, hf)
        assert len(pool) == 4
        assert list(pool.columns) == POOL_COLUMNS
        assert set(pool[SOURCE_COL]) == {"deepspcas9", "deephf"}
        assert (pool["split_assignment"] == "train").all()
        assert (pool["label_transform"] ==
                config_mod.LABEL_HARMONIZATION["method"]).all()

    def test_shared_guide_marking(self):
        shared_guide = "A" * 20
        dsp = make_dsp_frame([_seq(0, shared_guide), _seq(1)], [0.4, 0.5])
        hf = make_hf_frame([shared_guide, "C" * 20], ["NGG", "NGG"],
                           [0.7, 0.8])
        pool = build_training_pool(dsp, hf)
        shared = pool[pool["duplicate_status"] == "shared_guide_cross_source"]
        assert len(shared) == 2
        assert set(shared[SOURCE_COL]) == {"deepspcas9", "deephf"}

    def test_shared_guide_set(self):
        dsp = make_dsp_frame([_seq(0, "A" * 20), _seq(1)], [0.4, 0.5])
        hf = make_hf_frame(["A" * 20, "C" * 20], ["NGG", "NGG"], [0.7, 0.8])
        assert shared_guide_set(dsp, hf) == {"A" * 20}

    def test_exact_cross_source_30mers_kept_not_averaged(self):
        seq = _seq(0, "G" * 20)
        dsp = make_dsp_frame([seq], [0.4])
        hf = pd.DataFrame({
            NORMALIZED_SEQ_COL: [seq],
            ACTIVITY_COL: [0.9],
            SOURCE_COL: "deephf",
            "guide_sequence": ["G" * 20],
            "pam": ["GGG"],
        })
        pool = build_training_pool(dsp, hf)
        assert len(pool) == 2
        assert pool[ACTIVITY_COL].tolist() == [0.4, 0.9]
        scan = duplicate_and_conflict_scan(pool)
        # one exact 30-mer co-occurring across the two sources (2 rows)
        assert scan["cross_source_exact_30mer_rows"] == 1


class TestDuplicateScan:
    def test_within_source_label_conflict_detected(self):
        seq = _seq(0)
        dsp = pd.DataFrame({
            NORMALIZED_SEQ_COL: [seq, seq],
            ACTIVITY_COL: [0.4, 0.4],
            SOURCE_COL: "deepspcas9",
            "guide_sequence": [seq[4:24]] * 2,
            "pam": ["NGG"] * 2,
        })
        scan = duplicate_and_conflict_scan(dsp)
        assert scan["within_source_duplicate_rows"] > 0

    def test_no_false_conflict_when_labels_agree(self):
        seq = _seq(0)
        dsp = pd.DataFrame({
            NORMALIZED_SEQ_COL: [seq, seq],
            ACTIVITY_COL: [0.4, 0.4],
            SOURCE_COL: "deepspcas9",
            "guide_sequence": [seq[4:24]] * 2,
            "pam": ["NGG"] * 2,
        })
        scan = duplicate_and_conflict_scan(dsp)
        assert scan["within_source_label_conflicts"] == 0


class TestCanonicalValidate:
    def test_drops_bad_length_and_homopolymer(self):
        good = "AAAA" + "ACGT" * 5 + "GGGAAA"   # valid, no homopolymer run
        df = pd.DataFrame({
            NORMALIZED_SEQ_COL: ["A" * 30,              # homopolymer guide
                                 "AAAA" + "C" * 20 + "GGGAAA",  # homopolymer guide
                                 good,
                                 "AAAA" + "ACGT" * 5 + "GGG"],   # bad length (27)
            ACTIVITY_COL: [0.1, 0.2, 0.3, 0.4],
        })
        out = canonical_validate(df, NORMALIZED_SEQ_COL)
        assert len(out) == 1  # only the valid, non-homopolymer 30-mer
        assert out.iloc[0]["activity_label"] == 0.3


class TestInventory:
    def test_inventory_table_fields(self):
        inv = dataset_inventory_table()
        assert {"Dataset", "Included", "N sequences"} <= set(inv.columns)
        included = inv[inv["Included"] == "YES"]
        assert set(included["Dataset"]) == set(allowed_additional_datasets())

    def test_only_deephf_allowed(self):
        assert allowed_additional_datasets() == ["DeepHF"]

    def test_inventory_nonempty(self):
        assert len(DATASET_INVENTORY) > 5


class TestSha256:
    def test_known_hexdigest(self, tmp_path):
        p = tmp_path / "probe.txt"
        p.write_bytes(b"phase10\n")
        assert sha256_file(str(p)) == \
            "0881f70b58c0267b93c1312b4f5e457b7933fd5a2e9ac3518fb5da72a4db3fe5"


class TestStats:
    def _errs(self):
        rng = np.random.default_rng(7)
        b = np.abs(rng.normal(0.1, 0.04, 810))
        d = b - np.abs(rng.normal(0.012, 0.005, 810))
        return b, d

    def test_mean_pairwise_diff_sign(self):
        b, d = self._errs()
        assert mean_pairwise_diff(b, d) > 0

    def test_cohens_dz_zero_when_equal(self):
        b = np.abs(np.random.default_rng(0).normal(0.1, 0.02, 100))
        assert cohens_dz(b, b) == 0.0

    def test_bootstrap_ci_contains_point(self):
        b, d = self._errs()
        ci = bootstrap_ci_meandiff(b, d, n_boot=200, seed=42)
        assert ci["ci_lower"] <= ci["point"] <= ci["ci_upper"]
        assert ci["ci_halfwidth"] >= 0
        assert ci["n_boot"] == 200

    def test_bootstrap_ci_excludes_zero_for_large_effect(self):
        b = np.abs(np.random.default_rng(0).normal(0.1, 0.02, 810))
        d = b * 0.9
        ci = bootstrap_ci_meandiff(b, d, n_boot=200, seed=1)
        assert ci["ci_lower"] > 0

    def test_paired_wilcoxon_zero_diff_guard(self):
        b = np.ones(50)
        r = paired_wilcoxon(b, b)
        assert r["p"] == pytest.approx(1.0)

    def test_paired_t_detects_direction(self):
        b, d = self._errs()
        r = paired_t_test(b, d)
        assert r["t"] > 0
        assert r["p"] < 0.05

    def test_paired_report_keys_and_verdict_inputs(self):
        b, d = self._errs()
        rep = paired_report(b, d, n_boot=100, seed=42)
        assert set(rep) == {"n_samples", "mean_pairwise_diff", "cohens_dz",
                            "paired_t", "paired_wilcoxon", "bootstrap_ci",
                            "verdict_inputs"}
        vi = rep["verdict_inputs"]
        assert vi["primary_d"] == rep["mean_pairwise_diff"]
        assert vi["ci_lower"] == rep["bootstrap_ci"]["ci_lower"]
        assert vi["ci_upper"] == rep["bootstrap_ci"]["ci_upper"]


class TestVerdict:
    def test_supported(self):
        assert decide_verdict(0.03, 0.01, 0.05,
                              {"xgboost": 0.0, "cnn": -0.03}) == "SUPPORTED"

    def test_supported_blocked_by_contradiction(self):
        assert decide_verdict(0.03, 0.01, 0.05,
                              {"xgboost": 0.05, "cnn": -0.03}) == \
            "PARTIAL_SUPPORT"

    def test_partial_small_effect(self):
        assert decide_verdict(0.005, 0.001, 0.01,
                              {"xgboost": 0.0}) == "PARTIAL_SUPPORT"

    def test_not_supported(self):
        assert decide_verdict(-0.01, -0.02, 0.001,
                              {"xgboost": -0.00}) == "NOT_SUPPORTED"

    def test_insufficient_wide_ci(self):
        assert decide_verdict(0.01, -0.05, 0.07,
                              {"xgboost": 0.0}) == "INSUFFICIENT_EVIDENCE"

    def test_pre_registration_sanity(self):
        # The frozen rule must be tied to EFFECT/CONTRADICTION constants.
        assert PRIMARY_MODEL == "cnn"
        assert PRIMARY_ARM == "D1"
        assert ARMS == ["B", "D1"]
        assert EFFECT_THRESHOLD == 0.01
        assert CONTRADICTION_THRESHOLD == 0.02
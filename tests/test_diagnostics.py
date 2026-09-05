"""
Unit tests for the Phase 9A diagnostics utilities.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.stats import (
    descriptive_stats,
    proportion_near_zero,
    proportion_above,
    ks_test,
    welch_ttest,
    cohens_d,
    histogram_data,
    ecdf_data
)
from src.diagnostics.sequence_analysis import (
    gc_content_array,
    nucleotide_frequency_array,
    mean_positional_frequency_matrix,
    positional_entropy,
    mean_kmer_frequencies,
    sequence_length_summary,
    ambiguous_character_summary,
    duplicate_summary,
    exact_sequence_overlap,
    top_differing_kmers,
    kmer_profile_correlation
)
from src.diagnostics.model_diagnostics import (
    prediction_summary,
    error_by_bins,
    model_agreement,
    prediction_vs_true_regression
)


SEQ_A = "ACGT" + "TACCGGTAATTTACCGGATG" + "GGG" + "TCA"
SEQ_B = "TTGA" + "CCCGGTTAAACCCGGTTTAA" + "AGG" + "CCT"
assert len(SEQ_A) == len(SEQ_B) == 30


class TestStats:
    def test_descriptive_stats(self):
        v = [1.0, 2.0, 3.0, 4.0]
        s = descriptive_stats(v)
        assert s['n'] == 4
        assert s['mean'] == pytest.approx(2.5)
        assert s['median'] == pytest.approx(2.5)
        assert s['min'] == 1.0 and s['max'] == 4.0
        assert s['q25'] == pytest.approx(np.percentile(v, 25))

    def test_descriptive_stats_empty_raises(self):
        with pytest.raises(ValueError):
            descriptive_stats([])

    def test_proportions(self):
        v = [0.0, 0.04, 0.5, 0.85, 0.95]
        assert proportion_near_zero(v, 0.05) == pytest.approx(2 / 5)
        assert proportion_above(v, 0.8) == pytest.approx(2 / 5)

    def test_ks_identical(self):
        a = np.linspace(0, 1, 100)
        res = ks_test(a, a)
        assert res['statistic'] == pytest.approx(0.0)
        assert res['p_value'] >= 0.99

    def test_ks_separated(self):
        a = np.random.default_rng(0).normal(0, 1, 200)
        b = np.random.default_rng(1).normal(3, 1, 200)
        assert ks_test(a, b)['p_value'] < 1e-6

    def test_welch_identical(self):
        a = np.linspace(0, 1, 50)
        res = welch_ttest(a, a)
        assert abs(res['t_statistic']) < 1e-12
        assert res['p_value'] == pytest.approx(1.0, abs=1e-6)

    def test_cohens_d_zero_for_identical(self):
        a = np.random.default_rng(2).normal(0, 1, 100)
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_histogram_data(self):
        h = histogram_data(np.arange(10), bins=5)
        assert len(h['bin_edges']) == len(h['counts_density']) + 1

    def test_ecdf_data_monotone(self):
        e = ecdf_data(np.random.default_rng(3).random(1000), n_points=50)
        assert len(e['x']) == 50
        x = np.asarray(e['x']); y = np.asarray(e['y'])
        assert np.all(np.diff(x) >= 0)
        assert np.all((y > 0) & (y <= 1))


class TestSequenceAnalysis:
    def test_gc_content(self):
        assert gc_content_array(["ACGT" * 2])[0] == pytest.approx(0.5)
        assert gc_content_array(["AAATTT"])[0] == pytest.approx(0.0)

    def test_nucleotide_frequency_rows_sum_one(self):
        m = nucleotide_frequency_array([SEQ_A, SEQ_B])
        assert m.shape == (2, 4)
        assert np.allclose(m.sum(axis=1), 1.0)

    def test_positional_frequency_matrix(self):
        pfm = mean_positional_frequency_matrix([SEQ_A, SEQ_B], 30)
        assert pfm.shape == (4, 30)
        assert np.allclose(pfm.sum(axis=0), 1.0)

    def test_positional_entropy_bounds(self):
        # Uniform position -> entropy ~1; constant position -> ~0.
        uniform = np.full((4, 5), 0.25)
        const = np.zeros((4, 5)); const[0, :] = 1.0
        assert np.allclose(positional_entropy(uniform), 1.0, atol=1e-9)
        assert np.allclose(positional_entropy(const), 0.0, atol=1e-9)

    def test_positional_entropy_shape_error(self):
        with pytest.raises(ValueError):
            positional_entropy(np.zeros((4,)))

    def test_mean_kmer_frequencies(self):
        f2 = mean_kmer_frequencies([SEQ_A, SEQ_B], k=2)
        assert f2.shape == (16,)
        assert np.isclose(f2.sum(), 1.0)
        f3 = mean_kmer_frequencies([SEQ_A, SEQ_B], k=3)
        assert f3.shape == (64,)

    def test_length_and_ambiguous_summaries(self):
        ls = sequence_length_summary([SEQ_A, SEQ_B])
        assert ls['uniform'] is True and ls['mean'] == 30
        am = ambiguous_character_summary([SEQ_A, SEQ_B])
        assert am['total_ambiguous_characters'] == 0

    def test_duplicates(self):
        d = duplicate_summary(["A", "A", "B"])
        assert d['unique'] == 2
        assert d['duplicate_rate'] == pytest.approx(1 / 3)

    def test_exact_overlap(self):
        o = exact_sequence_overlap(["A", "B"], ["B", "C"])
        assert o['shared_unique'] == 1
        assert o['jaccard'] == pytest.approx(1 / 3)
        assert o['in_a_only'] == 1 and o['in_b_only'] == 1

    def test_top_differing_kmers(self):
        top = top_differing_kmers(["AAAAAA"], ["CCCCCC"], k=2, top_k=3)
        assert len(top) == 3
        assert all('kmer' in t and 'difference' in t for t in top)
        assert top[0]['kmer'] in ('AA',)

    def test_kmer_profile_correlation_identical(self):
        r = kmer_profile_correlation([SEQ_A] * 20, [SEQ_A] * 20, k=2)
        assert r['pearson_across_kmers'] == pytest.approx(1.0)


class TestModelDiagnostics:
    def test_prediction_summary(self):
        y_true = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_pred = y_true + 0.05
        s = prediction_summary(y_true, y_pred)
        assert s['n'] == 5
        assert s['mae'] == pytest.approx(0.05)
        assert s['bias_mean_pred_minus_true'] == pytest.approx(0.05)
        assert s['pearson_r'] == pytest.approx(1.0)

    def test_prediction_summary_shape_mismatch(self):
        with pytest.raises(ValueError):
            prediction_summary([0.1, 0.2], [0.1])

    def test_error_by_bins(self):
        y_true = np.array([0.1, 0.2, 0.5, 0.8, 0.95])
        y_pred = y_true.copy()
        bins = error_by_bins(y_true, y_pred, y_true, n_bins=5)
        assert sum(b['n'] for b in bins) == 5
        assert all(b['mae'] == pytest.approx(0.0) for b in bins if b['n'] > 0)

    def test_error_by_bins_mismatch(self):
        with pytest.raises(ValueError):
            error_by_bins([0.1], [0.1], [0.1, 0.2])

    def test_model_agreement_identical(self):
        a = np.linspace(0, 1, 40)
        ag = model_agreement(a, a)
        assert ag['pearson'] == pytest.approx(1.0)
        assert ag['spearman'] == pytest.approx(1.0)
        assert ag['mae_between_predictions'] == pytest.approx(0.0)

    def test_agreement_mismatch(self):
        with pytest.raises(ValueError):
            model_agreement(np.zeros(3), np.zeros(4))

    def test_prediction_vs_true_regression_linear(self):
        y_true = np.linspace(0, 1, 50)
        y_pred = 0.5 * y_true + 0.2
        reg = prediction_vs_true_regression(y_true, y_pred)
        assert reg['slope'] == pytest.approx(0.5)
        assert reg['intercept'] == pytest.approx(0.2)
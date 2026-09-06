"""Phase 11 - targeted dataset analysis: pure unit tests.

These tests exercise the deterministic helpers of
src/dataset_analysis/analysis.py on synthetic training-domain data only.
No model training, no Moreno-Mateos data, no external anything.
"""

import unittest

import numpy as np
import pandas as pd

from src.dataset_analysis.analysis import (
    ACTIVITY_EDGES,
    arm_descriptor,
    classify_shift_addressed,
    coverage_by_bin,
    coverage_expansion,
    decision_gate,
    duplicate_and_conflict_report,
    high_activity_coverage,
    hypothesis_assessment,
    js_divergence,
    label_quality_audit,
    pool_composition_table,
    sequence_diversity_report,
    summarize_generated_external,
    wasserstein_1d,
)


POOL_COLS = ['source_dataset', 'original_sequence', 'normalized_sequence',
             'activity_label', 'label_transform', 'experimental_context',
             'duplicate_status', 'split_assignment', 'guide_sequence', 'pam']


def make_pool(n_dsp=5, n_hf=4, dsp_bases='AT', hf_bases='GC'):
    """Synthetic pool of 30-mer rows with a valid subset of the 10 provenance
    columns required by the audit/analysis helpers. Base characters cycle so
    exact row counts are obtained regardless of parity."""
    import itertools
    dsp_cycle = itertools.cycle(dsp_bases)
    hf_cycle = itertools.cycle(hf_bases)
    dsp = [next(dsp_cycle) * 30 for _ in range(n_dsp)]
    hf = [next(hf_cycle) * 30 for _ in range(n_hf)]
    n = n_dsp + n_hf
    rows = []
    for i, (s, seq) in enumerate(zip(['deepspcas9'] * n_dsp + ['deephf'] * n_hf,
                                     dsp + hf)):
        rows.append({
            'source_dataset': s,
            'original_sequence': 'A' * 20 + 'AGG' if i % 2 else seq,
            'normalized_sequence': seq,
            'activity_label': float(0.1 + (i % 5) * 0.2),
            'label_transform': 'raw',
            'experimental_context': 'ctx',
            'duplicate_status': 'single',
            'split_assignment': 'train',
            'guide_sequence': seq[4:24],
            'pam': seq[24:27],
        })
    return pd.DataFrame(rows, columns=POOL_COLS)


class CoverageTests(unittest.TestCase):
    def test_coverage_fractions_sum_to_one(self):
        labels = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
        bins = coverage_by_bin(labels)
        self.assertAlmostEqual(sum(b['fraction'] for b in bins), 1.0, places=9)
        self.assertEqual(sum(b['n'] for b in bins), len(labels))

    def test_coverage_uses_canonical_edges(self):
        labels = [0.5, 0.5]
        bins = coverage_by_bin(labels)
        self.assertEqual([b['bin'] for b in bins], ['[0.00, 0.20]', '[0.20, 0.40]',
                                                    '[0.40, 0.60]', '[0.60, 0.80]',
                                                    '[0.80, 1.00]'])
        self.assertEqual(ACTIVITY_EDGES, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_coverage_empty_raises(self):
        with self.assertRaises(ValueError):
            coverage_by_bin([])

    def test_coverage_last_bin_is_closed(self):
        # 1.0 must land in the last bin, not be dropped
        bins = coverage_by_bin([1.0])
        self.assertEqual(bins[-1]['n'], 1)

    def test_expansion_flags_material_bins(self):
        base = [0.1] * 10
        added = [0.9] * 10
        exp = coverage_expansion(base, added)
        expanded = [b['bin'] for b in exp['bins'] if b['expanded']]
        self.assertIn('[0.80, 1.00]', expanded)
        # 10 base rows + 10 added rows: bin [0.8, 1.0] covers 0.5 of the
        # combined pool vs 0.0 of the base
        self.assertAlmostEqual(exp['combined_coverage']['max_delta_fraction'],
                               0.5)

    def test_expansion_no_flag_when_distributions_similar(self):
        base = [0.3, 0.3, 0.3, 0.7]
        added = [0.3, 0.3, 0.3, 0.7]
        exp = coverage_expansion(base, added)
        self.assertEqual([b['bin'] for b in exp['bins'] if b['expanded']], [])


class DiversityTests(unittest.TestCase):
    def test_js_divergence_identical_is_zero(self):
        p = [0.25, 0.25, 0.25, 0.25]
        self.assertAlmostEqual(js_divergence(p, p), 0.0, places=10)

    def test_js_divergence_orthogonal_is_max(self):
        self.assertAlmostEqual(js_divergence([1.0, 0.0], [0.0, 1.0]), 1.0,
                               places=10)

    def test_js_divergence_bounded_and_symmetric(self):
        rng = np.random.RandomState(0)
        a = rng.randint(1, 20, size=8).astype(float)
        b = rng.randint(1, 20, size=8).astype(float)
        ab = js_divergence(a, b)
        self.assertGreaterEqual(ab, 0.0)
        self.assertLessEqual(ab, 1.0)
        self.assertAlmostEqual(ab, js_divergence(b, a), places=12)

    def test_js_divergence_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            js_divergence([0.5, 0.5], [1.0])

    def test_wasserstein_1d_known(self):
        self.assertAlmostEqual(wasserstein_1d([0, 0], [1, 1]), 1.0, places=9)
        self.assertAlmostEqual(wasserstein_1d([0, 1], [0, 1]), 0.0, places=9)

    def test_diversity_report_structure_and_overlap(self):
        a = ['A' * 30] * 4 + ['T' * 30] * 4
        b = ['G' * 30] * 8
        r = sequence_diversity_report(a, b, add_name='deephf')
        self.assertEqual(r['overlap']['exact_30mer']['shared_unique'], 0)
        self.assertIn('top_differing_3mers', r['kmer_3'])
        self.assertIn('wasserstein_1d', r['gc'])

    def test_diversity_report_detects_shared_sequences(self):
        shared = 'ACGT' * 7 + 'AC'
        a = [shared, 'A' * 30]
        b = [shared, 'G' * 30]
        r = sequence_diversity_report(a, b)
        self.assertEqual(r['overlap']['exact_30mer']['shared_unique'], 1)
        self.assertEqual(r['overlap']['guide_20mer_overlap_count'], 1)


class CompositionTests(unittest.TestCase):
    def test_pool_composition_counts(self):
        pool = make_pool(n_dsp=4, n_hf=3)
        comp = pool_composition_table(pool)
        self.assertEqual(comp['deepspcas9']['retained_count'], 4)
        self.assertEqual(comp['deephf']['retained_count'], 3)
        self.assertAlmostEqual(comp['deepspcas9']['fraction_of_pool'], 4 / 7)
        self.assertEqual(comp['deepspcas9']['unique_sequences'], 2)
        self.assertEqual(comp['deepspcas9']['duplicate_count'], 0)

    def test_composition_duplicate_stats(self):
        seqs = ['A' * 30, 'A' * 30, 'G' * 30]
        pool = make_pool(n_dsp=3, n_hf=0)
        comp = pool_composition_table(pool)
        self.assertEqual(comp['deepspcas9']['retained_count'], 3)


class HighActivityAndAuditTests(unittest.TestCase):
    def test_high_activity_coverage_adds_tail(self):
        base = [0.4] * 8
        added = [0.9] * 4
        ha = high_activity_coverage(base, added, [0.8])
        row = ha['rows']['0.8']
        self.assertEqual(row['base_fraction'], 0.0)
        self.assertEqual(row['added_fraction'], 1.0)
        self.assertAlmostEqual(row['combined_delta_fraction'], 1 / 3)

    def test_label_quality_audit_counts_nan(self):
        df = pd.DataFrame({
            'activity_label': [0.1, np.nan, 0.3, np.nan]},
            index=range(4))
        df.attrs['n_mapped_valid'] = 100
        df.attrs['n_unlabeled_dropped'] = 7
        pool = pd.DataFrame({'normalized_sequence': ['A' * 30] * 3,
                             'activity_label': [0.1, 0.2, 0.3]})
        audit = label_quality_audit(df['activity_label'], df, pool)
        self.assertEqual(audit['n_unlabeled_dropped'], 7)
        self.assertEqual(audit['nan_in_retained_labels'], 2)
        self.assertEqual(audit['pool_nan_labels'], 0)

    def test_duplicate_conflict_report_basic(self):
        pool = make_pool(n_dsp=3, n_hf=3, dsp_bases='AT', hf_bases='GC')
        r = duplicate_and_conflict_report(pool)
        self.assertEqual(r['n_rows'], 6)
        self.assertEqual(r['label_col_null'], 0)
        self.assertTrue(r['policy_verified'])


class ClassificationTests(unittest.TestCase):
    def test_classify_shift_directions(self):
        up = classify_shift_addressed('x', 0.1, 0.5, 'increase_needed')
        self.assertIn('addressed', up['status'])
        down = classify_shift_addressed('y', 0.5, 0.1, 'decrease_needed')
        self.assertIn('addressed', down['status'])
        wrong_dir = classify_shift_addressed('z', 0.5, 0.1, 'increase_needed')
        self.assertEqual(wrong_dir['status'], 'not addressed')
        with self.assertRaises(ValueError):
            classify_shift_addressed('w', 1.0, 0.0, 'sideways')


# ------------------------------------------------------------ hypotheses
HYP_EVIDENCE = dict(
    sample_ratio=6.62,
    added_samples_fraction=0.849,
    coverage_expanded_bins=[
        {'bin': '[0.60, 0.80]', 'expanded': True},
        {'bin': '[0.80, 1.00]', 'expanded': True}],
    high_coverage_threshold=0.8,
    high_base_fraction=0.025,
    high_added_fraction=0.33,
    kmer_js_divergence=0.06,
    gc_cohens_d=0.4,
    experimental_contexts=2,
    unique_domains=2,
    external_effect_size=0.0022,
    external_ci_excludes_zero=False,
)


class HypothesisTests(unittest.TestCase):
    def test_requires_all_keys(self):
        ev = dict(HYP_EVIDENCE)
        del ev['sample_ratio']
        with self.assertRaises(KeyError):
            hypothesis_assessment(ev)

    def test_verdicts_in_allowed_set(self):
        h = hypothesis_assessment(HYP_EVIDENCE)
        for key in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            verdict = h[key]
            self.assertIn(verdict, ['SUPPORTED', 'PARTIALLY SUPPORTED',
                                    'NOT SUPPORTED', 'UNTESTED',
                                    'NOT IDENTIFIABLE'])

    def test_sample_size_only_strongest_lever(self):
        ev = dict(HYP_EVIDENCE)
        ev.update(kmer_js_divergence=0.001, gc_cohens_d=0.01)
        h = hypothesis_assessment(ev)
        self.assertEqual(h['H1'], 'SUPPORTED')
        self.assertEqual(h['H3'], 'NOT SUPPORTED')

    def test_coverage_only_second_lever(self):
        ev = dict(HYP_EVIDENCE)
        ev.update(coverage_expanded_bins=[
            {'bin': '[0.80, 1.00]', 'expanded': True}])
        h = hypothesis_assessment(ev)
        self.assertEqual(h['H2'], 'SUPPORTED')

    def test_no_expansion_not_supported(self):
        # No material bin expansion: with sample_ratio>1 the claim of
        # "materially increased activity-range coverage" is only partially
        # supported (rule keeps PARTIALLY for any added mass, NOT only when
        # nothing at all changed).
        ev = dict(HYP_EVIDENCE)
        ev.update(coverage_expanded_bins=[
            {'bin': '[0.40, 0.60]', 'expanded': False}])
        h = hypothesis_assessment(ev)
        self.assertEqual(h['H2'], 'PARTIALLY SUPPORTED')


class GateTests(unittest.TestCase):
    def test_gate_D_when_effect_not_meaningful(self):
        ev = dict(h=hypothesis_assessment(HYP_EVIDENCE),
                  external_effect_size=0.0022,
                  external_ci_excludes_zero=False,
                  effect_threshold=0.01)
        gate = decision_gate(ev)
        self.assertEqual(gate['gate'], 'D')

    def test_gate_D_when_ci_crosses_zero(self):
        ev = dict(h=hypothesis_assessment(HYP_EVIDENCE),
                  external_effect_size=0.012,
                  external_ci_excludes_zero=False,
                  effect_threshold=0.01)
        self.assertEqual(decision_gate(ev)['gate'], 'D')

    def test_gate_A_diversity_stronger(self):
        h = hypothesis_assessment(dict(
            HYP_EVIDENCE, coverage_expanded_bins=[
                {'bin': '[0.40, 0.60]', 'expanded': False}]))
        ev = dict(h=h, external_effect_size=0.012,
                  external_ci_excludes_zero=True, effect_threshold=0.01)
        self.assertEqual(decision_gate(ev)['gate'], 'A')

    def test_gate_B_coverage_stronger(self):
        h = hypothesis_assessment(dict(
            HYP_EVIDENCE, kmer_js_divergence=0.01, gc_cohens_d=0.05))
        ev = dict(h=h, external_effect_size=0.012,
                  external_ci_excludes_zero=True, effect_threshold=0.01)
        self.assertEqual(decision_gate(ev)['gate'], 'B')

    def test_gate_C_confounded(self):
        ev = dict(h=hypothesis_assessment(HYP_EVIDENCE),
                  external_effect_size=0.012,
                  external_ci_excludes_zero=True, effect_threshold=0.01)
        self.assertEqual(decision_gate(ev)['gate'], 'C')

    def test_gate_requires_h(self):
        with self.assertRaises(KeyError):
            decision_gate({})


class AttributionTests(unittest.TestCase):
    def test_arm_descriptor(self):
        self.assertIn('DeepSpCas9 training split only', arm_descriptor('B')['composition'])
        self.assertIn('56,894', arm_descriptor('D1')['composition'])
        with self.assertRaises(ValueError):
            arm_descriptor('X')

    def test_summarize_generated_external_quotes(self):
        ext = {'cnn': {'B': {'mae': 0.25, 'rmse': 0.3, 'r2': 0.1,
                             'pearson': 0.2, 'spearman': 0.3,
                             'prediction_std': 0.1}}}
        s = summarize_generated_external('cnn', 'B', ext)
        self.assertEqual(s['mae'], 0.25)
        self.assertIn('locked', s['source'])


if __name__ == '__main__':
    unittest.main()
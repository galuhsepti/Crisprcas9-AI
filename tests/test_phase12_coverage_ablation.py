"""Phase 12 - controlled data-coverage ablation: unit tests.

Pure synthetic-data tests. No model training, no external data, and the
Moreno-Mateos freeze is respected (no such data is ever referenced).
"""

import unittest

import numpy as np
import pandas as pd

from src.coverage_ablation.matching import (
    activity_matching_report,
    construct_arm_pools,
    decide_gate,
    feasibility_audit,
    pool_equivalence_audit,
    sample_size_report,
    stratified_subsample,
    sequence_matching_report,
)
from src.coverage_ablation.analysis import (
    arm_pool_metadata,
    high_activity_summary,
)
from src.coverage_ablation.stats import contrast_report, cohens_dz


def make_frame(seqs, labels, source, context='ctx'):
    n = len(seqs)
    return pd.DataFrame({
        'normalized_sequence': seqs,
        'activity_label': labels,
        'source_dataset': [source] * n,
        'experimental_context': [context] * n,
        'guide_sequence': [s[4:24] for s in seqs],
        'pam': [s[24:27] for s in seqs],
        'original_sequence': list(seqs),
        'label_transform': ['raw'] * n,
        'duplicate_status': ['single'] * n,
        'split_assignment': ['train'] * n,
    })


def canonical_like(n):
    seqs = [('A' * 12 if i % 3 else 'T' * 12) + 'AGG' + 'A' * 15
            for i in range(n)]
    labels = [0.3 + 0.6 * (i / max(1, n - 1)) * 0.5 for i in range(n)]
    return make_frame(seqs, labels, 'deepspcas9')


class SampleSizeTests(unittest.TestCase):
    def test_report_abs_and_pct(self):
        r = sample_size_report(100, 110)
        self.assertEqual(r['abs_diff'], 10)
        self.assertAlmostEqual(r['rel_pct'], 10.0)
        self.assertTrue(r['within_proposed_tolerance'])

    def test_report_large_mismatch(self):
        r = sample_size_report(1000, 6000)
        self.assertAlmostEqual(r['rel_pct'], 500.0)
        self.assertFalse(r['within_proposed_tolerance'])


class ActivityMatchingTests(unittest.TestCase):
    def test_identical_distribution(self):
        a = [0.1] * 4 + [0.4] * 4 + [0.7] * 4
        r = activity_matching_report(a, a)
        self.assertAlmostEqual(r['wasserstein_1d'], 0.0, places=8)
        self.assertAlmostEqual(r['cohens_d'], 0.0, places=8)
        self.assertAlmostEqual(r['bins']['max_abs_bin_delta'], 0.0)
        self.assertAlmostEqual(r['quantiles']['max_abs_quantile_delta'], 0.0)

    def test_different_tail_same_mean(self):
        # same mean but different tails must be detected as mismatched
        a = [0.2] * 8 + [0.8] * 2          # mean 0.32, heavy high tail
        b = [0.32] * 10                     # same mean, flat
        r = activity_matching_report(a, b)
        self.assertAlmostEqual(float(np.mean(a)), float(np.mean(b)), places=6)
        self.assertGreater(r['wasserstein_1d'], 0.1)
        self.assertGreater(r['bins']['max_abs_bin_delta'], 0.05)
        self.assertGreater(r['quantiles']['max_abs_quantile_delta'], 0.1)

    def test_quantiles_keys(self):
        a = list(np.linspace(0.05, 0.95, 200))
        r = activity_matching_report(a, a)
        self.assertEqual(sorted(r['quantiles']['a'].keys()),
                         ['p10', 'p25', 'p5', 'p50', 'p75', 'p90', 'p95'])


class SequenceMatchingTests(unittest.TestCase):
    def test_identical_sequences(self):
        seqs = ['A' * 30] * 3 + ['T' * 30] * 3
        r = sequence_matching_report(seqs, seqs)
        self.assertAlmostEqual(r['gc']['cohens_d'], 0.0, places=8)
        self.assertAlmostEqual(r['kmer_3']['js_divergence'], 0.0, places=8)
        self.assertEqual(r['unique_sequences']['a'],
                         r['unique_sequences']['b'])

    def test_different_composition_detected(self):
        # GC 0.20/0.40 vs GC 0.70/0.90 - domains far apart, within-group
        # variance non-zero so Cohen's d is well defined.
        a = ['G' * 6 + 'A' * 24] * 3 + ['G' * 12 + 'A' * 18] * 2
        b = ['G' * 21 + 'A' * 9] * 3 + ['G' * 27 + 'A' * 3] * 2
        r = sequence_matching_report(a, b)
        self.assertGreater(abs(r['gc']['cohens_d']), 1.0)


class PoolEquivalenceTests(unittest.TestCase):
    def test_equivalence_audit_sections(self):
        pa = canonical_like(20)
        pb = canonical_like(22)
        audit = pool_equivalence_audit(pa, pb)
        self.assertEqual(sorted(audit.keys()),
                         ['activity', 'domain', 'sample_size', 'sequence'])
        self.assertEqual(audit['sample_size']['n_a'], 20)
        self.assertEqual(audit['sample_size']['n_b'], 22)


class FeasibilityTests(unittest.TestCase):
    def test_infeasible_when_only_wrong_domain_extra(self):
        can = canonical_like(20)
        ext = {'deephf': make_frame(['G' * 30] * 30, [0.9] * 30, 'deephf')}
        frames = {
            'canonical': {'source': 'deepspcas9', 'n': 20, 'context': 'X'},
            'additional': {'deephf': {'source': 'deephf', 'n': 30,
                                      'context': 'Y'}},
        }
        f = feasibility_audit(frames, can, ext)
        self.assertEqual(f['decision'], 'INFEASIBLE')
        self.assertIn(1, f['triggered_stop_conditions'])  # no matched control
        self.assertIn(9, f['triggered_stop_conditions'])  # sample size
        self.assertEqual(decide_gate(f)['gate'], 'F')

    def test_feasible_when_same_domain_extra_exists(self):
        can = canonical_like(5)
        ext = {'deepspcas9_more': canonical_like(50)}
        frames = {
            'canonical': {'source': 'deepspcas9', 'n': 5, 'context': 'X'},
            'additional': {'deepspcas9_more': {'source': 'deepspcas9',
                                               'n': 50, 'context': 'X'}},
        }
        f = feasibility_audit(frames, can, ext)
        self.assertEqual(f['decision'], 'FEASIBLE')
        self.assertEqual(f['triggered_stop_conditions'], [])


class SubsampleTests(unittest.TestCase):
    def test_stratified_subsample_deterministic(self):
        df = canonical_like(200)
        s1 = stratified_subsample(df, 100, seed=42)
        s2 = stratified_subsample(df, 100, seed=42)
        self.assertTrue(s1['normalized_sequence'].equals(
            s2['normalized_sequence']))
        self.assertEqual(len(s1), 100)

    def test_stratified_subsample_preserves_profile(self):
        df = canonical_like(400)
        s = stratified_subsample(df, 100, seed=42)
        r = activity_matching_report(df['activity_label'], s['activity_label'])
        self.assertGreater(r['bins']['max_abs_bin_delta'], 0.0)
        self.assertLess(r['bins']['max_abs_bin_delta'], 0.25)

    def test_subsample_never_exceeds_available(self):
        df = canonical_like(50)
        s = stratified_subsample(df, 1000, seed=1)
        self.assertLessEqual(len(s), 50)


class ArmConstructionTests(unittest.TestCase):
    def test_stop9_when_no_same_domain_extra(self):
        can = canonical_like(20)
        extras = {'deephf': make_frame(['G' * 30] * 80, [0.9] * 80, 'deephf')}
        with self.assertRaises(ValueError) as ctx:
            construct_arm_pools(can, extras, sizes={'B': 100, 'C': 100,
                                                    'D': 100})
        self.assertIn('STOP-9', str(ctx.exception))

    def test_builds_all_arms_when_feasible(self):
        can = canonical_like(20)
        extras = {'deepspcas9_more': canonical_like(400)}
        pools = construct_arm_pools(can, extras,
                                    sizes={'B': 200, 'C': 150, 'D': 150})
        self.assertEqual(set(pools.keys()), {'A', 'B', 'C', 'D', 'E'})
        self.assertEqual(len(pools['A']), 20)
        self.assertGreater(len(pools['B']), 20)
        self.assertEqual(len(pools['E']), 20 + 400)
        # A unchanged (reference integrity)
        self.assertTrue(pools['A']['normalized_sequence'].equals(
            can['normalized_sequence'].reset_index(drop=True)))


class MetadataAndSummaryTests(unittest.TestCase):
    def test_arm_metadata_required_fields(self):
        pool = make_frame(['A' * 30] * 4 + ['G' * 30] * 2,
                          [0.5] * 6, 'deepspcas9')
        meta = arm_pool_metadata(pool, 'A', 'canonical', 42, {}, {'dsp': 'x'})
        for k in ['arm', 'source_datasets', 'n_samples', 'n_unique_sequences',
                  'activity', 'gc', 'source_proportions',
                  'selection_method', 'random_seed']:
            self.assertIn(k, meta)
        self.assertEqual(meta['n_samples'], 6)
        self.assertEqual(meta['n_unique_sequences'], 2)

    def test_high_activity_summary(self):
        s = high_activity_summary([0.2, 0.85, 0.9], [0.2, 0.2, 0.2, 0.2])
        self.assertEqual(s['n_high'], 2)
        self.assertAlmostEqual(s['fraction_high'], 2 / 3)
        self.assertEqual(s['baseline_fraction_high'], 0.0)

    def test_high_activity_enrichment(self):
        s = high_activity_summary([0.2, 0.9], [0.2, 0.2, 0.4, 0.6])
        self.assertAlmostEqual(s['fraction_high'], 0.5)
        self.assertAlmostEqual(s['baseline_fraction_high'], 0.0)


class StatsTests(unittest.TestCase):
    def test_contrast_mae_sign_and_ci(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        pa = [1.5, 2.5, 3.5, 4.5, 5.5]   # worse arm A
        pb = [1.02, 2.01, 3.03, 4.0, 5.0]  # better arm B
        r = contrast_report(y, pa, pb, metric='mae')
        self.assertLess(r['delta_b_minus_a'], 0)       # B better (MAE down)
        self.assertLess(r['mean_pairwise_diff'], 0)    # mean(|eB|-|eA|) < 0
        self.assertLessEqual(r['ci_upper'], r['mean_pairwise_diff'] + 1e-6 +
                             r['ci_upper'] - r['ci_lower'] + 1e-6)  # sane CI

    def test_contrast_r2(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        pa = [1.02, 2.01, 3.0, 4.0, 5.0]
        pb = [1.5, 2.5, 3.5, 4.5, 5.5]
        r = contrast_report(y, pa, pb, metric='r2')
        self.assertGreater(r['value_a'], r['value_b'])

    def test_contrast_shape_validation(self):
        with self.assertRaises(ValueError):
            contrast_report([1, 2, 3], [1, 2], [1, 2, 3])

    def test_cohens_dz(self):
        d = cohens_dz([1.0, 1.0, 1.0])
        self.assertEqual(d, 0.0)


if __name__ == '__main__':
    unittest.main()
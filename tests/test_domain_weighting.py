"""Unit tests for the Phase 9C domain-adaptation weighting utilities."""

import numpy as np
import pytest

from src.adaptation import (
    inverse_density_weights,
    uniform_weights,
    combine_weights,
    label_reweighting_weights,
    gc_reweighting_weights,
    combined_reweighting_weights,
    weight_summary,
)


@pytest.fixture
def bimodal_labels():
    """Dense cluster near 0.05, sparse tail near 0.95 (mirrors DeepSpCas9
    vs Moreno-Mateos label imbalance)."""
    rng = np.random.RandomState(0)
    low = rng.normal(0.05, 0.02, size=3600)
    high = rng.normal(0.95, 0.02, size=400)
    return np.clip(np.concatenate([low, high]), 0.001, 0.999)


class TestInverseDensityWeights:
    def test_returns_same_length(self, bimodal_labels):
        w = inverse_density_weights(bimodal_labels)
        assert w.shape == bimodal_labels.shape

    def test_mean_approximately_one(self, bimodal_labels):
        w = inverse_density_weights(bimodal_labels)
        assert abs(np.mean(w) - 1.0) < 0.5

    def test_weights_actually_vary(self, bimodal_labels):
        """Regression guard: the weights must not collapse to all-1 (the
        original 1/p-scale clipping bug wiped all variation out)."""
        w = inverse_density_weights(bimodal_labels)
        assert np.max(w) > 1.5
        assert np.min(w) < 0.9

    def test_upweights_sparse_tail(self, bimodal_labels):
        w = inverse_density_weights(bimodal_labels)
        tail_idx = bimodal_labels > 0.9
        dense_idx = bimodal_labels < 0.1
        assert np.mean(w[tail_idx]) > np.mean(w[dense_idx])
        assert np.min(w) > 0 and np.max(w) > 1.5

    def test_clip_respected(self):
        rng = np.random.RandomState(0)
        y = np.concatenate([
            np.repeat(0.5, 900),
            np.linspace(0.0, 1.0, 100),
        ])
        w = inverse_density_weights(y, clip_min=0.1, clip_max=5.0)
        assert np.all(w >= 0.1 - 1e-12)
        assert np.all(w <= 5.0 + 1e-12)

    def test_constant_covariate_is_identity(self):
        y = np.full(50, 0.4)
        w = inverse_density_weights(y)
        assert np.allclose(w, 1.0)

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            inverse_density_weights(np.array([], dtype=float))
        with pytest.raises(ValueError):
            inverse_density_weights(np.array([0.0, np.nan, 1.0]))
        with pytest.raises(ValueError):
            inverse_density_weights(np.array([0.0, 1.0]), n_bins=0)
        with pytest.raises(ValueError):
            inverse_density_weights(np.array([0.0, 1.0]), clip_min=0.0)
        with pytest.raises(ValueError):
            inverse_density_weights(np.array([0.0, 1.0]), clip_min=2.0, clip_max=1.0)


class TestArmConvenience:
    def test_uniform(self, bimodal_labels):
        w = uniform_weights(bimodal_labels)
        assert np.allclose(w, 1.0)

    def test_label_weights_match_inverse_density(self, bimodal_labels):
        w1 = label_reweighting_weights(bimodal_labels)
        w2 = inverse_density_weights(bimodal_labels)
        assert np.allclose(w1, w2)

    def test_gc_weights_shape_and_scale(self):
        gc = np.linspace(0.2, 0.8, 500)
        w = gc_reweighting_weights(gc)
        assert w.shape == gc.shape
        assert abs(np.mean(w) - 1.0) < 0.5

    def test_combined_product(self, bimodal_labels):
        gc = np.linspace(0.3, 0.7, len(bimodal_labels))
        wl = label_reweighting_weights(bimodal_labels)
        wg = gc_reweighting_weights(gc)
        w_lg = combined_reweighting_weights(bimodal_labels, gc)
        expected = wl * wg
        expected = expected / np.mean(expected)
        expected = np.clip(expected, 0.1, 10.0)
        assert np.allclose(w_lg, expected)

    def test_lg_emphasises_tail(self, bimodal_labels):
        gc = np.linspace(0.3, 0.7, len(bimodal_labels))
        w = combined_reweighting_weights(bimodal_labels, gc)
        tail_idx = bimodal_labels > 0.9
        dense_idx = bimodal_labels < 0.1
        assert np.mean(w[tail_idx]) > np.mean(w[dense_idx])


class TestCombineWeights:
    def test_requires_at_least_one_array(self):
        with pytest.raises(ValueError):
            combine_weights()

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            combine_weights(np.ones(3), np.ones(4))

    def test_composes_and_normalises(self):
        a = np.array([1.0, 1.0, 10.0, 1.0])
        b = np.array([1.0, 5.0, 1.0, 1.0])
        combo = combine_weights(a, b, clip_min=0.1, clip_max=10.0)
        assert abs(np.mean(combo) - 1.0) < 0.5
        assert np.argmax(combo) == 2  # both large weights align on index 2
        assert combo[2] > combo[1]


class TestWeightSummary:
    def test_uniform(self):
        s = weight_summary(np.ones(100))
        assert s['mean'] == 1.0
        assert s['ess_ratio'] == pytest.approx(1.0)

    def test_reweighting_reduces_ess(self):
        w = np.array([5.0] * 10 + [0.2] * 90)
        s = weight_summary(w)
        assert s['effective_sample_size'] < 100
        assert s['ess_ratio'] < 1.0
        assert s['n'] == 100

    def test_mean_honoured(self):
        values = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        s = weight_summary(values)
        assert np.isclose(s['mean'], np.mean(values))
        assert s['min'] == 0.5 and s['max'] == 2.5

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            weight_summary(np.array([1.0, -1.0]))
"""
Unit tests for CNN attribution utilities (Phase 8).
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interpretability.cnn_attribution import (
    position_saliency,
    integrated_gradients,
    attribution_by_region
)


def make_onehot(seqs, L=30):
    nuc_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    X = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            X[i, j, nuc_idx[c]] = 1.0
    return X


@pytest.fixture
def onehot_batch():
    rng = np.random.default_rng(0)
    seqs = [''.join(rng.choice(list('ACGT'), 30)) for _ in range(8)]
    return make_onehot(seqs)


class SinglePositionNet(nn.Module):
    """Linear network whose output only depends on column 3."""

    def __init__(self, L=30):
        super().__init__()
        self.lin = nn.Linear(L * 4, 1, bias=False)
        # Layout is channel-major: flat index = c * L + position.
        with torch.no_grad():
            self.lin.weight.zero_()
            idx = [c * L + 3 for c in range(4)]
            vals = torch.tensor([1.0, -2.0, 3.0, -4.0])
            self.lin.weight[:, idx] = vals

    def forward(self, x):
        b = x.shape[0]
        return self.lin(x.reshape(b, -1))


class TestPositionSaliency:
    """Tests for position_saliency."""

    def test_shape_and_normalization(self, onehot_batch):
        net = SinglePositionNet()
        profile = position_saliency(net, onehot_batch, batch_size=4)
        assert profile.shape == (30,)
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0)
        assert profile.sum() == pytest.approx(1.0, abs=1e-6)

    def test_concentration_at_sensitive_position(self, onehot_batch):
        net = SinglePositionNet()
        profile = position_saliency(net, onehot_batch)
        assert profile[3] == pytest.approx(1.0, abs=1e-6)
        assert np.all(profile[:3] < 1e-9)
        assert np.all(profile[4:] < 1e-9)

    def test_constant_network_zero_sensitivity(self, onehot_batch):
        # A network with zero weights has zero sensitivity everywhere; this is
        # reported honestly as an all-zero (finite) profile.
        net = SinglePositionNet()
        with torch.no_grad():
            net.lin.weight.zero_()
        profile = position_saliency(net, onehot_batch)
        assert np.all(np.isfinite(profile))
        assert np.all(np.abs(profile) < 1e-9)


class TestIntegratedGradients:
    """Tests for integrated_gradients."""

    def test_shape_and_normalization(self, onehot_batch):
        net = SinglePositionNet()
        profile = integrated_gradients(net, onehot_batch, steps=10)
        assert profile.shape == (30,)
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0)
        assert profile.sum() == pytest.approx(1.0, abs=1e-6)

    def test_zero_baseline_linear_matches_saliency(self, onehot_batch):
        # For a linear net with zero baseline, IG == saliency in expectation
        # over the one-hot support (the profile concentrates at position 3).
        net = SinglePositionNet()
        ig = integrated_gradients(net, onehot_batch, steps=10)
        assert ig[3] == pytest.approx(1.0, abs=1e-6)

    def test_invalid_steps_raises(self, onehot_batch):
        net = SinglePositionNet()
        with pytest.raises(ValueError):
            integrated_gradients(net, onehot_batch, steps=0)


class TestAttributionByRegion:
    """Tests for attribution_by_region."""

    def test_fractions_sum_to_one(self):
        profile = np.ones(30)
        res = attribution_by_region(profile)
        assert set(res) == {'guide', 'pam', 'flanks_5p', 'flanks_3p'}
        total = sum(res.values())
        assert total == pytest.approx(1.0)
        assert res['guide'] == pytest.approx(20 / 30)
        assert res['pam'] == pytest.approx(3 / 30)
        assert res['flanks_5p'] == pytest.approx(4 / 30)
        assert res['flanks_3p'] == pytest.approx(3 / 30)

    def test_zero_total_returns_zeros(self):
        res = attribution_by_region(np.zeros(30))
        assert all(v == 0.0 for v in res.values())

    def test_negative_profile_rejected(self):
        with pytest.raises(ValueError):
            attribution_by_region(np.full(30, -0.1))

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            attribution_by_region(np.ones(20))

    def test_unknown_region_rejected(self):
        with pytest.raises(ValueError):
            attribution_by_region(np.ones(30), regions=['guide', 'nope'])
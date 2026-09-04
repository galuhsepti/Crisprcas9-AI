"""
Unit tests for the CNN model.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn import CNNModel, CRISPRsvGN
import torch


@pytest.fixture
def small_data():
    """Small synthetic one-hot data for fast tests."""
    np.random.seed(42)
    n = 40
    X = np.zeros((n, 30, 4), dtype=np.float32)
    for i in range(n):
        for j in range(30):
            X[i, j, np.random.randint(4)] = 1.0
    y = np.random.rand(n).astype(np.float32) * X.sum(axis=(1, 2))[:n] * 0.1
    # Make a small deterministic relationship
    y = (X[:, 10, 0] * 0.8 + X[:, 15, 1] * 0.5 + 0.1 * np.random.randn(n)).astype(np.float32)
    return X, y


class TestCRISPRsvGN:
    """Test the PyTorch network module."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        net = CRISPRsvGN(input_length=30, n_nucleotides=4)
        x = torch.randn(8, 4, 30)
        out = net(x)
        assert out.shape == (8, 1)

    def test_forward_handles_batch_numpy_layout(self):
        """Test network accepts (batch, length, 4) layout via wrapper transpose."""
        net = CRISPRsvGN(input_length=30, n_nucleotides=4)
        x = torch.randn(8, 30, 4)
        # The forward method transposes channel-first layout
        out = net(x)
        assert out.shape == (8, 1)


class TestCNNModel:
    """Test CNNModel wrapper."""

    def test_init(self):
        """Test model initialization."""
        model = CNNModel(input_length=30, use_cpu_threads=1)
        assert model.input_length == 30
        assert not model.is_fitted

    def test_fit_predict(self, small_data):
        """Test fit and predict on small data."""
        X, y = small_data
        model = CNNModel(epochs=3, batch_size=8, use_cpu_threads=1, random_state=42)
        history = model.fit(X, y, verbose=False)
        assert model.is_fitted
        assert 'train_loss' in history
        assert model.predict(X[:5]).shape == (5,)

    def test_fit_with_validation(self, small_data):
        """Test early stopping with validation set."""
        X, y = small_data
        model = CNNModel(epochs=50, batch_size=8, patience=2, use_cpu_threads=1, random_state=42)
        history = model.fit(X[:30], y[:30], X[30:], y[30:], verbose=False)
        assert 'val_loss' in history
        assert history['early_stopping'] is True
        assert model.best_val_loss is not None

    def test_predict_before_fit(self, small_data):
        """Test predict before fitting."""
        X, _ = small_data
        model = CNNModel(use_cpu_threads=1)
        with pytest.raises(ValueError):
            model.predict(X)

    def test_save_load(self, tmp_path, small_data):
        """Test save/load."""
        X, y = small_data
        model = CNNModel(epochs=2, batch_size=8, use_cpu_threads=1, random_state=42)
        model.fit(X, y, verbose=False)
        path = tmp_path / "cnn.pt"
        model.save_model(str(path))
        loaded = CNNModel.load_model(str(path))
        assert loaded.is_fitted
        np.testing.assert_allclose(model.predict(X[:5]), loaded.predict(X[:5]), atol=1e-5)

    def test_get_params(self):
        """Test parameter extraction."""
        model = CNNModel(epochs=100, learning_rate=0.0005)
        params = model.get_params()
        assert params['epochs'] == 100
        assert params['learning_rate'] == 0.0005

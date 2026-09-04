"""
Convolutional Neural Network (CNN) model for CRISPR-Cas9 sgRNA activity
prediction using PyTorch.

Architecture (from config.yaml):
  - Conv1D layer 1: 64 filters, kernel 3, ReLU
  - (max) pooling
  - Conv1D layer 2: 32 filters, kernel 3, ReLU
  - (max) pooling
  - Dense layer: 64 units, dropout
  - Output: 1 unit, linear (regression)

This model is the primary model of the thesis. Unlike the RF/XGB baselines,
early stopping on the validation set is used (best model by validation
loss retained) because this is the main model, not a baseline. Moreno-Mateos
remains held out and is only used for final evaluation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import logging
import time

logger = logging.getLogger(__name__)


class CRISPRsvGN(nn.Module):
    """
    Multi-convolutional neural network for sgRNA activity prediction.
    
    Inspired by the original CRISPRpred(CNN) architecture: parallel
    convolutional branches over different receptive fields, with
    concatenated features fed to dense layers.
    """

    def __init__(
        self,
        input_length: int = 30,
        n_nucleotides: int = 4,
        conv_n_filters: Any = 64,
        conv_kernel_sizes: List[int] = [5, 7, 9],
        dense_units: int = 64,
        dropout_rate: float = 0.3,
        n_classes: int = 1
    ):
        """
        Initialize the CNN.
        
        Args:
            input_length: Length of input sequence (30)
            n_nucleotides: Alphabet size (4: A, C, G, T)
            conv_n_filters: Number of filters per branch. Either an int
                (broadcast to every branch) or a list of length equal to
                len(conv_kernel_sizes).
            conv_kernel_sizes: Kernel size for each parallel branch
            dense_units: Number of units in dense layer
            dropout_rate: Dropout probability
            n_classes: Number of output units (1 for regression)
        """
        super().__init__()

        if isinstance(conv_n_filters, int):
            conv_n_filters = [conv_n_filters] * len(conv_kernel_sizes)
        if len(conv_n_filters) != len(conv_kernel_sizes):
            raise ValueError(
                "len(conv_n_filters) must equal len(conv_kernel_sizes)"
            )

        # Parallel convolutional branches, one per receptive field.
        # Each branch: Conv1d -> ReLU -> MaxPool1d -> AdaptiveAvgPool1d
        self.conv_branches = nn.ModuleList()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        n_conv_outputs = 0
        for k, n_filters in zip(conv_kernel_sizes, conv_n_filters):
            n_conv_outputs += n_filters
            self.conv_branches.append(
                nn.Sequential(
                    nn.Conv1d(n_nucleotides, n_filters, kernel_size=k,
                              padding=k // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )

        # Dense layers
        self.dense1 = nn.Linear(n_conv_outputs, dense_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(dense_units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_nucleotides, input_length)
               or (batch, input_length, n_nucleotides) which is transposed.
               
        Returns:
            Tensor of shape (batch, 1)
        """
        # Ensure channel-first layout: (batch, 4, length)
        if x.dim() == 3 and x.shape[1] != 4:
            x = x.permute(0, 2, 1)

        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(x)                 # (batch, filters, L/2)
            out = self.global_pool(out)     # (batch, filters, 1)
            branch_outputs.append(out.squeeze(-1))  # (batch, filters)

        # Concatenate along feature dim
        conv_out = torch.cat(branch_outputs, dim=1)  # (batch, 3*filters)

        dense = self.dropout(self.relu(self.dense1(conv_out)))
        return self.output(dense)


class CNNModel:
    """
    CNN model wrapper for sgRNA activity prediction.
    """

    def __init__(
        self,
        input_length: int = 30,
        n_nucleotides: int = 4,
        conv_n_filters: Any = 64,
        conv_kernel_sizes: List[int] = [5, 7, 9],
        dense_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        optimizer: str = 'adam',
        loss: str = 'mse',
        random_state: int = 42,
        use_cpu_threads: int = 4
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_length: Input sequence length (30)
            n_nucleotides: Alphabet size (4)
            conv_n_filters: Filters per branch
            conv_kernel_sizes: Kernels per branch
            dense_units: Dense layer units
            dropout_rate: Dropout probability
            learning_rate: Optimizer learning rate
            batch_size: Training batch size
            epochs: Maximum number of epochs
            patience: Early stopping patience (on validation loss)
            optimizer: Optimizer name
            loss: Loss name
            random_state: Random seed
            use_cpu_threads: Number of CPU threads
        """
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if use_cpu_threads > 0:
            torch.set_num_threads(use_cpu_threads)

        self.input_length = input_length
        self.n_nucleotides = n_nucleotides
        self.conv_n_filters = conv_n_filters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.optimizer_name = optimizer
        self.loss_name = loss
        self.random_state = random_state

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.network = CRISPRsvGN(
            input_length=input_length,
            n_nucleotides=n_nucleotides,
            conv_n_filters=conv_n_filters,
            conv_kernel_sizes=conv_kernel_sizes,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            n_classes=1
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.training_history = {}
        self.is_fitted = False

    def _to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert numpy arrays to tensors with correct layout.
        
        Input X shape: (n, input_length, 4) -> transpose to (n, 4, input_length).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3 and X.shape[-1] == self.n_nucleotides:
            X = X.transpose(0, 2, 1)  # (n, 4, length)
        Xt = torch.from_numpy(X).to(self.device)
        yt = None
        if y is not None:
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            yt = torch.from_numpy(y).to(self.device)
        return Xt, yt

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the CNN with early stopping on validation loss.
        
        Args:
            X_train: One-hot encoded training features (n, 30, 4)
            y_train: Training targets
            X_val: One-hot encoded validation features
            y_val: Validation targets
            verbose: Print progress
            
        Returns:
            Dictionary with training history
        """
        start_time = time.time()

        Xt, yt = self._to_tensor(X_train, y_train)
        train_dataset = TensorDataset(Xt, yt)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            Xvt, yvt = self._to_tensor(X_val, y_val)
            val_dataset = TensorDataset(Xvt, yvt)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                pred = self.network(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)

            # Validation loss
            avg_val_loss = None
            if val_loader is not None:
                self.network.eval()
                val_loss_sum = 0.0
                v_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred = self.network(xb)
                        loss = self.criterion(pred, yb)
                        val_loss_sum += loss.item()
                        v_batches += 1
                avg_val_loss = val_loss_sum / v_batches
                history['val_loss'].append(avg_val_loss)

            if verbose:
                val_str = f" | val_loss: {avg_val_loss:.4f}" if avg_val_loss is not None else ""
                print(f"Epoch {epoch}/{self.epochs} | train_loss: {avg_train_loss:.4f}{val_str}")

            # Early stopping on validation loss (main model)
            if val_loader is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} (patience {self.patience})")
                        break

        # Restore best model (by validation loss)
        if best_state is not None:
            self.network.load_state_dict(best_state)

        self.best_val_loss = best_val_loss
        self.training_history = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': best_val_loss,
            'best_epoch': len(history['val_loss']) if history['val_loss'] else len(history['train_loss']),
            'total_epochs_run': len(history['train_loss']),
            'early_stopping': val_loader is not None,
            'training_time': time.time() - start_time
        }
        self.is_fitted = True

        logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sgRNA activity.
        
        Args:
            X: One-hot encoded features (n, 30, 4)
            
        Returns:
            Predicted values (n,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        self.network.eval()
        Xt, _ = self._to_tensor(X)
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xt), self.batch_size):
                batch = Xt[i:i + self.batch_size]
                preds.append(self.network(batch).cpu().numpy())
        return np.concatenate(preds).ravel()

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        torch.save(
            {
                'state_dict': self.network.state_dict(),
                'hyperparameters': self.get_params(),
                'training_history': self.training_history
            },
            filepath
        )
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'CNNModel':
        """
        Load a trained model.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded CNNModel instance
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        hp = checkpoint['hyperparameters']
        instance = cls(**hp)
        instance.network.load_state_dict(checkpoint['state_dict'])
        instance.training_history = checkpoint['training_history']
        instance.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'input_length': self.input_length,
            'n_nucleotides': self.n_nucleotides,
            'conv_n_filters': self.conv_n_filters,
            'conv_kernel_sizes': self.conv_kernel_sizes,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'optimizer': self.optimizer_name,
            'loss': self.loss_name,
            'random_state': self.random_state
        }
